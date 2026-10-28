"""Heavy work is bounded without holding threads; questions and health checks never wait for
it (spec: http-api, "Heavy work is bounded").

The HTTP tests send real concurrent requests (one client thread each through the
``TestClient``) to an application whose fakes block on events, so every request is in flight
at a known moment.
"""

import threading
import time
from collections.abc import Callable

import anyio
import anyio.to_thread
import httpx2
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import JobLimiter
from app.domain.cancellation import NEVER_CANCELLED, Cancellation
from app.domain.errors import DependencyUnavailable, ProcessingFailed, ServiceError
from app.domain.types import OcrPage
from tests.api.conftest import ClientFactory, LogLines
from tests.api.support import (
    BLOCK_S,
    CONTRACT_PAGES,
    PDF,
    PNG,
    BlockingLLM,
    BlockingOcr,
    Fakes,
    RecordedSpool,
)

HEAVY_REQUESTS = 70
"""More than AnyIO's 40 default worker threads, in which the old limiter waited."""

BUSY_WAIT_S = 1.0
TOLERANCE_S = 0.5
"""Slack for scheduling 70 client threads; well below a second ``BUSY_WAIT_S``."""

QUICK_S = 0.5
"""A health check or an unhindered request answers within this."""

QUESTION = "who signed the contract with the notary"

QUESTIONS = 4
"""``MAX_CONCURRENT_QUESTIONS`` of the question tests."""

type Timed = tuple[httpx2.Response, float]


def _pdf_form(data: bytes = PDF) -> dict[str, tuple[str, bytes, str]]:
    return {"pdf_file": ("a.pdf", data, "application/pdf")}


def _timed(call: Callable[[], httpx2.Response]) -> Timed:
    started = time.perf_counter()
    response = call()
    return response, time.perf_counter() - started


def _until(condition: Callable[[], bool]) -> bool:
    """Poll ``condition`` until it holds; ``False`` after ``BLOCK_S``."""
    deadline = time.monotonic() + BLOCK_S
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


class _Calls:
    """Runs calls in client threads and collects their timed responses."""

    def __init__(self) -> None:
        self.results: list[Timed] = []
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    def start(self, call: Callable[[], httpx2.Response]) -> None:
        def run() -> None:
            timed = _timed(call)
            with self._lock:
                self.results.append(timed)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self._threads.append(thread)

    def join(self) -> list[Timed]:
        for thread in self._threads:
            thread.join(BLOCK_S)
        assert len(self.results) == len(self._threads), "a request did not finish"
        return self.results


def _limiter(client: TestClient) -> JobLimiter:
    limiter: JobLimiter = client.app.state.limiter  # type: ignore[attr-defined]
    return limiter


# --- the limiter itself ---------------------------------------------------------------------


class _Hold:
    """A job that blocks until ``release`` is set (whatever its ``cancel``); keeps its token."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancel: Cancellation | None = None

    def __call__(self, cancel: Cancellation) -> str:
        self.cancel = cancel
        self.entered.set()
        assert self.release.wait(BLOCK_S)
        return "held"

    async def wait_entered(self) -> None:
        assert await anyio.to_thread.run_sync(self.entered.wait, BLOCK_S)


def test_busy_request_takes_and_releases_no_slot() -> None:
    limiter = JobLimiter(1, 0.05, max_questions=1)
    hold = _Hold()
    results: dict[str, str] = {}

    async def holder() -> None:
        results["holder"] = await limiter.run_job(hold)

    async def scenario() -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(holder)
            await hold.wait_entered()
            with pytest.raises(DependencyUnavailable) as busy:
                await limiter.run_job(lambda _: "never")
            results["busy"] = busy.value.code
            assert limiter.free_slots == 0  # the holder's slot is still held
            assert limiter.waiting_jobs == 0
            hold.release.set()

    anyio.run(scenario)
    assert results == {"holder": "held", "busy": "service_busy"}
    assert limiter.free_slots == 1


def test_free_slot_is_taken_without_waiting_even_with_no_wait() -> None:
    limiter = JobLimiter(1, 0.0, max_questions=1)

    assert anyio.run(limiter.run_job, lambda _: "done") == "done"
    assert limiter.free_slots == 1


def test_cancelled_job_frees_its_slot_only_when_its_thread_returns() -> None:
    limiter = JobLimiter(1, 0.0, max_questions=1)
    hold = _Hold()

    async def scenario() -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(limiter.run_job, hold)
            await hold.wait_entered()
            group.cancel_scope.cancel()  # like a disconnect or a shutdown
            with anyio.CancelScope(shield=True):
                await anyio.sleep(0.05)
                assert limiter.free_slots == 0  # the thread still runs the job
                hold.release.set()

    anyio.run(scenario)
    assert limiter.free_slots == 1


def test_waiter_cancelled_before_a_slot_frees_takes_none() -> None:
    limiter = JobLimiter(1, BLOCK_S, max_questions=1)
    hold = _Hold()

    async def scenario() -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(limiter.run_job, hold)
            await hold.wait_entered()
            with anyio.move_on_after(0.05):
                await limiter.run_job(lambda _: "never")
            assert limiter.waiting_jobs == 0
            assert limiter.free_slots == 0
            hold.release.set()

    anyio.run(scenario)
    assert limiter.free_slots == 1


# --- over HTTP ------------------------------------------------------------------------------


def test_saturated_heavy_requests_are_busy_after_the_wait_and_health_answers_at_once(
    make_client: ClientFactory, fakes: Fakes
) -> None:
    ocr = BlockingOcr()
    fakes.ocr = ocr
    client = make_client(max_concurrent_jobs=1, busy_wait_s=BUSY_WAIT_S)
    limiter = _limiter(client)
    holder, others = _Calls(), _Calls()
    try:
        holder.start(lambda: client.post("/api/extract", files=_pdf_form()))
        assert ocr.entered.wait(BLOCK_S)
        for index in range(HEAVY_REQUESTS - 1):
            path = "/api/extract" if index % 2 else "/api/ingest"
            others.start(lambda path=path: client.post(path, files=_pdf_form()))  # type: ignore[misc]
        assert _until(lambda: limiter.waiting_jobs == HEAVY_REQUESTS - 1)
        ready, ready_s = _timed(lambda: client.get("/health/ready"))
        alive, alive_s = _timed(lambda: client.get("/health"))
        rejected = others.join()
    finally:
        ocr.release.set()
    ((held, _),) = holder.join()

    assert (ready.status_code, alive.status_code) == (200, 200)
    assert ready_s < QUICK_S
    assert alive_s < QUICK_S
    assert held.status_code == 200
    assert {response.status_code for response, _ in rejected} == {503}
    assert {response.json()["code"] for response, _ in rejected} == {"service_busy"}
    slowest = max(seconds for _, seconds in rejected)
    assert slowest < BUSY_WAIT_S + TOLERANCE_S, f"a busy answer took {slowest:.2f} s"
    assert ocr.calls == 1  # no rejected request reached the OCR
    assert limiter.free_slots == 1


def test_slow_model_answers_delay_neither_readiness_nor_jobs(
    make_client: ClientFactory, fakes: Fakes
) -> None:
    llm = BlockingLLM()
    fakes.llm = llm
    client = make_client(
        max_concurrent_jobs=1, max_concurrent_questions=QUESTIONS, busy_wait_s=BLOCK_S
    )
    assert client.post("/api/ingest", files=_pdf_form()).status_code == 200
    questions = _Calls()
    try:
        for _ in range(QUESTIONS + 30):
            questions.start(lambda: client.post("/api/ask", json={"question": QUESTION}))
        assert llm.waiting(QUESTIONS)
        assert _until(lambda: _limiter(client).waiting_questions == 30)
        ready, ready_s = _timed(lambda: client.get("/health/ready"))
        extracted, extract_s = _timed(lambda: client.post("/api/extract", files=_pdf_form()))
    finally:
        llm.release.set()
    answers = questions.join()

    assert ready.status_code == 200
    assert ready_s < QUICK_S
    assert extracted.status_code == 200
    assert extract_s < QUICK_S
    assert {answer.status_code for answer, _ in answers} == {200}
    assert llm.most_in_flight == QUESTIONS  # the others waited without a thread
    assert _limiter(client).free_question_slots == QUESTIONS


def test_questions_over_their_limit_are_busy_after_the_wait(
    make_client: ClientFactory, fakes: Fakes
) -> None:
    llm = BlockingLLM()
    fakes.llm = llm
    client = make_client(
        max_concurrent_jobs=1, max_concurrent_questions=QUESTIONS, busy_wait_s=BUSY_WAIT_S
    )
    assert client.post("/api/ingest", files=_pdf_form()).status_code == 200
    held, extra = _Calls(), _Calls()
    try:
        for _ in range(QUESTIONS):
            held.start(lambda: client.post("/api/ask", json={"question": QUESTION}))
        assert llm.waiting(QUESTIONS)
        for _ in range(20):
            extra.start(lambda: client.post("/api/ask", json={"question": QUESTION}))
        extracted, extract_s = _timed(lambda: client.post("/api/extract", files=_pdf_form()))
        rejected = extra.join()
    finally:
        llm.release.set()
    answered = held.join()

    assert {response.status_code for response, _ in answered} == {200}
    assert {response.status_code for response, _ in rejected} == {503}
    assert {response.json()["code"] for response, _ in rejected} == {"service_busy"}
    assert {response.headers["Retry-After"] for response, _ in rejected} == {"1"}
    fastest = min(seconds for _, seconds in rejected)
    slowest = max(seconds for _, seconds in rejected)
    assert fastest >= BUSY_WAIT_S - 0.05  # they waited for a question slot first
    assert slowest < BUSY_WAIT_S + TOLERANCE_S, f"a busy answer took {slowest:.2f} s"
    assert llm.most_in_flight == QUESTIONS
    assert extracted.status_code == 200  # jobs do not wait for question slots
    assert extract_s < QUICK_S
    limiter = _limiter(client)
    assert (limiter.free_question_slots, limiter.waiting_questions) == (QUESTIONS, 0)


def test_question_and_job_slots_are_separate() -> None:
    limiter = JobLimiter(1, 0.05, max_questions=1)
    hold = _Hold()

    def late() -> str:
        assert hold.release.wait(BLOCK_S)
        return "late"

    async def scenario() -> list[str]:
        results: list[str] = []
        async with anyio.create_task_group() as group:
            group.start_soon(limiter.run_job, hold)
            await hold.wait_entered()
            results.append(await limiter.run_question(lambda: "answered"))  # no job slot needed

            async def held_question() -> None:
                results.append(await limiter.run_question(late))

            group.start_soon(held_question)
            while limiter.free_question_slots:
                await anyio.sleep(0.001)
            with pytest.raises(DependencyUnavailable) as busy:
                await limiter.run_question(lambda: "never")
            results.append(busy.value.code)
            assert limiter.free_slots == 0
            hold.release.set()
        return results

    assert anyio.run(scenario) == ["answered", "service_busy", "late"]
    assert (limiter.free_slots, limiter.free_question_slots) == (1, 1)


class _SwitchableOcr:
    """Raises ``error`` when it is set, else returns the contract pages."""

    def __init__(self) -> None:
        self.error: ServiceError | None = None

    def recognize(
        self, pdf: bytes, language: str | None = None, *, cancel: Cancellation = NEVER_CANCELLED
    ) -> list[OcrPage]:
        if self.error is not None:
            raise self.error
        return list(CONTRACT_PAGES)


def test_slot_is_freed_after_failing_jobs(make_client: ClientFactory, fakes: Fakes) -> None:
    ocr = _SwitchableOcr()
    fakes.ocr = ocr
    client = make_client(max_concurrent_jobs=1, busy_wait_s=0.1)

    statuses = [
        client.post("/api/extract", files=_pdf_form(PNG)).status_code,
        client.post("/api/ingest", files=_pdf_form(b"")).status_code,
    ]
    ocr.error = ProcessingFailed("page 2 could not be read", code="ocr_failed")
    statuses.append(client.post("/api/extract", files=_pdf_form()).status_code)
    statuses.append(client.post("/api/ingest", files=_pdf_form()).status_code)
    ocr.error = None
    statuses.append(client.post("/api/extract", files=_pdf_form()).status_code)

    assert statuses == [415, 400, 500, 500, 200]
    assert _limiter(client).free_slots == 1


def test_busy_request_closes_its_upload(
    make_client: ClientFactory, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    ocr = BlockingOcr()
    fakes.ocr = ocr
    client = make_client(max_concurrent_jobs=1, busy_wait_s=0.1)
    holder = _Calls()
    try:
        holder.start(lambda: client.post("/api/extract", files=_pdf_form()))
        assert ocr.entered.wait(BLOCK_S)
        RecordedSpool.instances = []
        monkeypatch.setattr("starlette.formparsers.SpooledTemporaryFile", RecordedSpool)
        busy = client.post("/api/ingest", files=_pdf_form())
    finally:
        ocr.release.set()
    ((held, _),) = holder.join()

    assert held.status_code == 200
    assert busy.status_code == 503
    assert len(RecordedSpool.instances) == 1
    assert RecordedSpool.instances[0].closed


def test_hanging_readiness_check_is_unavailable_after_its_timeout(
    make_client: ClientFactory, fakes: Fakes, logs: LogLines
) -> None:
    release = threading.Event()

    def hang() -> str:
        release.wait(BLOCK_S)
        return "ok"

    fakes.readiness = (("vector_store", hang), ("llm", lambda: "configured"))
    client = make_client(readiness_timeout_s=0.2)
    try:
        ready, ready_s = _timed(lambda: client.get("/health/ready"))
        alive, alive_s = _timed(lambda: client.get("/health"))
    finally:
        release.set()

    assert ready.status_code == 503
    assert ready.json() == {
        "status": "not_ready",
        "checks": {"vector_store": "unavailable", "llm": "configured"},
    }
    assert 0.2 <= ready_s < 0.2 + QUICK_S
    assert alive.status_code == 200
    assert alive_s < QUICK_S
    (failed,) = logs.with_message("readiness check failed")
    assert failed["check"] == "vector_store"
    assert failed["code"] == "readiness_timeout"
