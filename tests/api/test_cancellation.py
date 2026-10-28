"""A client that disconnects cancels its job (spec: http-api, "Abandoned jobs stop early").

The limiter tests drive ``JobLimiter.run_job`` with a scripted ``receive``. The HTTP tests call
the application as an ASGI server would, on the event loop of its ``TestClient``, with a
``receive`` that delivers the form and then reports ``http.disconnect`` at a chosen moment (the
``TestClient`` itself never disconnects). ``test_uvicorn_edge`` repeats this with real Uvicorn.
"""

import threading
import time
from collections.abc import Callable

import anyio
import anyio.to_thread
import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Scope

from app.api.dependencies import BUSY_MESSAGE, JobLimiter
from app.domain.cancellation import Cancellation
from app.domain.errors import DependencyUnavailable, ProcessingFailed
from tests.api.conftest import ClientFactory, LogLines
from tests.api.support import BLOCK_S, PDF, BlockingOcr, Fakes, PagedOcr, multipart

QUICK_S = 0.5
"""A request whose client left ends within this once it is told."""


def _until(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + BLOCK_S
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


class _Client:
    """A scripted ASGI ``receive``: optional messages, then ``http.disconnect`` once ``leave()``
    is called (it blocks until then, as a server's does once the body has been read)."""

    def __init__(self, *messages: Message) -> None:
        self._messages = list(messages)
        self._gone = threading.Event()
        self.calls = 0

    def leave(self) -> None:
        self._gone.set()

    async def receive(self) -> Message:
        self.calls += 1
        if self._messages:
            return self._messages.pop(0)
        while not self._gone.is_set():  # polled: a cancelled wait leaves no thread behind
            await anyio.sleep(0.005)
        return {"type": "http.disconnect"}


class _Hold:
    """A job that blocks until ``release`` is set, ignoring its token (an uninterruptible
    step); records the token it got."""

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


# --- the limiter ----------------------------------------------------------------------------


def test_disconnect_while_waiting_gives_up_the_wait_without_a_slot() -> None:
    limiter = JobLimiter(1, BLOCK_S, max_questions=1)
    hold = _Hold()
    client = _Client()
    ran: list[str] = []

    async def scenario() -> float:
        async with anyio.create_task_group() as group:
            group.start_soon(limiter.run_job, hold)
            await hold.wait_entered()
            begin = time.perf_counter()
            group.start_soon(_leave_when, client, lambda: limiter.waiting_jobs == 1)
            with pytest.raises(ProcessingFailed) as caught:
                await limiter.run_job(lambda _: ran.append("job"), client.receive)
            took = time.perf_counter() - begin
            assert caught.value.code == "request_cancelled"
            assert limiter.waiting_jobs == 0
            assert limiter.free_slots == 0  # still the holder's
            hold.release.set()
        return took

    took = anyio.run(scenario)
    assert took < QUICK_S
    assert ran == []
    assert limiter.free_slots == 1


def test_disconnect_cancels_the_running_job_whose_slot_frees_when_it_returns() -> None:
    limiter = JobLimiter(1, 0.0, max_questions=1)
    hold = _Hold()
    client = _Client()

    async def scenario() -> None:
        async with anyio.create_task_group() as group:

            async def leave_then_release() -> None:
                await hold.wait_entered()
                client.leave()
                assert await anyio.to_thread.run_sync(
                    _until, lambda: hold.cancel is not None and hold.cancel.cancelled
                )
                assert limiter.free_slots == 0  # the thread still runs
                hold.release.set()

            group.start_soon(leave_then_release)
            with pytest.raises(ProcessingFailed) as caught:
                # The job returns normally, but its client is gone: nobody gets the answer.
                await limiter.run_job(hold, client.receive)
            assert caught.value.code == "request_cancelled"

    anyio.run(scenario)
    assert limiter.free_slots == 1


def test_without_a_disconnect_the_watch_ends_with_the_job() -> None:
    limiter = JobLimiter(1, 0.0, max_questions=1)
    client = _Client({"type": "http.request", "body": b"", "more_body": False})
    tokens: list[Cancellation] = []

    def job(cancel: Cancellation) -> str:
        tokens.append(cancel)
        return "done"

    assert anyio.run(limiter.run_job, job, client.receive) == "done"
    (token,) = tokens
    assert not token.cancelled
    assert client.calls == 2  # an empty body message is skipped; the next wait was stopped
    assert limiter.free_slots == 1


def test_a_failing_job_keeps_its_error_and_frees_its_slot() -> None:
    limiter = JobLimiter(1, 0.0, max_questions=1)

    def job(cancel: Cancellation) -> str:
        raise ProcessingFailed("page 2 could not be read", code="ocr_failed")

    with pytest.raises(ProcessingFailed) as caught:
        anyio.run(limiter.run_job, job, _Client().receive)
    assert caught.value.code == "ocr_failed"
    assert limiter.free_slots == 1


class _Polling:
    """A job that checks its token every few milliseconds for up to ``BLOCK_S`` seconds, as a
    use case does between pages; records whether it saw the token cancelled."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.saw_cancel = False

    def __call__(self, cancel: Cancellation) -> str:
        self.entered.set()
        deadline = time.monotonic() + BLOCK_S
        while time.monotonic() < deadline:
            if cancel.cancelled:
                self.saw_cancel = True
                return "stopped"
            time.sleep(0.005)
        return "ran to the end"


@pytest.mark.parametrize("watched", [True, False], ids=["with-receive", "without-receive"])
def test_cancelling_the_request_task_cancels_the_job(watched: bool) -> None:
    # The server cancels request tasks at shutdown, once its graceful timeout is over: the
    # job's thread must return at its next check instead of holding the shutdown.
    limiter = JobLimiter(1, 0.0, max_questions=1)
    job = _Polling()
    receive = _Client().receive if watched else None

    async def scenario() -> float:
        begin = time.perf_counter()
        with anyio.move_on_after(0.1) as outer:
            await limiter.run_job(job, receive)
        assert outer.cancelled_caught  # the caller gets its cancellation, not a result
        return time.perf_counter() - begin

    took = anyio.run(scenario)
    assert job.entered.is_set()
    assert job.saw_cancel
    assert took < QUICK_S
    assert limiter.free_slots == 1


def test_a_receive_that_raises_is_taken_as_a_disconnect() -> None:
    limiter = JobLimiter(1, 0.0, max_questions=1)
    job = _Polling()

    async def broken() -> Message:
        assert await anyio.to_thread.run_sync(job.entered.wait, BLOCK_S)
        raise RuntimeError("the transport failed")

    with pytest.raises(ProcessingFailed) as caught:  # not an ExceptionGroup, not a 500
        anyio.run(limiter.run_job, job, broken)
    assert caught.value.code == "request_cancelled"
    assert job.saw_cancel
    assert limiter.free_slots == 1


def test_busy_wait_ending_after_the_client_left_is_a_cancellation() -> None:
    # The wait for a slot times out as the client leaves: nobody can read the 503.
    limiter = JobLimiter(1, 0.0, max_questions=1)
    client = _Client()
    ran: list[str] = []

    async def take_then_time_out(wait_s: float) -> None:
        with anyio.CancelScope(shield=True):  # the timeout fires before the watcher's cancel
            client.leave()
            await anyio.sleep(0.05)  # the watcher sees the disconnect meanwhile
        raise DependencyUnavailable(BUSY_MESSAGE, code="service_busy")

    limiter._jobs.take = take_then_time_out  # type: ignore[method-assign]
    with pytest.raises(ProcessingFailed) as caught:
        anyio.run(limiter.run_job, lambda _: ran.append("job"), client.receive)
    assert caught.value.code == "request_cancelled"
    assert ran == []


def test_busy_wait_without_a_disconnect_stays_busy() -> None:
    limiter = JobLimiter(1, 0.0, max_questions=1)
    hold = _Hold()

    async def scenario() -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(limiter.run_job, hold)
            await hold.wait_entered()
            with pytest.raises(DependencyUnavailable) as caught:
                await limiter.run_job(lambda _: "never", _Client().receive)
            assert caught.value.code == "service_busy"
            hold.release.set()

    anyio.run(scenario)
    assert hold.cancel is not None
    assert not hold.cancel.cancelled
    assert limiter.free_slots == 1


async def _leave_when(client: _Client, condition: Callable[[], bool]) -> None:
    assert await anyio.to_thread.run_sync(_until, condition)
    client.leave()


# --- over HTTP ------------------------------------------------------------------------------


def _scope(path: str, body: bytes, content_type: str, request_id: str) -> Scope:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", content_type.encode()),
            (b"content-length", str(len(body)).encode()),
            (b"x-request-id", request_id.encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }


def _post_then_leave(
    client: TestClient, path: str, request_id: str, leave_when: Callable[[], bool]
) -> tuple[list[Message], float]:
    """Post a PDF to ``path`` and disconnect once ``leave_when()`` holds; what the application
    sent, and how long it took to end after the disconnect."""
    body, content_type = multipart(PDF)
    peer = _Client({"type": "http.request", "body": body, "more_body": False})
    sent: list[Message] = []
    left: list[float] = []

    async def send(message: Message) -> None:
        sent.append(message)

    async def leave() -> None:
        assert await anyio.to_thread.run_sync(_until, leave_when)
        left.append(time.perf_counter())
        peer.leave()

    async def scenario() -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(leave)
            await client.app(_scope(path, body, content_type, request_id), peer.receive, send)

    assert client.portal is not None
    client.portal.call(scenario)
    return sent, time.perf_counter() - left[0]


@pytest.mark.parametrize("path", ["/api/extract", "/api/ingest"])
def test_client_leaving_stops_the_job_after_the_page_in_progress(
    make_client: ClientFactory, fakes: Fakes, logs: LogLines, path: str
) -> None:
    ocr = PagedOcr(pages=10, page_s=0.05)
    fakes.ocr = ocr
    client = make_client(max_concurrent_jobs=1)
    logs.clear()

    sent, took = _post_then_leave(client, path, "gone-1", lambda: ocr.recognised >= 2)

    assert took < QUICK_S
    assert ocr.finished.is_set()
    assert 2 <= ocr.recognised <= 3  # the page in progress finishes; no later one starts
    assert sent[0]["status"] == 499  # sent nowhere: the client is gone
    limiter: JobLimiter = client.app.state.limiter  # type: ignore[attr-defined]
    assert limiter.free_slots == 1
    (failed,) = logs.with_message("request failed")
    assert (failed["code"], failed["status"], failed["request_id"]) == (
        "request_cancelled",
        499,
        "gone-1",
    )
    (access,) = logs.with_message("request")
    assert (access["route"], access["status"]) == (path, 499)
    assert not logs.with_message("unexpected error")
    assert client.post(path, files={"pdf_file": ("a.pdf", PDF, "application/pdf")}).is_success


def test_client_leaving_while_waiting_for_a_slot_takes_none(
    make_client: ClientFactory, fakes: Fakes, logs: LogLines
) -> None:
    ocr = BlockingOcr()
    fakes.ocr = ocr
    client = make_client(max_concurrent_jobs=1, busy_wait_s=BLOCK_S)
    limiter: JobLimiter = client.app.state.limiter  # type: ignore[attr-defined]
    held: list[int] = []
    holder = threading.Thread(
        target=lambda: held.append(
            client.post(
                "/api/extract", files={"pdf_file": ("a.pdf", PDF, "application/pdf")}
            ).status_code
        )
    )
    holder.start()
    try:
        assert ocr.entered.wait(BLOCK_S)
        logs.clear()
        sent, took = _post_then_leave(
            client, "/api/ingest", "gone-2", lambda: limiter.waiting_jobs == 1
        )
        assert limiter.waiting_jobs == 0
        assert limiter.free_slots == 0  # the holder's
    finally:
        ocr.release.set()
        holder.join(BLOCK_S)

    assert took < QUICK_S
    assert sent[0]["status"] == 499
    assert ocr.calls == 1  # the request that left never reached the OCR
    assert held == [200]
    assert limiter.free_slots == 1
    (access,) = [line for line in logs.with_message("request") if line["request_id"] == "gone-2"]
    assert access["status"] == 499
