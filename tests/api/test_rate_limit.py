"""Per-client rate limits and job caps at the HTTP edge (spec: client-rate-limiting; http-api,
"Rate limits are reported like other errors").

The HTTP tests run the application over fakes with a fake clock (``create_app(clock=...)``), so
a refill is a matter of moving time, not of waiting; a second address is a second test client
over the same running application. The raw ASGI tests exercise the middleware alone, as the
body limit's do. One test drives a real Uvicorn process, for the burst over real connections.
"""

import asyncio
import threading
import time
from collections.abc import Callable, Iterator

import httpx2
import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from app.api.rate_limit import (
    CAP_MESSAGE,
    PRUNE_INTERVAL_S,
    RATE_MESSAGE,
    RateLimitMiddleware,
    RouteLimits,
)
from app.main import create_app
from tests.api.conftest import LogLines
from tests.api.served_app import PAGE_LINE
from tests.api.support import BLOCK_S, PDF, BlockingOcr, Fakes, api_settings
from tests.api.uvicorn_server import served
from tests.fakes.ocr import FakeOcrEngine

ADDRESS = "10.0.0.1"
OTHER_ADDRESS = "10.0.0.2"
QUESTION = "who signed the contract"
QUICK_S = 0.5
"""A request refused at once answers within this."""


class FakeClock:
    """Monotonic seconds that move only when a test says so."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


type Limited = Callable[..., TestClient]


@pytest.fixture
def make_limited(fakes: Fakes, clock: FakeClock) -> Iterator[Limited]:
    """Start an application over ``fakes`` and ``clock`` with settings from
    ``api_settings(**overrides)``; its client connects from ``ADDRESS``."""
    clients: list[TestClient] = []

    def make(**overrides: object) -> TestClient:
        active = api_settings(**overrides)
        app = create_app(settings=active, container=fakes.container(active), clock=clock)
        client = TestClient(app, client=(ADDRESS, 50000))
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def _other_client(client: TestClient) -> TestClient:
    """A client of ``OTHER_ADDRESS`` over the same running application (no second lifespan)."""
    return TestClient(client.app, client=(OTHER_ADDRESS, 50000))


def _upload(client: TestClient, path: str = "/api/extract", **headers: str) -> httpx2.Response:
    return client.post(path, files={"pdf_file": ("a.pdf", PDF, "application/pdf")}, headers=headers)


def _ask(client: TestClient) -> httpx2.Response:
    return client.post("/api/ask", json={"question": QUESTION})


def _ocr_calls(fakes: Fakes) -> int:
    assert isinstance(fakes.ocr, FakeOcrEngine)
    return len(fakes.ocr.calls)


def _until(condition: Callable[[], bool]) -> bool:
    """Poll ``condition`` until it holds; ``False`` after ``BLOCK_S``."""
    deadline = time.monotonic() + BLOCK_S
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


class _InThreads:
    """Runs calls in client threads and collects their responses in the order started."""

    def __init__(self) -> None:
        self._threads: list[threading.Thread] = []
        self._responses: dict[int, httpx2.Response] = {}

    def start(self, call: Callable[[], httpx2.Response]) -> None:
        index = len(self._threads)

        def run() -> None:
            self._responses[index] = call()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self._threads.append(thread)

    def join(self) -> list[httpx2.Response]:
        for thread in self._threads:
            thread.join(BLOCK_S)
        assert len(self._responses) == len(self._threads), "a request did not finish"
        return [self._responses[index] for index in range(len(self._threads))]


# --- over HTTP ------------------------------------------------------------------------------


def test_requests_over_the_burst_are_429_with_retry_after_and_run_no_ocr(
    make_limited: Limited, fakes: Fakes
) -> None:
    client = make_limited(rate_limit_uploads_per_minute=60, rate_limit_uploads_burst=3)

    admitted = [_upload(client) for _ in range(3)]
    refused = [_upload(client, **{"X-Request-ID": f"over-{n}"}) for n in range(2)]

    assert [response.status_code for response in admitted] == [200, 200, 200]
    assert [response.status_code for response in refused] == [429, 429]
    assert refused[0].json() == {
        "detail": RATE_MESSAGE,
        "code": "rate_limited",
        "request_id": "over-0",
    }
    assert refused[0].headers["Retry-After"] == "1"  # a token refills every second at 60/min
    assert _ocr_calls(fakes) == 3  # the refused requests never reached the use case


def test_a_limited_request_is_logged_once_with_its_code_and_in_the_access_line(
    make_limited: Limited, logs: LogLines
) -> None:
    client = make_limited(rate_limit_uploads_per_minute=60, rate_limit_uploads_burst=1)
    assert _upload(client).status_code == 200
    logs.clear()

    response = _upload(client, **{"X-Request-ID": "limited-1"})

    assert response.status_code == 429
    assert response.headers["X-Request-ID"] == "limited-1"
    assert int(response.headers["Retry-After"]) >= 1
    (failed,) = logs.with_message("request failed")
    assert (failed["code"], failed["category"], failed["status"]) == (
        "rate_limited",
        "invalid_input",
        429,
    )
    assert failed["request_id"] == "limited-1"
    (access,) = logs.with_message("request")
    assert (access["method"], access["status"], access["request_id"]) == (
        "POST",
        429,
        "limited-1",
    )


def test_tokens_refill_with_time_up_to_the_burst(make_limited: Limited, clock: FakeClock) -> None:
    client = make_limited(rate_limit_uploads_per_minute=60, rate_limit_uploads_burst=3)
    assert [_upload(client).status_code for _ in range(4)] == [200, 200, 200, 429]

    clock.advance(1.0)  # one token at 60 per minute
    assert [_upload(client).status_code for _ in range(2)] == [200, 429]

    clock.advance(600.0)  # full again, and no fuller than the burst
    assert [_upload(client).status_code for _ in range(4)] == [200, 200, 200, 429]


def test_retry_after_is_the_wait_for_one_token_rounded_up(
    make_limited: Limited, clock: FakeClock
) -> None:
    # A token every 10 s.
    client = make_limited(rate_limit_uploads_per_minute=6, rate_limit_uploads_burst=1)
    assert _upload(client).status_code == 200

    assert _upload(client).headers["Retry-After"] == "10"
    clock.advance(4.5)
    assert _upload(client).headers["Retry-After"] == "6"  # 5.5 s to go, rounded up
    clock.advance(5.5)
    assert _upload(client).status_code == 200


def test_route_classes_are_independent(make_limited: Limited) -> None:
    client = make_limited(
        rate_limit_uploads_per_minute=1,
        rate_limit_uploads_burst=1,
        rate_limit_questions_per_minute=1,
        rate_limit_questions_burst=1,
    )

    assert _ask(client).status_code == 200
    assert _ask(client).status_code == 429  # the question allowance is spent ...
    assert _upload(client).status_code == 200  # ... and an extraction is still accepted
    assert _upload(client, "/api/ingest").status_code == 429  # the uploads share a bucket
    assert _ask(client).status_code == 429


def test_other_clients_are_unaffected(make_limited: Limited) -> None:
    client = make_limited(rate_limit_uploads_per_minute=1, rate_limit_uploads_burst=1)
    other = _other_client(client)

    assert [_upload(client).status_code for _ in range(2)] == [200, 429]
    assert [_upload(other).status_code for _ in range(2)] == [200, 429]
    assert _upload(client).status_code == 429


@pytest.mark.parametrize(
    "path", ["/api/extract/", "/api/%65xtract", "/api/extract?x=1", "/api/ingest"]
)
def test_spellings_of_an_upload_route_share_its_bucket(make_limited: Limited, path: str) -> None:
    """Review probe: a trailing slash, a percent-encoded path, a query string or the other
    upload route must not open a fresh bucket for the same client."""
    client = make_limited(rate_limit_uploads_per_minute=1, rate_limit_uploads_burst=1)

    assert _upload(client).status_code == 200
    assert (
        client.post(
            path, files={"pdf_file": ("a.pdf", PDF, "application/pdf")}, follow_redirects=False
        ).status_code
        == 429
    )


def test_probes_are_never_limited(make_limited: Limited) -> None:
    client = make_limited(
        rate_limit_uploads_per_minute=1,
        rate_limit_uploads_burst=1,
        rate_limit_questions_per_minute=1,
        rate_limit_questions_burst=1,
        max_jobs_per_client=1,
    )

    statuses = {
        client.get(path).status_code
        for _ in range(50)
        for path in ("/health", "/health/ready", "/health/ready?x=1")
    }

    assert statuses == {200}


def test_a_client_at_its_job_cap_is_refused_at_once_while_another_gets_a_slot(
    make_limited: Limited, fakes: Fakes
) -> None:
    ocr = BlockingOcr()
    fakes.ocr = ocr
    client = make_limited(max_jobs_per_client=2, max_concurrent_jobs=4, busy_wait_s=BLOCK_S)
    other = _other_client(client)
    running = _InThreads()
    try:
        for _ in range(2):
            running.start(lambda: _upload(client))
        assert _until(lambda: ocr.calls == 2)
        began = time.perf_counter()
        third = _upload(client, **{"X-Request-ID": "capped"})
        took_s = time.perf_counter() - began
        running.start(lambda: _upload(other))
        assert _until(lambda: ocr.calls == 3)  # the other address got a slot at once
    finally:
        ocr.release.set()
    completed = running.join()

    assert third.status_code == 429
    assert third.json() == {"detail": CAP_MESSAGE, "code": "rate_limited", "request_id": "capped"}
    assert third.headers["Retry-After"] == "1"
    assert took_s < QUICK_S  # refused without waiting for a job slot
    assert [response.status_code for response in completed] == [200, 200, 200]
    assert _upload(client).status_code == 200  # its jobs ended: the count is back down


def test_a_rate_of_zero_turns_the_limit_off(make_limited: Limited, fakes: Fakes) -> None:
    client = make_limited(rate_limit_uploads_per_minute=0, rate_limit_uploads_burst=1)

    assert {_upload(client).status_code for _ in range(10)} == {200}
    assert _ocr_calls(fakes) == 10


def test_a_cap_of_zero_turns_the_job_cap_off(make_limited: Limited, fakes: Fakes) -> None:
    ocr = BlockingOcr()
    fakes.ocr = ocr
    client = make_limited(max_jobs_per_client=0, max_concurrent_jobs=3, busy_wait_s=BLOCK_S)
    running = _InThreads()
    try:
        for _ in range(3):
            running.start(lambda: _upload(client))
        assert _until(lambda: ocr.calls == 3)  # all three run at once
    finally:
        ocr.release.set()

    assert [response.status_code for response in running.join()] == [200, 200, 200]


# --- the middleware on its own, with a raw ASGI exchange -----------------------------------


def _scope(client: tuple[str, int] | None) -> Scope:
    scope: Scope = {"type": "http", "method": "POST", "path": "/api/extract", "headers": []}
    if client is not None:
        scope["client"] = client
    return scope


async def _ok(scope: Scope, receive: Receive, send: Send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def _disconnect() -> Message:
    return {"type": "http.disconnect"}


async def _exchange(app: RateLimitMiddleware, scope: Scope) -> tuple[int, str | None]:
    """Run one request through ``app``: its status and ``Retry-After`` header."""
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, _disconnect, send)
    headers = dict(sent[0].get("headers", []))
    retry_after = headers.get(b"retry-after")
    return int(sent[0]["status"]), retry_after.decode() if retry_after else None


def _status(app: RateLimitMiddleware, scope: Scope) -> int:
    return asyncio.run(_exchange(app, scope))[0]


def test_a_request_without_a_client_counts_as_one_address(clock: FakeClock) -> None:
    limits = RouteLimits("uploads", per_minute=60, burst=2)
    app = RateLimitMiddleware(_ok, limits=lambda scope: limits, clock=clock)

    assert [_status(app, _scope(None)) for _ in range(3)] == [200, 200, 429]


def test_a_request_of_an_unlimited_route_is_passed_through(clock: FakeClock) -> None:
    app = RateLimitMiddleware(_ok, limits=lambda scope: None, clock=clock)

    assert {_status(app, _scope((ADDRESS, 1))) for _ in range(5)} == {200}
    assert app.tracked == 0


def test_the_cap_counts_requests_in_flight_and_is_checked_before_the_bucket(
    clock: FakeClock,
) -> None:
    release = asyncio.Event()

    async def holding(scope: Scope, receive: Receive, send: Send) -> None:
        await release.wait()
        await _ok(scope, receive, send)

    limits = RouteLimits("uploads", per_minute=60, burst=2, max_jobs=1)
    app = RateLimitMiddleware(holding, limits=lambda scope: limits, clock=clock)
    outcomes: list[tuple[int, str | None]] = []

    async def scenario() -> None:
        first = asyncio.create_task(_exchange(app, _scope((ADDRESS, 1))))
        await asyncio.sleep(0)  # the first request is inside the application, holding
        outcomes.append(await _exchange(app, _scope((ADDRESS, 2))))  # at the cap
        release.set()
        outcomes.append(await first)
        outcomes.append(await _exchange(app, _scope((ADDRESS, 3))))  # the second token
        outcomes.append(await _exchange(app, _scope((ADDRESS, 4))))  # the bucket is empty

    asyncio.run(scenario())

    # The capped request was not charged a token: the second token was still there after it.
    assert outcomes == [(429, "1"), (200, None), (200, None), (429, "1")]


def test_a_failed_request_frees_its_place_under_the_cap(clock: FakeClock) -> None:
    async def failing(scope: Scope, receive: Receive, send: Send) -> None:
        raise RuntimeError("boom")

    limits = RouteLimits("uploads", per_minute=0, burst=1, max_jobs=1)
    app = RateLimitMiddleware(failing, limits=lambda scope: limits, clock=clock)

    for _ in range(2):  # admitted both times: a 429 would be sent instead of raising
        with pytest.raises(RuntimeError, match="boom"):
            _status(app, _scope((ADDRESS, 1)))


def test_idle_full_buckets_are_pruned_and_others_kept(clock: FakeClock) -> None:
    limits = RouteLimits("uploads", per_minute=1, burst=2)  # a token refills every 60 s
    app = RateLimitMiddleware(_ok, limits=lambda scope: limits, clock=clock)
    for n in range(50):
        _status(app, _scope((f"10.1.0.{n}", 1)))  # one token left each: full in 60 s
    _status(app, _scope((OTHER_ADDRESS, 1)))
    _status(app, _scope((OTHER_ADDRESS, 1)))  # no token left: full in 120 s
    assert app.tracked == 51

    clock.advance(PRUNE_INTERVAL_S)
    _status(app, _scope((ADDRESS, 1)))  # the first request after the interval prunes

    assert app.tracked == 2  # the other address, still refilling, and this one


# --- over real connections ------------------------------------------------------------------


def test_the_burst_holds_over_real_connections() -> None:
    settings = {
        "rate_limit_uploads_per_minute": 1,
        "rate_limit_uploads_burst": 2,
        "max_concurrent_jobs": 1,
    }
    env = {"SERVED_APP_PAGES": "1", "SERVED_APP_PAGE_S": "0.01"}
    with served(settings, env=env) as running:
        base_url = f"http://127.0.0.1:{running.port}"
        with httpx2.Client(base_url=base_url, timeout=30.0) as client:
            responses = [
                client.post(
                    "/api/extract",
                    files={"pdf_file": ("a.pdf", PDF, "application/pdf")},
                    headers={"X-Request-ID": f"real-{n}"},
                )
                for n in range(3)
            ]
        access = running.wait_for(
            lambda line: line.get("message") == "request" and line.get("status") == 429
        )
        pages = [line for line in running.lines() if line.get("message") == PAGE_LINE]

    assert [response.status_code for response in responses] == [200, 200, 429]
    assert responses[2].headers["Retry-After"] == "60"
    assert responses[2].json() == {
        "detail": RATE_MESSAGE,
        "code": "rate_limited",
        "request_id": "real-2",
    }
    assert access is not None
    assert access["request_id"] == "real-2"
    assert len(pages) == 2  # no page was recognised for the refused request
