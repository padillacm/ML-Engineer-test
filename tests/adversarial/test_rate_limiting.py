"""Flood scenarios of the client-rate-limiting spec, end to end over the HTTP application.

The unit coverage of the limiter lives in ``tests/api/test_rate_limit.py``; these tests replay
the spec scenarios as a client flooding the service would: a burst of uploads from one address
(``Burst exhausted``, ``Other clients unaffected``, ``Probes are never limited``), a flood
against the job cap while slow jobs run (``One client floods the jobs``), the limits switched
off (``Limit off``) and the two route classes (``Route classes are independent``). The clock
is the fake one of the rate-limit tests, so a refill is a matter of moving time.
"""

import threading
import time
from collections.abc import Callable

import httpx2
from fastapi.testclient import TestClient

from tests.api.support import BLOCK_S, PDF, BlockingOcr, Fakes
from tests.api.test_rate_limit import FakeClock, Limited
from tests.fakes.ocr import FakeOcrEngine

OTHER_ADDRESS = "10.0.0.2"
QUESTION = "who signed the contract"
FLOOD = 30
QUICK_S = 0.5
"""A request refused at once answers within this."""


def _upload(client: TestClient, path: str = "/api/extract") -> httpx2.Response:
    return client.post(path, files={"pdf_file": ("a.pdf", PDF, "application/pdf")})


def _ask(client: TestClient) -> httpx2.Response:
    return client.post("/api/ask", json={"question": QUESTION})


def _other_client(client: TestClient) -> TestClient:
    """A client of ``OTHER_ADDRESS`` over the same running application."""
    return TestClient(client.app, client=(OTHER_ADDRESS, 50000))


def _ocr_calls(fakes: Fakes) -> int:
    assert isinstance(fakes.ocr, FakeOcrEngine)
    return len(fakes.ocr.calls)


def _until(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + BLOCK_S
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _in_thread(call: Callable[[], httpx2.Response]) -> Callable[[], httpx2.Response]:
    """Start ``call`` in a thread; the returned function joins it and gives its response."""
    responses: list[httpx2.Response] = []
    thread = threading.Thread(target=lambda: responses.append(call()), daemon=True)
    thread.start()

    def join() -> httpx2.Response:
        thread.join(BLOCK_S)
        assert responses, "the request did not finish"
        return responses[0]

    return join


def test_a_flood_of_uploads_is_cut_after_the_burst_while_another_address_is_served(
    make_limited: Limited, clock: FakeClock, fakes: Fakes
) -> None:
    client = make_limited(rate_limit_uploads_per_minute=60, rate_limit_uploads_burst=5)
    other = _other_client(client)

    flood = [_upload(client) for _ in range(FLOOD)]
    probes = {client.get(path).status_code for path in ("/health", "/health/ready")}
    served = [_upload(other).status_code for _ in range(5)]

    assert [response.status_code for response in flood] == [200] * 5 + [429] * (FLOOD - 5)
    for refused in flood[5:]:
        assert refused.json()["code"] == "rate_limited"
        assert int(refused.headers["Retry-After"]) >= 1
    assert probes == {200}
    assert served == [200] * 5
    assert _ocr_calls(fakes) == 10  # no OCR ran for a refused request

    clock.advance(60.0)  # a minute later the bucket is full again, and no fuller
    assert [_upload(client).status_code for _ in range(6)] == [200] * 5 + [429]


def test_a_flood_against_the_job_cap_is_refused_at_once_while_the_slow_jobs_run(
    make_limited: Limited, fakes: Fakes
) -> None:
    ocr = BlockingOcr()  # a slow OCR: its pages come out only when released
    fakes.ocr = ocr
    client = make_limited(max_jobs_per_client=2, max_concurrent_jobs=4, busy_wait_s=BLOCK_S)
    other = _other_client(client)
    try:
        running = [_in_thread(lambda: _upload(client)) for _ in range(2)]
        assert _until(lambda: ocr.calls == 2)
        began = time.perf_counter()
        flood = [_upload(client, "/api/ingest" if n % 2 else "/api/extract") for n in range(5)]
        flood_s = time.perf_counter() - began
        served = _in_thread(lambda: _upload(other))
        assert _until(lambda: ocr.calls == 3)  # the other address got a slot at once
    finally:
        ocr.release.set()

    assert [response.status_code for response in flood] == [429] * 5
    assert {response.json()["code"] for response in flood} == {"rate_limited"}
    assert {response.headers["Retry-After"] for response in flood} == {"1"}
    assert flood_s < QUICK_S  # refused without waiting for a job slot
    assert [join().status_code for join in running] == [200, 200]
    assert served().status_code == 200
    assert _upload(client).status_code == 200  # the jobs ended: the count is back down


def test_a_rate_and_a_cap_of_zero_turn_the_limits_off(make_limited: Limited, fakes: Fakes) -> None:
    client = make_limited(
        rate_limit_uploads_per_minute=0,
        rate_limit_uploads_burst=1,
        rate_limit_questions_per_minute=0,
        rate_limit_questions_burst=1,
        max_jobs_per_client=0,
    )

    uploads = {_upload(client).status_code for _ in range(FLOOD)}
    questions = {_ask(client).status_code for _ in range(FLOOD)}

    assert uploads == questions == {200}
    assert _ocr_calls(fakes) == FLOOD


def test_questions_and_uploads_are_limited_independently(make_limited: Limited) -> None:
    client = make_limited(
        rate_limit_uploads_per_minute=1,
        rate_limit_uploads_burst=3,
        rate_limit_questions_per_minute=1,
        rate_limit_questions_burst=3,
    )

    questions = [_ask(client).status_code for _ in range(4)]
    uploads = [_upload(client, path).status_code for path in ("/api/extract", "/api/ingest") * 2]

    assert questions == [200, 200, 200, 429]  # the question allowance is spent ...
    assert uploads == [200, 200, 200, 429]  # ... the uploads share their own bucket
    assert _ask(client).status_code == 429
    assert client.get("/health").status_code == 200
