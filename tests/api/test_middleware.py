"""Request id, access line and body limit (spec: observability, http-api)."""

import asyncio
import logging
import re

import pytest
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api import middleware
from app.api.middleware import DRAIN_LIMIT_BYTES, BodyLimitMiddleware, RequestContextMiddleware
from app.domain.errors import DependencyUnavailable
from app.main import FORM_ALLOWANCE_BYTES
from app.observability import request_id_var
from tests.api.conftest import ClientFactory, LogLines
from tests.api.support import MIB, PDF, Fakes, chunked, multipart
from tests.fakes.ocr import FakeOcrEngine

GENERATED = re.compile(r"[0-9a-f]{32}")
BODY_LIMIT = MIB + FORM_ALLOWANCE_BYTES


def test_incoming_request_id_is_echoed_and_on_every_line(
    client: TestClient, logs: LogLines
) -> None:
    logs.clear()

    response = client.post(
        "/api/extract",
        files={"pdf_file": ("a.pdf", PDF, "application/pdf")},
        headers={"X-Request-ID": "abc-123"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "abc-123"
    service_lines = [line for line in logs.lines() if str(line["logger"]).startswith("app")]
    assert sorted(str(line["message"]) for line in service_lines) == [
        "extraction completed",
        "request",
    ]
    assert all(line["request_id"] == "abc-123" for line in service_lines)


def test_malformed_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "x" * 200})

    assert GENERATED.fullmatch(response.headers["X-Request-ID"])


def test_request_without_id_gets_a_generated_one(client: TestClient) -> None:
    first = client.get("/health").headers["X-Request-ID"]
    second = client.get("/health").headers["X-Request-ID"]

    assert GENERATED.fullmatch(first)
    assert first != second


def test_access_line_has_method_route_template_status_and_duration(
    client: TestClient, logs: LogLines
) -> None:
    client.post("/api/extract", files={"pdf_file": ("a.pdf", b"nope", "application/pdf")})

    (access,) = logs.with_message("request")
    assert access["method"] == "POST"
    assert access["route"] == "/api/extract"
    assert access["status"] == 415
    assert isinstance(access["duration_ms"], float)
    (failed,) = logs.with_message("request failed")
    assert (failed["code"], failed["category"]) == ("not_a_pdf", "invalid_input")


def _store_down() -> str:
    raise DependencyUnavailable("the vector store could not be reached", "vector_store_unavailable")


def test_successful_probes_are_not_access_logged_at_info(
    client: TestClient, logs: LogLines
) -> None:
    """The container's healthcheck probes readiness every 15 s; client requests stay logged."""
    assert client.get("/health").status_code == 200
    assert client.get("/health/ready").status_code == 200
    client.post("/api/extract", files={"pdf_file": ("a.pdf", b"nope", "application/pdf")})

    assert [line["route"] for line in logs.with_message("request")] == ["/api/extract"]


def test_a_failing_readiness_probe_is_access_logged(
    make_client: ClientFactory, fakes: Fakes, logs: LogLines
) -> None:
    fakes.readiness = (("vector_store", _store_down),)
    client = make_client()

    assert client.get("/health/ready").status_code == 503

    (access,) = logs.with_message("request")
    assert (access["level"], access["route"], access["status"]) == ("INFO", "/health/ready", 503)


def test_probes_are_access_logged_at_debug(make_client: ClientFactory, logs: LogLines) -> None:
    client = make_client(log_level="DEBUG")

    client.get("/health")
    client.get("/health/ready")

    access = [line for line in logs.with_message("request") if line["logger"] == "app.access"]
    assert [(line["level"], line["route"], line["status"]) for line in access] == [
        ("DEBUG", "/health", 200),
        ("DEBUG", "/health/ready", 200),
    ]


def test_without_probe_routes_every_request_is_logged_at_info(logs: LogLines) -> None:
    async def ok(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    logging.getLogger().setLevel(logging.INFO)  # restored by the restore_logging fixture
    client = TestClient(RequestContextMiddleware(ok))

    client.get("/health")

    (access,) = logs.with_message("request")
    assert (access["level"], access["route"]) == ("INFO", middleware.UNMATCHED_ROUTE)


def test_declared_body_over_the_limit_is_413_before_anything_is_parsed(
    client: TestClient, fakes: Fakes, logs: LogLines
) -> None:
    body, content_type = multipart(PDF + b"0" * BODY_LIMIT)

    response = client.post(
        "/api/extract",
        content=body,
        headers={"Content-Type": content_type, "X-Request-ID": "big-1"},
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "the request body is larger than this endpoint accepts",
        "code": "too_large",
        "request_id": "big-1",
    }
    assert isinstance(fakes.ocr, FakeOcrEngine)
    assert fakes.ocr.calls == []
    (failed,) = logs.with_message("request failed")
    assert failed["status"] == 413


def test_streamed_body_over_the_limit_is_413(client: TestClient, fakes: Fakes) -> None:
    body, content_type = multipart(PDF + b"0" * BODY_LIMIT)

    response = client.post(
        "/api/extract",
        content=chunked(body)(),
        headers={"Content-Type": content_type},
    )

    assert response.status_code == 413
    assert response.json()["code"] == "too_large"
    assert isinstance(fakes.ocr, FakeOcrEngine)
    assert fakes.ocr.calls == []


def test_streamed_file_one_byte_over_the_limit_is_413(client: TestClient, fakes: Fakes) -> None:
    body, content_type = multipart(PDF + b"0" * (MIB + 1 - len(PDF)))

    response = client.post(
        "/api/extract", content=chunked(body)(), headers={"Content-Type": content_type}
    )

    assert response.status_code == 413
    assert response.json()["code"] == "too_large"
    assert isinstance(fakes.ocr, FakeOcrEngine)
    assert fakes.ocr.calls == []


# --- the middleware on its own, with a raw ASGI exchange -----------------------------------


def _scope(headers: list[tuple[bytes, bytes]]) -> Scope:
    return {"type": "http", "method": "POST", "path": "/x", "headers": headers}


def _run(app: object, scope: Scope, messages: list[Message]) -> tuple[list[Message], int]:
    """Run ``app`` on ``messages``; return what it sent and how many messages it received."""
    sent: list[Message] = []
    received = 0

    async def receive() -> Message:
        nonlocal received
        received += 1
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))  # type: ignore[operator]
    return sent, received


async def _read_everything(scope: Scope, receive: Receive, send: Send) -> None:
    while True:
        message = await receive()
        if not message.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"read"})


def _chunks(count: int, size: int) -> list[Message]:
    return [
        {"type": "http.request", "body": b"x" * size, "more_body": index < count - 1}
        for index in range(count)
    ]


def test_streamed_body_stops_reaching_the_application_once_over_the_limit() -> None:
    seen = _Seen(_read_everything)
    # As in FastAPI, the exception handlers run inside the body limit.
    limited = BodyLimitMiddleware(_with_http_exception_handling(seen), limit=lambda scope: 250)
    app = RequestContextMiddleware(limited)

    sent, received = _run(app, _scope([]), _chunks(10, 100))

    assert sent[0]["status"] == 413
    assert seen.received == 3  # 300 bytes: the third chunk crossed the 250-byte limit
    assert received == 10  # the rest was drained (read and discarded) ...
    assert _ends(sent) == [b'{"code":"too_large"}', b""]  # ... after the response was sent


def test_streamed_body_far_over_the_limit_is_drained_only_up_to_the_drain_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(middleware, "DRAIN_LIMIT_BYTES", 250)
    limited = BodyLimitMiddleware(
        _with_http_exception_handling(_read_everything), limit=lambda scope: 250
    )
    app = RequestContextMiddleware(limited)

    sent, received = _run(app, _scope([]), _chunks(10, 100))

    assert sent[0]["status"] == 413
    assert received == 6  # 600 bytes read: over 250 + 250, so the rest is cut


def test_body_within_the_limit_is_passed_through() -> None:
    app = BodyLimitMiddleware(_read_everything, limit=lambda scope: 1000)

    sent, received = _run(app, _scope([]), _chunks(10, 100))

    assert sent[0]["status"] == 200
    assert received == 10


def test_declared_length_over_the_limit_is_refused_before_reading_then_drained() -> None:
    seen = _Seen(_read_everything)
    app = BodyLimitMiddleware(seen, limit=lambda scope: 100)

    sent, received = _run(app, _scope([(b"content-length", b"200")]), _chunks(2, 100))

    assert sent[0]["status"] == 413
    assert seen.received == 0  # the application never runs
    assert received == 2  # the body was drained after the response was sent
    assert sent[1]["more_body"] is True
    assert sent[-1] == {"type": "http.response.body", "body": b"", "more_body": False}


def test_declared_length_beyond_the_drain_limit_is_cut_without_reading() -> None:
    app = BodyLimitMiddleware(_read_everything, limit=lambda scope: 100)
    declared = str(100 + DRAIN_LIMIT_BYTES + 1).encode()

    sent, received = _run(app, _scope([(b"content-length", declared)]), _chunks(2, 100))

    assert sent[0]["status"] == 413
    assert received == 0


def test_client_waiting_for_100_continue_is_not_drained() -> None:
    app = BodyLimitMiddleware(_read_everything, limit=lambda scope: 100)
    headers = [(b"content-length", b"200"), (b"expect", b"100-continue")]

    sent, received = _run(app, _scope(headers), _chunks(2, 100))

    assert sent[0]["status"] == 413
    assert received == 0  # refused on its declared length: never told to continue


def test_client_told_to_continue_is_drained_once_its_body_is_arriving() -> None:
    # The server sends 100 Continue at the application's first read; the client then sends.
    limited = BodyLimitMiddleware(
        _with_http_exception_handling(_read_everything), limit=lambda scope: 250
    )

    sent, received = _run(limited, _scope([(b"expect", b"100-continue")]), _chunks(10, 100))

    assert sent[0]["status"] == 413
    assert received == 10  # the rest was drained, as without the header
    assert sent[-1]["more_body"] is False


@pytest.mark.parametrize(
    ("headers", "messages"),
    [
        pytest.param([(b"content-length", b"200")], _chunks(2, 100), id="declared"),
        pytest.param([], _chunks(10, 100), id="streamed"),
    ],
)
def test_every_early_413_closes_the_connection(
    headers: list[tuple[bytes, bytes]], messages: list[Message]
) -> None:
    # Without it a keep-alive server reads (and discards) the rest of the body itself, however
    # large and however slowly it comes, to keep the connection.
    limited = BodyLimitMiddleware(
        _with_http_exception_handling(_read_everything), limit=lambda scope: 100
    )

    sent, _ = _run(limited, _scope(headers), messages)

    assert sent[0]["status"] == 413
    connection = [value for key, value in sent[0]["headers"] if key.lower() == b"connection"]
    assert connection == [b"close"]


def test_connection_header_of_the_application_is_replaced_on_a_413() -> None:
    async def keeping(scope: Scope, receive: Receive, send: Send) -> None:
        while (await receive()).get("more_body", False):
            pass
        raise AssertionError("unreachable: the third chunk is over the limit")

    async def handled(scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await keeping(scope, receive, send)
        except HTTPException as error:
            await send(
                {
                    "type": "http.response.start",
                    "status": error.status_code,
                    "headers": [(b"Connection", b"keep-alive")],
                }
            )
            await send({"type": "http.response.body", "body": b""})

    app = BodyLimitMiddleware(handled, limit=lambda scope: 250)

    sent, _ = _run(app, _scope([]), _chunks(10, 100))

    assert sent[0]["headers"] == [(b"connection", b"close")]


def test_body_within_the_limit_keeps_the_connection() -> None:
    app = BodyLimitMiddleware(_read_everything, limit=lambda scope: 1000)

    sent, _ = _run(app, _scope([(b"content-length", b"1000")]), _chunks(10, 100))

    assert sent[0]["status"] == 200
    assert sent[0]["headers"] == []


def test_drain_stops_at_a_disconnect() -> None:
    app = BodyLimitMiddleware(_read_everything, limit=lambda scope: 100)

    sent, received = _run(app, _scope([(b"content-length", b"300")]), _chunks(3, 100)[:1])

    assert sent[0]["status"] == 413
    assert received == 2  # one chunk, then the disconnect
    assert sent[-1]["more_body"] is False


def test_drain_gives_up_after_its_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(middleware, "DRAIN_TIMEOUT_S", 0.05)
    app = BodyLimitMiddleware(_read_everything, limit=lambda scope: 100)
    sent: list[Message] = []

    async def stalled() -> Message:
        await asyncio.sleep(10)  # a client that stopped sending
        raise AssertionError("unreachable")

    async def send(message: Message) -> None:
        sent.append(message)

    async def run() -> float:
        begin = asyncio.get_running_loop().time()
        await app(_scope([(b"content-length", b"200")]), stalled, send)
        return asyncio.get_running_loop().time() - begin

    took = asyncio.run(run())

    assert took < 1.0
    assert sent[0]["status"] == 413
    assert sent[-1]["more_body"] is False


def test_no_limit_before_the_settings_are_loaded() -> None:
    app = BodyLimitMiddleware(_read_everything, limit=lambda scope: None)

    sent, _ = _run(app, _scope([(b"content-length", b"1000")]), _chunks(10, 100))

    assert sent[0]["status"] == 200


def test_line_break_in_request_id_is_replaced() -> None:
    seen: list[str | None] = []

    async def record(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(request_id_var.get())
        await _read_everything(scope, receive, send)

    sent, _ = _run(
        RequestContextMiddleware(record), _scope([(b"x-request-id", b"abc\ndef")]), _chunks(1, 1)
    )

    assert seen[0] is not None
    assert GENERATED.fullmatch(seen[0])
    assert (b"x-request-id", seen[0].encode()) in sent[0]["headers"]
    assert request_id_var.get() is None


def test_lifespan_scope_passes_through() -> None:
    calls: list[str] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        calls.append(scope["type"])

    async def run() -> None:
        async def receive() -> Message:
            return {"type": "lifespan.startup"}

        async def send(message: Message) -> None:
            return None

        await RequestContextMiddleware(BodyLimitMiddleware(inner, lambda scope: 1))(
            {"type": "lifespan"}, receive, send
        )

    asyncio.run(run())

    assert calls == ["lifespan"]


class _Seen:
    """Counts the ``receive`` calls the wrapped application makes."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.received = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def counted() -> Message:
            self.received += 1
            return await receive()

        await self.app(scope, counted, send)


def _ends(sent: list[Message]) -> list[bytes]:
    """The bodies of the response's last two messages."""
    return [bytes(message.get("body", b"")) for message in sent[-2:]]


def _with_http_exception_handling(app: ASGIApp) -> ASGIApp:
    """What FastAPI's exception middleware does with an ``HTTPException`` from ``receive``."""

    async def handled(scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await app(scope, receive, send)
        except HTTPException as error:
            await JSONResponse({"code": "too_large"}, status_code=error.status_code)(
                scope, receive, send
            )

    return handled


def test_failure_after_the_response_started_is_raised_to_the_server() -> None:
    async def half_sent(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("stream broke")

    with pytest.raises(RuntimeError, match="stream broke"):
        _run(RequestContextMiddleware(half_sent), _scope([]), _chunks(1, 1))


def test_request_cancelled_before_a_response_is_logged_as_499(
    logs: LogLines, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    async def cancelled(scope: Scope, receive: Receive, send: Send) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        _run(RequestContextMiddleware(cancelled), _scope([]), [])

    [access] = logs.with_message("request")
    assert access["status"] == 499
