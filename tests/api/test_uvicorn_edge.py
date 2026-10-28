"""The HTTP edge under a real Uvicorn process: the early 413 and its drain, and disconnects.

What the in-process tests cannot show: how a real server and a real socket behave. One server
(``tests.api.served_app``, Uvicorn's default HTTP implementation, the drain's time bound cut to
``DRAIN_S``) serves the whole module.

Without the drain, a server closing a connection with part of the body unread resets it, and
the client's send fails before it reads the 413 (measured on Linux for a body 900 KiB over the
limit). A body more than ``DRAIN_LIMIT_BYTES`` over its limit is still cut. Every early 413
closes the connection: on a keep-alive connection Uvicorn would otherwise read and discard the
rest of the body itself, however large (10^15 declared bytes: 20 GB accepted in 8 s) and however
slowly it comes.
"""

import socket
import time
from collections.abc import Iterator

import pytest

from app.api.middleware import DRAIN_LIMIT_BYTES
from app.main import FORM_ALLOWANCE_BYTES
from tests.api.served_app import PAGE_LINE
from tests.api.support import MIB, PDF, multipart
from tests.api.uvicorn_server import Server, served

LIMIT = MIB + FORM_ALLOWANCE_BYTES
"""The upload routes' body limit with ``MAX_UPLOAD_MB=1``."""

PAGES = 10
WAIT_S = 10.0
DRAIN_S = 1.0
"""The served application's ``DRAIN_TIMEOUT_S``."""
CLOSE_SLACK_S = 1.5
"""Time allowed between the drain's end and the client seeing the connection closed."""
SENT_AFTER_CLOSE_BYTES = 64 * MIB
"""More than the socket buffers of both ends hold: sending this much fails once the server has
closed (before, Uvicorn read gigabytes)."""


@pytest.fixture(scope="module")
def server() -> Iterator[Server]:
    settings = {"max_upload_mb": 1, "max_concurrent_jobs": 1}
    env = {"SERVED_APP_PAGES": str(PAGES), "SERVED_APP_DRAIN_TIMEOUT_S": str(DRAIN_S)}
    with served(settings, env=env) as running:
        yield running


def _head(
    content_type: str,
    length: int | None,
    request_id: str,
    *,
    close: bool = True,
    expect: bool = False,
) -> bytes:
    lines = [
        "POST /api/extract HTTP/1.1",
        "Host: 127.0.0.1",
        f"Content-Type: {content_type}",
        f"X-Request-ID: {request_id}",
        "Transfer-Encoding: chunked" if length is None else f"Content-Length: {length}",
    ]
    if close:
        lines.append("Connection: close")
    if expect:
        lines.append("Expect: 100-continue")
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


def _response(sock: socket.socket) -> bytes:
    """Everything the server sends until it closes the connection."""
    data = b""
    while chunk := sock.recv(65536):
        data += chunk
    return data


def _oversized_form(over: int) -> tuple[bytes, str]:
    body, content_type = multipart(PDF + b"0" * (LIMIT + over - 200))
    return body, content_type


def _send_body(sock: socket.socket, body: bytes, *, chunked: bool) -> None:
    for start in range(0, len(body), 64 * 1024):
        piece = body[start : start + 64 * 1024]
        if chunked:
            piece = f"{len(piece):x}\r\n".encode() + piece + b"\r\n"
        sock.sendall(piece)  # no reset: the server reads the whole body
    if chunked:
        sock.sendall(b"0\r\n\r\n")


@pytest.mark.parametrize("connection", ["close", "keep-alive"])
@pytest.mark.parametrize("framing", ["content-length", "chunked"])
def test_body_slightly_over_the_limit_is_drained_so_the_client_reads_its_413(
    server: Server, framing: str, connection: str
) -> None:
    body, content_type = _oversized_form(900 * 1024)
    assert LIMIT < len(body) <= LIMIT + DRAIN_LIMIT_BYTES
    length = len(body) if framing == "content-length" else None
    head = _head(content_type, length, f"drain-{framing}", close=connection == "close")

    with socket.create_connection(("127.0.0.1", server.port), timeout=WAIT_S) as sock:
        sock.sendall(head)
        _send_body(sock, body, chunked=length is None)
        response = _response(sock)  # ends: the server closes even a keep-alive connection

    assert response.startswith(b"HTTP/1.1 413 ")
    assert b"connection: close\r\n" in response.lower()
    assert b'"code":"too_large"' in response


def test_client_told_to_continue_reads_its_413(server: Server) -> None:
    # Drained like any other once its body arrives (test_middleware shows the drain itself: on
    # loopback a streamed near miss was not seen to reset even undrained).
    body, content_type = _oversized_form(900 * 1024)

    with socket.create_connection(("127.0.0.1", server.port), timeout=WAIT_S) as sock:
        sock.sendall(_head(content_type, None, "drain-expect", expect=True))
        assert sock.recv(65536).startswith(b"HTTP/1.1 100 ")  # at the application's first read
        _send_body(sock, body, chunked=True)
        response = _response(sock)

    assert response.startswith(b"HTTP/1.1 413 ")
    assert b'"code":"too_large"' in response


def test_body_far_over_the_drain_limit_is_cut_at_once(server: Server) -> None:
    declared = LIMIT + DRAIN_LIMIT_BYTES + 1

    with socket.create_connection(("127.0.0.1", server.port), timeout=WAIT_S) as sock:
        sock.sendall(_head("multipart/form-data; boundary=x", declared, "cut-1"))
        sock.sendall(b"0" * 64 * 1024)  # and then nothing: the server must not wait for more
        begin = time.perf_counter()
        response = _response(sock)  # ends when the server closes, without draining
        took = time.perf_counter() - begin

    assert response.startswith(b"HTTP/1.1 413 ")
    assert took < 1.0


def test_far_oversized_body_on_a_keep_alive_connection_is_not_read(server: Server) -> None:
    # Spec http-api, "Far oversized upload": the 413 is sent and the connection is closed.
    with socket.create_connection(("127.0.0.1", server.port), timeout=WAIT_S) as sock:
        sock.sendall(_head("multipart/form-data; boundary=x", 10**15, "cut-2", close=False))
        sock.sendall(b"0" * 64 * 1024)
        begin = time.perf_counter()
        response = _response(sock)  # ends when the server closes
        took = time.perf_counter() - begin
        accepted = _sent_until_refused(sock)

    assert response.startswith(b"HTTP/1.1 413 ")
    assert took < 1.0
    assert accepted is not None, "the server still reads the body"


def _sent_until_refused(sock: socket.socket) -> int | None:
    """Bytes the connection takes before a send fails (reset or aborted once the server has
    closed); ``None`` when it takes ``SENT_AFTER_CLOSE_BYTES``."""
    sent = 0
    try:
        while sent < SENT_AFTER_CLOSE_BYTES:
            sock.sendall(b"0" * MIB)
            sent += MIB
    except ConnectionError:
        return sent
    return None


def test_slow_sender_of_a_rejected_body_is_closed_after_the_drain_timeout(
    server: Server,
) -> None:
    # A near miss, so it is drained, but one byte at a time: the drain's time bound ends it.
    with socket.create_connection(("127.0.0.1", server.port), timeout=WAIT_S) as sock:
        sock.sendall(_head("multipart/form-data; boundary=x", LIMIT + 1000, "drip", close=False))
        sock.settimeout(0.1)
        begin = time.perf_counter()
        response = b""
        took: float | None = None
        while took is None and time.perf_counter() - begin < DRAIN_S + 5 * CLOSE_SLACK_S:
            try:
                sock.sendall(b"0")
                chunk = sock.recv(65536)
            except TimeoutError:
                continue
            except OSError:  # reset: a byte arrived as the server closed
                chunk = b""
            if chunk:
                response += chunk
            else:
                took = time.perf_counter() - begin

    assert response.startswith(b"HTTP/1.1 413 ")  # sent at once, before the drain
    assert took is not None, "the connection is still open"
    assert DRAIN_S - 0.2 <= took <= DRAIN_S + CLOSE_SLACK_S


def test_client_leaving_mid_job_stops_it_and_is_logged_as_499(server: Server) -> None:
    body, content_type = multipart(PDF)

    with socket.create_connection(("127.0.0.1", server.port), timeout=WAIT_S) as sock:
        sock.sendall(_head(content_type, len(body), "gone-1") + body)
        assert server.wait_for(
            lambda line: line.get("message") == PAGE_LINE and line.get("request_id") == "gone-1"
        )
    access = server.wait_for(
        lambda line: line.get("message") == "request" and line.get("request_id") == "gone-1"
    )

    assert access is not None
    assert (access["route"], access["status"]) == ("/api/extract", 499)
    lines = [line for line in server.lines() if line.get("request_id") == "gone-1"]
    failed = [line for line in lines if line.get("message") == "request failed"]
    assert [(line["code"], line["status"]) for line in failed] == [("request_cancelled", 499)]
    assert sum(line.get("message") == PAGE_LINE for line in lines) < PAGES
    assert not [line for line in lines if line.get("level") == "ERROR"]
