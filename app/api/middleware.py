"""Pure ASGI middleware: request id, access line, unexpected errors and the body-size limit.

``RequestContextMiddleware`` (outermost) accepts or generates the request id, sets it in
``request_context`` for the whole request (handlers, exception handlers and the worker
threads of sync handlers, which copy the context), returns it in ``X-Request-ID``, turns an
exception that escapes the application into a 500 ``internal_error`` (logged with its stack
trace) and writes one access line per request: at INFO, except a successful request (status
below 400) to one of its ``probe_routes`` (the health endpoints), which is logged at DEBUG. The
container's healthcheck probes readiness every 15 s, and an orchestrator's probes as often, so
at INFO they would bury the client requests; a failing probe (503) stays at INFO, next to the
readiness warning naming the failed check, and ``LOG_LEVEL=DEBUG`` shows every probe. It also
adds ``PROTECTIVE_HEADERS`` (``Cache-Control: no-store``, ``X-Content-Type-Options: nosniff``,
``Referrer-Policy: no-referrer``) to every response, success or error (its own 500 included),
of the API routes: the paths under ``/api/`` and the health routes. Those responses are JSON
about one request (never to be cached or sniffed) and nothing links from them; the audit viewer
sets its own caching and security headers, which are left alone, as is any other path.

``BodyLimitMiddleware`` applies the limit ``limit(scope)`` chooses for the request's path (the
upload routes accept a PDF, the others a small JSON body or none): it rejects a request whose
declared ``Content-Length`` exceeds it with 413 ``too_large`` before reading the body (a value of
more than 18 significant digits is over any limit), and counts the bytes of bodies without one
(chunked), raising a 413 ``HTTPException`` from ``receive`` once they exceed it, so the
application never reads a body past the limit.

Every early 413 carries ``Connection: close``, so the server closes the connection once the
response is complete instead of reading and discarding the rest of the body itself to keep the
connection alive (on a keep-alive connection Uvicorn would otherwise read a body of any declared
size, for as long as the client sends it). Before completing the response the middleware drains
the rest of the body (reads and discards it), so a client still sending reaches its end and
reads the 413 instead of a connection reset: closed with unread data, the connection is reset
(measured on Linux). Only a body at most ``DRAIN_LIMIT_BYTES`` over the limit is drained, and
for at most ``DRAIN_TIMEOUT_S``; a larger one (known at once from its declared length) is cut,
as is a client that sent ``Expect: 100-continue`` and has not started its body (refused on its
declared length, it was never told to continue). Once body bytes have arrived (the server sends
``100 Continue`` at the application's first read) the client is sending, and it is drained like
any other. The response is sent before draining, with only its end held back, because the
server stops delivering the body once a response is complete; a client that reads while it
sends sees the 413 at once.
"""

import logging
import time
from collections.abc import Callable, Collection

import anyio
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import (
    error_response,
    internal_error_response,
    log_failure,
    log_unexpected,
)
from app.domain.errors import Category
from app.observability import accept_request_id, request_context

REQUEST_ID_HEADER = "X-Request-ID"
TOO_LARGE_MESSAGE = "the request body is larger than this endpoint accepts"
UNMATCHED_ROUTE = "unmatched"

_REQUEST_ID_KEY = REQUEST_ID_HEADER.lower().encode("latin-1")
_CONNECTION_KEY = b"connection"

PROTECTIVE_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"cache-control", b"no-store"),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
)
"""On every response of the API routes (``protected_path``), replacing any of the same name."""
_PROTECTIVE_KEYS = frozenset(key for key, _ in PROTECTIVE_HEADERS)
API_PREFIX = "/api/"
HEALTH_PATH = "/health"
_TOO_LARGE_STATUS = 413
_FIRST_ERROR_STATUS = 400
_CANCELLED_STATUS = 499  # the request was cancelled before any response (client gone, shutdown)
_MAX_LENGTH_DIGITS = 18
"""A declared length with more significant digits than this (an exabyte or more) is taken as
``_OVER_ANY_LIMIT``; it also keeps ``int()`` away from Python's 4,300-digit conversion limit."""
_OVER_ANY_LIMIT: int = 10**_MAX_LENGTH_DIGITS
"""Above every 18-digit number, so above any configurable limit."""

DRAIN_LIMIT_BYTES = 1024 * 1024
"""How far over its limit a body may be and still be drained after an early 413. 1 MiB covers
a file a little over ``MAX_UPLOAD_MB`` or a question a little over 64 KiB, and draining reads at
most that much more than an accepted request (about 0.1 s at 100 Mbit/s). A body further over is
not a near miss, and reading it would only spend bandwidth."""

DRAIN_TIMEOUT_S = 5.0
"""Longest time spent draining one rejected body, as nginx's ``lingering_timeout`` (5 s): long
enough for a just-over upload on a fast link, short enough that a slow sender holds a connection
(no thread, no job slot) only briefly."""

_access_logger = logging.getLogger("app.access")


class RequestContextMiddleware:
    """Request id, ``X-Request-ID`` response header, unexpected errors and the access line.

    ``probe_routes``: route templates whose successful requests are logged at DEBUG.
    """

    def __init__(self, app: ASGIApp, probe_routes: Collection[str] = ()) -> None:
        self.app = app
        self.probe_routes = frozenset(probe_routes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = accept_request_id(Headers(scope=scope).get(REQUEST_ID_HEADER))
        protected = protected_path(route_path(scope))
        status = 500
        started = False

        async def send_with_id(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                started = True
                status = int(message["status"])
                replaced = {_REQUEST_ID_KEY, *(_PROTECTIVE_KEYS if protected else ())}
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() not in replaced
                ]
                headers.append((_REQUEST_ID_KEY, request_id.encode("latin-1")))
                if protected:
                    headers.extend(PROTECTIVE_HEADERS)
                message["headers"] = headers
            await send(message)

        begin = time.perf_counter()
        with request_context(request_id):
            try:
                await self.app(scope, receive, send_with_id)
            except anyio.get_cancelled_exc_class():
                if not started:
                    status = _CANCELLED_STATUS
                raise
            except Exception as error:
                if started:
                    raise  # the response is under way; the server closes the connection
                log_unexpected(error)
                await internal_error_response()(scope, receive, send_with_id)
            finally:
                route = _route_template(scope)
                _access_logger.log(
                    self._access_level(route, status),
                    "request",
                    extra={
                        "method": scope.get("method"),
                        "route": route,
                        "status": status,
                        "duration_ms": round((time.perf_counter() - begin) * 1000, 1),
                    },
                )

    def _access_level(self, route: str, status: int) -> int:
        """DEBUG for a successful probe, INFO for every other request."""
        if route in self.probe_routes and status < _FIRST_ERROR_STATUS:
            return logging.DEBUG
        return logging.INFO


class BodyLimitMiddleware:
    """413 ``too_large`` for bodies above ``limit(scope)`` bytes (``None``: no limit yet).

    ``limit`` is called per request with its scope, because the limit depends on the path and
    the settings are loaded in the lifespan.
    """

    def __init__(self, app: ASGIApp, limit: Callable[[Scope], int | None]) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = self.limit(scope) if scope["type"] == "http" else None
        if limit is None:
            await self.app(scope, receive, send)
            return
        body = _BodyCount(limit, _drain_allowed(scope))

        async def receive_within_limit() -> Message:
            message = await receive()
            body.count(message)
            if body.over:
                raise StarletteHTTPException(_TOO_LARGE_STATUS, TOO_LARGE_MESSAGE)
            return message

        async def send_then_drain(message: Message) -> None:
            if body.over and message["type"] == "http.response.start":
                message = _closing(message)
            elif (
                body.over
                and message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                # Everything but the end of the response goes out now; the server stops
                # delivering the body once the response is complete.
                await send({**message, "more_body": True})
                await body.drain(receive)
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return
            await send(message)

        declared = _content_length(scope)
        if declared is not None and declared > limit:
            body.reject(declared)
            log_failure("too_large", str(Category.INVALID_INPUT), _TOO_LARGE_STATUS)
            await error_response(_TOO_LARGE_STATUS, TOO_LARGE_MESSAGE, "too_large")(
                scope, receive, send_then_drain
            )
            return
        await self.app(scope, receive_within_limit, send_then_drain)


class _BodyCount:
    """How much of one request body has been received, and whether the rest may be drained."""

    def __init__(self, limit: int, drain_allowed: bool) -> None:
        self.limit = limit
        self.received = 0
        self.complete = False
        self.over = False
        self._drain_allowed = drain_allowed
        self._declared: int | None = None

    def count(self, message: Message) -> None:
        if message["type"] == "http.request":
            self.received += len(message.get("body", b""))
            self.complete = not message.get("more_body", False)
            self.over = self.over or self.received > self.limit
        else:  # http.disconnect: nothing more will arrive
            self.complete = True

    def reject(self, declared: int) -> None:
        """Reject the body before reading it, for its declared length."""
        self.over = True
        self._declared = declared

    async def drain(self, receive: Receive) -> None:
        """Read and discard the rest of the body while it stays within ``DRAIN_LIMIT_BYTES``
        over the limit, for at most ``DRAIN_TIMEOUT_S``; a client waiting for ``100 Continue``
        only once its body has started arriving."""
        if self.complete or not (self._drain_allowed or self.received > 0):
            return
        most = self.limit + DRAIN_LIMIT_BYTES
        if self._declared is not None and self._declared > most:
            return  # it would be cut anyway: do not read any of it
        with anyio.move_on_after(DRAIN_TIMEOUT_S):
            while not self.complete and self.received <= most:
                self.count(await receive())


def _drain_allowed(scope: Scope) -> bool:
    """Whether the client sends its body unasked: one that sent ``Expect: 100-continue`` sends
    it only once told to continue, which a request refused before reading is not."""
    expect = Headers(scope=scope).get("expect", "")
    return expect.strip().lower() != "100-continue"


def _closing(message: Message) -> Message:
    """``message`` (a response start) with ``Connection: close`` as its only connection header."""
    headers = [
        (key, value) for key, value in message.get("headers", []) if key.lower() != _CONNECTION_KEY
    ]
    headers.append((_CONNECTION_KEY, b"close"))
    return {**message, "headers": headers}


def _content_length(scope: Scope) -> int | None:
    """The declared length; ``None`` when absent or not a decimal number (the server rejects
    those), a value over any limit when it has more than 18 significant digits."""
    value = Headers(scope=scope).get("content-length")
    if value is None:
        return None
    digits = value.strip()
    if not (digits.isascii() and digits.isdigit()):
        return None
    significant = digits.lstrip("0")
    if len(significant) > _MAX_LENGTH_DIGITS:
        return _OVER_ANY_LIMIT
    return int(significant or "0")


def route_path(scope: Scope) -> str:
    """The request path without the ``root_path`` prefix and trailing slashes ("/" at least)."""
    path = str(scope.get("path", ""))
    root = str(scope.get("root_path", ""))
    if root and path.startswith(root):
        path = path[len(root) :]
    return path.rstrip("/") or "/"


def protected_path(path: str) -> bool:
    """Whether ``path`` (as ``route_path`` gives it) is an API route, whose responses carry
    ``PROTECTIVE_HEADERS``: under ``/api/``, or ``/health`` and the paths under it."""
    return path.startswith(API_PREFIX) or path == HEALTH_PATH or path.startswith(HEALTH_PATH + "/")


def _route_template(scope: Scope) -> str:
    """The matched route's path template ("/api/extract"), never the raw path."""
    path = getattr(scope.get("route"), "path", None)
    return path if isinstance(path, str) else UNMATCHED_ROUTE
