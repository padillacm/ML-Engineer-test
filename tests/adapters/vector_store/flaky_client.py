"""A ``QdrantClient`` stand-in that raises scripted errors before delegating to a real client."""

import threading
from collections.abc import Callable
from typing import Any, cast

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

type Scripted = BaseException | Callable[[], BaseException] | None


class FlakyClient:
    """Delegates to ``inner``; ``failures[method]`` scripts each call.

    ``None`` passes through; an exception is raised; a callable is called and what it returns is
    raised (so a test can, for example, advance a fake clock as the call "takes time").
    """

    def __init__(
        self, inner: QdrantClient, failures: dict[str, list[Scripted]] | None = None
    ) -> None:
        self.inner = inner
        self.failures = failures if failures is not None else {}
        self.calls: dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        target = getattr(self.inner, name)
        if not callable(target):
            return target
        return self._wrap(name, target)

    def _wrap(self, name: str, target: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls[name] = self.calls.get(name, 0) + 1
            script = self.failures.get(name)
            if script:
                error = script.pop(0)
                if callable(error) and not isinstance(error, BaseException):
                    error = error()
                if error is not None:
                    raise error
            return target(*args, **kwargs)

        return call

    def as_client(self) -> QdrantClient:
        return cast(QdrantClient, self)


def connect_error() -> ResponseHandlingException:
    """What qdrant-client raises when the server refuses the connection."""
    return ResponseHandlingException(httpx.ConnectError("connection refused"))


def read_timeout() -> ResponseHandlingException:
    """What qdrant-client raises when the server does not answer within the timeout."""
    return ResponseHandlingException(httpx.ReadTimeout("timed out"))


def http_status(status: int, body: bytes = b'{"status":{"error":"bad request"}}') -> Exception:
    """What qdrant-client raises on a non-2xx response."""
    return UnexpectedResponse(status, "Error", body, httpx.Headers())


class SerializedClient:
    """Delegates to ``inner`` one call at a time, so threads can share the ``:memory:`` client.

    Port operations still interleave between client calls, which is what concurrency tests need.
    """

    def __init__(self, inner: QdrantClient) -> None:
        self.inner = inner
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        target = getattr(self.inner, name)
        if not callable(target):
            return target

        def call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return target(*args, **kwargs)

        return call

    def as_client(self) -> QdrantClient:
        return cast(QdrantClient, self)
