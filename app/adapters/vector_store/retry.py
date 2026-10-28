"""Bounded retries of Qdrant calls and the mapping of client exceptions to service errors.

What qdrant-client raises (1.20, REST transport, which the adapter uses):

- ``ResponseHandlingException`` wrapping the ``httpx`` exception in ``.source`` when the request
  could not be sent or the response not read: ``httpx.TimeoutException`` (connect, read, write,
  pool timeouts) or another ``httpx.TransportError`` (``ConnectError``, ``ReadError``,
  ``RemoteProtocolError``, ...). Retried; the timeout family maps to ``DependencyTimeout``.
- ``UnexpectedResponse`` with status 502, 503 or 504 (a proxy or the server is briefly
  unavailable): retried as a connection failure.
- ``ResponseHandlingException`` wrapping anything else (for example a response that does not
  parse), ``UnexpectedResponse`` with any other non-2xx status (4xx, 500, ...) and
  ``QdrantException`` such as ``ResourceExhaustedResponse`` (429 with Retry-After): not retried,
  ``DependencyUnavailable`` at once.

Also classified, for other transports and injected clients: bare ``httpx`` transport errors, the
builtin ``TimeoutError`` / ``ConnectionError``, and ``grpc.RpcError`` with status
``DEADLINE_EXCEEDED`` (timeout) or ``UNAVAILABLE`` (connection). Any other exception is a
programming error and propagates unchanged.

Besides the number of attempts, an overall deadline bounds one call: no new attempt starts once
the deadline has passed (counting the backoff before it), and running out of time fails with
``DependencyTimeout``. A call therefore ends no later than the deadline plus one request timeout.

Service errors carry fixed messages; the original exception (which may quote a request or a
response body) is kept only as ``__cause__``.
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import grpc  # type: ignore[import-untyped]  # grpcio ships no type stubs
import httpx
from qdrant_client.common.client_exceptions import QdrantException
from qdrant_client.http.exceptions import (
    ApiException,
    ResponseHandlingException,
    UnexpectedResponse,
)

from app.domain.errors import DependencyTimeout, DependencyUnavailable

UNAVAILABLE_CODE = "vector_store_unavailable"
TIMEOUT_CODE = "vector_store_timeout"

_UNREACHABLE_MESSAGE = "the vector store could not be reached"
_TIMEOUT_MESSAGE = "the vector store did not answer in time"
_REJECTED_MESSAGE = "the vector store rejected the request"
_TRANSIENT_STATUSES = frozenset({502, 503, 504})


class Failure(Enum):
    """How a raised exception is handled."""

    CONNECTION = "connection"  # retried, then DependencyUnavailable
    TIMEOUT = "timeout"  # retried, then DependencyTimeout
    REJECTED = "rejected"  # not retried, DependencyUnavailable
    OTHER = "other"  # not a dependency failure: propagates unchanged


def classify(error: BaseException) -> Failure:
    """Return how ``error``, raised by a qdrant-client call, is handled."""
    if isinstance(error, ResponseHandlingException):
        source = error.source
        kind = _classify_transport(source)
        return Failure.REJECTED if kind is Failure.OTHER else kind
    if isinstance(error, UnexpectedResponse) and error.status_code in _TRANSIENT_STATUSES:
        return Failure.CONNECTION
    if isinstance(error, ApiException | QdrantException):
        return Failure.REJECTED
    if isinstance(error, grpc.RpcError):
        return _classify_grpc(error)
    return _classify_transport(error)


def _classify_transport(error: BaseException) -> Failure:
    if isinstance(error, httpx.TimeoutException | TimeoutError):
        return Failure.TIMEOUT
    if isinstance(error, httpx.TransportError | ConnectionError):
        return Failure.CONNECTION
    return Failure.OTHER


def _classify_grpc(error: grpc.RpcError) -> Failure:
    code_of = getattr(error, "code", None)
    code = code_of() if callable(code_of) else None
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return Failure.TIMEOUT
    if code == grpc.StatusCode.UNAVAILABLE:
        return Failure.CONNECTION
    return Failure.REJECTED


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """``attempts`` tries in total; the n-th wait (from 0) is ``U(0, min(cap, base * 2**n))``.

    No try starts ``deadline_s`` or more after the first one began.
    """

    attempts: int = 3
    backoff_base_s: float = 0.2
    backoff_cap_s: float = 2.0
    deadline_s: float = 20.0

    def delay(self, retry_index: int, uniform: float) -> float:
        """Full-jitter delay before retry ``retry_index``; ``uniform`` is drawn from [0, 1)."""
        return uniform * min(self.backoff_cap_s, self.backoff_base_s * 2.0**retry_index)


class Retrier:
    """Runs a call under a ``RetryPolicy`` and maps its failures to service errors."""

    def __init__(
        self,
        policy: RetryPolicy,
        *,
        sleep: Callable[[float], None] = time.sleep,
        uniform: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._policy = policy
        self._sleep = sleep
        self._uniform = uniform
        self._clock = clock

    def call[T](self, operation: Callable[[], T]) -> T:
        """Return ``operation()``, retrying connection errors, timeouts and 502/503/504.

        Raises ``DependencyUnavailable`` (``vector_store_unavailable``) when the store refuses the
        call or stays unreachable, and ``DependencyTimeout`` (``vector_store_timeout``) when the
        last attempt timed out or the deadline left no time for another; both are raised from
        the original exception.
        """
        deadline = self._clock() + self._policy.deadline_s
        attempt = 0
        while True:
            try:
                return operation()
            except Exception as error:
                kind = classify(error)
                if kind is Failure.OTHER:
                    raise
                attempt += 1
                if kind is Failure.REJECTED:
                    raise DependencyUnavailable(_REJECTED_MESSAGE, UNAVAILABLE_CODE) from error
                if attempt >= self._policy.attempts:
                    if kind is Failure.TIMEOUT:
                        raise DependencyTimeout(_TIMEOUT_MESSAGE, TIMEOUT_CODE) from error
                    raise DependencyUnavailable(_UNREACHABLE_MESSAGE, UNAVAILABLE_CODE) from error
                delay = self._policy.delay(attempt - 1, self._uniform())
                if self._clock() + delay >= deadline:
                    raise DependencyTimeout(_TIMEOUT_MESSAGE, TIMEOUT_CODE) from error
                self._sleep(delay)
