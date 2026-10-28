"""``LLMClient`` for any server that speaks the OpenAI chat-completions API (Ollama, hosted APIs).

Retry policy: connection-level transport errors (``httpx.NetworkError``: connect, read, write and
close errors, plus ``httpx.RemoteProtocolError``), every ``httpx.TimeoutException``, HTTP 429 and
HTTP 5xx are retried up to ``attempts`` tries in total. Before a retry the client waits for the
response's ``Retry-After`` (seconds or an HTTP date, capped at ``retry_after_cap_s``) or, without
one, for a full-jitter exponential backoff. Any other HTTP status, a misconfigured request
(``httpx.UnsupportedProtocol``, ``LocalProtocolError``, proxy errors) and a malformed 2xx body fail
at once.

Every call also has an overall deadline (``total_timeout_s``, see ``LLMClientConfig``): no attempt
starts and no wait is taken past it, each attempt's timeouts are cut to the time left, the body is
read in chunks with the deadline checked after each one, and running out raises
``DependencyTimeout("LLM request exceeded its deadline", "llm_timeout")``.

Error messages are fixed text with attempt counts, status codes and exception type names only:
never the key, the prompt or the response body. The original exception is chained as
``__cause__`` for logs; the key travels only in the ``Authorization`` header, never in the URL.
"""

import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Self

import httpx

from app.adapters.llm.config import LLMClientConfig
from app.domain.errors import DependencyTimeout, DependencyUnavailable
from app.domain.types import ChatMessage

_UNAVAILABLE = "llm_unavailable"
_TIMEOUT = "llm_timeout"
_RETRYABLE_TRANSPORT_ERRORS = (httpx.NetworkError, httpx.RemoteProtocolError)
_DEADLINE_MESSAGE = "LLM request exceeded its deadline"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class OpenAICompatibleClient:
    """Chat completions over one pooled ``httpx.Client``; close it, or use it as a context manager.

    ``transport`` replaces the network (tests pass ``httpx.MockTransport``); ``sleep``, ``rng``
    (returns a float in ``[0, 1)``), ``clock`` (aware ``datetime`` now, for ``Retry-After``
    dates) and ``monotonic`` (seconds, for the overall deadline) are injectable so retry timing is
    deterministic under test.
    """

    def __init__(
        self,
        config: LLMClientConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
        clock: Callable[[], datetime] = _utc_now,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._sleep = sleep
        self._rng = rng
        self._clock = clock
        self._monotonic = monotonic
        headers = {"Accept": "application/json"}
        if config.api_key is not None:
            headers["Authorization"] = f"Bearer {config.api_key.reveal()}"
        self._http = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(
                config.read_timeout_s,
                connect=config.connect_timeout_s,
                read=config.read_timeout_s,
                write=config.read_timeout_s,
                pool=config.connect_timeout_s,
            ),
            transport=transport,
            follow_redirects=False,
        )

    @property
    def config(self) -> LLMClientConfig:
        return self._config

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self._config.base_url!r}, "
            f"model={self._config.model!r}, "
            f"api_key={'set' if self._config.api_key is not None else 'none'})"
        )

    __str__ = __repr__

    def close(self) -> None:
        """Release pooled connections; the client cannot be used afterwards."""
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        payload = {
            "model": self._config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }
        deadline = self._monotonic() + self._config.total_timeout_s
        attempts = self._config.attempts
        cause: BaseException | None = None
        for attempt in range(1, attempts + 1):
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise DependencyTimeout(_DEADLINE_MESSAGE, _TIMEOUT) from cause
            outcome = self._attempt(payload, deadline, self._timeout(remaining))
            if isinstance(outcome, str):
                return outcome
            cause = outcome.cause
            remaining = deadline - self._monotonic()
            if outcome.past_deadline or remaining <= 0:
                raise DependencyTimeout(_DEADLINE_MESSAGE, _TIMEOUT) from cause
            if attempt == attempts:
                raise outcome.final_error(attempts) from cause
            delay = outcome.retry_after
            if delay is None:
                delay = self._backoff(attempt)
            if delay >= remaining:
                # The wait alone would use up the deadline; fail now instead of sleeping to fail.
                raise DependencyTimeout(_DEADLINE_MESSAGE, _TIMEOUT) from cause
            self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _timeout(self, remaining: float) -> httpx.Timeout:
        """The configured per-operation timeouts, none longer than the time left."""
        connect = min(self._config.connect_timeout_s, remaining)
        read = min(self._config.read_timeout_s, remaining)
        return httpx.Timeout(read, connect=connect, read=read, write=read, pool=connect)

    def _attempt(
        self, payload: dict[str, object], deadline: float, timeout: httpx.Timeout
    ) -> "str | _Transient":
        """One request: the reply, a transient failure to retry, or a raised permanent error."""
        try:
            with self._http.stream(
                "POST", self._config.completions_url, json=payload, timeout=timeout
            ) as response:
                if self._monotonic() >= deadline:
                    return _Transient(detail="deadline", past_deadline=True)
                if response.is_success:
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body += chunk
                        if self._monotonic() >= deadline:
                            return _Transient(detail="deadline", past_deadline=True)
                    return _parse_content(bytes(body), response.status_code)
        except httpx.TimeoutException as error:
            return _Transient(timed_out=True, detail="timeout", cause=error)
        except _RETRYABLE_TRANSPORT_ERRORS as error:
            return _Transient(detail=f"connection error: {type(error).__name__}", cause=error)
        except httpx.HTTPError as error:
            raise DependencyUnavailable(
                f"LLM request failed ({type(error).__name__})", _UNAVAILABLE
            ) from error
        status = response.status_code
        if status == httpx.codes.TOO_MANY_REQUESTS or response.is_server_error:
            return _Transient(detail=f"status {status}", retry_after=self._retry_after(response))
        raise DependencyUnavailable(f"LLM request failed (status {status})", _UNAVAILABLE)

    def _backoff(self, attempt: int) -> float:
        ceiling = min(
            self._config.backoff_cap_s, self._config.backoff_base_s * 2.0 ** min(attempt - 1, 64)
        )
        return self._rng() * ceiling

    def _retry_after(self, response: httpx.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        value = value.strip()
        seconds: float
        try:
            seconds = float(value)
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
            except (TypeError, ValueError, IndexError):
                return None
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            seconds = (when - self._clock()).total_seconds()
        if not math.isfinite(seconds):
            return None
        return min(max(seconds, 0.0), self._config.retry_after_cap_s)


@dataclass(frozen=True, slots=True)
class _Transient:
    """A failure worth retrying, with what the final error reports if it is the last one."""

    detail: str
    timed_out: bool = False
    cause: BaseException | None = None
    retry_after: float | None = None
    past_deadline: bool = False

    def final_error(self, attempts: int) -> DependencyTimeout | DependencyUnavailable:
        if self.timed_out:
            return DependencyTimeout(f"LLM request timed out after {attempts} attempts", _TIMEOUT)
        return DependencyUnavailable(
            f"LLM request failed after {attempts} attempts ({self.detail})", _UNAVAILABLE
        )


def _parse_content(body: bytes, status: int) -> str:
    malformed = DependencyUnavailable(f"LLM response is malformed (status {status})", _UNAVAILABLE)
    try:
        document = json.loads(body)
    except ValueError as error:
        raise malformed from error
    try:
        content = document["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise malformed from error
    if not isinstance(content, str):
        raise malformed
    return content
