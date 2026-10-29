"""In-memory injection detector: scores by marker phrases, records calls, injectable failure."""

from collections.abc import Iterable, Sequence

from app.domain.errors import ProcessingFailed, ServiceError

INJECTION_MARKERS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignora las instrucciones anteriores",
    "ignore as instruções anteriores",
    "ignore les instructions précédentes",
    "ignoriere die vorherigen anweisungen",
    "system prompt",
    "you are now",
)
"""Phrases the fake scores as injections (case-insensitive substring match)."""


class FakeInjectionDetector:
    """Scores ``high`` for a text containing a marker phrase and ``low`` otherwise.

    ``markers`` replaces the default phrases. A queued ``ServiceError`` is raised by the next
    call instead of scoring (then cleared), and ``fail_with`` raises on every call.
    """

    def __init__(
        self,
        markers: Iterable[str] = INJECTION_MARKERS,
        *,
        high: float = 0.99,
        low: float = 0.01,
        configured: bool = True,
        fail_with: ServiceError | None = None,
    ) -> None:
        self._markers = tuple(marker.casefold() for marker in markers)
        self._high = high
        self._low = low
        self._configured = configured
        self._fail_with = fail_with
        self._next_error: ServiceError | None = None
        self.calls: list[list[str]] = []
        self.query_calls: list[str] = []
        self.closed = False

    @property
    def configured(self) -> bool:
        return self._configured

    def fail_next(self, error: ServiceError | None = None) -> None:
        """Raise ``error`` (default: ``detector_failed``) on the next call only."""
        self._next_error = error if error is not None else _failure()

    def screen(self, texts: Sequence[str]) -> list[float]:
        self.calls.append(list(texts))
        if self._next_error is not None:
            error, self._next_error = self._next_error, None
            raise error
        if self._fail_with is not None:
            raise self._fail_with
        return [self._score(text) for text in texts]

    def screen_query(self, text: str) -> float:
        """Recorded in ``query_calls`` and in ``calls``, scored as by ``screen``."""
        self.query_calls.append(text)
        return self.screen([text])[0]

    def close(self) -> None:
        self.closed = True

    def _score(self, text: str) -> float:
        folded = text.casefold()
        return self._high if any(marker in folded for marker in self._markers) else self._low


def _failure() -> ProcessingFailed:
    return ProcessingFailed("the injection detector failed", "detector_failed")
