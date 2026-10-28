"""Cooperative cancellation of a long job.

The HTTP edge creates one ``Cancellation`` per extraction or ingestion request and cancels it
when the client disconnects; the use cases and the OCR engine call ``raise_if_cancelled``
between pages (and before each embedding batch), so an abandoned job stops after the step in
progress. Python threads cannot be interrupted, so a step that has started always finishes.
Thread-safe: it is cancelled on the event loop and checked in a worker thread.
"""

import threading

from app.domain.errors import ProcessingFailed

CANCELLED_MESSAGE = "the request was cancelled"


class Cancellation:
    """A one-way flag: once cancelled it stays cancelled."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise ``ProcessingFailed`` (code ``request_cancelled``) once cancelled."""
        if self._event.is_set():
            raise ProcessingFailed(CANCELLED_MESSAGE, code="request_cancelled")


class _NeverCancelled(Cancellation):
    def cancel(self) -> None:
        raise TypeError("the shared default token cannot be cancelled; create a Cancellation")


NEVER_CANCELLED: Cancellation = _NeverCancelled()
"""Shared default for callers without a request (scripts, benchmarks): never cancelled."""
