"""In-memory chat-completion client with queued answers and injectable failures."""

from collections import deque
from collections.abc import Iterable, Sequence

from app.domain.errors import ServiceError
from app.domain.types import ChatMessage


class FakeLLMClient:
    """Replies with queued answers, then with ``default``; a queued error is raised instead."""

    def __init__(
        self,
        answers: Iterable[str | ServiceError] = (),
        default: str = "fake answer",
    ) -> None:
        self._queue: deque[str | ServiceError] = deque(answers)
        self._default = default
        self.calls: list[list[ChatMessage]] = []

    def enqueue(self, answer: str | ServiceError) -> None:
        """Add an answer, or an error to raise, after those already queued."""
        self._queue.append(answer)

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        answer = self._queue.popleft() if self._queue else self._default
        if isinstance(answer, ServiceError):
            raise answer
        return answer
