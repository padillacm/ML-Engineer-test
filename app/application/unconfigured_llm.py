"""The language-model port of a service started without a language model."""

from collections.abc import Sequence

from app.domain.errors import DependencyUnavailable
from app.domain.types import ChatMessage

NOT_CONFIGURED_MESSAGE = "no language model is configured"


class UnconfiguredLLM:
    """``LLMClient`` that fails every call with ``DependencyUnavailable`` (``llm_unavailable``).

    It stands in for the model when none is configured, so ``AskQuestion`` still refuses
    questions that retrieve nothing relevant (it never calls the model then) and fails only
    when it would send a question to the model.
    """

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Raise ``DependencyUnavailable`` (code ``llm_unavailable``); nothing is sent."""
        del messages
        raise DependencyUnavailable(NOT_CONFIGURED_MESSAGE, code="llm_unavailable")
