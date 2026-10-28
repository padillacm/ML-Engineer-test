"""Use case: answer a question only from the stored documents, or refuse."""

import logging
import math
import time
from typing import Literal

from app.application.ports import Embedder, LLMClient, VectorStore
from app.domain.errors import ConfigurationError, InvalidInput
from app.domain.prompting import (
    REFUSAL,
    build_messages,
    is_refusal,
    select_context,
    shown_sources,
)
from app.domain.types import Answer

DEFAULT_THRESHOLD = 0.808
DEFAULT_TOP_K = 5
DEFAULT_MAX_CHUNKS = 3
DEFAULT_MAX_CONTEXT_CHARS = 6000

OUTCOME_MESSAGE = "question completed"

type RefusalReason = Literal["below_threshold", "no_context", "model"]
"""Why a question was refused: no hit at or above the threshold (none at all included), no
page fitting the context budget, or the model replying that the documents do not answer it."""

_logger = logging.getLogger(__name__)


class AskQuestion:
    """Retrieve the chunks most similar to a question and have the model answer from them.

    Hits scoring below ``threshold`` are discarded. When none is left (including an empty
    store), or when not even one page fits the context budget, the fixed ``REFUSAL`` is
    returned with no sources and the model is not called. Otherwise the parent pages of the
    kept hits (``select_context``) are sent with the grounding instructions
    (``build_messages``). The sources are the texts of the kept
    hits that the model saw (``shown_sources``), exactly as stored, in rank order. When the
    model's reply is a refusal (``is_refusal``: the ``NOT_IN_DOCUMENTS`` sentinel, or nothing),
    the answer is ``REFUSAL`` with no sources, since no retrieved text supports it. Port errors
    (for instance ``DependencyUnavailable`` or ``DependencyTimeout`` from the model) propagate
    unchanged.

    A successful call logs one ``OUTCOME_MESSAGE`` line: ``hits``, ``hits_kept``,
    ``top_score`` (best score to 4 decimals, null without hits), ``refused``,
    ``refusal_reason`` (a ``RefusalReason``, null when answered), ``sources``, ``embed_ms``,
    ``search_ms``, ``llm_ms`` (only when the model was called) and ``duration_ms``; never the
    question, the answer or a source text.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            raise ConfigurationError(f"threshold must be a number within 0..1, got {threshold!r}")
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ConfigurationError(f"threshold must be within 0..1, got {threshold}")
        for name, value in (
            ("top_k", top_k),
            ("max_chunks", max_chunks),
            ("max_context_chars", max_context_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"{name} must be an integer >= 1, got {value!r}")
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._threshold = float(threshold)
        self._top_k = top_k
        self._max_chunks = max_chunks
        self._max_context_chars = max_context_chars

    def __call__(self, question: str) -> Answer:
        """Answer ``question``.

        A question that is empty or only whitespace raises ``InvalidInput`` (code
        ``invalid_question``) before anything is embedded, searched or sent to the model.
        """
        if not question.strip():
            raise InvalidInput("question must not be blank", code="invalid_question")
        started = time.perf_counter()
        fields: dict[str, int | float | str | None] = {}
        answer, reason = self._answer(question, fields)
        _logger.info(
            OUTCOME_MESSAGE,
            extra={
                **fields,
                "refused": reason is not None,
                "refusal_reason": reason,
                "sources": len(answer.sources),
                "duration_ms": _elapsed_ms(started, time.perf_counter()),
            },
        )
        return answer

    def _answer(
        self, question: str, fields: dict[str, int | float | str | None]
    ) -> tuple[Answer, RefusalReason | None]:
        """The answer and why it is a refusal (``None`` when answered); fills the log fields."""
        self._store.ensure_collection(self._embedder.identity)
        begin = time.perf_counter()
        vector = self._embedder.embed_query(question)
        fields["embed_ms"] = _elapsed_ms(begin, time.perf_counter())
        begin = time.perf_counter()
        hits = self._store.search(vector, self._top_k)
        fields["search_ms"] = _elapsed_ms(begin, time.perf_counter())
        kept = [hit for hit in hits if hit.score >= self._threshold]
        fields["hits"] = len(hits)
        fields["hits_kept"] = len(kept)
        fields["top_score"] = round(max(hit.score for hit in hits), 4) if hits else None
        if not kept:
            return Answer(REFUSAL, ()), "below_threshold"

        pages = select_context(kept, max_chunks=self._max_chunks, max_chars=self._max_context_chars)
        if not pages:
            return Answer(REFUSAL, ()), "no_context"
        begin = time.perf_counter()
        reply = self._llm.complete(build_messages(question, pages)).strip()
        fields["llm_ms"] = _elapsed_ms(begin, time.perf_counter())
        if is_refusal(reply):
            return Answer(REFUSAL, ()), "model"
        return Answer(reply, shown_sources(kept, pages, max_chunks=self._max_chunks)), None


def _elapsed_ms(start: float, end: float) -> float:
    """Milliseconds between two ``time.perf_counter`` readings, to one decimal."""
    return round((end - start) * 1000, 1)
