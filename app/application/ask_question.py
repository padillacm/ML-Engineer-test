"""Use case: answer a question only from the stored documents, or refuse."""

import math

from app.application.ports import Embedder, LLMClient, VectorStore
from app.domain.errors import ConfigurationError
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
        """Answer ``question``; raises ``ValueError`` when it is blank (a caller's bug)."""
        if not question.strip():
            raise ValueError("question must not be blank")
        self._store.ensure_collection(self._embedder.identity)
        hits = self._store.search(self._embedder.embed_query(question), self._top_k)
        kept = [hit for hit in hits if hit.score >= self._threshold]
        if not kept:
            return Answer(REFUSAL, ())

        pages = select_context(kept, max_chunks=self._max_chunks, max_chars=self._max_context_chars)
        if not pages:
            return Answer(REFUSAL, ())
        reply = self._llm.complete(build_messages(question, pages)).strip()
        if is_refusal(reply):
            return Answer(REFUSAL, ())
        return Answer(reply, shown_sources(kept, pages, max_chunks=self._max_chunks))
