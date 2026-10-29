"""Use case: answer a question only from the stored documents, or refuse.

The question crosses two trust boundaries, the documents and the model, and each is guarded
here. Before anything is embedded, the injection detector screens the question (a question that
is itself an instruction to the model is refused, D7). After the model replies, two pure checks
from ``app.domain.guardrails`` run over the answer in this order (D9): the refusal sentinel
first (a refusal carries nothing to check), then the support share (a fabricated answer is
refused before any further reading of it), then the output filter (a supported answer can still
leak the instructions or point at a document that was never retrieved). The outcome line names
the reason and the numbers, never the texts.
"""

import logging
import math
import time
from typing import Literal

from app.application.no_detector import NoDetector
from app.application.ports import Embedder, InjectionDetector, LLMClient, VectorStore
from app.domain.errors import ConfigurationError, InvalidInput
from app.domain.guardrails import (
    DEFAULT_INJECTION_THRESHOLD,
    DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
    DEFAULT_SUPPORT_MIN_SHARE,
    output_filter,
    support_share,
    validate_threshold,
)
from app.domain.prompting import (
    REFUSAL,
    SYSTEM_PROMPT,
    ContextPage,
    build_messages,
    is_refusal,
    render_context,
    select_context,
    shown_sources,
)
from app.domain.types import Answer

DEFAULT_THRESHOLD = 0.808
DEFAULT_TOP_K = 5
DEFAULT_MAX_CHUNKS = 3
DEFAULT_MAX_CONTEXT_CHARS = 6000

OUTCOME_MESSAGE = "question completed"

type RefusalReason = Literal[
    "question_flagged", "below_threshold", "no_context", "model", "unsupported", "output_filtered"
]
"""Why a question was refused: the detector flagged the question; no hit at or above the
threshold (none at all included); no page fitting the context budget; the model replying that
the documents do not answer it; too few of the answer's content tokens in the context; or the
answer leaking the instructions or citing a URL, document or page the context does not hold."""

_logger = logging.getLogger(__name__)


class AskQuestion:
    """Retrieve the chunks most similar to a question and have the model answer from them.

    The question is screened by the injection ``detector`` first: a score at or above
    ``injection_threshold`` returns the fixed ``REFUSAL`` with no sources before anything is
    embedded, searched or sent to the model. Hits scoring below ``threshold`` are discarded.
    When none is left (including an empty store), or when not even one page fits the context
    budget, ``REFUSAL`` is returned with no sources and the model is not called. Otherwise the
    parent pages of the kept hits (``select_context``, at most ``max_chunks_per_document`` of
    one document when several matched) are sent as delimited passages with the grounding
    instructions (``build_messages``). The sources are the texts of the kept hits that the
    model saw (``shown_sources``), exactly as stored, in rank order. When the model's reply is a
    refusal (``is_refusal``: the ``NOT_IN_DOCUMENTS`` sentinel, or nothing), the answer is
    ``REFUSAL`` with no sources, since no retrieved text supports it. A reply whose
    ``support_share`` against the pages shown is below ``support_min_share``, or that the
    ``output_filter`` rejects, is replaced by ``REFUSAL`` with no sources too. Port errors (for
    instance ``DependencyUnavailable`` or ``DependencyTimeout`` from the model) propagate
    unchanged.

    A successful call logs one ``OUTCOME_MESSAGE`` line: ``injection_score`` (to 3 decimals),
    ``hits``, ``hits_kept``, ``top_score`` (best score to 4 decimals, null without hits),
    ``refused``, ``refusal_reason`` (a ``RefusalReason``, null when answered), ``sources``,
    ``embed_ms``, ``search_ms``, ``llm_ms`` (only when the model was called),
    ``support_share`` (to 3 decimals, only when computed), ``filter_reason`` (the rule the
    output filter applied, only then) and ``duration_ms``; never the question, the answer, a
    source text or the context.
    """

    def __init__(  # noqa: PLR0913 - every tunable of the use case, each validated
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        *,
        detector: InjectionDetector | None = None,
        injection_threshold: float = DEFAULT_INJECTION_THRESHOLD,
        support_min_share: float = DEFAULT_SUPPORT_MIN_SHARE,
        max_chunks_per_document: int = DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
    ) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            raise ConfigurationError(f"threshold must be a number within 0..1, got {threshold!r}")
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ConfigurationError(f"threshold must be within 0..1, got {threshold}")
        for name, value in (
            ("top_k", top_k),
            ("max_chunks", max_chunks),
            ("max_context_chars", max_context_chars),
            ("max_chunks_per_document", max_chunks_per_document),
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
        self._detector: InjectionDetector = detector if detector is not None else NoDetector()
        self._injection_threshold = validate_threshold("injection_threshold", injection_threshold)
        self._support_min_share = validate_threshold("support_min_share", support_min_share)
        self._max_chunks_per_document = max_chunks_per_document

    def __call__(self, question: str) -> Answer:
        """Answer ``question``.

        A question that is empty or only whitespace raises ``InvalidInput`` (code
        ``invalid_question``) before anything is screened, embedded, searched or sent to the
        model.
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
        score = self._detector.screen_query(question)
        fields["injection_score"] = round(score, 3)
        if score >= self._injection_threshold:
            return Answer(REFUSAL, ()), "question_flagged"

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

        pages = select_context(
            kept,
            max_chunks=self._max_chunks,
            max_chars=self._max_context_chars,
            max_chunks_per_document=self._max_chunks_per_document,
        )
        if not pages:
            return Answer(REFUSAL, ()), "no_context"
        begin = time.perf_counter()
        reply = self._llm.complete(build_messages(question, pages)).strip()
        fields["llm_ms"] = _elapsed_ms(begin, time.perf_counter())
        if is_refusal(reply):
            return Answer(REFUSAL, ()), "model"

        reason = self._check_reply(reply, pages, fields)
        if reason is not None:
            return Answer(REFUSAL, ()), reason
        sources = shown_sources(
            kept,
            pages,
            max_chunks=self._max_chunks,
            max_chunks_per_document=self._max_chunks_per_document,
        )
        return Answer(reply, sources), None

    def _check_reply(
        self,
        reply: str,
        pages: list[ContextPage],
        fields: dict[str, int | float | str | None],
    ) -> RefusalReason | None:
        """Why ``reply`` must not be returned as it is, judged against the ``pages`` shown.

        The support share comes first: an answer the context does not support is refused
        whatever else it says. The output filter then catches what a supported answer can
        still do wrong (quote the instructions, cite a foreign URL, document or page).
        """
        texts = [page.text for page in pages]
        # The share is judged against the context as the model saw it, labels included: the
        # model often copies a passage label ("<<<passage 1: document <id>, page 1>>>") into a
        # correct answer, and the id's hex segments would otherwise count as unsupported
        # tokens (BENCHMARKS 22: 10 of the 12 false refusals of the measurement).
        share = support_share(reply, [render_context(pages)])
        if share is not None:
            fields["support_share"] = round(share, 3)
            if share < self._support_min_share:
                return "unsupported"
        shown = [(page.document_id, page.page + 1) for page in pages]
        rule = output_filter(reply, texts, SYSTEM_PROMPT, shown_pages=shown)
        if rule is not None:
            fields["filter_reason"] = rule
            return "output_filtered"
        return None


def _elapsed_ms(start: float, end: float) -> float:
    """Milliseconds between two ``time.perf_counter`` readings, to one decimal."""
    return round((end - start) * 1000, 1)
