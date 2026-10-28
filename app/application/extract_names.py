"""Use case: find the person names printed in a PDF, locate them and match them to a request."""

import logging
import time
from collections.abc import Sequence

from app.application.ports import NameRecognizer, OcrEngine
from app.domain.cancellation import NEVER_CANCELLED, Cancellation
from app.domain.errors import ProcessingFailed
from app.domain.geometry import PageText, clamp, union
from app.domain.matching import DEFAULT_THRESHOLD, match_names, normalize
from app.domain.names import clean_mentions, title_case_caps
from app.domain.types import ExtractionResult, NameMention, NamePair, OcrPage

OUTCOME_MESSAGE = "extraction completed"
CANCELLED_MESSAGE = "extraction cancelled"

_logger = logging.getLogger(__name__)


class ExtractNames:
    """Extract person-name mentions from a PDF and fuzzy-match them to requested names.

    Only the ``OcrEngine`` and ``NameRecognizer`` ports and domain code are used. For every page,
    in page order, the OCR words are joined into the page text, the recogniser reads its
    ``title_case_caps`` copy (same length, so its spans index the original text), the domain
    span rules turn the spans into word ranges, and each mention is reported with the original
    words of its range. Its box is the union of those word boxes clamped to the page, in PDF
    points with a top-left origin, x growing right and y growing down, on the 0-based page.
    The span rules use the language the OCR engine reports for the page (``OcrPage.language``),
    not the caller's hint, so a detected language counts as much as a given one.

    Matches are computed once per distinct extracted name (distinct after ``normalize``, first
    spelling and first-occurrence order kept). Port errors propagate unchanged.

    A successful call logs one ``OUTCOME_MESSAGE`` line with counts and durations only
    (``size_bytes``, ``pages``, ``names_found``, ``requested_names``, ``matches``, ``ocr_ms``,
    ``ner_ms``, ``duration_ms``), never a name or any text of the document.

    ``cancel`` is passed to the OCR engine and checked again before the recogniser reads each
    page, so an abandoned request stops after the step in progress with ``ProcessingFailed``
    (code ``request_cancelled``). A cancelled call logs one ``CANCELLED_MESSAGE`` line instead
    of the outcome line (``code``, ``size_bytes``, ``duration_ms``).
    """

    def __init__(
        self,
        ocr: OcrEngine,
        recognizer: NameRecognizer,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        match_names([], [], threshold)  # raises ConfigurationError on an invalid threshold
        self._ocr = ocr
        self._recognizer = recognizer
        self._threshold = float(threshold)

    def __call__(
        self,
        pdf: bytes,
        requested: Sequence[NamePair],
        language: str | None = None,
        *,
        cancel: Cancellation = NEVER_CANCELLED,
    ) -> ExtractionResult:
        """Return the mentions of ``pdf`` in reading order and their matches to ``requested``.

        ``language`` is the hint passed to the OCR engine; the title rule (French "M.") follows
        the language each page reports having been read with.
        """
        started = time.perf_counter()
        try:
            pages = self._ocr.recognize(pdf, language, cancel=cancel)
            recognized = time.perf_counter()
            mentions: list[NameMention] = []
            for page in pages:
                cancel.raise_if_cancelled()
                mentions.extend(self._page_mentions(page))
        except ProcessingFailed as error:
            if error.code == "request_cancelled":
                _log_cancelled(len(pdf), started)
            raise
        names = tuple(mentions)
        found = time.perf_counter()
        matches = (
            tuple(match_names(_distinct([m.name for m in names]), requested, self._threshold))
            if requested
            else ()
        )
        _logger.info(
            OUTCOME_MESSAGE,
            extra={
                "size_bytes": len(pdf),
                "pages": len(pages),
                "names_found": len(names),
                "requested_names": len(requested),
                "matches": len(matches),
                "ocr_ms": _elapsed_ms(started, recognized),
                "ner_ms": _elapsed_ms(recognized, found),
                "duration_ms": _elapsed_ms(started, time.perf_counter()),
            },
        )
        return ExtractionResult(names=names, matches=matches)

    def _page_mentions(self, page: OcrPage) -> list[NameMention]:
        page_text = PageText.from_words(page.words)
        if not page_text.text:
            return []
        spans = self._recognizer.find_persons(title_case_caps(page_text.text))
        mentions: list[NameMention] = []
        for name, word_range in clean_mentions(page_text, spans, language=page.language):
            words = page_text.words[word_range.start_word : word_range.end_word]
            box = clamp(union([word.box for word in words]), page.width, page.height)
            mentions.append(NameMention(name=name, box=box))
        return mentions


def _log_cancelled(size_bytes: int, started: float) -> None:
    _logger.info(
        CANCELLED_MESSAGE,
        extra={
            "code": "request_cancelled",
            "size_bytes": size_bytes,
            "duration_ms": _elapsed_ms(started, time.perf_counter()),
        },
    )


def _distinct(names: Sequence[str]) -> list[str]:
    """Names distinct by ``normalize``, keeping the first spelling in first-occurrence order."""
    seen: set[str] = set()
    distinct: list[str] = []
    for name in names:
        key = normalize(name)
        if key not in seen:
            seen.add(key)
            distinct.append(name)
    return distinct


def _elapsed_ms(start: float, end: float) -> float:
    """Milliseconds between two ``time.perf_counter`` readings, to one decimal."""
    return round((end - start) * 1000, 1)
