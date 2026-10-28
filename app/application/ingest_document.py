"""Use case: turn an uploaded PDF into stored, retrievable chunks identified by its content."""

import hashlib
import logging
import time
import uuid
from collections.abc import Sequence

from app.application.ports import Embedder, OcrEngine, VectorStore
from app.domain.cancellation import NEVER_CANCELLED, Cancellation
from app.domain.chunking import ChunkingConfig, chunk_page, clean_page_text
from app.domain.errors import ConfigurationError, ProcessingFailed
from app.domain.types import Chunk, IngestResult, OcrPage

DOCUMENT_NAMESPACE = uuid.UUID("6f1c3a52-2d4e-5b8a-9c71-0e4d2b9f8a13")
"""Fixed namespace of document identifiers: ``uuid5(DOCUMENT_NAMESPACE, sha256 hex digest)``.

Changing it changes every document identifier, so already stored documents would no longer be
recognised as indexed and would be stored a second time.
"""

DEFAULT_BATCH_SIZE = 32

OUTCOME_MESSAGE = "ingestion completed"
CANCELLED_MESSAGE = "ingestion cancelled"

_logger = logging.getLogger(__name__)


def document_id_for(content_hash: str) -> str:
    """The document identifier derived from the SHA-256 hex digest of a file's bytes."""
    return str(uuid.uuid5(DOCUMENT_NAMESPACE, content_hash))


class IngestDocument:
    """Recognise, chunk, embed and store a PDF, unless its exact bytes are already stored.

    The document identifier comes from the SHA-256 of the bytes, so the same file (under any
    name) is never stored twice and different files never share an identifier. When chunks are
    already stored for it, the stored count is reported and neither OCR nor the embedder runs.

    Each page's text is rebuilt from its OCR words, one line per recognised line and a blank
    line between recognised paragraphs, and cleaned with ``clean_page_text`` (so words
    hyphenated across a line break are joined and paragraphs stay separated). That cleaned
    text is both what the chunk offsets refer to and the page text stored with the chunks.
    Every vector is computed before anything is written, so a failing embedder leaves the store
    untouched. Port errors propagate unchanged.

    A successful call logs one ``OUTCOME_MESSAGE`` line: ``size_bytes``, ``document_id``,
    ``already_indexed``, ``chunks_stored`` and ``duration_ms``, plus ``pages`` and ``ocr_ms``
    when OCR ran and ``embed_ms`` and ``store_ms`` when chunks were embedded and stored. No
    text of the document is logged.

    ``cancel`` is passed to the OCR engine and checked again before the OCR results are
    processed, before each embedding batch and just before storing; once storing has begun
    it runs to the end, so a cancelled ingest never leaves a partly stored document. A
    cancelled call raises ``ProcessingFailed`` (code ``request_cancelled``) and logs one
    ``CANCELLED_MESSAGE`` line instead of the outcome line (``code``, ``size_bytes``,
    ``document_id``, ``duration_ms``).
    """

    def __init__(
        self,
        ocr: OcrEngine,
        embedder: Embedder,
        store: VectorStore,
        chunking: ChunkingConfig | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ConfigurationError(f"batch_size must be an integer >= 1, got {batch_size!r}")
        self._ocr = ocr
        self._embedder = embedder
        self._store = store
        self._chunking = chunking if chunking is not None else ChunkingConfig()
        self._batch_size = batch_size

    def __call__(self, pdf: bytes, *, cancel: Cancellation = NEVER_CANCELLED) -> IngestResult:
        started = time.perf_counter()
        content_hash = hashlib.sha256(pdf).hexdigest()
        document_id = document_id_for(content_hash)
        self._store.ensure_collection(self._embedder.identity)
        existing = self._store.document_chunk_count(document_id)
        timings: dict[str, int | float] = {}
        if existing > 0:
            result = IngestResult(document_id, existing, already_indexed=True)
        else:
            try:
                result = self._index(pdf, document_id, content_hash, timings, cancel)
            except ProcessingFailed as error:
                if error.code == "request_cancelled":
                    _log_cancelled(len(pdf), document_id, started)
                raise
        _logger.info(
            OUTCOME_MESSAGE,
            extra={
                "size_bytes": len(pdf),
                "document_id": result.document_id,
                "already_indexed": result.already_indexed,
                "chunks_stored": result.chunks_stored,
                **timings,
                "duration_ms": _elapsed_ms(started, time.perf_counter()),
            },
        )
        return result

    def _index(
        self,
        pdf: bytes,
        document_id: str,
        content_hash: str,
        timings: dict[str, int | float],
        cancel: Cancellation,
    ) -> IngestResult:
        """Recognise, chunk, embed and store ``pdf``; record page count and stage timings."""
        begin = time.perf_counter()
        pages = self._ocr.recognize(pdf, cancel=cancel)
        timings["pages"] = len(pages)
        timings["ocr_ms"] = _elapsed_ms(begin, time.perf_counter())
        cancel.raise_if_cancelled()
        page_texts = {page.page: clean_page_text(_raw_page_text(page)) for page in pages}
        chunks: list[Chunk] = []
        for page_index, text in sorted(page_texts.items()):
            chunks.extend(chunk_page(text, page_index, self._chunking, len(chunks)))
        if not chunks:
            return IngestResult(document_id, 0, already_indexed=False)

        begin = time.perf_counter()
        vectors = self._embed([chunk.text for chunk in chunks], cancel)
        timings["embed_ms"] = _elapsed_ms(begin, time.perf_counter())
        stored_pages = {chunk.page for chunk in chunks}
        # Last check: once upsert starts it runs to the end, never leaving a partial document.
        cancel.raise_if_cancelled()
        begin = time.perf_counter()
        self._store.upsert(
            document_id,
            chunks,
            vectors,
            page_texts={page: text for page, text in page_texts.items() if page in stored_pages},
            content_hash=content_hash,
        )
        timings["store_ms"] = _elapsed_ms(begin, time.perf_counter())
        return IngestResult(document_id, len(chunks), already_indexed=False)

    def _embed(self, texts: Sequence[str], cancel: Cancellation) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            cancel.raise_if_cancelled()
            vectors.extend(self._embedder.embed_passages(texts[start : start + self._batch_size]))
        return vectors


def _log_cancelled(size_bytes: int, document_id: str, started: float) -> None:
    _logger.info(
        CANCELLED_MESSAGE,
        extra={
            "code": "request_cancelled",
            "size_bytes": size_bytes,
            "document_id": document_id,
            "duration_ms": _elapsed_ms(started, time.perf_counter()),
        },
    )


def _raw_page_text(page: OcrPage) -> str:
    """The page's non-blank words, keeping the paragraphs and lines recognised on the page.

    Words are joined by spaces within a line, lines by a line break within a paragraph and
    paragraphs by a blank line, so ``clean_page_text`` keeps the paragraphs.
    """
    paragraphs: list[list[list[str]]] = []
    previous: tuple[int, int] | None = None
    for word in page.words:
        if not word.text.strip():
            continue
        if previous is None or word.paragraph != previous[0]:
            paragraphs.append([[]])
        elif word.line != previous[1]:
            paragraphs[-1].append([])
        paragraphs[-1][-1].append(word.text)
        previous = (word.paragraph, word.line)
    return "\n\n".join("\n".join(" ".join(words) for words in lines) for lines in paragraphs)


def _elapsed_ms(start: float, end: float) -> float:
    """Milliseconds between two ``time.perf_counter`` readings, to one decimal."""
    return round((end - start) * 1000, 1)
