"""Use case: turn an uploaded PDF into stored, retrievable chunks identified by its content."""

import hashlib
import uuid
from collections.abc import Sequence

from app.application.ports import Embedder, OcrEngine, VectorStore
from app.domain.chunking import ChunkingConfig, chunk_page, clean_page_text
from app.domain.errors import ConfigurationError
from app.domain.types import Chunk, IngestResult, OcrPage

DOCUMENT_NAMESPACE = uuid.UUID("6f1c3a52-2d4e-5b8a-9c71-0e4d2b9f8a13")
"""Fixed namespace of document identifiers: ``uuid5(DOCUMENT_NAMESPACE, sha256 hex digest)``.

Changing it changes every document identifier, so already stored documents would no longer be
recognised as indexed and would be stored a second time.
"""

DEFAULT_BATCH_SIZE = 32


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

    def __call__(self, pdf: bytes) -> IngestResult:
        content_hash = hashlib.sha256(pdf).hexdigest()
        document_id = document_id_for(content_hash)
        self._store.ensure_collection(self._embedder.identity)
        existing = self._store.document_chunk_count(document_id)
        if existing > 0:
            return IngestResult(document_id, existing, already_indexed=True)

        page_texts = {
            page.page: clean_page_text(_raw_page_text(page)) for page in self._ocr.recognize(pdf)
        }
        chunks: list[Chunk] = []
        for page_index, text in sorted(page_texts.items()):
            chunks.extend(chunk_page(text, page_index, self._chunking, len(chunks)))
        if not chunks:
            return IngestResult(document_id, 0, already_indexed=False)

        vectors = self._embed([chunk.text for chunk in chunks])
        stored_pages = {chunk.page for chunk in chunks}
        self._store.upsert(
            document_id,
            chunks,
            vectors,
            page_texts={page: text for page, text in page_texts.items() if page in stored_pages},
            content_hash=content_hash,
        )
        return IngestResult(document_id, len(chunks), already_indexed=False)

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embedder.embed_passages(texts[start : start + self._batch_size]))
        return vectors


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
