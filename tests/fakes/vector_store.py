"""In-memory vector store with the identity binding and idempotent, completeness-aware upsert.

Like the Qdrant adapter it keeps each chunk's ``flagged`` mark, never returns a flagged chunk
from ``search`` and counts, deletes and repairs flagged chunks like any other.
"""

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.domain.errors import IncompatibleEmbedding, ServiceError
from app.domain.types import Chunk, EmbeddingIdentity, SearchHit


@dataclass(frozen=True, slots=True)
class _StoredChunk:
    chunk: Chunk
    vector: list[float]
    page_text: str
    content_hash: str
    chunk_total: int


class FakeVectorStore:
    """Pure-Python store; cosine similarity is clamped to 0..1, best hit first.

    Like the real adapter, ``upsert`` writes chunk by chunk without deleting first and removes
    the document's other chunks only after writing all of them, so a write that fails part-way
    leaves an incomplete document (``document_chunk_count`` 0) or another writer's complete one.
    A chunk stored with ``flagged`` set is kept (and counted) but excluded from every search.
    Safe to share between threads.
    """

    def __init__(self) -> None:
        self._identity: EmbeddingIdentity | None = None
        self._documents: dict[str, dict[int, _StoredChunk]] = {}
        self._failures_left = 0
        self._failure: ServiceError | None = None
        self._partial_write: tuple[int, ServiceError] | None = None
        self._lock = threading.RLock()

    def fail_next(self, n: int, exc: ServiceError) -> None:
        """Make the next ``n`` calls to any port method raise ``exc`` without touching state."""
        with self._lock:
            self._failures_left = n
            self._failure = exc

    def fail_upsert_after(self, stored: int, exc: ServiceError) -> None:
        """Make the next ``upsert`` store its first ``stored`` chunks, then raise ``exc``."""
        with self._lock:
            self._partial_write = (stored, exc)

    def drop_chunk(self, document_id: str, sequence: int) -> None:
        """Test helper: lose one stored chunk behind the store's back."""
        with self._lock:
            self._documents.get(document_id, {}).pop(sequence, None)

    def ensure_collection(self, identity: EmbeddingIdentity) -> None:
        with self._lock:
            self._maybe_fail()
            if self._identity is None:
                self._identity = identity
            elif self._identity != identity:
                raise IncompatibleEmbedding(
                    "collection was created with another embedding identity"
                )

    def upsert(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        page_texts: Mapping[int, str],
        content_hash: str,
    ) -> None:
        with self._lock:
            self._maybe_fail()
            _validate(self._bound_identity(), chunks, vectors, page_texts, content_hash)
            partial, self._partial_write = self._partial_write, None
        total = len(chunks)
        for written, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            if partial is not None and written >= partial[0]:
                raise partial[1]
            entry = _StoredChunk(
                chunk, [float(x) for x in vector], page_texts[chunk.page], content_hash, total
            )
            with self._lock:  # one chunk at a time, so concurrent writers interleave
                self._documents.setdefault(document_id, {})[chunk.sequence] = entry
        if partial is not None:
            raise partial[1]
        keep = {chunk.sequence for chunk in chunks}
        with self._lock:
            stored = self._documents.get(document_id, {})
            for sequence in [sequence for sequence in stored if sequence not in keep]:
                del stored[sequence]
            if not stored:
                self._documents.pop(document_id, None)

    def search(self, vector: Sequence[float], top_k: int) -> list[SearchHit]:
        with self._lock:
            self._maybe_fail()
            identity = self._bound_identity()
            if top_k < 1:
                raise ValueError(f"top_k must be >= 1, got {top_k}")
            if len(vector) != identity.dimension:
                raise IncompatibleEmbedding("query dimension differs from the collection's")
            hits = [
                SearchHit(
                    document_id,
                    entry.chunk.page,
                    entry.chunk.sequence,
                    entry.chunk.text,
                    entry.page_text,
                    _cosine(vector, entry.vector),
                )
                for document_id, entries in self._documents.items()
                for entry in entries.values()
                if not entry.chunk.flagged
            ]
        hits.sort(key=lambda hit: (-hit.score, hit.document_id, hit.sequence))
        return hits[:top_k]

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self._maybe_fail()
            self._documents.pop(document_id, None)

    def document_chunk_count(self, document_id: str) -> int:
        with self._lock:
            self._maybe_fail()
            self._bound_identity()
            entries = list(self._documents.get(document_id, {}).values())
        totals = {entry.chunk_total for entry in entries}
        if len(totals) != 1:
            return 0
        (total,) = totals
        return total if len(entries) == total else 0

    def chunks_of(self, document_id: str) -> list[Chunk]:
        """Test helper: the stored chunks of ``document_id``, in sequence order."""
        with self._lock:
            entries = self._documents.get(document_id, {})
            return [entries[sequence].chunk for sequence in sorted(entries)]

    def flagged_sequences(self, document_id: str) -> set[int]:
        """Test helper: the sequences of the flagged chunks of ``document_id``."""
        with self._lock:
            entries = self._documents.get(document_id, {})
            return {sequence for sequence, entry in entries.items() if entry.chunk.flagged}

    def content_hashes(self, document_id: str) -> set[str]:
        """Test helper: the content hashes stored with the chunks of ``document_id``."""
        with self._lock:
            return {entry.content_hash for entry in self._documents.get(document_id, {}).values()}

    def count(self) -> int:
        with self._lock:
            self._maybe_fail()
            return sum(len(entries) for entries in self._documents.values())

    def _bound_identity(self) -> EmbeddingIdentity:
        if self._identity is None:
            raise RuntimeError("ensure_collection must be called before using the store")
        return self._identity

    def _maybe_fail(self) -> None:
        if self._failures_left > 0 and self._failure is not None:
            self._failures_left -= 1
            raise self._failure


def _validate(
    identity: EmbeddingIdentity,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    page_texts: Mapping[int, str],
    content_hash: str,
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")
    if len({chunk.sequence for chunk in chunks}) != len(chunks):
        raise ValueError("chunk sequences must be unique within a document")
    if any(len(vector) != identity.dimension for vector in vectors):
        raise IncompatibleEmbedding("vector dimension differs from the collection's")
    missing = {chunk.page for chunk in chunks} - set(page_texts)
    if missing:
        raise ValueError(f"page_texts lacks the text of pages {sorted(missing)}")
    if not content_hash:
        raise ValueError("content_hash must not be empty")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return min(1.0, max(0.0, dot / norm))
