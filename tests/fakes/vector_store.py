"""In-memory vector store with the identity binding and replace-on-reingest rules."""

import math
from collections.abc import Sequence

from app.domain.errors import IncompatibleEmbedding, ServiceError
from app.domain.types import Chunk, EmbeddingIdentity, SearchHit


class FakeVectorStore:
    """Pure-Python store; cosine similarity is clamped to 0..1, best hit first."""

    def __init__(self) -> None:
        self._identity: EmbeddingIdentity | None = None
        self._documents: dict[str, list[tuple[Chunk, list[float]]]] = {}
        self._failures_left = 0
        self._failure: ServiceError | None = None

    def fail_next(self, n: int, exc: ServiceError) -> None:
        """Make the next ``n`` calls to any port method raise ``exc`` without touching state."""
        self._failures_left = n
        self._failure = exc

    def ensure_collection(self, identity: EmbeddingIdentity) -> None:
        self._maybe_fail()
        if self._identity is None:
            self._identity = identity
        elif self._identity != identity:
            raise IncompatibleEmbedding("collection was created with another embedding identity")

    def upsert(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        self._maybe_fail()
        identity = self._bound_identity()
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len({chunk.sequence for chunk in chunks}) != len(chunks):
            raise ValueError("chunk sequences must be unique within a document")
        if any(len(vector) != identity.dimension for vector in vectors):
            raise IncompatibleEmbedding("vector dimension differs from the collection's")
        self._documents[document_id] = [
            (chunk, [float(value) for value in vector])
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def search(self, vector: Sequence[float], top_k: int) -> list[SearchHit]:
        self._maybe_fail()
        identity = self._bound_identity()
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if len(vector) != identity.dimension:
            raise IncompatibleEmbedding("query dimension differs from the collection's")
        hits = [
            SearchHit(document_id, chunk.page, chunk.sequence, chunk.text, _cosine(vector, stored))
            for document_id, entries in self._documents.items()
            for chunk, stored in entries
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.document_id, hit.sequence))
        return hits[:top_k]

    def delete_document(self, document_id: str) -> None:
        self._maybe_fail()
        self._documents.pop(document_id, None)

    def count(self) -> int:
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


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return min(1.0, max(0.0, dot / norm))
