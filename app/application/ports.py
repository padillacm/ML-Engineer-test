"""Contracts through which the service core talks to external technologies.

Every port is a structural ``Protocol``: adapters and test fakes satisfy it without inheriting
from it. Ports speak only core types and standard Python types. Adapters wrap vendor exceptions
into ``ServiceError`` subclasses with fixed messages; the failure contract of each port lists
which ones.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.domain.types import (
    ChatMessage,
    Chunk,
    EmbeddingIdentity,
    OcrPage,
    SearchHit,
    TextSpan,
)


@runtime_checkable
class OcrEngine(Protocol):
    """Recognises text and word positions in a PDF.

    The adapter rasterises the pages and converts the engine's pixel boxes to page space, so no
    pixel coordinate or rasteriser type crosses this boundary.

    Failures: ``ProcessingFailed`` (code ``ocr_failed``) when a page cannot be read.
    """

    def recognize(self, pdf: bytes, language: str | None = None) -> list[OcrPage]:
        """Return one ``OcrPage`` per page of ``pdf``, in page order.

        ``language`` is an engine-neutral language hint; ``None`` lets the adapter decide.
        """
        ...


@runtime_checkable
class NameRecognizer(Protocol):
    """Finds person names in text.

    Failures: ``ProcessingFailed`` when the model cannot process the text.
    """

    def find_persons(self, text: str) -> list[TextSpan]:
        """Return character spans into ``text``, one per person name, in order of appearance."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into embedding vectors; query and passage prefixes live inside the adapter.

    Failures: ``DependencyUnavailable`` or ``DependencyTimeout`` when the model or the remote
    service cannot answer.
    """

    @property
    def identity(self) -> EmbeddingIdentity:
        """What makes vectors from this embedder interchangeable with another's."""
        ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector of ``identity.dimension`` floats per text, in the same order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Return the vector of a search query, in the same space as the passage vectors."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunk vectors and searches them by cosine similarity.

    A collection is bound to one ``EmbeddingIdentity``. Chunk ids are derived by the store from
    ``document_id`` and ``Chunk.sequence``, so they never collide across documents.

    Failures: ``IncompatibleEmbedding`` when the identity or a vector dimension does not match
    the collection (nothing is modified); ``DependencyUnavailable`` or ``DependencyTimeout`` when
    the backing service cannot answer.

    Programming errors raise ``ValueError``/``RuntimeError``, not ``ServiceError``: ``upsert``
    raises ``ValueError`` when ``chunks`` and ``vectors`` differ in length or when two chunks
    share a ``sequence`` (nothing is modified); ``search`` raises ``ValueError`` when
    ``top_k < 1``; ``upsert`` and ``search`` raise ``RuntimeError`` when called before
    ``ensure_collection``.
    """

    def ensure_collection(self, identity: EmbeddingIdentity) -> None:
        """Create the collection for ``identity``, or verify that an existing one has the same."""
        ...

    def upsert(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Store ``chunks`` with their ``vectors`` (parallel sequences) for ``document_id``.

        All chunks previously stored under the same ``document_id`` are replaced; other
        documents are untouched.
        """
        ...

    def search(self, vector: Sequence[float], top_k: int) -> list[SearchHit]:
        """Return up to ``top_k`` nearest chunks, best first, scored by cosine in 0.0..1.0.

        A negative cosine is reported as 0.0.
        """
        ...

    def delete_document(self, document_id: str) -> None:
        """Remove every chunk of ``document_id``; unknown documents are ignored."""
        ...

    def count(self) -> int:
        """Return the number of chunks stored across all documents."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    """Completes a chat conversation with a language model.

    Retries and timeouts are the adapter's concern.

    Failures: ``DependencyUnavailable`` (code ``llm_unavailable``) and ``DependencyTimeout``
    (code ``llm_timeout``) are the only errors it raises.
    """

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return the assistant's reply to ``messages``."""
        ...
