"""Contracts through which the service core talks to external technologies.

Every port is a structural ``Protocol``: adapters and test fakes satisfy it without inheriting
from it. Ports speak only core types and standard Python types. Adapters wrap vendor exceptions
into ``ServiceError`` subclasses with fixed messages; the failure contract of each port lists
which ones.
"""

from collections.abc import Mapping, Sequence
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

    Failures: ``ProcessingFailed`` (code ``ner_failed``) when the model cannot process the text.
    """

    def find_persons(self, text: str) -> list[TextSpan]:
        """Return character spans into ``text``, one per person name, in order of appearance."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into embedding vectors; query and passage prefixes live inside the adapter.

    Failures: ``ProcessingFailed`` (code ``embedding_failed``) when the model itself fails on the
    input (an inference error, or a text over the model's limit); ``DependencyUnavailable`` or
    ``DependencyTimeout`` when the model or the remote service cannot answer.
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
    share a ``sequence``, when ``page_texts`` lacks the page of a chunk or when ``content_hash``
    is empty (nothing is modified); ``search`` raises ``ValueError`` when ``top_k < 1``;
    ``upsert``, ``search`` and ``document_chunk_count`` raise ``RuntimeError`` when called before
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
        *,
        page_texts: Mapping[int, str],
        content_hash: str,
    ) -> None:
        """Store ``chunks`` with their ``vectors`` (parallel sequences) for ``document_id``.

        ``page_texts`` maps each chunk's page index to the cleaned text of that page; it is
        stored with the chunk and returned as ``SearchHit.page_text``. ``content_hash``
        identifies the source file's bytes and is stored with every chunk, and so is the chunk
        total ``len(chunks)`` of this write.

        Idempotent and safe against concurrent writers of the same document: chunks are
        overwritten by id without deleting the document first, and only once all of them are
        written are the document's other chunks (those beyond the new chunk count) removed; other
        documents are untouched. If writing fails part-way, nothing is removed and the error is
        raised: the document may then be incomplete (``document_chunk_count`` returns 0) or still
        hold a complete copy written by another writer; storing it again completes it.
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

    def document_chunk_count(self, document_id: str) -> int:
        """Return the number of chunks of a completely stored ``document_id``.

        Complete means every stored chunk of the document records the same chunk total and that
        many chunks are stored. Returns 0 when the document is unknown or incomplete (for example
        after an interrupted ``upsert``).
        """
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
