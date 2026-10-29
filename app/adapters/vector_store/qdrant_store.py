"""Qdrant implementation of the ``VectorStore`` port (design D3).

Collections: one per embedding identity, named ``{prefix}_{sha256(identity)[:12]}`` where the
identity is the canonical JSON of the ``EmbeddingIdentity`` fields (sorted keys, no spaces). The
same JSON is written in the collection's ``metadata`` (key ``embedding_identity``, Qdrant >= 1.16)
when the adapter creates it; ``ensure_collection`` refuses a collection whose recorded identity,
vector size or distance differs, without modifying it. A store instance binds to the first
identity it is opened with, like the port's fakes.

Concurrent first use: the first binding runs under a lock, so simultaneous first calls in one
process check and create the collection once; later calls read the binding without the lock.
Separate processes (replicas) still race on the creation. Qdrant 1.20 then answers the losing
create with 409, and for some milliseconds after a create is accepted the collection exists but
``get_collection`` answers 500 ``0 of 0 read operations failed`` (its shards serve no reads yet).
A 409 is therefore taken for "created elsewhere" and the collection verified, and the
``get_collection`` calls of the binding wait that 500 out (``_SETTLE_*``: up to 10 tries, backoff
0.05-0.5 s, within the deadline). Creating the existing ``document_id`` index again succeeds.

Point ids: ``uuid5(CHUNK_ID_NAMESPACE, f"{document_id}:{sequence}")``. The namespace is a fixed
constant (itself ``uuid5(NAMESPACE_URL, "urn:docintel:vector-store:chunk-id")``); changing it
would orphan every stored point, so it must never change.

Payload of every point: ``document_id``, ``page``, ``sequence``, ``text``, ``page_text``,
``start``, ``end``, ``content_hash`` and ``chunk_total`` (the number of chunks of the write that
stored it), plus ``flagged: true`` on the chunks the injection detector flagged (D6). The field
is written only when true, so points stored before it existed (no field) read as not flagged
and stay searchable. ``search`` adds ``must_not: [flagged = true]`` to every query, so a flagged
chunk is never returned whatever its similarity; counting, deletion and the chunk total ignore
the flag, so a document with flagged chunks is complete, idempotent and repairable like any
other. Payload indexes (server mode only, the embedded local mode has none and filters without
them): keyword on ``document_id`` for deletion and counting, bool on ``flagged`` for the search
filter; both are created on first use and verified on later ones.

Upsert is idempotent and never deletes before writing: it overwrites the document's points by id
in batches with ``wait=True``, then deletes only the document's points that this write did not
store (those beyond the new chunk count). When a batch fails nothing is deleted: another writer
of the same document may have completed, and the partial state is recognisable because the
points' ``chunk_total`` and their count disagree. ``document_chunk_count`` reports only complete
documents (every point agrees on ``chunk_total`` and there are that many), 0 otherwise.

Every client call goes through ``Retrier`` (see ``retry``): connection errors, timeouts and HTTP
502/503/504 are retried within an attempt limit and a deadline, other client errors fail at once,
all as ``DependencyUnavailable`` / ``DependencyTimeout`` with fixed messages. The readiness ping
tries once, through a second REST client whose timeout is ``ping_timeout_s``, so a probe of a hung
server returns within about that time.

Lost collection: when a call on the bound collection fails because the collection no longer
exists (the server lost its storage, or someone dropped it), the collection is recreated once
with the bound identity (under the same lock, and verified like a first binding if another
client recreated it first) and the call is retried once.

Thread safety: one store may be shared by threads, as the REST client may; the ``:memory:``
local mode is meant for single-threaded tests.
"""

import dataclasses
import hashlib
import json
import logging
import math
import random
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.adapters.vector_store.config import QdrantStoreConfig
from app.adapters.vector_store.retry import Retrier, RetryPolicy
from app.domain.errors import DependencyUnavailable, IncompatibleEmbedding
from app.domain.types import Chunk, EmbeddingIdentity, SearchHit

CHUNK_ID_NAMESPACE = uuid.UUID("af2c2b75-7ad5-5066-9b3f-0a70af9f6a3f")
IDENTITY_METADATA_KEY = "embedding_identity"
DOCUMENT_ID_FIELD = "document_id"
CHUNK_TOTAL_FIELD = "chunk_total"
FLAGGED_FIELD = "flagged"
"""Payload field of a chunk the injection detector flagged; present (``True``) only on those."""

_PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
    (DOCUMENT_ID_FIELD, models.PayloadSchemaType.KEYWORD),
    (FLAGGED_FIELD, models.PayloadSchemaType.BOOL),
)

_HASH_LENGTH = 12
_HTTP_CONFLICT = 409
_HTTP_NOT_FOUND = 404
_HTTP_SERVER_ERROR = 500
_MISSING_COLLECTION_MARKER = b"Not found: Collection"
_INITIALISING_MARKER = b"read operations failed"
# get_collection while binding: waits out a collection that another client is still creating.
_SETTLE_ATTEMPTS = 10
_SETTLE_BACKOFF_BASE_S = 0.05
_SETTLE_BACKOFF_CAP_S = 0.5
_SUPPORTED_METRIC = "cosine"

_logger = logging.getLogger(__name__)


def canonical_identity(identity: EmbeddingIdentity) -> str:
    """The identity as canonical JSON: the string hashed into the name and stored as metadata."""
    return json.dumps(dataclasses.asdict(identity), sort_keys=True, separators=(",", ":"))


def collection_name(prefix: str, identity: EmbeddingIdentity) -> str:
    """``{prefix}_{first 12 hex digits of sha256(canonical_identity(identity))}``."""
    digest = hashlib.sha256(canonical_identity(identity).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:_HASH_LENGTH]}"


def chunk_id(document_id: str, sequence: int) -> str:
    """Deterministic point id of chunk ``sequence`` of ``document_id``."""
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, f"{document_id}:{sequence}"))


class QdrantVectorStore:
    """``VectorStore`` backed by Qdrant through ``qdrant-client`` (REST, or local ``:memory:``).

    ``client`` replaces the client built from ``config`` (tests inject failing clients);
    ``sleep``, ``uniform`` and ``clock`` drive the retry backoff and deadline.
    """

    def __init__(
        self,
        config: QdrantStoreConfig,
        *,
        client: QdrantClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
        uniform: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._client = client if client is not None else _build_client(config)
        # The readiness ping has a client of its own with a short timeout, so a probe of a hung
        # server ends within ``ping_timeout_s``; an injected client (tests) serves both.
        self._probe_client = (
            self._client if client is not None or config.url is None else _build_probe(config)
        )
        self._retrier = Retrier(
            RetryPolicy(
                config.attempts, config.backoff_base_s, config.backoff_cap_s, config.deadline_s
            ),
            sleep=sleep,
            uniform=uniform,
            clock=clock,
        )
        # Readiness probes try once: a down server must not hold a probe for the whole deadline.
        self._probe = Retrier(
            RetryPolicy(1, 0.0, 0.0, config.deadline_s), sleep=sleep, uniform=uniform, clock=clock
        )
        self._settle = Retrier(
            RetryPolicy(
                _SETTLE_ATTEMPTS, _SETTLE_BACKOFF_BASE_S, _SETTLE_BACKOFF_CAP_S, config.deadline_s
            ),
            transient=_is_initialising,
            sleep=sleep,
            uniform=uniform,
            clock=clock,
        )
        # Serialises binding and recreation; the binding is one tuple, so readers never see half.
        self._lock = threading.Lock()
        self._binding: tuple[EmbeddingIdentity, str] | None = None

    @property
    def config(self) -> QdrantStoreConfig:
        """The settings this store was built with."""
        return self._config

    @property
    def identity(self) -> EmbeddingIdentity | None:
        """The bound embedding identity, ``None`` before ``ensure_collection``."""
        binding = self._binding
        return None if binding is None else binding[0]

    @property
    def collection(self) -> str | None:
        """Name of the bound collection, ``None`` before ``ensure_collection``."""
        binding = self._binding
        return None if binding is None else binding[1]

    def close(self) -> None:
        """Release the connections of the client and of the probe client."""
        try:
            self._client.close()
        finally:
            if self._probe_client is not self._client:
                self._probe_client.close()

    def ping(self) -> None:
        """Check that the server answers, with one request and no retry (readiness probe).

        Lists the collections, which needs no bound collection and writes nothing, through the
        probe client, whose timeout is ``ping_timeout_s``. Raises ``DependencyUnavailable``
        (``vector_store_unavailable``) or ``DependencyTimeout`` (``vector_store_timeout``) like
        every other call.
        """
        self._probe.call(self._probe_client.get_collections)

    def ensure_collection(self, identity: EmbeddingIdentity) -> None:
        """Bind to the collection of ``identity``, verifying or creating it (see the port).

        The first binding holds the lock (double-checked), so concurrent first calls reach the
        server once; once bound, a call only compares identities.
        """
        binding = self._binding
        if binding is None:
            with self._lock:
                binding = self._binding
                if binding is None:
                    binding = self._bind(identity)
                    self._binding = binding
        if binding[0] != identity:
            raise IncompatibleEmbedding("the store is bound to another embedding identity")

    def upsert(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        page_texts: Mapping[int, str],
        content_hash: str,
    ) -> None:
        identity, name = self._bound()
        _validate_upsert(identity, chunks, vectors, page_texts, content_hash)
        total = len(chunks)
        points = [
            models.PointStruct(
                id=chunk_id(document_id, chunk.sequence),
                vector=[float(value) for value in vector],
                payload={
                    DOCUMENT_ID_FIELD: document_id,
                    "page": chunk.page,
                    "sequence": chunk.sequence,
                    "text": chunk.text,
                    "page_text": page_texts[chunk.page],
                    "start": chunk.start,
                    "end": chunk.end,
                    "content_hash": content_hash,
                    CHUNK_TOTAL_FIELD: total,
                    # Only when flagged: an absent field is "not flagged", as in older points.
                    **({FLAGGED_FIELD: True} if chunk.flagged else {}),
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        batch = self._config.upsert_batch
        for start in range(0, len(points), batch):
            self._on_collection(
                partial(
                    self._client.upsert,
                    collection_name=name,
                    points=points[start : start + batch],
                    wait=True,
                )
            )
        # Only now that every point is written: drop what an earlier, longer version left.
        stale = models.Filter(
            must=[_document_condition(document_id)],
            must_not=[models.HasIdCondition(has_id=[point.id for point in points])],
        )
        self._delete_matching(name, stale)

    def search(self, vector: Sequence[float], top_k: int) -> list[SearchHit]:
        identity, name = self._bound()
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if len(vector) != identity.dimension:
            raise IncompatibleEmbedding("query dimension differs from the collection's")
        query = [float(value) for value in vector]
        response = self._on_collection(
            lambda: self._client.query_points(
                collection_name=name,
                query=query,
                query_filter=unflagged_filter(),
                limit=top_k,
                with_payload=True,
            )
        )
        return [_to_hit(point) for point in response.points]

    def delete_document(self, document_id: str) -> None:
        _, name = self._bound()
        self._delete_matching(name, models.Filter(must=[_document_condition(document_id)]))

    def document_chunk_count(self, document_id: str) -> int:
        _, name = self._bound()
        document = models.Filter(must=[_document_condition(document_id)])
        points, _ = self._on_collection(
            lambda: self._client.scroll(
                collection_name=name,
                scroll_filter=document,
                limit=1,
                with_payload=[CHUNK_TOTAL_FIELD],
                with_vectors=False,
            )
        )
        if not points:
            return 0
        total = (points[0].payload or {}).get(CHUNK_TOTAL_FIELD)
        if not isinstance(total, int) or isinstance(total, bool) or total < 1:
            return 0  # stored before chunk_total existed, or corrupt: incomplete
        if self._count(name, document) != total:
            return 0
        agreeing = models.Filter(
            must=[
                _document_condition(document_id),
                models.FieldCondition(key=CHUNK_TOTAL_FIELD, match=models.MatchValue(value=total)),
            ]
        )
        return total if self._count(name, agreeing) == total else 0

    def count(self) -> int:
        _, name = self._bound()
        return self._on_collection(
            lambda: self._client.count(collection_name=name, exact=True)
        ).count

    def _call[T](self, operation: Callable[[], T]) -> T:
        return self._retrier.call(operation)

    def _on_collection[T](self, operation: Callable[[], T]) -> T:
        """``_call`` an operation on the bound collection; recreate the collection once if lost."""
        identity, name = self._bound()
        try:
            return self._call(operation)
        except (DependencyUnavailable, ValueError) as error:
            if not _is_missing_collection(error, name):
                raise
        _logger.warning("vector store collection %s is missing; recreating it", name)
        with self._lock:
            self._open(name, identity)
        return self._call(operation)

    def _count(self, name: str, count_filter: models.Filter) -> int:
        return self._on_collection(
            lambda: self._client.count(collection_name=name, count_filter=count_filter, exact=True)
        ).count

    def _delete_matching(self, name: str, points_filter: models.Filter) -> None:
        selector = models.FilterSelector(filter=points_filter)
        self._on_collection(
            lambda: self._client.delete(collection_name=name, points_selector=selector, wait=True)
        )

    def _bound(self) -> tuple[EmbeddingIdentity, str]:
        binding = self._binding
        if binding is None:
            raise RuntimeError("ensure_collection must be called before using the store")
        return binding

    def _bind(self, identity: EmbeddingIdentity) -> tuple[EmbeddingIdentity, str]:
        """Open the collection of ``identity``; the caller holds the lock."""
        if identity.metric != _SUPPORTED_METRIC:
            raise IncompatibleEmbedding("the vector store supports only the cosine metric")
        name = collection_name(self._config.collection_prefix, identity)
        self._open(name, identity)
        return identity, name

    def _open(self, name: str, identity: EmbeddingIdentity) -> None:
        """Verify collection ``name`` if it exists, else create it; then ensure its index."""
        expected = canonical_identity(identity)
        if self._call(lambda: self._client.collection_exists(name)):
            self._verify(name, identity, expected)
        else:
            self._create(name, identity, expected)
        self._ensure_payload_indexes(name)

    def _create(self, name: str, identity: EmbeddingIdentity, expected: str) -> None:
        params = models.VectorParams(size=identity.dimension, distance=models.Distance.COSINE)
        try:
            self._call(
                lambda: self._client.create_collection(
                    collection_name=name,
                    vectors_config=params,
                    metadata={IDENTITY_METADATA_KEY: expected},
                )
            )
        except DependencyUnavailable as error:
            # Another process created it between the existence check and the create call.
            if not _is_conflict(error):
                raise
            self._verify(name, identity, expected)

    def _verify(self, name: str, identity: EmbeddingIdentity, expected: str) -> None:
        info = self._settle.call(lambda: self._client.get_collection(name))
        metadata = info.config.metadata or {}
        if metadata.get(IDENTITY_METADATA_KEY) != expected:
            raise IncompatibleEmbedding("the collection records another embedding identity")
        vectors = info.config.params.vectors
        if (
            not isinstance(vectors, models.VectorParams)
            or vectors.size != identity.dimension
            or vectors.distance != models.Distance.COSINE
        ):
            raise IncompatibleEmbedding("the collection's vector size or distance differs")

    def _ensure_payload_indexes(self, name: str) -> None:
        """Create each missing index of ``_PAYLOAD_INDEXES``; creating an existing one is safe.

        A collection created before the ``flagged`` index existed gains it here on first use.
        """
        if self._config.location is not None:
            return  # local mode: payload indexes have no effect (and qdrant-client warns)
        info = self._settle.call(lambda: self._client.get_collection(name))
        present = info.payload_schema or {}
        for field, schema in _PAYLOAD_INDEXES:
            if field in present:
                continue
            self._call(
                partial(
                    self._client.create_payload_index,
                    collection_name=name,
                    field_name=field,
                    field_schema=schema,
                    wait=True,
                )
            )


def _build_client(config: QdrantStoreConfig) -> QdrantClient:
    if config.location is not None:
        return QdrantClient(location=config.location)
    return QdrantClient(
        url=config.url,
        api_key=config.api_key,
        timeout=math.ceil(config.timeout_s),
        prefer_grpc=False,
    )


def _build_probe(config: QdrantStoreConfig) -> QdrantClient:
    """The readiness client: ``ping_timeout_s`` per request, no start-up version check (the
    main client already runs one)."""
    return QdrantClient(
        url=config.url,
        api_key=config.api_key,
        timeout=math.ceil(config.ping_timeout_s),
        prefer_grpc=False,
        check_compatibility=False,
    )


def _is_missing_collection(error: Exception, name: str) -> bool:
    """Whether ``error`` says that collection ``name`` does not exist.

    The server answers 404 ``Not found: Collection `name` doesn't exist!``; the embedded local
    mode raises ``ValueError("Collection <name> not found")``.
    """
    if isinstance(error, ValueError):
        return str(error) == f"Collection {name} not found"
    cause = error.__cause__
    return (
        isinstance(cause, UnexpectedResponse)
        and cause.status_code == _HTTP_NOT_FOUND
        and _MISSING_COLLECTION_MARKER in cause.content
    )


def _is_conflict(error: Exception) -> bool:
    """Whether ``error`` is the 409 that creating an existing collection answers."""
    cause = error.__cause__
    return isinstance(cause, UnexpectedResponse) and cause.status_code == _HTTP_CONFLICT


def _is_initialising(error: BaseException) -> bool:
    """Whether ``error`` is the 500 of a collection whose shards serve no reads yet.

    The server answers ``Service internal error: 0 of 0 read operations failed`` for some
    milliseconds after another client's create was accepted.
    """
    return (
        isinstance(error, UnexpectedResponse)
        and error.status_code == _HTTP_SERVER_ERROR
        and _INITIALISING_MARKER in error.content
    )


def unflagged_filter() -> models.Filter:
    """The search filter: ``must_not: [flagged = true]``.

    A point without the field does not match the condition, so it passes the filter: points
    written before the field existed stay searchable.
    """
    return models.Filter(
        must_not=[models.FieldCondition(key=FLAGGED_FIELD, match=models.MatchValue(value=True))]
    )


def _document_condition(document_id: str) -> models.FieldCondition:
    return models.FieldCondition(key=DOCUMENT_ID_FIELD, match=models.MatchValue(value=document_id))


def _validate_upsert(
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


def _to_hit(point: models.ScoredPoint) -> SearchHit:
    payload: dict[str, Any] = point.payload or {}
    return SearchHit(
        document_id=str(payload[DOCUMENT_ID_FIELD]),
        page=int(payload["page"]),
        sequence=int(payload["sequence"]),
        text=str(payload["text"]),
        page_text=str(payload["page_text"]),
        score=min(1.0, max(0.0, float(point.score))),
    )
