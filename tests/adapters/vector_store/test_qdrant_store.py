"""Adapter-specific behaviour of ``QdrantVectorStore`` beyond the port contract."""

import dataclasses
import json
import os
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from app.adapters.vector_store import QdrantStoreConfig, QdrantVectorStore
from app.adapters.vector_store.qdrant_store import (
    CHUNK_ID_NAMESPACE,
    IDENTITY_METADATA_KEY,
    canonical_identity,
    chunk_id,
    collection_name,
)
from app.domain.errors import DependencyTimeout, DependencyUnavailable, IncompatibleEmbedding
from app.domain.types import Chunk, EmbeddingIdentity
from tests.adapters.vector_store.conftest import memory_store
from tests.adapters.vector_store.flaky_client import (
    FlakyClient,
    connect_error,
    http_status,
    read_timeout,
)
from tests.adapters.vector_store.servers import refused_url, silent_server
from tests.application.contracts import make_chunks, one_hot, upsert

IDENTITY = EmbeddingIdentity("model-a", 8, "e5", True, "cosine")
MEMORY = QdrantStoreConfig(location=":memory:")


def store_on(client: QdrantClient | FlakyClient, **overrides: Any) -> QdrantVectorStore:
    config = dataclasses.replace(MEMORY, **overrides)
    real = client.as_client() if isinstance(client, FlakyClient) else client
    return QdrantVectorStore(config, client=real, sleep=lambda _: None, uniform=lambda: 0.5)


def ready_store(**overrides: Any) -> tuple[QdrantVectorStore, FlakyClient]:
    flaky = FlakyClient(QdrantClient(location=":memory:"))
    store = store_on(flaky, **overrides)
    store.ensure_collection(IDENTITY)
    return store, flaky


# --- collection naming and identity metadata ------------------------------------------------


def test_collection_name_is_prefix_and_twelve_hex_digits_of_the_identity() -> None:
    name = collection_name("chunks", IDENTITY)

    assert re.fullmatch(r"chunks_[0-9a-f]{12}", name)
    assert name == collection_name("chunks", dataclasses.replace(IDENTITY))
    assert canonical_identity(IDENTITY) == (
        '{"dimension":8,"metric":"cosine","model_id":"model-a","normalized":true,'
        '"prefix_scheme":"e5"}'
    )


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "model-b"},
        {"dimension": 16},
        {"prefix_scheme": "none"},
        {"normalized": False},
    ],
)
def test_every_identity_field_changes_the_collection_name(change: dict[str, Any]) -> None:
    other = dataclasses.replace(IDENTITY, **change)

    assert collection_name("chunks", other) != collection_name("chunks", IDENTITY)


def test_collection_is_created_with_the_identity_in_its_metadata() -> None:
    client = QdrantClient(location=":memory:")
    store = store_on(client, collection_prefix="docs")

    store.ensure_collection(IDENTITY)

    assert store.collection == collection_name("docs", IDENTITY)
    info = client.get_collection(store.collection)
    assert info.config.metadata == {IDENTITY_METADATA_KEY: canonical_identity(IDENTITY)}
    vectors = info.config.params.vectors
    assert isinstance(vectors, models.VectorParams)
    assert vectors.size == 8
    assert vectors.distance == models.Distance.COSINE


def test_a_model_change_opens_a_new_collection_and_leaves_the_old_one() -> None:
    client = QdrantClient(location=":memory:")
    old = store_on(client)
    old.ensure_collection(IDENTITY)
    upsert(old, "doc", make_chunks("old model"), [one_hot(8, 0)])

    new = store_on(client)
    new.ensure_collection(dataclasses.replace(IDENTITY, model_id="model-b"))

    assert new.collection != old.collection
    assert new.count() == 0
    assert old.count() == 1


def _precreate(client: QdrantClient, size: int, metadata: dict[str, object] | None) -> str:
    name = collection_name("chunks", IDENTITY)
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
        metadata=metadata,
    )
    client.upsert(name, [models.PointStruct(id=1, vector=one_hot(size, 0), payload={})])
    return name


@pytest.mark.parametrize(
    ("size", "metadata"),
    [
        (8, {IDENTITY_METADATA_KEY: canonical_identity(dataclasses.replace(IDENTITY, metric="x"))}),
        (8, None),
        (8, {"other": "value"}),
        (4, {IDENTITY_METADATA_KEY: canonical_identity(IDENTITY)}),
    ],
    ids=["other-identity", "no-metadata", "no-identity-key", "other-size"],
)
def test_an_existing_collection_that_does_not_match_is_refused_and_untouched(
    size: int, metadata: dict[str, object] | None
) -> None:
    client = QdrantClient(location=":memory:")
    name = _precreate(client, size, metadata)
    store = store_on(client)

    with pytest.raises(IncompatibleEmbedding):
        store.ensure_collection(IDENTITY)

    assert client.count(name).count == 1
    assert client.get_collection(name).config.metadata == metadata
    with pytest.raises(RuntimeError):
        store.count()


def test_an_existing_matching_collection_is_reused() -> None:
    client = QdrantClient(location=":memory:")
    name = _precreate(client, 8, {IDENTITY_METADATA_KEY: canonical_identity(IDENTITY)})
    store = store_on(client)

    store.ensure_collection(IDENTITY)

    assert store.collection == name
    assert store.count() == 1


def test_a_collection_created_concurrently_is_verified_and_used() -> None:
    inner = QdrantClient(location=":memory:")
    _precreate(inner, 8, {IDENTITY_METADATA_KEY: canonical_identity(IDENTITY)})
    # The existence check ran before another process created the collection, so creating it
    # now answers 409 Conflict.
    flaky = FlakyClient(_ExistsFalse(inner).as_client(), {"create_collection": [http_status(409)]})
    store = store_on(flaky)

    store.ensure_collection(IDENTITY)

    assert flaky.calls["create_collection"] == 1
    assert store.count() == 1


class _ExistsFalse(FlakyClient):
    def collection_exists(self, name: str) -> bool:
        return False


def test_a_failed_create_that_is_not_a_conflict_propagates() -> None:
    flaky = FlakyClient(
        QdrantClient(location=":memory:"), {"create_collection": [http_status(500)]}
    )
    store = store_on(flaky)

    with pytest.raises(DependencyUnavailable):
        store.ensure_collection(IDENTITY)


def test_a_metric_other_than_cosine_is_incompatible() -> None:
    with pytest.raises(IncompatibleEmbedding):
        memory_store().ensure_collection(dataclasses.replace(IDENTITY, metric="dot"))


def test_reopening_a_bound_store_with_the_same_identity_is_a_no_op() -> None:
    store, flaky = ready_store()
    calls = dict(flaky.calls)

    store.ensure_collection(dataclasses.replace(IDENTITY))

    assert flaky.calls == calls


def test_other_methods_before_ensure_collection_are_programming_errors() -> None:
    store = memory_store()

    with pytest.raises(RuntimeError):
        store.count()
    with pytest.raises(RuntimeError):
        store.delete_document("doc")


# --- ids and payload -------------------------------------------------------------------------


def test_chunk_ids_are_uuid5_of_document_and_sequence() -> None:
    assert chunk_id("doc-a", 0) == str(uuid.uuid5(CHUNK_ID_NAMESPACE, "doc-a:0"))
    assert chunk_id("doc-a", 0) == chunk_id("doc-a", 0)
    assert len({chunk_id("doc-a", 0), chunk_id("doc-b", 0), chunk_id("doc-a", 1)}) == 3
    assert (
        uuid.uuid5(uuid.NAMESPACE_URL, "urn:docintel:vector-store:chunk-id") == CHUNK_ID_NAMESPACE
    )


def test_points_are_stored_under_their_chunk_ids_with_the_full_payload() -> None:
    store, flaky = ready_store()
    chunks = [Chunk("first half", 0, 0, 0, 10), Chunk("second", 1, 1, 4, 10)]
    page_texts = {0: "first half and more", 1: "the second page"}
    store.upsert(
        "doc", chunks, [one_hot(8, 0), one_hot(8, 1)], page_texts=page_texts, content_hash="h1"
    )
    assert store.collection is not None

    points = flaky.inner.retrieve(
        store.collection, [chunk_id("doc", 0), chunk_id("doc", 1)], with_payload=True
    )

    payloads = {str(point.id): point.payload for point in points}
    assert payloads == {
        chunk_id("doc", 0): {
            "document_id": "doc",
            "page": 0,
            "sequence": 0,
            "text": "first half",
            "page_text": "first half and more",
            "start": 0,
            "end": 10,
            "content_hash": "h1",
            "chunk_total": 2,
        },
        chunk_id("doc", 1): {
            "document_id": "doc",
            "page": 1,
            "sequence": 1,
            "text": "second",
            "page_text": "the second page",
            "start": 4,
            "end": 10,
            "content_hash": "h1",
            "chunk_total": 2,
        },
    }
    hit = store.search(one_hot(8, 1), top_k=1)[0]
    assert (hit.document_id, hit.page, hit.sequence, hit.text, hit.page_text) == (
        "doc",
        1,
        1,
        "second",
        "the second page",
    )


def test_upserting_no_chunks_removes_the_document() -> None:
    store, _ = ready_store()
    upsert(store, "doc", make_chunks("a", "b"), [one_hot(8, 0), one_hot(8, 1)])

    upsert(store, "doc", [], [])

    assert store.document_chunk_count("doc") == 0


def test_points_are_written_in_batches() -> None:
    store, flaky = ready_store(upsert_batch=2)
    chunks = make_chunks("a", "b", "c", "d", "e")

    upsert(store, "doc", chunks, [one_hot(8, i) for i in range(5)])

    assert flaky.calls["upsert"] == 3
    assert store.document_chunk_count("doc") == 5


def test_rewriting_a_document_never_deletes_before_writing() -> None:
    store, flaky = ready_store(upsert_batch=2)
    upsert(store, "doc", make_chunks("a", "b", "c"), [one_hot(8, i) for i in range(3)])
    flaky.calls.clear()

    upsert(store, "doc", make_chunks("a", "b", "c"), [one_hot(8, i) for i in range(3)])

    assert flaky.calls == {"upsert": 2, "delete": 1}  # the delete comes last
    assert store.document_chunk_count("doc") == 3


def test_a_shorter_version_removes_only_the_chunks_beyond_its_count() -> None:
    store, flaky = ready_store()
    upsert(store, "doc", make_chunks("a", "b", "c", "d"), [one_hot(8, i) for i in range(4)])
    upsert(store, "other", make_chunks("x", "y"), [one_hot(8, 6), one_hot(8, 7)])

    upsert(store, "doc", make_chunks("new a", "new b"), [one_hot(8, 0), one_hot(8, 1)])

    assert store.collection is not None
    points, _ = flaky.inner.scroll(store.collection, limit=10, with_payload=True)
    stored = sorted((p.payload["document_id"], p.payload["sequence"]) for p in points if p.payload)
    assert stored == [("doc", 0), ("doc", 1), ("other", 0), ("other", 1)]
    assert store.document_chunk_count("doc") == 2


@pytest.mark.parametrize(
    "error", [connect_error(), http_status(400)], ids=["connection", "rejected"]
)
def test_a_failed_batch_deletes_nothing_and_leaves_the_document_incomplete(
    error: Exception,
) -> None:
    store, flaky = ready_store(upsert_batch=2, attempts=1)
    upsert(store, "other", make_chunks("kept"), [one_hot(8, 7)])
    flaky.failures["upsert"] = [None, error]
    flaky.calls.clear()

    with pytest.raises(DependencyUnavailable) as raised:
        upsert(
            store, "doc", make_chunks("a", "b", "c", "d", "e"), [one_hot(8, i) for i in range(5)]
        )

    assert raised.value.__cause__ is error
    assert "delete" not in flaky.calls
    assert store.count() == 3  # the first batch of doc, plus other
    assert store.document_chunk_count("doc") == 0
    assert store.document_chunk_count("other") == 1


def test_the_reviewed_race_a_failed_rewrite_keeps_a_completed_document() -> None:
    """A completed ingest, then a concurrent ingest of the same file failing on its 2nd batch."""
    store, flaky = ready_store(upsert_batch=1, attempts=1)
    chunks, vectors = make_chunks("a", "b"), [one_hot(8, 0), one_hot(8, 1)]
    upsert(store, "doc", chunks, vectors)
    flaky.failures["upsert"] = [None, http_status(500)]

    with pytest.raises(DependencyUnavailable):
        upsert(store, "doc", chunks, vectors)

    assert store.document_chunk_count("doc") == 2


@pytest.mark.parametrize(
    "payload",
    [{}, {"chunk_total": "2"}, {"chunk_total": True}, {"chunk_total": 0}],
    ids=["legacy-no-total", "string", "bool", "zero"],
)
def test_points_without_a_valid_chunk_total_count_as_incomplete(payload: dict[str, Any]) -> None:
    store, flaky = ready_store()
    upsert(store, "doc", make_chunks("a"), [one_hot(8, 0)])
    assert store.collection is not None
    flaky.inner.overwrite_payload(
        store.collection,
        payload={"document_id": "doc", "sequence": 0, **payload},
        points=[chunk_id("doc", 0)],
    )

    assert store.document_chunk_count("doc") == 0


def test_points_disagreeing_on_the_chunk_total_count_as_incomplete() -> None:
    store, flaky = ready_store()
    upsert(store, "doc", make_chunks("a", "b"), [one_hot(8, 0), one_hot(8, 1)])
    assert store.collection is not None
    flaky.inner.set_payload(
        store.collection, payload={"chunk_total": 3}, points=[chunk_id("doc", 1)]
    )

    assert store.document_chunk_count("doc") == 0


# --- lost collection -------------------------------------------------------------------------


def test_a_collection_dropped_behind_the_store_is_recreated_with_its_identity() -> None:
    store, flaky = ready_store()
    upsert(store, "doc", make_chunks("a"), [one_hot(8, 0)])
    assert store.collection is not None
    flaky.inner.delete_collection(store.collection)

    assert store.document_chunk_count("doc") == 0
    upsert(store, "doc", make_chunks("again"), [one_hot(8, 1)])

    assert store.document_chunk_count("doc") == 1
    assert [hit.text for hit in store.search(one_hot(8, 1), top_k=1)] == ["again"]
    info = flaky.inner.get_collection(store.collection)
    assert info.config.metadata == {IDENTITY_METADATA_KEY: canonical_identity(IDENTITY)}


@pytest.mark.parametrize("method", ["count", "query_points", "upsert", "delete", "scroll"])
def test_every_operation_recovers_from_a_lost_collection(method: str) -> None:
    store, flaky = ready_store()
    assert store.collection is not None
    flaky.inner.delete_collection(store.collection)
    operations: dict[str, Callable[[], object]] = {
        "count": store.count,
        "query_points": lambda: store.search(one_hot(8, 0), top_k=1),
        "upsert": lambda: upsert(store, "doc", make_chunks("a"), [one_hot(8, 0)]),
        "delete": lambda: store.delete_document("doc"),
        "scroll": lambda: store.document_chunk_count("doc"),
    }

    operations[method]()

    assert flaky.inner.collection_exists(store.collection)


def _missing(name: str) -> Exception:
    """What the server answers when collection ``name`` does not exist."""
    body = {"status": {"error": f"Not found: Collection `{name}` doesn't exist!"}}
    return http_status(404, json.dumps(body).encode())


def test_a_server_404_for_the_collection_triggers_recreation() -> None:
    store, flaky = ready_store()
    assert store.collection is not None
    flaky.inner.delete_collection(store.collection)
    flaky.failures["count"] = [_missing(store.collection)]

    assert store.count() == 0

    assert flaky.calls["count"] == 2
    assert flaky.calls["create_collection"] == 2  # once in ensure_collection, once recreating
    assert flaky.inner.collection_exists(store.collection)


def test_recreation_is_attempted_once_then_the_error_propagates() -> None:
    store, flaky = ready_store()
    assert store.collection is not None
    flaky.failures["count"] = [_missing(store.collection), _missing(store.collection)]

    with pytest.raises(DependencyUnavailable):
        store.count()

    assert flaky.calls["count"] == 2
    assert flaky.calls["create_collection"] == 1  # it still existed: not recreated


def test_another_404_is_not_taken_for_a_lost_collection() -> None:
    store, flaky = ready_store()
    flaky.failures["count"] = [http_status(404, b'{"status":{"error":"Not found: Point"}}')]

    with pytest.raises(DependencyUnavailable):
        store.count()

    assert flaky.calls["count"] == 1


# --- retries and error mapping ---------------------------------------------------------------


def test_one_connection_error_is_retried_and_the_call_succeeds() -> None:
    sleeps: list[float] = []
    flaky = FlakyClient(QdrantClient(location=":memory:"))
    store = QdrantVectorStore(
        MEMORY, client=flaky.as_client(), sleep=sleeps.append, uniform=lambda: 0.5
    )
    store.ensure_collection(IDENTITY)
    upsert(store, "doc", make_chunks("a"), [one_hot(8, 0)])
    flaky.failures["count"] = [connect_error()]

    assert store.count() == 1
    assert flaky.calls["count"] == 2
    assert sleeps == [pytest.approx(0.1)]


def test_exhausted_connection_errors_fail_as_unavailable_after_the_configured_attempts() -> None:
    sleeps: list[float] = []
    flaky = FlakyClient(QdrantClient(location=":memory:"))
    config = dataclasses.replace(MEMORY, attempts=4, backoff_base_s=0.2, backoff_cap_s=0.5)
    store = QdrantVectorStore(
        config, client=flaky.as_client(), sleep=sleeps.append, uniform=lambda: 1.0
    )
    store.ensure_collection(IDENTITY)
    errors = [connect_error() for _ in range(4)]
    flaky.failures["count"] = list(errors)

    with pytest.raises(DependencyUnavailable) as raised:
        store.count()

    assert raised.value.code == "vector_store_unavailable"
    assert raised.value.__cause__ is errors[-1]
    assert flaky.calls["count"] == 4
    assert sleeps == [pytest.approx(0.2), pytest.approx(0.4), pytest.approx(0.5)]


def test_exhausted_timeouts_fail_as_timeout() -> None:
    store, flaky = ready_store()
    flaky.failures["query_points"] = [read_timeout() for _ in range(3)]

    with pytest.raises(DependencyTimeout) as raised:
        store.search(one_hot(8, 0), top_k=1)

    assert raised.value.code == "vector_store_timeout"
    assert flaky.calls["query_points"] == 3


def test_a_rejected_request_fails_at_once_without_retry_or_payload_text() -> None:
    store, flaky = ready_store()
    flaky.failures["upsert"] = [http_status(400, b'{"status":{"error":"SECRET page text"}}')]

    with pytest.raises(DependencyUnavailable) as raised:
        store.upsert(
            "doc",
            make_chunks("SECRET chunk"),
            [one_hot(8, 0)],
            page_texts={0: "SECRET page text"},
            content_hash="h",
        )

    assert flaky.calls["upsert"] == 1
    assert raised.value.code == "vector_store_unavailable"
    assert "SECRET" not in str(raised.value)
    assert "SECRET" not in repr(raised.value)


@pytest.mark.parametrize("status", [502, 503, 504])
def test_gateway_and_unavailable_statuses_are_retried(status: int) -> None:
    store, flaky = ready_store()
    flaky.failures["count"] = [http_status(status)]

    assert store.count() == 0
    assert flaky.calls["count"] == 2


def test_the_deadline_stops_retries_before_the_attempts_are_used_up() -> None:
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    def slow_failure() -> Exception:
        now[0] += 4.0  # each attempt hangs for 4 s before failing
        return connect_error()

    flaky = FlakyClient(QdrantClient(location=":memory:"))
    config = dataclasses.replace(MEMORY, attempts=10, deadline_s=10.0, backoff_base_s=1.0)
    store = QdrantVectorStore(
        config, client=flaky.as_client(), sleep=sleep, uniform=lambda: 1.0, clock=lambda: now[0]
    )
    store.ensure_collection(IDENTITY)
    flaky.failures["count"] = [slow_failure for _ in range(10)]

    with pytest.raises(DependencyTimeout) as raised:
        store.count()

    assert raised.value.code == "vector_store_timeout"
    # t=0 try (fails at 4), wait 1 -> t=5 try (fails at 9), next wait 2 would start at 11 > 10.
    assert flaky.calls["count"] == 2
    assert now[0] == pytest.approx(9.0)


def test_programming_errors_from_the_client_are_not_wrapped() -> None:
    store, flaky = ready_store()
    flaky.failures["count"] = [TypeError("bug")]

    with pytest.raises(TypeError):
        store.count()
    assert flaky.calls["count"] == 1


# --- real transport failures (what qdrant-client actually raises) ----------------------------


def test_a_refused_connection_maps_to_unavailable_after_retries() -> None:

    url = refused_url()
    client = QdrantClient(url=url, timeout=10, check_compatibility=False)
    config = QdrantStoreConfig(url=url, attempts=2, timeout_s=10)
    store = QdrantVectorStore(config, client=client, sleep=lambda _: None)

    with pytest.raises(DependencyUnavailable) as raised:
        store.ensure_collection(IDENTITY)

    assert raised.value.code == "vector_store_unavailable"
    cause = raised.value.__cause__
    assert type(cause).__name__ == "ResponseHandlingException"
    assert isinstance(getattr(cause, "source", None), Exception)
    assert type(cause.source).__name__ == "ConnectError"  # type: ignore[union-attr]


def test_a_server_that_never_answers_maps_to_timeout() -> None:

    with silent_server() as url:
        client = QdrantClient(url=url, timeout=1, check_compatibility=False)
        config = QdrantStoreConfig(url=url, attempts=1, timeout_s=1)
        store = QdrantVectorStore(config, client=client, sleep=lambda _: None)

        with pytest.raises(DependencyTimeout) as raised:
            store.ensure_collection(IDENTITY)

    assert type(raised.value.__cause__.source).__name__ == "ReadTimeout"  # type: ignore[union-attr]


@pytest.mark.filterwarnings("ignore:Failed to obtain server version:UserWarning")
def test_a_store_building_its_own_client_fails_fast_on_a_refused_connection() -> None:
    store = QdrantVectorStore(QdrantStoreConfig(url=refused_url(), timeout_s=1, attempts=1))
    started = time.monotonic()

    # Linux refuses at once; Windows retries the SYN for ~2 s, so the 1 s connect timeout ends it.
    with pytest.raises((DependencyUnavailable, DependencyTimeout)) as raised:
        store.ensure_collection(IDENTITY)

    assert time.monotonic() - started < 3.0
    assert raised.value.code in {"vector_store_unavailable", "vector_store_timeout"}
    store.close()


@pytest.mark.filterwarnings("ignore:Failed to obtain server version:UserWarning")
def test_the_configured_timeout_reaches_the_client_the_store_builds() -> None:
    # qdrant-client's own default timeout is 5 s; with timeout_s=1 the call gives up after ~1 s.
    with silent_server() as url:
        store = QdrantVectorStore(QdrantStoreConfig(url=url, timeout_s=1, attempts=1))
        started = time.monotonic()

        with pytest.raises(DependencyTimeout) as raised:
            store.ensure_collection(IDENTITY)

        elapsed = time.monotonic() - started
        store.close()

    assert 0.5 < elapsed < 3.0
    assert raised.value.code == "vector_store_timeout"


# --- server only -----------------------------------------------------------------------------


@pytest.mark.qdrant_server
def test_the_server_collection_has_a_keyword_index_on_document_id(
    server_store: QdrantVectorStore,
) -> None:
    server_store.ensure_collection(IDENTITY)
    server_store.ensure_collection(IDENTITY)
    assert server_store.collection is not None

    client = QdrantClient(url=os.environ["QDRANT_URL"])
    info = client.get_collection(server_store.collection)
    client.close()

    schema = info.payload_schema["document_id"]
    assert schema.data_type == models.PayloadSchemaType.KEYWORD


@pytest.mark.qdrant_server
def test_a_collection_lost_on_the_server_is_recreated_with_its_index(
    server_store: QdrantVectorStore,
) -> None:
    server_store.ensure_collection(IDENTITY)
    upsert(server_store, "doc", make_chunks("before"), [one_hot(8, 0)])
    assert server_store.collection is not None
    admin = QdrantClient(url=os.environ["QDRANT_URL"])
    admin.delete_collection(server_store.collection)

    assert server_store.document_chunk_count("doc") == 0
    upsert(server_store, "doc", make_chunks("after"), [one_hot(8, 1)])

    assert server_store.document_chunk_count("doc") == 1
    info = admin.get_collection(server_store.collection)
    admin.close()
    assert info.config.metadata == {IDENTITY_METADATA_KEY: canonical_identity(IDENTITY)}
    assert info.payload_schema["document_id"].data_type == models.PayloadSchemaType.KEYWORD
