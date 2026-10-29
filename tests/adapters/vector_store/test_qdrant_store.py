"""Adapter-specific behaviour of ``QdrantVectorStore`` beyond the port contract."""

import dataclasses
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from functools import partial
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from app.adapters.vector_store import QdrantStoreConfig, QdrantVectorStore
from app.adapters.vector_store.qdrant_store import (
    CHUNK_ID_NAMESPACE,
    FLAGGED_FIELD,
    IDENTITY_METADATA_KEY,
    canonical_identity,
    chunk_id,
    collection_name,
    unflagged_filter,
)
from app.domain.errors import DependencyTimeout, DependencyUnavailable, IncompatibleEmbedding
from app.domain.types import Chunk, EmbeddingIdentity
from tests.adapters.vector_store.conftest import memory_store
from tests.adapters.vector_store.flaky_client import (
    FlakyClient,
    SerializedClient,
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


# --- concurrent first use ----------------------------------------------------------------------


def _initialising() -> Exception:
    """What ``get_collection`` answers while another client's new collection has no live shards."""
    body = b'{"status":{"error":"Service internal error: 0 of 0 read operations failed"}}'
    return http_status(500, body)


def _created_elsewhere(client: QdrantClient) -> Callable[[], Exception]:
    """A scripted create: another process creates the collection first, so this one gets 409."""

    def create() -> Exception:
        _precreate(client, 8, {IDENTITY_METADATA_KEY: canonical_identity(IDENTITY)})
        return http_status(409, b'{"status":{"error":"Wrong input: Collection already exists!"}}')

    return create


def _concurrently(calls: list[Callable[[], object]]) -> list[BaseException | None]:
    """Start every call at the same moment in its own thread; what each raised, or ``None``."""
    barrier = threading.Barrier(len(calls))
    outcomes: list[BaseException | None] = [None] * len(calls)

    def run(index: int) -> None:
        barrier.wait()
        try:
            calls[index]()
        except BaseException as error:  # noqa: BLE001 - reported to the test
            outcomes[index] = error

    threads = [threading.Thread(target=run, args=(index,)) for index in range(len(calls))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes


class _SlowExists(FlakyClient):
    """The existence check takes a while, so concurrent callers all pass it before any create."""

    def collection_exists(self, name: str) -> bool:
        self.calls["collection_exists"] = self.calls.get("collection_exists", 0) + 1
        time.sleep(0.05)
        return bool(self.inner.collection_exists(name))


def test_concurrent_first_calls_on_one_store_check_and_create_the_collection_once() -> None:
    flaky = _SlowExists(SerializedClient(QdrantClient(location=":memory:")).as_client())
    store = store_on(flaky)

    outcomes = _concurrently([lambda: store.ensure_collection(IDENTITY)] * 8)

    assert outcomes == [None] * 8
    assert flaky.calls["collection_exists"] == 1
    assert flaky.calls["create_collection"] == 1
    assert store.collection == collection_name("chunks", IDENTITY)


def test_a_bound_store_does_not_take_the_binding_lock() -> None:
    store, _ = ready_store()

    with store._lock:  # held by "another thread": a bound store must not wait for it
        store.ensure_collection(IDENTITY)
        with pytest.raises(IncompatibleEmbedding):
            store.ensure_collection(dataclasses.replace(IDENTITY, model_id="model-b"))


def test_a_create_that_loses_to_another_process_waits_for_the_collection_and_verifies_it() -> None:
    inner = QdrantClient(location=":memory:")
    flaky = FlakyClient(
        inner,
        {
            "create_collection": [_created_elsewhere(inner)],
            "get_collection": [_initialising(), _initialising()],
        },
    )
    store = store_on(flaky)

    store.ensure_collection(IDENTITY)

    assert flaky.calls["create_collection"] == 1
    assert flaky.calls["get_collection"] == 3
    assert store.count() == 1  # the point the other process stored


def test_a_collection_found_while_another_process_creates_it_is_waited_for() -> None:
    inner = QdrantClient(location=":memory:")
    _precreate(inner, 8, {IDENTITY_METADATA_KEY: canonical_identity(IDENTITY)})
    flaky = FlakyClient(inner, {"get_collection": [_initialising()]})
    store = store_on(flaky)

    store.ensure_collection(IDENTITY)

    assert flaky.calls["get_collection"] == 2
    assert "create_collection" not in flaky.calls


def test_a_collection_created_elsewhere_with_another_identity_is_refused() -> None:
    inner = QdrantClient(location=":memory:")

    def create() -> Exception:
        _precreate(inner, 16, {IDENTITY_METADATA_KEY: "other"})
        return http_status(409)

    store = store_on(FlakyClient(inner, {"create_collection": [create]}))

    with pytest.raises(IncompatibleEmbedding):
        store.ensure_collection(IDENTITY)
    assert store.collection is None


def test_a_collection_that_never_becomes_readable_fails_after_the_settle_attempts() -> None:
    inner = QdrantClient(location=":memory:")
    _precreate(inner, 8, {IDENTITY_METADATA_KEY: canonical_identity(IDENTITY)})
    flaky = FlakyClient(inner, {"get_collection": [_initialising] * 20})
    store = store_on(flaky)

    with pytest.raises(DependencyUnavailable) as raised:
        store.ensure_collection(IDENTITY)

    assert raised.value.code == "vector_store_unavailable"
    assert flaky.calls["get_collection"] == 10
    assert store.collection is None


def test_another_server_error_while_verifying_fails_at_once() -> None:
    inner = QdrantClient(location=":memory:")
    _precreate(inner, 8, {IDENTITY_METADATA_KEY: canonical_identity(IDENTITY)})
    flaky = FlakyClient(inner, {"get_collection": [http_status(500)]})
    store = store_on(flaky)

    with pytest.raises(DependencyUnavailable):
        store.ensure_collection(IDENTITY)
    assert flaky.calls["get_collection"] == 1


def test_a_failed_first_binding_can_be_retried() -> None:
    flaky = FlakyClient(
        QdrantClient(location=":memory:"), {"collection_exists": [http_status(500)]}
    )
    store = store_on(flaky)

    with pytest.raises(DependencyUnavailable):
        store.ensure_collection(IDENTITY)
    store.ensure_collection(IDENTITY)

    assert store.collection == collection_name("chunks", IDENTITY)


def test_a_lost_collection_recreated_elsewhere_first_is_waited_for_and_verified() -> None:
    store, flaky = ready_store()
    assert store.collection is not None
    flaky.inner.delete_collection(store.collection)
    flaky.failures["create_collection"] = [_created_elsewhere(flaky.inner)]
    flaky.failures["get_collection"] = [_initialising()]

    assert store.count() == 1  # the point the other process stored

    assert flaky.calls["create_collection"] == 2  # binding, then the recreation that lost


def test_concurrent_calls_on_a_lost_collection_recreate_it_once() -> None:
    flaky = FlakyClient(SerializedClient(QdrantClient(location=":memory:")).as_client())
    store = store_on(flaky)
    store.ensure_collection(IDENTITY)
    assert store.collection is not None
    flaky.inner.delete_collection(store.collection)

    outcomes = _concurrently([store.count] * 6)

    assert outcomes == [None] * 6
    assert flaky.calls["create_collection"] == 2  # binding, then one recreation


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


# --- flagged chunks (D6) ---------------------------------------------------------------------


def _ids(points: list[models.Record] | list[models.ScoredPoint]) -> set[str]:
    return {str(point.id) for point in points}


def test_flagged_is_written_only_on_flagged_chunks() -> None:
    store, flaky = ready_store()
    chunks = [Chunk("clean", 0, 0, 0, 5), Chunk("injected", 0, 1, 6, 14, flagged=True)]
    upsert(store, "doc", chunks, [one_hot(8, 0), one_hot(8, 1)])
    assert store.collection is not None

    points = flaky.inner.retrieve(
        store.collection, [chunk_id("doc", 0), chunk_id("doc", 1)], with_payload=True
    )

    payloads = {str(point.id): point.payload or {} for point in points}
    assert FLAGGED_FIELD not in payloads[chunk_id("doc", 0)]
    assert payloads[chunk_id("doc", 1)][FLAGGED_FIELD] is True


def test_the_search_filter_excludes_only_points_flagged_true() -> None:
    assert unflagged_filter() == models.Filter(
        must_not=[models.FieldCondition(key="flagged", match=models.MatchValue(value=True))]
    )


def _write_raw_point(
    client: QdrantClient,
    collection: str,
    document_id: str,
    sequence: int,
    payload: dict[str, Any],
    vector: list[float],
) -> None:
    client.upsert(
        collection,
        points=[
            models.PointStruct(
                id=chunk_id(document_id, sequence),
                vector=vector,
                payload={
                    "document_id": document_id,
                    "page": 0,
                    "sequence": sequence,
                    "text": f"chunk {sequence}",
                    "page_text": "page",
                    "start": 0,
                    "end": 7,
                    "content_hash": "h",
                    "chunk_total": 3,
                    **payload,
                },
            )
        ],
        wait=True,
    )


def test_points_written_before_the_flagged_field_existed_are_all_searchable() -> None:
    store, flaky = ready_store()
    assert store.collection is not None
    for sequence in range(3):  # the payload shape of points stored before this change
        _write_raw_point(flaky.inner, store.collection, "old", sequence, {}, one_hot(8, sequence))

    hits = store.search(one_hot(8, 1), top_k=10)

    assert [hit.sequence for hit in hits] == [1, 0, 2] or {hit.sequence for hit in hits} == {
        0,
        1,
        2,
    }
    assert hits[0].sequence == 1
    assert store.document_chunk_count("old") == 3


def test_a_point_flagged_false_explicitly_is_searchable_and_flagged_true_is_not() -> None:
    store, flaky = ready_store()
    assert store.collection is not None
    _write_raw_point(flaky.inner, store.collection, "doc", 0, {"flagged": False}, one_hot(8, 0))
    _write_raw_point(flaky.inner, store.collection, "doc", 1, {"flagged": True}, one_hot(8, 1))
    _write_raw_point(flaky.inner, store.collection, "doc", 2, {}, one_hot(8, 2))

    assert [hit.sequence for hit in store.search(one_hot(8, 0), top_k=1)] == [0]
    assert [hit.sequence for hit in store.search(one_hot(8, 1), top_k=1)] != [1]
    assert {hit.sequence for hit in store.search(one_hot(8, 1), top_k=10)} == {0, 2}
    assert store.document_chunk_count("doc") == 3


def test_a_flagged_chunk_is_excluded_whatever_its_score() -> None:
    store, _ = ready_store()
    chunks = [Chunk("injected", 0, 0, 0, 8, flagged=True), Chunk("other", 0, 1, 9, 14)]
    upsert(store, "doc", chunks, [one_hot(8, 0), one_hot(8, 5)])

    hits = store.search(one_hot(8, 0), top_k=5)  # identical to the flagged vector

    assert [(hit.sequence, hit.score) for hit in hits] == [(1, 0.0)]


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
def test_the_server_collection_has_a_bool_index_on_flagged(
    server_store: QdrantVectorStore,
) -> None:
    server_store.ensure_collection(IDENTITY)
    server_store.ensure_collection(IDENTITY)  # creating the existing index again is a no-op
    assert server_store.collection is not None

    client = QdrantClient(url=os.environ["QDRANT_URL"])
    info = client.get_collection(server_store.collection)
    client.close()

    assert info.payload_schema[FLAGGED_FIELD].data_type == models.PayloadSchemaType.BOOL
    assert info.payload_schema["document_id"].data_type == models.PayloadSchemaType.KEYWORD


@pytest.mark.qdrant_server
def test_a_server_collection_without_the_flagged_index_gains_it_on_first_use(
    server_store: QdrantVectorStore,
) -> None:
    # A collection created by the previous version: identity metadata and the document_id
    # index only, with points that have no flagged field.
    name = collection_name(server_store.config.collection_prefix, IDENTITY)
    admin = QdrantClient(url=os.environ["QDRANT_URL"])
    admin.create_collection(
        name,
        vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE),
        metadata={IDENTITY_METADATA_KEY: canonical_identity(IDENTITY)},
    )
    admin.create_payload_index(name, "document_id", models.PayloadSchemaType.KEYWORD, wait=True)
    _write_raw_point(admin, name, "old", 0, {}, one_hot(8, 0))

    server_store.ensure_collection(IDENTITY)
    upsert(
        server_store,
        "new",
        [Chunk("injected", 0, 0, 0, 8, flagged=True), Chunk("clean", 0, 1, 9, 14)],
        [one_hot(8, 1), one_hot(8, 2)],
    )

    info = admin.get_collection(name)
    admin.close()
    assert info.payload_schema[FLAGGED_FIELD].data_type == models.PayloadSchemaType.BOOL
    hits = server_store.search(one_hot(8, 1), top_k=10)
    assert {(hit.document_id, hit.sequence) for hit in hits} == {("old", 0), ("new", 1)}


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
    assert info.payload_schema[FLAGGED_FIELD].data_type == models.PayloadSchemaType.BOOL


def _prefixed_collections(prefix: str) -> list[str]:
    admin = QdrantClient(url=os.environ["QDRANT_URL"])
    names = [c.name for c in admin.get_collections().collections if c.name.startswith(prefix)]
    admin.close()
    return names


@pytest.mark.qdrant_server
def test_simultaneous_first_calls_on_one_store_all_bind_one_collection(
    server_store: QdrantVectorStore,
) -> None:
    # The service's case: one store per process, twelve request threads binding it at once.
    outcomes = _concurrently([lambda: server_store.ensure_collection(IDENTITY)] * 12)

    assert outcomes == [None] * 12
    assert _prefixed_collections(f"{server_store.config.collection_prefix}_") == [
        server_store.collection
    ]


@pytest.mark.qdrant_server
def test_simultaneous_first_calls_from_separate_stores_all_bind_one_collection(
    server_store: QdrantVectorStore,
) -> None:
    # Replicas: each store has its own client, so only the server orders the creates (Qdrant
    # answers the losers 409 and, until the new collection's shards serve reads, 500).
    stores = [QdrantVectorStore(server_store.config) for _ in range(12)]
    try:
        outcomes = _concurrently([partial(store.ensure_collection, IDENTITY) for store in stores])
    finally:
        for store in stores:
            store.close()

    assert outcomes == [None] * 12
    expected = collection_name(server_store.config.collection_prefix, IDENTITY)
    assert _prefixed_collections(f"{server_store.config.collection_prefix}_") == [expected]
    server_store.ensure_collection(IDENTITY)
    info = QdrantClient(url=os.environ["QDRANT_URL"]).get_collection(expected)
    assert info.config.metadata == {IDENTITY_METADATA_KEY: canonical_identity(IDENTITY)}
    assert info.payload_schema["document_id"].data_type == models.PayloadSchemaType.KEYWORD
