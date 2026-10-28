"""``QdrantVectorStore`` against the ``VectorStore`` contract: embedded mode and a real server."""

import dataclasses
import os
from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient, models

from app.adapters.vector_store import QdrantStoreConfig, QdrantVectorStore
from app.adapters.vector_store.qdrant_store import chunk_id
from app.application.ports import VectorStore
from tests.adapters.vector_store.conftest import memory_store
from tests.adapters.vector_store.flaky_client import (
    FlakyClient,
    Scripted,
    SerializedClient,
    connect_error,
)
from tests.application.contracts import VectorStoreContract


class _QdrantHooks(VectorStoreContract):
    """Contract hooks that reach the Qdrant data through a second client."""

    def raw_client(self, store: QdrantVectorStore) -> QdrantClient:
        raise NotImplementedError

    def sibling(
        self, store: QdrantVectorStore, client: QdrantClient, *, upsert_batch: int, attempts: int
    ) -> QdrantVectorStore:
        """Another store on the same collection, through ``client``."""
        config = dataclasses.replace(store.config, upsert_batch=upsert_batch, attempts=attempts)
        sibling = QdrantVectorStore(config, client=client, sleep=lambda _: None)
        assert store.identity is not None
        sibling.ensure_collection(store.identity)
        return sibling

    def drop_chunk(self, store: VectorStore, document_id: str, sequence: int) -> None:
        assert isinstance(store, QdrantVectorStore)
        assert store.collection is not None
        self.raw_client(store).delete(
            store.collection,
            points_selector=models.PointIdsList(points=[chunk_id(document_id, sequence)]),
            wait=True,
        )

    def writer_failing_after(self, store: VectorStore, stored: int) -> VectorStore:
        assert isinstance(store, QdrantVectorStore)
        flaky = FlakyClient(self.raw_client(store))
        writer = self.sibling(store, flaky.as_client(), upsert_batch=1, attempts=1)
        script: list[Scripted] = [None] * stored
        flaky.failures["upsert"] = [*script, connect_error()]
        return writer

    def other_writer(self, store: VectorStore) -> VectorStore:
        assert isinstance(store, QdrantVectorStore)
        return self.sibling(store, self.raw_client(store), upsert_batch=1, attempts=3)


class TestQdrantMemoryStoreContract(_QdrantHooks):
    @pytest.fixture
    def store(self) -> VectorStore:
        # Serialized so the concurrency contract can share the embedded client between threads.
        self.client = SerializedClient(QdrantClient(location=":memory:")).as_client()
        return QdrantVectorStore(QdrantStoreConfig(location=":memory:"), client=self.client)

    def raw_client(self, store: QdrantVectorStore) -> QdrantClient:
        return self.client


@pytest.mark.qdrant_server
class TestQdrantServerStoreContract(_QdrantHooks):
    @pytest.fixture
    def store(self, server_store: QdrantVectorStore) -> VectorStore:
        return server_store

    @pytest.fixture(autouse=True)
    def _admin(self) -> Iterator[None]:
        self.admin = QdrantClient(url=os.environ["QDRANT_URL"])
        yield
        self.admin.close()

    def raw_client(self, store: QdrantVectorStore) -> QdrantClient:
        return self.admin


def test_store_satisfies_the_port() -> None:
    assert isinstance(memory_store(), VectorStore)
