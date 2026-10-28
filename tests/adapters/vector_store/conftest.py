"""Stores for the adapter tests: embedded ``:memory:`` on the host, the compose server when set."""

import dataclasses
import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from qdrant_client import QdrantClient

from app.adapters.vector_store import QdrantStoreConfig, QdrantVectorStore


def memory_store(**overrides: Any) -> QdrantVectorStore:
    """A fresh embedded store; ``overrides`` are ``QdrantStoreConfig`` fields."""
    config = dataclasses.replace(QdrantStoreConfig(location=":memory:"), **overrides)
    return QdrantVectorStore(config)


@pytest.fixture
def server_store() -> Iterator[QdrantVectorStore]:
    """A store on the ``QDRANT_URL`` server with a prefix unique to the test.

    Every collection with that prefix is dropped afterwards, so tests sharing the server never
    see each other's points.
    """
    url = os.environ["QDRANT_URL"]
    prefix = f"t{uuid.uuid4().hex[:10]}"
    store = QdrantVectorStore(QdrantStoreConfig(url=url, collection_prefix=prefix))
    yield store
    store.close()
    admin = QdrantClient(url=url)
    for collection in admin.get_collections().collections:
        if collection.name.startswith(f"{prefix}_"):
            admin.delete_collection(collection.name)
    admin.close()
