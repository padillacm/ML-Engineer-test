"""Runs the port contracts against the in-memory fakes.

Real adapters join the same contracts in later blocks by adding a subclass here (or a
parametrised factory) that overrides the same fixtures.
"""

import pytest

from app.application.ports import Embedder, VectorStore
from app.domain.types import EmbeddingIdentity
from tests.application.contracts import EmbedderContract, VectorStoreContract
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.vector_store import FakeVectorStore


class TestFakeEmbedderContract(EmbedderContract):
    @pytest.fixture
    def embedder(self) -> Embedder:
        return FakeEmbedder()


class TestNormalizedFakeEmbedderOfOtherDimensionContract(EmbedderContract):
    """Normalised branch of the unit-length contract with a non-default identity."""

    @pytest.fixture
    def embedder(self) -> Embedder:
        embedder = FakeEmbedder(EmbeddingIdentity("fake-unit", 32, "none", True, "cosine"))
        assert embedder.identity.normalized
        return embedder


class TestUnnormalizedFakeEmbedderContract(EmbedderContract):
    @pytest.fixture
    def embedder(self) -> Embedder:
        return FakeEmbedder(EmbeddingIdentity("fake-raw", 32, "none", False, "cosine"))


class TestFakeVectorStoreContract(VectorStoreContract):
    @pytest.fixture
    def store(self) -> VectorStore:
        return FakeVectorStore()
