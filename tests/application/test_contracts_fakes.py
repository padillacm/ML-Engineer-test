"""Runs the port contracts against the in-memory fakes.

Real adapters join the same contracts in later blocks by adding a subclass here (or a
parametrised factory) that overrides the same fixtures.
"""

import pytest

from app.application.ports import Embedder, OcrEngine, VectorStore
from app.domain.types import Box, EmbeddingIdentity, OcrPage, PageWord
from tests.application.contracts import EmbedderContract, OcrEngineContract, VectorStoreContract
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.ocr import FakeOcrEngine
from tests.fakes.vector_store import FakeVectorStore


class TestFakeOcrEngineContract(OcrEngineContract):
    @pytest.fixture
    def engine(self) -> OcrEngine:
        return FakeOcrEngine(
            [
                OcrPage(0, 595.0, 842.0, (PageWord("Memo", Box(0, 72.0, 72.0, 40.0, 12.0)),)),
                OcrPage(1, 842.0, 595.0, ()),
            ]
        )

    @pytest.fixture
    def sample_pdf(self) -> bytes:
        return b"%PDF-1.7 two pages"

    @pytest.fixture
    def expected_pages(self) -> int:
        return 2


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
