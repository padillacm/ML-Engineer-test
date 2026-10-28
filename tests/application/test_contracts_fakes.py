"""Runs the port contracts against the in-memory fakes.

Real adapters join the same contracts in later blocks by adding a subclass here (or a
parametrised factory) that overrides the same fixtures.
"""

import pytest

from app.application.ports import Embedder, LLMClient, NameRecognizer, OcrEngine, VectorStore
from app.domain.errors import DependencyUnavailable
from app.domain.types import Box, EmbeddingIdentity, OcrPage, PageWord
from tests.application.contracts import (
    EmbedderContract,
    LLMClientContract,
    NameRecognizerContract,
    OcrEngineContract,
    VectorStoreContract,
)
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm import FakeLLMClient
from tests.fakes.names import FakeNameRecognizer
from tests.fakes.ocr import FakeOcrEngine
from tests.fakes.vector_store import FakeVectorStore


class TestFakeOcrEngineContract(OcrEngineContract):
    @pytest.fixture
    def engine(self) -> OcrEngine:
        return FakeOcrEngine(
            [
                OcrPage(
                    0, 595.0, 842.0, (PageWord("Memo", Box(0, 72.0, 72.0, 40.0, 12.0), 0, 0),), None
                ),
                OcrPage(1, 842.0, 595.0, (), None),
            ]
        )

    @pytest.fixture
    def sample_pdf(self) -> bytes:
        return b"%PDF-1.7 two pages"

    @pytest.fixture
    def expected_pages(self) -> int:
        return 2


class TestFakeNameRecognizerContract(NameRecognizerContract):
    @pytest.fixture
    def recognizer(self) -> NameRecognizer:
        return FakeNameRecognizer(["Maria Gonzalez", "Peter Smith", "Peter"])


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


class TestFakeLLMClientContract(LLMClientContract):
    @pytest.fixture
    def client(self) -> LLMClient:
        return FakeLLMClient()

    @pytest.fixture
    def failing_client(self) -> LLMClient:
        return FakeLLMClient([DependencyUnavailable("LLM is down", "llm_unavailable")])
