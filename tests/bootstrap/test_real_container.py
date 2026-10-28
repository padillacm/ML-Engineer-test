"""The real container, with both models and the embedded Qdrant (container suite only)."""

import os

import pytest

from app.adapters.embeddings.embedder import E5Embedder
from app.adapters.ner.recognizer import TransformersNameRecognizer
from app.adapters.ocr.engine import TesseractOcrEngine
from app.adapters.vector_store.qdrant_store import QdrantVectorStore
from app.application.unconfigured_llm import UnconfiguredLLM
from app.bootstrap import build_container
from app.settings import load_settings

pytestmark = [pytest.mark.ner_model, pytest.mark.embedding_model]


def test_real_container_builds_every_adapter_and_is_ready() -> None:
    settings = load_settings(
        _env_file=None,
        ner_model_path=os.environ["NER_MODEL_PATH"],
        embedding_model_path=os.environ["EMBEDDING_MODEL_PATH"],
        embedding_model_revision=os.environ.get("EMBEDDING_MODEL_REVISION", "unknown"),
        qdrant_url=None,
        qdrant_location=":memory:",
        llm_base_url=None,
        llm_model=None,
    )

    container = build_container(settings)
    try:
        assert isinstance(container.extract._ocr, TesseractOcrEngine)
        assert isinstance(container.extract._recognizer, TransformersNameRecognizer)
        assert isinstance(container.ingest._embedder, E5Embedder)
        assert isinstance(container.ingest._store, QdrantVectorStore)
        assert isinstance(container.ask._llm, UnconfiguredLLM)
        assert {name: check() for name, check in container.readiness} == {
            "vector_store": "ok",
            "llm": "not_configured",
        }
    finally:
        container.close()
