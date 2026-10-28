"""Fake adapter factories that record the configuration each adapter was built with."""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.adapters.embeddings.config import E5EmbedderConfig
from app.adapters.llm.config import LLMClientConfig
from app.adapters.ner.config import TransformersNerConfig
from app.adapters.ocr.engine import TesseractOcrConfig
from app.adapters.vector_store.config import QdrantStoreConfig
from app.application.ports import Embedder, OcrEngine
from app.bootstrap import AdapterFactories
from app.domain.types import ChatMessage
from app.settings import Settings, load_settings
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm import FakeLLMClient
from tests.fakes.names import FakeNameRecognizer
from tests.fakes.ocr import FakeOcrEngine
from tests.fakes.vector_store import FakeVectorStore


def model_directory(root: Path, name: str) -> Path:
    """A directory that passes the model-directory checks (the files are placeholders)."""
    directory = root / name
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "sentencepiece.bpe.model").write_bytes(b"placeholder")
    (directory / "model.safetensors").write_bytes(b"placeholder")
    return directory


def bootstrap_settings(models: Path, **overrides: Any) -> Settings:
    """Settings whose model paths are placeholder directories under ``models``."""
    values: dict[str, Any] = {
        "ner_model_path": str(models / "ner"),
        "embedding_model_path": str(models / "e5"),
        "embedding_model_revision": "rev-1",
        "qdrant_url": None,
        "qdrant_location": ":memory:",
        "llm_base_url": None,
        "llm_model": None,
    }
    values.update(overrides)
    return load_settings(_env_file=None, **values)


class RecordingStore(FakeVectorStore):
    """``FakeVectorStore`` with the readiness probe and ``close`` of the real adapter."""

    def __init__(self, events: list[str], *, down: Exception | None = None) -> None:
        super().__init__()
        self.events = events
        self.down = down

    def ping(self) -> None:
        self.events.append("ping")
        if self.down is not None:
            raise self.down

    def close(self) -> None:
        self.events.append("store.close")


class RecordingRecognizer(FakeNameRecognizer):
    """``FakeNameRecognizer`` with ``close``."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def close(self) -> None:
        self.events.append("recognizer.close")


class RecordingLLM(FakeLLMClient):
    """``FakeLLMClient`` with ``close``."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(default="an answer")
        self.events = events

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.events.append("llm.complete")
        return super().complete(messages)

    def close(self) -> None:
        self.events.append("llm.close")


class Recorder:
    """Builds fakes in place of the adapters and records configs, builds and closes."""

    def __init__(self, *, store_down: Exception | None = None) -> None:
        self.configs: dict[str, object] = {}
        self.built: Counter[str] = Counter()
        self.events: list[str] = []
        self.store_down = store_down
        self.failing: dict[str, Exception] = {}

    def factories(self) -> AdapterFactories:
        return AdapterFactories(
            ocr=self._ocr,
            recognizer=self._recognizer,
            embedder=self._embedder,
            store=self._store,
            llm=self._llm,
        )

    def _record(self, name: str, config: object) -> None:
        self.configs[name] = config
        self.built[name] += 1
        if name in self.failing:
            raise self.failing[name]

    def _ocr(self, config: TesseractOcrConfig) -> OcrEngine:
        self._record("ocr", config)
        return FakeOcrEngine()

    def _recognizer(self, config: TransformersNerConfig) -> RecordingRecognizer:
        self._record("recognizer", config)
        return RecordingRecognizer(self.events)

    def _embedder(self, config: E5EmbedderConfig) -> Embedder:
        self._record("embedder", config)
        return FakeEmbedder()

    def _store(self, config: QdrantStoreConfig) -> RecordingStore:
        self._record("store", config)
        return RecordingStore(self.events, down=self.store_down)

    def _llm(self, config: LLMClientConfig) -> RecordingLLM:
        self._record("llm", config)
        return RecordingLLM(self.events)
