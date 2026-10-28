"""Shared fixtures of the NER adapter tests."""

from pathlib import Path

import pytest

from tests.adapters.ner.tiny_model import build_tiny_model


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A local model directory with the tiny hand-wired model (built once per session)."""
    return build_tiny_model(tmp_path_factory.mktemp("tiny-ner"))


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A directory with the file names the configuration checks for (contents are dummies)."""
    for name in ("config.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "sentencepiece.bpe.model").write_bytes(b"spm")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    return tmp_path
