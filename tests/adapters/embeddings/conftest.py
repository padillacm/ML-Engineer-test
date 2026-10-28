"""Shared fixtures of the embedding adapter tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from tests.adapters.embeddings.tiny_model import build_tiny_model


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A local model directory with the tiny seeded encoder (built once per session)."""
    return build_tiny_model(tmp_path_factory.mktemp("tiny-e5"))


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A directory with the file names the configuration checks for (contents are dummies)."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    return tmp_path


@pytest.fixture(autouse=True)
def _restore_torch_threads() -> Iterator[None]:
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)
