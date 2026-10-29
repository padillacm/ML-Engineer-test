"""Shared fixtures of the injection detector adapter tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from tests.adapters.injection.tiny_model import build_tiny_model


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A local model directory with the tiny seeded classifier (built once per session)."""
    return build_tiny_model(tmp_path_factory.mktemp("tiny-prompt-guard"))


@pytest.fixture(autouse=True)
def _restore_torch_threads() -> Iterator[None]:
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)
