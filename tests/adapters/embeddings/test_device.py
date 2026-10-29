"""The embedder's device setting and its GPU out-of-memory handling (spec: inference-device)."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.adapters.embeddings import E5Embedder, E5EmbedderConfig
from app.adapters.embeddings.embedder import LoadedEncoder
from app.domain.errors import ConfigurationError, ProcessingFailed
from tests.adapters.oom import count_cleanups, out_of_memory

REVISION = "rev123"


def test_the_device_defaults_to_the_cpu(model_dir: Path) -> None:
    assert E5EmbedderConfig(model_dir, REVISION).device == "cpu"


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_a_cuda_device_is_accepted(model_dir: Path, device: str) -> None:
    assert E5EmbedderConfig(model_dir, REVISION, device=device).device == device


@pytest.mark.parametrize("device", ["gpu", "auto", "cuda:a"])
def test_an_unknown_device_is_invalid(model_dir: Path, device: str) -> None:
    with pytest.raises(ConfigurationError, match="device must be"):
        E5EmbedderConfig(model_dir, REVISION, device=device)


@pytest.mark.parametrize("query", [True, False], ids=["query", "passages"])
def test_out_of_memory_is_embedding_failed_after_the_cleanup_and_the_next_call_works(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch, *, query: bool
) -> None:
    cleanups = count_cleanups(monkeypatch)
    failures = [out_of_memory()]

    def encode(texts: Sequence[str]) -> list[list[float]]:
        if failures:
            raise failures.pop()
        return [[1.0, 0.0] for _ in texts]

    embedder = E5Embedder(
        E5EmbedderConfig(model_dir, REVISION), model_factory=lambda _: LoadedEncoder(2, encode)
    )

    def call() -> object:
        return embedder.embed_query("text") if query else embedder.embed_passages(["a", "b"])

    with pytest.raises(ProcessingFailed) as caught:
        call()

    assert caught.value.code == "embedding_failed"
    assert cleanups == [1]
    assert call()
    assert cleanups == [1]
