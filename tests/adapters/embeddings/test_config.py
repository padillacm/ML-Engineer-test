"""Validation of ``E5EmbedderConfig``."""

import dataclasses
from pathlib import Path

import pytest

from app.adapters.embeddings import E5EmbedderConfig
from app.domain.errors import ConfigurationError

REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"


def test_defaults_are_the_benchmarked_configuration(model_dir: Path) -> None:
    config = E5EmbedderConfig(model_dir, REVISION)

    assert (config.max_length, config.batch_size, config.threads) == (512, 16, 2)
    assert config.max_chars == 50_000
    assert config.pool_size == 4
    assert config.path == model_dir


def test_config_is_frozen(model_dir: Path) -> None:
    config = E5EmbedderConfig(str(model_dir), REVISION)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.threads = 4  # type: ignore[misc]


@pytest.mark.parametrize("revision", ["", "   ", " abc", 123])
def test_revision_must_be_a_clean_string(model_dir: Path, revision: object) -> None:
    with pytest.raises(ConfigurationError, match="revision"):
        E5EmbedderConfig(model_dir, revision)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["max_length", "batch_size", "threads", "max_chars", "pool_size"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_sizes_must_be_positive_integers(model_dir: Path, field: str, value: object) -> None:
    with pytest.raises(ConfigurationError, match=field):
        E5EmbedderConfig(model_dir, REVISION, **{field: value})  # type: ignore[arg-type]


def test_missing_directory_is_invalid(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        E5EmbedderConfig(tmp_path / "absent", REVISION)


@pytest.mark.parametrize(
    ("removed", "message"),
    [
        ("config.json", "config.json"),
        ("tokenizer.json", "tokenizer file"),
        ("model.safetensors", "weights"),
    ],
)
def test_required_files_must_exist(model_dir: Path, removed: str, message: str) -> None:
    (model_dir / removed).unlink()

    with pytest.raises(ConfigurationError, match=message):
        E5EmbedderConfig(model_dir, REVISION)


def test_a_sentencepiece_model_is_an_accepted_tokenizer_file(model_dir: Path) -> None:
    (model_dir / "tokenizer.json").unlink()
    (model_dir / "sentencepiece.bpe.model").write_bytes(b"spm")

    E5EmbedderConfig(model_dir, REVISION)


def test_pickled_weights_alone_are_refused(model_dir: Path) -> None:
    (model_dir / "model.safetensors").unlink()
    (model_dir / "pytorch_model.bin").write_bytes(b"pickle")

    with pytest.raises(ConfigurationError, match=r"model\.safetensors"):
        E5EmbedderConfig(model_dir, REVISION)


@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
def test_config_json_must_be_a_json_object(model_dir: Path, content: str) -> None:
    (model_dir / "config.json").write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="JSON"):
        E5EmbedderConfig(model_dir, REVISION)
