"""Validation of ``TransformersNerConfig``."""

import dataclasses
from pathlib import Path

import pytest

from app.adapters.ner import TransformersNerConfig
from app.domain.errors import ConfigurationError


def test_defaults_are_the_benchmarked_configuration(model_dir: Path) -> None:
    config = TransformersNerConfig(model_dir)

    assert config.labels == ("PER",)
    assert config.aggregation == "first"
    assert (config.stride, config.threads, config.batch_size) == (64, 2, 8)
    assert config.max_chars == 50_000
    assert config.pool_size == 8
    assert config.path == model_dir


def test_a_string_path_is_accepted(model_dir: Path) -> None:
    assert TransformersNerConfig(str(model_dir)).path == model_dir


def test_is_frozen(model_dir: Path) -> None:
    config = TransformersNerConfig(model_dir)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.stride = 32  # type: ignore[misc]


def test_missing_directory_is_an_invalid_configuration(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as caught:
        TransformersNerConfig(tmp_path / "no-such-model")

    assert caught.value.code == "invalid_configuration"
    assert "not found" in caught.value.message


def test_a_file_is_not_a_model_directory(model_dir: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        TransformersNerConfig(model_dir / "config.json")


@pytest.mark.parametrize(
    ("removed", "message"),
    [
        (("config.json",), "config.json"),
        (("sentencepiece.bpe.model",), "tokenizer"),
        (("model.safetensors",), "weights"),
    ],
)
def test_missing_model_files_are_invalid(
    model_dir: Path, removed: tuple[str, ...], message: str
) -> None:
    for name in removed:
        (model_dir / name).unlink()

    with pytest.raises(ConfigurationError, match=message):
        TransformersNerConfig(model_dir)


def test_alternative_tokenizer_and_weight_files_are_accepted(model_dir: Path) -> None:
    (model_dir / "sentencepiece.bpe.model").rename(model_dir / "tokenizer.json")
    (model_dir / "model.safetensors").rename(model_dir / "pytorch_model.bin")

    assert TransformersNerConfig(model_dir).path == model_dir


@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
def test_unreadable_config_json_is_invalid(model_dir: Path, content: str) -> None:
    (model_dir / "config.json").write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=r"config\.json"):
        TransformersNerConfig(model_dir)


@pytest.mark.parametrize("field", ["stride", "threads", "batch_size", "max_chars", "pool_size"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_counts_must_be_positive_integers(model_dir: Path, field: str, value: object) -> None:
    with pytest.raises(ConfigurationError, match=field):
        TransformersNerConfig(model_dir, **{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("labels", [(), ("",), ("  ",), ["PER"], (1,)])
def test_labels_must_be_a_non_empty_tuple_of_names(model_dir: Path, labels: object) -> None:
    with pytest.raises(ConfigurationError, match="labels"):
        TransformersNerConfig(model_dir, labels=labels)  # type: ignore[arg-type]


@pytest.mark.parametrize("aggregation", ["none", "FIRST", ""])
def test_aggregation_must_group_entities(model_dir: Path, aggregation: str) -> None:
    with pytest.raises(ConfigurationError, match="aggregation"):
        TransformersNerConfig(model_dir, aggregation=aggregation)


@pytest.mark.parametrize("aggregation", ["simple", "first", "average", "max"])
def test_grouping_aggregations_are_accepted(model_dir: Path, aggregation: str) -> None:
    assert TransformersNerConfig(model_dir, aggregation=aggregation).aggregation == aggregation


def test_stride_must_be_smaller_than_the_declared_input_length(model_dir: Path) -> None:
    (model_dir / "tokenizer_config.json").write_text('{"model_max_length": 64}', encoding="utf-8")

    assert TransformersNerConfig(model_dir, stride=63).stride == 63
    with pytest.raises(ConfigurationError, match="stride"):
        TransformersNerConfig(model_dir, stride=64)


@pytest.mark.parametrize("declared", ['{"model_max_length": "big"}', "{}"])
def test_stride_is_checked_at_load_time_when_no_length_is_declared(
    model_dir: Path, declared: str
) -> None:
    (model_dir / "tokenizer_config.json").write_text(declared, encoding="utf-8")

    assert TransformersNerConfig(model_dir, stride=10_000).stride == 10_000


def test_without_tokenizer_config_the_stride_is_checked_at_load_time(model_dir: Path) -> None:
    (model_dir / "tokenizer_config.json").unlink()

    assert TransformersNerConfig(model_dir, stride=10_000).stride == 10_000
