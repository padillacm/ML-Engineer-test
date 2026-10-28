"""Settings of the ``transformers`` name recogniser, validated when they are created."""

import json
from dataclasses import dataclass
from pathlib import Path

from app.domain.errors import ConfigurationError

AGGREGATIONS: frozenset[str] = frozenset({"simple", "first", "average", "max"})
"""Aggregation strategies that group sub-word tokens into labelled entities.

``"none"`` is excluded: it yields one ``B-``/``I-`` tag per token instead of entity groups.
"""

_CONFIG_FILE = "config.json"
_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
_TOKENIZER_FILES = ("tokenizer.json", "sentencepiece.bpe.model")
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


@dataclass(frozen=True, slots=True)
class TransformersNerConfig:
    """Where the model lives and how it runs; the defaults are the benchmarked configuration.

    ``model_path`` is a local directory with ``config.json``, the tokenizer files
    (``tokenizer.json`` or ``sentencepiece.bpe.model``) and the weights (``model.safetensors``
    or ``pytorch_model.bin``); nothing is ever downloaded. ``labels`` are the entity groups
    reported (the model's ``PER``), ``aggregation`` and ``stride`` (overlap, in tokens, between
    the windows of a text longer than the model input) are passed to the ``transformers``
    pipeline, ``threads`` is the torch intra-op thread count and ``batch_size`` the number of
    windows of one long text run together. ``max_chars`` bounds the text of one call: a longer
    text is refused without running the model, because inference costs about 0.25 s per 1,000
    characters at 2 threads and holds the recogniser's lock meanwhile (a 200,000-character text
    took 75 s). 50,000 characters is several dense pages of text, far above one OCR page.
    Invalid values raise ``ConfigurationError``.
    """

    model_path: str | Path
    labels: tuple[str, ...] = ("PER",)
    aggregation: str = "first"
    stride: int = 64
    threads: int = 2
    batch_size: int = 8
    max_chars: int = 50_000

    def __post_init__(self) -> None:
        _require(
            isinstance(self.labels, tuple)
            and len(self.labels) > 0
            and all(isinstance(label, str) and label.strip() for label in self.labels),
            "labels must be a non-empty tuple of non-blank strings",
        )
        _require(
            self.aggregation in AGGREGATIONS,
            f"aggregation must be one of {sorted(AGGREGATIONS)}",
        )
        _require(_is_positive_int(self.stride), "stride must be a positive integer")
        _require(_is_positive_int(self.threads), "threads must be a positive integer")
        _require(_is_positive_int(self.batch_size), "batch_size must be a positive integer")
        _require(_is_positive_int(self.max_chars), "max_chars must be a positive integer")
        _validate_model_directory(self.path)
        max_length = _declared_max_length(self.path)
        _require(
            max_length is None or self.stride < max_length,
            f"stride must be smaller than the model input length ({max_length} tokens)",
        )

    @property
    def path(self) -> Path:
        """``model_path`` as a ``Path``."""
        return Path(self.model_path)


def _validate_model_directory(path: Path) -> None:
    _require(path.is_dir(), f"NER model directory not found: {path}")
    _require((path / _CONFIG_FILE).is_file(), f"NER model has no {_CONFIG_FILE}: {path}")
    _require(
        any((path / name).is_file() for name in _TOKENIZER_FILES),
        f"NER model has no tokenizer file ({' or '.join(_TOKENIZER_FILES)}): {path}",
    )
    _require(
        any((path / name).is_file() for name in _WEIGHT_FILES),
        f"NER model has no weights ({' or '.join(_WEIGHT_FILES)}): {path}",
    )
    _read_json(path / _CONFIG_FILE)


def _declared_max_length(path: Path) -> int | None:
    """``model_max_length`` from ``tokenizer_config.json``, when the file declares one.

    The loaded tokenizer's value is checked again when the model is loaded.
    """
    tokenizer_config = path / _TOKENIZER_CONFIG_FILE
    if not tokenizer_config.is_file():
        return None
    value = _read_json(tokenizer_config).get("model_max_length")
    return value if isinstance(value, int) and _is_positive_int(value) else None


def _read_json(path: Path) -> dict[str, object]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"NER model file is not readable JSON: {path}") from error
    _require(isinstance(content, dict), f"NER model file is not a JSON object: {path}")
    return dict(content)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)
