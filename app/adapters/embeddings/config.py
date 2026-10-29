"""Settings of the e5 embedder, validated when they are created."""

import json
from dataclasses import dataclass
from pathlib import Path

from app.domain.errors import ConfigurationError

_CONFIG_FILE = "config.json"
_TOKENIZER_FILES = ("tokenizer.json", "sentencepiece.bpe.model")
_WEIGHT_FILE = "model.safetensors"


@dataclass(frozen=True, slots=True)
class E5EmbedderConfig:
    """Where the e5 model lives and how it runs; the defaults are the benchmarked configuration.

    ``model_path`` is a local directory with ``config.json``, the tokenizer files
    (``tokenizer.json`` or ``sentencepiece.bpe.model``) and ``model.safetensors``; nothing is
    ever downloaded (pickled ``pytorch_model.bin`` weights are not accepted). ``revision`` is the
    Hub commit the files come from; it is recorded in the embedder identity, so vectors of two
    revisions are never mixed in one collection. ``max_length`` is the token limit of one input
    (longer inputs are truncated; an 800-character chunk is about 200 tokens), ``batch_size`` the
    number of texts run together, ``threads`` the torch intra-op thread count (process-wide:
    creating the embedder sets it for every torch computation in the process, and the last torch
    adapter created wins, so the service passes the one ``TORCH_THREADS`` to the embedder and the
    recogniser; the field stays for standalone use by benchmarks and scripts). ``max_chars``
    bounds one text: a longer one is refused without running the model, because it would be
    truncated anyway and it would hold one of the embedder's slots meanwhile. ``pool_size`` is
    the number of passage batches embedded at once: they share one copy of the model weights
    and one tokenizer (tokenising runs under one lock), and a call waits for a free slot; one
    more slot is reserved for queries, which also go before waiting passages. Each running
    batch uses ``threads`` intra-op threads, so up to ``pool_size`` x ``threads`` CPUs are busy.
    The default 4 is the smallest pool within 5 % of the best ingestion throughput on 16 CPUs
    with 2 threads (BENCHMARKS 18); the service derives the size from the CPUs when
    ``EMBEDDING_POOL_SIZE`` is unset. Invalid values raise ``ConfigurationError``.
    """

    model_path: str | Path
    revision: str
    max_length: int = 512
    batch_size: int = 16
    threads: int = 2
    max_chars: int = 50_000
    pool_size: int = 4

    def __post_init__(self) -> None:
        _require(
            isinstance(self.revision, str)
            and self.revision.strip() != ""
            and self.revision == self.revision.strip(),
            "revision must be a non-blank string without surrounding whitespace",
        )
        _require(_is_positive_int(self.max_length), "max_length must be a positive integer")
        _require(_is_positive_int(self.batch_size), "batch_size must be a positive integer")
        _require(_is_positive_int(self.threads), "threads must be a positive integer")
        _require(_is_positive_int(self.max_chars), "max_chars must be a positive integer")
        _require(_is_positive_int(self.pool_size), "pool_size must be a positive integer")
        _validate_model_directory(self.path)

    @property
    def path(self) -> Path:
        """``model_path`` as a ``Path``."""
        return Path(self.model_path)


def _validate_model_directory(path: Path) -> None:
    _require(path.is_dir(), f"embedding model directory not found: {path}")
    _require((path / _CONFIG_FILE).is_file(), f"embedding model has no {_CONFIG_FILE}: {path}")
    _require(
        any((path / name).is_file() for name in _TOKENIZER_FILES),
        f"embedding model has no tokenizer file ({' or '.join(_TOKENIZER_FILES)}): {path}",
    )
    _require(
        (path / _WEIGHT_FILE).is_file(), f"embedding model has no weights ({_WEIGHT_FILE}): {path}"
    )
    _read_json(path / _CONFIG_FILE)


def _read_json(path: Path) -> dict[str, object]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"embedding model file is not readable JSON: {path}") from error
    _require(isinstance(content, dict), f"embedding model file is not a JSON object: {path}")
    return dict(content)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)
