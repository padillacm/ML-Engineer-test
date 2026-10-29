"""Configuration of the Prompt Guard 2 detector adapter.

``model_path`` is a local directory holding a sequence-classification model and its tokenizer
(the image puts Llama Prompt Guard 2 86M under ``/models``). ``threads`` is torch's intra-op
thread count, process-wide and shared with the recogniser and the embedder (the service passes
``TORCH_THREADS``). ``pool_size`` is how many screenings may run at once on the one loaded
model; ``batch_size`` how many texts one forward pass scores. ``window_tokens`` is the longest
text the model reads at once (its position limit); a longer text is scored by windows of that
size that overlap by ``stride_tokens``, keeping the highest score. ``device`` is where the model
runs, ``"cpu"`` or ``"cuda[:<index>]"`` (in half precision there, with the softmax in full
precision; ``app.adapters.device``); the service resolves the ``DEVICE`` setting into it.
Invalid values raise ``ConfigurationError``.
"""

import math
from dataclasses import dataclass

from app.adapters.device import is_device_name
from app.domain.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class InjectionDetectorConfig:
    model_path: str
    threads: int = 2
    pool_size: int = 2
    batch_size: int = 16
    window_tokens: int = 512
    stride_tokens: int = 128
    device: str = "cpu"

    def __post_init__(self) -> None:
        _require(
            isinstance(self.model_path, str) and bool(self.model_path.strip()),
            "model_path must be a non-empty string",
        )
        _require(_is_positive_int(self.threads), "threads must be a positive integer")
        _require(_is_positive_int(self.pool_size), "pool_size must be a positive integer")
        _require(_is_positive_int(self.batch_size), "batch_size must be a positive integer")
        _require(_is_positive_int(self.window_tokens), "window_tokens must be a positive integer")
        _require(
            _is_positive_int(self.stride_tokens) and self.stride_tokens < self.window_tokens,
            "stride_tokens must be a positive integer smaller than window_tokens",
        )
        _require(is_device_name(self.device), "device must be 'cpu', 'cuda' or 'cuda:<index>'")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def _is_positive_int(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )
