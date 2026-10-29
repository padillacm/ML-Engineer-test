"""Where the in-process models run: the device helpers the three torch adapters share.

The composition root resolves the ``DEVICE`` setting once (``app.bootstrap.resolve_device``,
which asks ``gpu_visible`` and ``gpu_description``) and hands each adapter configuration the
resolved device name, ``"cpu"`` or ``"cuda[:n]"``; the adapters never probe the hardware
themselves, so the three always agree and the start-up line names the device they use.

On the GPU the embedder and the detector run in half precision (``model_dtype``): the e5 fp16
equivalence was measured (98.99 % top-1 agreement, BENCHMARKS 5) and fp16 halves the VRAM of
fp32. The recogniser stays in float32 there (design D2: its fp16 spans differed from the CPU's
on a few pages, BENCHMARKS 23), which its VRAM allows. Only the
model's forward pass is half precision: tokenising stays on the CPU, and every adapter brings
its outputs back to the CPU in float32 before the steps that compare scores or vectors
(pooling and normalisation of e5, the detector's softmax, the NER pipeline's aggregation), so
those steps are the CPU path's. ``place`` leaves a model on the CPU untouched (no dtype cast),
so the CPU path stays bit-identical to the one before the GPU support.

A CUDA out-of-memory error inside a call fails that call with the adapter's processing-failure
code; ``release_after_failure`` frees the allocator's cached blocks first (``oom_cleanup``),
so the memory the failed call held is returned and the next call can run.
"""

import re

import torch

CPU = "cpu"
"""The CPU device name."""
_DEVICE_NAME = re.compile(r"cpu|cuda(?::\d+)?")
_MIB = 1024 * 1024


def is_device_name(value: object) -> bool:
    """``value`` is ``"cpu"``, ``"cuda"`` or ``"cuda:<index>"``."""
    return isinstance(value, str) and _DEVICE_NAME.fullmatch(value) is not None


def is_cuda(device: str) -> bool:
    """The device is a CUDA GPU (``"cuda"`` or ``"cuda:<index>"``)."""
    return device == "cuda" or device.startswith("cuda:")


def model_dtype(device: str) -> torch.dtype:
    """The dtype a model runs in on ``device``: float16 on a GPU, float32 on the CPU."""
    return torch.float16 if is_cuda(device) else torch.float32


def place(model: torch.nn.Module, device: str, *, half: bool = True) -> None:
    """Move ``model`` (in place) to ``device`` in its dtype; on the CPU it stays as loaded.

    ``half=False`` keeps a model in float32 on the GPU too: the recogniser, whose fp16 spans
    differed from the CPU's at name boundaries on 6 of 1,030 corpus pages (BENCHMARKS 23).
    """
    if is_cuda(device):
        model.to(device=device, dtype=model_dtype(device) if half else torch.float32)


def gpu_visible() -> bool:
    """A CUDA GPU is visible to this process (``torch.cuda.is_available``)."""
    return bool(torch.cuda.is_available())


def gpu_description(index: int = 0) -> tuple[str, int]:
    """The name and the total memory (MiB) of GPU ``index``."""
    properties = torch.cuda.get_device_properties(index)
    return str(properties.name), int(properties.total_memory) // _MIB


def oom_cleanup() -> None:
    """Return the CUDA allocator's cached blocks to the device; nothing without a GPU (torch
    also skips it while CUDA is not initialised)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_after_failure(error: BaseException) -> None:
    """After a failed model call: ``oom_cleanup`` when ``error`` is a CUDA out-of-memory error."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        oom_cleanup()
