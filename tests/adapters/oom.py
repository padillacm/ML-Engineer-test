"""A counter in place of ``app.adapters.device.oom_cleanup``, for the adapters' OOM tests."""

import pytest
import torch


def count_cleanups(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the GPU cleanup with a recorder; the list grows by one per cleanup."""
    calls: list[int] = []
    monkeypatch.setattr("app.adapters.device.oom_cleanup", lambda: calls.append(1))
    return calls


def out_of_memory() -> torch.cuda.OutOfMemoryError:
    """The error torch raises when a CUDA allocation fails (constructible without a GPU)."""
    return torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
