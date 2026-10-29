"""The shared device helpers of the torch adapters (spec: inference-device)."""

import pytest
import torch

from app.adapters import device
from app.adapters.device import (
    is_cuda,
    is_device_name,
    model_dtype,
    oom_cleanup,
    place,
    release_after_failure,
)


@pytest.mark.parametrize("name", ["cpu", "cuda", "cuda:0", "cuda:12"])
def test_valid_device_names(name: str) -> None:
    assert is_device_name(name)


@pytest.mark.parametrize("name", ["", "CPU", "gpu", "cuda:", "cuda:x", "cuda:0 ", "mps", 0, None])
def test_invalid_device_names(name: object) -> None:
    assert not is_device_name(name)


def test_cuda_devices_are_told_from_the_cpu() -> None:
    assert is_cuda("cuda")
    assert is_cuda("cuda:1")
    assert not is_cuda("cpu")


def test_the_gpu_runs_in_half_precision_and_the_cpu_in_full() -> None:
    assert model_dtype("cuda:0") == torch.float16
    assert model_dtype("cpu") == torch.float32


def test_placing_on_the_cpu_leaves_the_model_as_loaded() -> None:
    model = torch.nn.Linear(2, 2).to(torch.float64)
    weight = model.weight

    place(model, "cpu")

    assert model.weight is weight
    assert model.weight.dtype == torch.float64


def test_placing_on_a_gpu_moves_the_model_in_half_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 2)
    calls: list[dict[str, object]] = []

    def to(**kwargs: object) -> torch.nn.Module:
        calls.append(kwargs)
        return model

    monkeypatch.setattr(model, "to", to)

    place(model, "cuda:0")

    assert calls == [{"device": "cuda:0", "dtype": torch.float16}]


def test_a_model_can_stay_in_full_precision_on_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(2, 2)
    calls: list[dict[str, object]] = []

    def to(**kwargs: object) -> torch.nn.Module:
        calls.append(kwargs)
        return model

    monkeypatch.setattr(model, "to", to)

    place(model, "cuda:0", half=False)

    assert calls == [{"device": "cuda:0", "dtype": torch.float32}]


def _counting(monkeypatch: pytest.MonkeyPatch, target: str) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr(target, lambda: calls.append(1))
    return calls


def test_cleanup_empties_the_cache_when_cuda_is_in_use(monkeypatch: pytest.MonkeyPatch) -> None:
    emptied = _counting(monkeypatch, "torch.cuda.empty_cache")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    oom_cleanup()

    assert emptied == [1]


def test_cleanup_does_nothing_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    emptied = _counting(monkeypatch, "torch.cuda.empty_cache")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    oom_cleanup()

    assert emptied == []


@pytest.mark.parametrize(
    ("error", "cleaned"),
    [
        (torch.cuda.OutOfMemoryError("CUDA out of memory"), 1),
        (RuntimeError("other"), 0),
        (ValueError("other"), 0),
    ],
)
def test_only_an_out_of_memory_error_triggers_the_cleanup(
    monkeypatch: pytest.MonkeyPatch, error: Exception, cleaned: int
) -> None:
    calls = _counting(monkeypatch, "app.adapters.device.oom_cleanup")

    release_after_failure(error)

    assert len(calls) == cleaned


def test_the_gpu_probe_reads_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    class Properties:
        name = "NVIDIA GeForce RTX 4070 Ti"
        total_memory = 12 * 1024**3 + 5

    seen: list[int] = []

    def properties(index: int) -> Properties:
        seen.append(index)
        return Properties()

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_properties", properties)

    assert device.gpu_visible() is True
    assert device.gpu_description(0) == ("NVIDIA GeForce RTX 4070 Ti", 12 * 1024)
    assert seen == [0]
