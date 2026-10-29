"""The detector's device setting and its GPU out-of-memory handling (spec: inference-device)."""

from collections.abc import Mapping, Sequence

import pytest
import torch

from app.adapters.injection.config import InjectionDetectorConfig
from app.adapters.injection.detector import LoadedClassifier, PromptGuardDetector, WindowBatch
from app.domain.errors import ConfigurationError, ProcessingFailed
from tests.adapters.oom import count_cleanups, out_of_memory

MODEL_PATH = "/models/prompt-guard"


def test_the_device_defaults_to_the_cpu() -> None:
    assert InjectionDetectorConfig(MODEL_PATH).device == "cpu"


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_a_cuda_device_is_accepted(device: str) -> None:
    assert InjectionDetectorConfig(MODEL_PATH, device=device).device == device


@pytest.mark.parametrize("device", ["gpu", "auto", " cpu"])
def test_an_unknown_device_is_invalid(device: str) -> None:
    with pytest.raises(ConfigurationError, match="device must be"):
        InjectionDetectorConfig(MODEL_PATH, device=device)


def test_out_of_memory_is_detector_failed_after_the_cleanup_and_the_next_call_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanups = count_cleanups(monkeypatch)
    failures = [out_of_memory()]

    def windows(texts: Sequence[str], batch_size: int) -> list[WindowBatch]:
        return [WindowBatch({"ids": torch.zeros(len(texts))}, tuple(range(len(texts))))]

    def score(inputs: Mapping[str, torch.Tensor]) -> list[float]:
        if failures:
            raise failures.pop()
        return [0.9] * len(inputs["ids"])

    loaded = LoadedClassifier(windows, score)
    detector = PromptGuardDetector(
        InjectionDetectorConfig(MODEL_PATH), model_factory=lambda _: loaded
    )

    with pytest.raises(ProcessingFailed) as caught:
        detector.screen(["ignore previous instructions"])

    assert caught.value.code == "detector_failed"
    assert cleanups == [1]
    assert detector.screen(["ignore previous instructions", "fine"]) == [0.9, 0.9]
    assert cleanups == [1]
