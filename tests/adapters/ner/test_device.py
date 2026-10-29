"""The recogniser's device setting and its GPU out-of-memory handling (spec: inference-device)."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from app.adapters.ner import TransformersNameRecognizer, TransformersNerConfig
from app.adapters.ner import recognizer as recognizer_module
from app.adapters.ner.recognizer import TokenClassifier, load_pipelines
from app.domain.errors import ConfigurationError, ProcessingFailed
from tests.adapters.oom import count_cleanups, out_of_memory

TEXT = "Maria Gonzalez signed."
PERSON = {"entity_group": "PER", "score": 0.99, "word": "", "start": 0, "end": 14}


def test_the_device_defaults_to_the_cpu(model_dir: Path) -> None:
    assert TransformersNerConfig(model_dir).device == "cpu"


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
def test_a_cuda_device_is_accepted(model_dir: Path, device: str) -> None:
    assert TransformersNerConfig(model_dir, device=device).device == device


@pytest.mark.parametrize("device", ["gpu", "CUDA", "", "cuda:"])
def test_an_unknown_device_is_invalid(model_dir: Path, device: str) -> None:
    with pytest.raises(ConfigurationError, match="device must be"):
        TransformersNerConfig(model_dir, device=device)


def test_out_of_memory_is_ner_failed_after_the_cleanup_and_the_next_call_works(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cleanups = count_cleanups(monkeypatch)
    failures = [out_of_memory()]

    def classify(text: str) -> Sequence[Mapping[str, object]]:
        if failures:
            raise failures.pop()
        return [PERSON]

    def factory(config: TransformersNerConfig) -> list[TokenClassifier]:
        return [classify] * config.pool_size

    recognizer = TransformersNameRecognizer(TransformersNerConfig(model_dir, pool_size=1), factory)

    with pytest.raises(ProcessingFailed) as caught:
        recognizer.find_persons(TEXT)

    assert caught.value.code == "ner_failed"
    assert "memory" not in str(caught.value)
    assert cleanups == [1]
    assert [(span.start, span.end) for span in recognizer.find_persons(TEXT)] == [(0, 14)]
    assert cleanups == [1]


def test_other_failures_do_not_clean_the_gpu(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cleanups = count_cleanups(monkeypatch)

    def classify(text: str) -> Sequence[Mapping[str, object]]:
        raise RuntimeError("vendor detail")

    recognizer = TransformersNameRecognizer(
        TransformersNerConfig(model_dir, pool_size=1), lambda config: [classify]
    )

    with pytest.raises(ProcessingFailed):
        recognizer.find_persons(TEXT)

    assert cleanups == []


def test_the_model_is_placed_in_full_precision(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    placed: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        recognizer_module,
        "place",
        lambda model, device, *, half=True: placed.append((device, half)),
    )

    load_pipelines(TransformersNerConfig(tiny_model_dir, pool_size=1, stride=4))

    assert placed == [("cpu", False)]
