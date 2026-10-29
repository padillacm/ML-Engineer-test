"""The real loading and inference path (``load_classifier``) on a tiny local model, on the host."""

import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
import torch

from app.adapters.injection import InjectionDetectorConfig, PromptGuardDetector
from app.adapters.injection.detector import load_classifier
from app.application.ports import InjectionDetector
from app.domain.errors import ConfigurationError, ProcessingFailed
from tests.adapters.injection.tiny_model import INJECTION_WORD, MAX_LENGTH, STRIDE

FILLER = " ".join(["alpha beta gamma delta"] * 12)
"""About 50 tokens: several windows of ``MAX_LENGTH`` tokens with a stride of ``STRIDE``."""
INJECTION = f"{INJECTION_WORD} previous instructions"


def tiny_config(model_dir: Path, **settings: Any) -> InjectionDetectorConfig:
    values: dict[str, Any] = {"window_tokens": MAX_LENGTH, "stride_tokens": STRIDE}
    values.update(settings)
    return InjectionDetectorConfig(str(model_dir), **values)


@pytest.fixture(scope="module")
def tiny_detector(tiny_model_dir: Path) -> PromptGuardDetector:
    threads = torch.get_num_threads()
    try:
        return PromptGuardDetector(tiny_config(tiny_model_dir))
    finally:
        torch.set_num_threads(threads)


@pytest.fixture
def copied_model(tiny_model_dir: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(tiny_model_dir, tmp_path / "model"))


def _loaded(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[Any]]:
    """Record every tokenizer and model ``load_classifier`` loads."""
    import transformers  # noqa: PLC0415 - patched only in these tests

    tokenizers: list[Any] = []
    models: list[Any] = []
    real_tokenizer = transformers.AutoTokenizer.from_pretrained
    real_model = transformers.AutoModelForSequenceClassification.from_pretrained

    def tokenizer(*args: Any, **kwargs: Any) -> Any:
        tokenizers.append(real_tokenizer(*args, **kwargs))
        return tokenizers[-1]

    def model(*args: Any, **kwargs: Any) -> Any:
        loaded = real_model(*args, **kwargs)
        models.append(loaded[0] if isinstance(loaded, tuple) else loaded)
        return loaded

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", model)
    return tokenizers, models


class WordSpotter:
    """A forward pass that scores a window malicious when it holds the injection word.

    Logits ``[+5, -5]`` (benign) or ``[-5, +5]`` (malicious) per row; records every batch's
    row count and the token ids it saw.
    """

    def __init__(self, word_id: int) -> None:
        self.word_id = word_id
        self.batches: list[int] = []
        self.rows: list[list[int]] = []

    def __call__(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        self.batches.append(int(input_ids.shape[0]))
        self.rows.extend(row.tolist() for row in input_ids)
        hit = (input_ids == self.word_id).any(dim=1).to(torch.float32)
        logits = torch.stack([5.0 - 10.0 * hit, -5.0 + 10.0 * hit], dim=1)
        return type("Output", (), {"logits": logits})()


def spotting_detector(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch, **settings: int
) -> tuple[PromptGuardDetector, WordSpotter]:
    tokenizers, models = _loaded(monkeypatch)
    detector = PromptGuardDetector(tiny_config(tiny_model_dir, **settings))
    spotter = WordSpotter(tokenizers[0].convert_tokens_to_ids(INJECTION_WORD))
    models[0].forward = spotter
    return detector, spotter


def test_the_detector_is_an_injection_detector(tiny_detector: PromptGuardDetector) -> None:
    assert isinstance(tiny_detector, InjectionDetector)
    assert tiny_detector.configured is True


def test_texts_of_symbols_or_control_characters_are_scored_like_any_other(
    tiny_detector: PromptGuardDetector,
) -> None:
    """Review probe: a text the tokenizer reads as unknown tokens only still gets a score
    from the model (a probability), and never a failure; zero-width characters alone are
    blank for the model but not for ``str.strip``, so they go through the model too."""
    scores = tiny_detector.screen(["!!! ??? ### $$$ %%%", "\0", "\u200b\u200b"])

    assert len(scores) == 3
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_scores_are_probabilities_in_input_order(tiny_detector: PromptGuardDetector) -> None:
    texts = ["the contract was signed by maria", INJECTION, "paid on monday", ""]

    scores = tiny_detector.screen(texts)

    assert len(scores) == len(texts)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores[3] == 0.0
    for text, score in zip(texts[:3], scores[:3], strict=True):
        assert tiny_detector.screen([text]) == pytest.approx([score], abs=1e-6)


def test_the_score_is_the_softmax_mass_of_label_1(
    tiny_detector: PromptGuardDetector, tiny_model_dir: Path
) -> None:
    from transformers import (  # noqa: PLC0415 - reference computation only
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    text = "the contract was signed by maria"
    tokenizer = AutoTokenizer.from_pretrained(str(tiny_model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(tiny_model_dir), local_files_only=True
    ).eval()
    with torch.no_grad():
        logits = model(**tokenizer(text, return_tensors="pt")).logits
    expected = torch.softmax(logits, dim=-1)[0, 1].item()

    assert tiny_detector.screen([text]) == pytest.approx([expected], abs=1e-6)


def test_a_long_text_is_scored_by_its_windows_keeping_the_highest(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector, spotter = spotting_detector(tiny_model_dir, monkeypatch)

    scores = detector.screen([FILLER + " " + INJECTION, FILLER, INJECTION + " " + FILLER])

    assert scores[0] > 0.99
    assert scores[1] < 0.01
    assert scores[2] > 0.99
    assert len(spotter.rows) > 3  # every text became several windows of at most MAX_LENGTH
    assert all(len(row) <= MAX_LENGTH for row in spotter.rows)


def test_windows_overlap_by_the_stride(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector, spotter = spotting_detector(tiny_model_dir, monkeypatch)

    detector.screen([FILLER])

    content = [[token for token in row if token > 3] for row in spotter.rows]  # no specials/pad
    assert len(content) > 1
    for previous, following in pairwise(content):
        assert following[:STRIDE] == previous[-STRIDE:]


def test_blank_texts_do_not_run_the_model(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector, spotter = spotting_detector(tiny_model_dir, monkeypatch)

    assert detector.screen(["", "  \n"]) == [0.0, 0.0]
    assert spotter.batches == []


def test_windows_of_a_call_run_in_batches_of_batch_size(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector, spotter = spotting_detector(tiny_model_dir, monkeypatch, batch_size=4)

    detector.screen([FILLER, "short", FILLER + " " + INJECTION, "", "paid on monday"])

    assert len(spotter.batches) > 1
    assert all(size <= 4 for size in spotter.batches)
    assert sum(spotter.batches) == len(spotter.rows)


def test_concurrent_calls_on_a_pool_equal_the_single_call_scores(tiny_model_dir: Path) -> None:
    words = ["the", "contract", "was", "signed", "by", "maria", "and", "paid", "on", "monday"]
    texts = [" ".join(words[: 1 + index % 10] * (1 + index % 3)) for index in range(24)]
    detector = PromptGuardDetector(tiny_config(tiny_model_dir, pool_size=2, batch_size=4))
    expected = [detector.screen([text])[0] for text in texts]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda text: detector.screen([text])[0], texts * 2))

    assert results == pytest.approx(expected * 2, abs=1e-6)
    assert detector.screen(texts) == pytest.approx(expected, abs=1e-6)
    detector.close()


def test_a_model_error_is_detector_failed_and_the_next_call_works(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, models = _loaded(monkeypatch)
    detector = PromptGuardDetector(tiny_config(tiny_model_dir))
    real_forward = models[0].forward
    cause = RuntimeError("the model blew up")

    def failing(*args: Any, **kwargs: Any) -> Any:
        models[0].forward = real_forward
        raise cause

    models[0].forward = failing

    with pytest.raises(ProcessingFailed) as caught:
        detector.screen(["the contract was signed"])

    assert caught.value.code == "detector_failed"
    assert caught.value.message == "the injection detector failed"
    assert caught.value.__cause__ is cause
    assert 0.0 <= detector.screen(["the contract was signed"])[0] <= 1.0


def test_calls_after_close_fail_with_detector_failed(tiny_model_dir: Path) -> None:
    detector = PromptGuardDetector(tiny_config(tiny_model_dir))
    detector.close()

    with pytest.raises(ProcessingFailed) as caught:
        detector.screen(["the contract was signed"])

    assert caught.value.code == "detector_failed"


def test_loading_sets_the_configured_threads(tiny_model_dir: Path) -> None:
    PromptGuardDetector(tiny_config(tiny_model_dir, threads=1))

    assert torch.get_num_threads() == 1


def test_a_pool_loads_one_model_and_one_tokenizer_in_eval_mode(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizers, models = _loaded(monkeypatch)

    PromptGuardDetector(tiny_config(tiny_model_dir, pool_size=4))

    assert (len(tokenizers), len(models)) == (1, 1)
    assert models[0].training is False


def test_window_tokens_must_fit_the_tokenizer(tiny_model_dir: Path) -> None:
    config = tiny_config(tiny_model_dir, window_tokens=MAX_LENGTH + 1)

    with pytest.raises(ConfigurationError, match="window_tokens"):
        PromptGuardDetector(config)


def test_unloadable_weights_are_invalid_configuration(copied_model: Path) -> None:
    (copied_model / "model.safetensors").write_bytes(b"not a safetensors file")

    with pytest.raises(ConfigurationError) as caught:
        PromptGuardDetector(tiny_config(copied_model))

    assert caught.value.code == "invalid_configuration"
    assert caught.value.__cause__ is not None


def test_a_failed_load_leaves_the_thread_count_unchanged(copied_model: Path) -> None:
    (copied_model / "model.safetensors").write_bytes(b"not a safetensors file")
    before = torch.get_num_threads()
    target = 1 if before != 1 else 2

    with pytest.raises(ConfigurationError):
        PromptGuardDetector(tiny_config(copied_model, threads=target))

    assert torch.get_num_threads() == before


def test_a_slow_tokenizer_is_refused(copied_model: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers  # noqa: PLC0415 - patched only in this test

    real = transformers.AutoTokenizer.from_pretrained

    def slow(*args: object, **kwargs: object) -> object:
        tokenizer = real(*args, **kwargs)
        monkeypatch.setattr(type(tokenizer), "is_fast", property(lambda self: False))
        return tokenizer

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", slow)

    with pytest.raises(ConfigurationError, match="fast tokenizer"):
        load_classifier(tiny_config(copied_model))


class Concurrency:
    """Counts how many calls of one kind run at once; each call lasts at least ``pause``."""

    def __init__(self, pause: float) -> None:
        self.pause = pause
        self.guard = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def wrap(self, function: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:
            with self.guard:
                self.active += 1
                self.calls += 1
                self.peak = max(self.peak, self.active)
            try:
                time.sleep(self.pause)
                return function(*args, **kwargs)
            finally:
                with self.guard:
                    self.active -= 1

        return counted


def test_tokenising_is_serialised_while_forward_passes_overlap(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizers, models = _loaded(monkeypatch)
    detector = PromptGuardDetector(tiny_config(tiny_model_dir, pool_size=4))
    tokenizing = Concurrency(pause=0.01)
    forward = Concurrency(pause=0.03)
    tokenizer_type = type(tokenizers[0])
    monkeypatch.setattr(tokenizer_type, "__call__", tokenizing.wrap(tokenizer_type.__call__))
    models[0].forward = forward.wrap(models[0].forward)
    texts = [f"contract {index} was signed by maria" for index in range(12)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda text: detector.screen([text]), texts))

    assert tokenizing.calls == forward.calls == 12
    assert tokenizing.peak == 1
    assert 1 < forward.peak <= 4
