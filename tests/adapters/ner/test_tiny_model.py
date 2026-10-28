"""The real loading and inference path (``load_pipelines``) on a tiny local model, on the host."""

import json
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from app.adapters.ner import TransformersNameRecognizer, TransformersNerConfig
from app.adapters.ner.recognizer import load_pipelines
from app.application.ports import NameRecognizer
from app.domain.errors import ConfigurationError
from tests.adapters.ner.tiny_model import MAX_LENGTH
from tests.application.contracts import NameRecognizerContract

SENTENCE = "Maria Gonzalez of Acme Corporation met in Madrid on 3 March. "


@pytest.fixture(autouse=True)
def _restore_torch_threads() -> Iterator[None]:
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def tiny_recognizer(tiny_model_dir: Path) -> TransformersNameRecognizer:
    threads = torch.get_num_threads()
    try:
        return TransformersNameRecognizer(TransformersNerConfig(tiny_model_dir, stride=4))
    finally:
        torch.set_num_threads(threads)


@pytest.fixture
def copied_model(tiny_model_dir: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(tiny_model_dir, tmp_path / "model"))


def covered(recognizer: NameRecognizer, text: str) -> list[str]:
    return [text[span.start : span.end] for span in recognizer.find_persons(text)]


class TestTinyModelContract(NameRecognizerContract):
    @pytest.fixture
    def recognizer(self, tiny_recognizer: TransformersNameRecognizer) -> NameRecognizer:
        return tiny_recognizer


def test_only_person_entities_are_reported(tiny_recognizer: TransformersNameRecognizer) -> None:
    assert covered(tiny_recognizer, SENTENCE) == ["Maria Gonzalez"]


def test_names_beyond_the_first_window_are_found(
    tiny_recognizer: TransformersNameRecognizer,
) -> None:
    text = SENTENCE * 6 + "The agreement was signed by Peter Smith."

    names = covered(tiny_recognizer, text)

    assert len(text.split()) > 2 * MAX_LENGTH
    assert names == ["Maria Gonzalez"] * 6 + ["Peter Smith"]


def test_loading_sets_the_configured_threads(tiny_model_dir: Path) -> None:
    TransformersNameRecognizer(TransformersNerConfig(tiny_model_dir, stride=4, threads=1))

    assert torch.get_num_threads() == 1


def test_stride_must_fit_the_loaded_tokenizer(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Skip the configuration's early check so the length of the loaded tokenizer decides.
    monkeypatch.setattr("app.adapters.ner.config._declared_max_length", lambda path: None)
    config = TransformersNerConfig(tiny_model_dir, stride=MAX_LENGTH)

    with pytest.raises(ConfigurationError, match="stride"):
        TransformersNameRecognizer(config)


def test_labels_unknown_to_the_model_are_invalid(tiny_model_dir: Path) -> None:
    config = TransformersNerConfig(tiny_model_dir, stride=4, labels=("PER", "MISC"))

    with pytest.raises(ConfigurationError, match="MISC"):
        TransformersNameRecognizer(config)


def test_unloadable_weights_are_invalid_configuration(copied_model: Path) -> None:
    (copied_model / "model.safetensors").write_bytes(b"not a safetensors file")

    with pytest.raises(ConfigurationError) as caught:
        TransformersNameRecognizer(TransformersNerConfig(copied_model, stride=4))

    assert caught.value.code == "invalid_configuration"
    assert caught.value.__cause__ is not None


def test_a_slow_tokenizer_is_refused(copied_model: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers  # noqa: PLC0415 - patched only in this test

    real = transformers.AutoTokenizer.from_pretrained

    def slow(*args: object, **kwargs: object) -> object:
        tokenizer = real(*args, **kwargs)
        monkeypatch.setattr(type(tokenizer), "is_fast", property(lambda self: False))
        return tokenizer

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", slow)

    with pytest.raises(ConfigurationError, match="fast tokenizer"):
        load_pipelines(TransformersNerConfig(copied_model, stride=4))


def test_weights_of_another_architecture_are_refused(copied_model: Path) -> None:
    # The BERT weights under a RoBERTa configuration: every tensor name differs, so transformers
    # would build a randomly initialised RoBERTa and only log a warning.
    config_file = copied_model / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config.update(model_type="roberta", architectures=["RobertaForTokenClassification"])
    config_file.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="do not match") as caught:
        TransformersNameRecognizer(TransformersNerConfig(copied_model, stride=4))

    assert caught.value.code == "invalid_configuration"


def test_truncated_weights_are_refused(copied_model: Path) -> None:
    weights = copied_model / "model.safetensors"
    tensors = load_file(weights)
    del tensors["classifier.weight"]
    save_file(tensors, weights, metadata={"format": "pt"})

    with pytest.raises(ConfigurationError, match="missing_keys"):
        TransformersNameRecognizer(TransformersNerConfig(copied_model, stride=4))


def test_a_failed_load_leaves_the_thread_count_unchanged(copied_model: Path) -> None:
    (copied_model / "model.safetensors").write_bytes(b"not a safetensors file")
    before = torch.get_num_threads()
    target = 1 if before != 1 else 2

    with pytest.raises(ConfigurationError):
        TransformersNameRecognizer(TransformersNerConfig(copied_model, stride=4, threads=target))

    assert torch.get_num_threads() == before


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


def test_a_pool_shares_one_model_and_one_tokenizer(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers  # noqa: PLC0415 - patched only in this test

    real_pipeline = transformers.pipeline
    real_model = transformers.AutoModelForTokenClassification.from_pretrained
    built: list[tuple[object, object]] = []
    loads: list[object] = []

    def recording_pipeline(*args: object, **kwargs: object) -> object:
        built.append((kwargs["model"], kwargs["tokenizer"]))
        return real_pipeline(*args, **kwargs)  # type: ignore[call-overload]

    def counting_model(*args: Any, **kwargs: Any) -> object:
        loads.append(args)
        return real_model(*args, **kwargs)

    monkeypatch.setattr(transformers, "pipeline", recording_pipeline)
    monkeypatch.setattr(
        transformers.AutoModelForTokenClassification, "from_pretrained", counting_model
    )

    classifiers = load_pipelines(TransformersNerConfig(tiny_model_dir, stride=4, pool_size=3))

    assert len(classifiers) == 3
    assert len(loads) == 1
    assert len(built) == 3
    assert len({id(model) for model, _ in built}) == 1
    assert len({id(tokenizer) for _, tokenizer in built}) == 1


def test_tokenizer_steps_are_serialised_while_forward_passes_overlap(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers  # noqa: PLC0415 - patched only in this test

    real_pipeline = transformers.pipeline
    built: list[Any] = []

    def recording_pipeline(*args: object, **kwargs: object) -> object:
        built.append(kwargs)
        return real_pipeline(*args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(transformers, "pipeline", recording_pipeline)
    classifiers = load_pipelines(TransformersNerConfig(tiny_model_dir, stride=4, pool_size=4))
    model, tokenizer = built[0]["model"], built[0]["tokenizer"]
    tokenizing = Concurrency(pause=0.01)
    grouping = Concurrency(pause=0.002)
    forward = Concurrency(pause=0.03)
    monkeypatch.setattr(type(tokenizer), "__call__", tokenizing.wrap(type(tokenizer).__call__))
    monkeypatch.setattr(
        type(tokenizer),
        "convert_tokens_to_string",
        grouping.wrap(type(tokenizer).convert_tokens_to_string),
    )
    model.forward = forward.wrap(model.forward)
    text = SENTENCE * 2

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda index: classifiers[index % 4](text), range(12)))

    assert all(result == results[0] for result in results)
    assert tokenizing.calls == 12
    assert grouping.calls >= 12
    assert (tokenizing.peak, grouping.peak) == (1, 1)
    assert forward.peak > 1


def test_concurrent_calls_on_a_pool_match_a_pool_of_one(tiny_model_dir: Path) -> None:
    texts = [SENTENCE * repeat + "Signed by Peter Smith." for repeat in range(1, 9)] * 4
    single = TransformersNameRecognizer(TransformersNerConfig(tiny_model_dir, stride=4))
    pooled = TransformersNameRecognizer(
        TransformersNerConfig(tiny_model_dir, stride=4, pool_size=4)
    )
    expected = [single.find_persons(text) for text in texts]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(pooled.find_persons, texts))

    assert results == expected
    pooled.close()
    single.close()
