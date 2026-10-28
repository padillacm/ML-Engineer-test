"""The real loading and inference path (``load_pipeline``) on a tiny local model, on the host."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from app.adapters.ner import TransformersNameRecognizer, TransformersNerConfig
from app.adapters.ner.recognizer import load_pipeline
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
        load_pipeline(TransformersNerConfig(copied_model, stride=4))


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
