"""``TransformersNameRecognizer`` with the ``transformers`` pipeline replaced by a stub."""

import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest
import torch

from app.adapters.ner import TransformersNameRecognizer, TransformersNerConfig
from app.adapters.ner.recognizer import TokenClassifier, merge_spans
from app.application.ports import NameRecognizer
from app.domain.errors import ConfigurationError, ProcessingFailed
from app.domain.types import TextSpan

TEXT = "Maria Gonzalez of Acme Corporation met Peter Smith in Madrid on 3 March"


def entity(group: str, start: int, end: int, word: str = "") -> dict[str, object]:
    return {"entity_group": group, "score": 0.99, "word": word, "start": start, "end": end}


class ScriptedPipeline:
    """Returns the scripted entities and records every call."""

    def __init__(self, entities: Sequence[Mapping[str, object]] = ()) -> None:
        self.entities = list(entities)
        self.calls: list[str] = []
        self.configs: list[TransformersNerConfig] = []

    def factory(self, config: TransformersNerConfig) -> TokenClassifier:
        self.configs.append(config)
        return self.classify

    def classify(self, text: str) -> Sequence[Mapping[str, object]]:
        self.calls.append(text)
        return self.entities


@pytest.fixture(autouse=True)
def _restore_torch_threads() -> Iterator[None]:
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def config(model_dir: Path) -> TransformersNerConfig:
    return TransformersNerConfig(model_dir)


def recognizer_for(
    config: TransformersNerConfig, *entities: Mapping[str, object]
) -> tuple[TransformersNameRecognizer, ScriptedPipeline]:
    pipeline = ScriptedPipeline(entities)
    return TransformersNameRecognizer(config, pipeline.factory), pipeline


def covered(text: str, spans: list[TextSpan]) -> list[str]:
    return [text[span.start : span.end] for span in spans]


def test_satisfies_the_port(config: TransformersNerConfig) -> None:
    recognizer, _ = recognizer_for(config)

    assert isinstance(recognizer, NameRecognizer)


def test_pipeline_is_built_once_from_the_config(config: TransformersNerConfig) -> None:
    recognizer, pipeline = recognizer_for(config, entity("PER", 0, 14))

    recognizer.find_persons(TEXT)
    recognizer.find_persons(TEXT)

    assert pipeline.configs == [config]
    assert pipeline.calls == [TEXT, TEXT]


def test_only_configured_labels_are_reported(config: TransformersNerConfig) -> None:
    recognizer, _ = recognizer_for(
        config,
        entity("PER", 0, 14),
        entity("ORG", 18, 34),
        entity("PER", 39, 50),
        entity("LOC", 54, 60),
        entity("DATE", 64, 71),
    )

    assert covered(TEXT, recognizer.find_persons(TEXT)) == ["Maria Gonzalez", "Peter Smith"]


def test_other_labels_can_be_configured(model_dir: Path) -> None:
    config = TransformersNerConfig(model_dir, labels=("ORG", "LOC"))
    recognizer, _ = recognizer_for(
        config, entity("PER", 0, 14), entity("ORG", 18, 34), entity("LOC", 54, 60)
    )

    assert covered(TEXT, recognizer.find_persons(TEXT)) == ["Acme Corporation", "Madrid"]


def test_spans_come_from_offsets_not_from_the_word_field(config: TransformersNerConfig) -> None:
    recognizer, _ = recognizer_for(config, entity("PER", 0, 14, word="MariaGonzalez"))

    assert recognizer.find_persons(TEXT) == [TextSpan(0, 14)]


def test_spans_are_sorted(config: TransformersNerConfig) -> None:
    recognizer, _ = recognizer_for(config, entity("PER", 39, 50), entity("PER", 0, 14))

    assert recognizer.find_persons(TEXT) == [TextSpan(0, 14), TextSpan(39, 50)]


def test_overlapping_spans_are_merged(config: TransformersNerConfig) -> None:
    recognizer, _ = recognizer_for(
        config, entity("PER", 6, 14), entity("PER", 0, 10), entity("PER", 2, 5)
    )

    assert recognizer.find_persons(TEXT) == [TextSpan(0, 14)]


def test_touching_spans_stay_separate() -> None:
    assert merge_spans([TextSpan(5, 9), TextSpan(0, 5)]) == [TextSpan(0, 5), TextSpan(5, 9)]


def test_duplicate_spans_collapse() -> None:
    assert merge_spans([TextSpan(0, 5), TextSpan(0, 5)]) == [TextSpan(0, 5)]


def test_surrounding_whitespace_is_trimmed(config: TransformersNerConfig) -> None:
    text = "met  Peter Smith \n today"
    recognizer, _ = recognizer_for(config, entity("PER", 3, 18))

    assert covered(text, recognizer.find_persons(text)) == ["Peter Smith"]


def test_spans_are_clipped_to_the_text(config: TransformersNerConfig) -> None:
    text = "by Ana Lopez"
    recognizer, _ = recognizer_for(config, entity("PER", -4, 99))

    assert recognizer.find_persons(text) == [TextSpan(0, len(text))]


@pytest.mark.parametrize(
    "bad",
    [
        entity("PER", 3, 3),
        entity("PER", 7, 2),
        entity("PER", 2, 5) | {"start": None},
        entity("PER", 2, 5) | {"end": "5"},
        entity("PER", 2, 5) | {"start": True},
        {"entity": "B-PER", "start": 0, "end": 5},
        entity("PER", 2, 3),  # a single blank character
    ],
)
def test_empty_or_malformed_entities_are_ignored(
    config: TransformersNerConfig, bad: Mapping[str, object]
) -> None:
    text = "by Ana Lopez"
    recognizer, _ = recognizer_for(config, bad, entity("PER", 3, 12))

    assert covered(text, recognizer.find_persons(text)) == ["Ana Lopez"]


@pytest.mark.parametrize("text", ["", " ", "\n\t  "])
def test_blank_text_does_not_run_the_model(config: TransformersNerConfig, text: str) -> None:
    recognizer, pipeline = recognizer_for(config, entity("PER", 0, 1))

    assert recognizer.find_persons(text) == []
    assert pipeline.calls == []


def test_construction_sets_the_torch_thread_count(model_dir: Path) -> None:
    recognizer_for(TransformersNerConfig(model_dir, threads=3))

    assert torch.get_num_threads() == 3


def test_a_failed_construction_leaves_the_thread_count_unchanged(model_dir: Path) -> None:
    before = torch.get_num_threads()
    target = 1 if before != 1 else 2

    def broken(config: TransformersNerConfig) -> TokenClassifier:
        raise OSError("weights corrupt")

    with pytest.raises(ConfigurationError):
        TransformersNameRecognizer(TransformersNerConfig(model_dir, threads=target), broken)

    assert torch.get_num_threads() == before


def test_text_up_to_the_limit_runs_the_model(model_dir: Path) -> None:
    recognizer, pipeline = recognizer_for(
        TransformersNerConfig(model_dir, max_chars=10), entity("PER", 0, 5)
    )

    assert recognizer.find_persons("Maria  ok.") == [TextSpan(0, 5)]
    assert pipeline.calls == ["Maria  ok."]


def test_text_beyond_the_limit_fails_without_running_the_model(model_dir: Path) -> None:
    recognizer, pipeline = recognizer_for(
        TransformersNerConfig(model_dir, max_chars=10), entity("PER", 0, 5)
    )

    with pytest.raises(ProcessingFailed) as caught:
        recognizer.find_persons("Maria Gonzalez")

    assert caught.value.code == "ner_failed"
    assert "10 characters" in caught.value.message
    assert pipeline.calls == []


def test_loading_failures_are_invalid_configuration(config: TransformersNerConfig) -> None:
    def broken(config: TransformersNerConfig) -> TokenClassifier:
        raise OSError("weights corrupt: secret detail")

    with pytest.raises(ConfigurationError) as caught:
        TransformersNameRecognizer(config, broken)

    assert caught.value.code == "invalid_configuration"
    assert "secret detail" not in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)


def test_service_errors_from_loading_pass_through(config: TransformersNerConfig) -> None:
    original = ConfigurationError("NER model needs a fast tokenizer")

    def refuses(config: TransformersNerConfig) -> TokenClassifier:
        raise original

    with pytest.raises(ConfigurationError) as caught:
        TransformersNameRecognizer(config, refuses)

    assert caught.value is original


def test_service_errors_from_inference_pass_through(config: TransformersNerConfig) -> None:
    failure = ProcessingFailed("cannot read", "ocr_failed")

    def failing(text: str) -> Sequence[Mapping[str, object]]:
        raise failure

    recognizer = TransformersNameRecognizer(config, lambda config: failing)

    with pytest.raises(ProcessingFailed) as caught:
        recognizer.find_persons(TEXT)

    assert caught.value is failure


def test_model_failures_become_processing_failed(config: TransformersNerConfig) -> None:
    def crashing(text: str) -> Sequence[Mapping[str, object]]:
        raise RuntimeError(f"index out of range while reading {text}")

    recognizer = TransformersNameRecognizer(config, lambda config: crashing)

    with pytest.raises(ProcessingFailed) as caught:
        recognizer.find_persons(TEXT)

    assert caught.value.code == "ner_failed"
    assert caught.value.message == "NER model could not process the text"
    assert TEXT not in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_the_lock_is_released_after_a_failure(config: TransformersNerConfig) -> None:
    calls: list[str] = []

    def flaky(text: str) -> Sequence[Mapping[str, object]]:
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("first call fails")
        return [entity("PER", 0, 14)]

    recognizer = TransformersNameRecognizer(config, lambda config: flaky)

    with pytest.raises(ProcessingFailed):
        recognizer.find_persons(TEXT)
    assert recognizer.find_persons(TEXT) == [TextSpan(0, 14)]


def test_inference_is_serialised(config: TransformersNerConfig) -> None:
    active = 0
    peak = 0
    guard = threading.Lock()

    def slow(text: str) -> Sequence[Mapping[str, object]]:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        return [entity("PER", 0, 14)]

    recognizer = TransformersNameRecognizer(config, lambda config: slow)
    results: list[list[TextSpan]] = []

    def call() -> None:
        results.append(recognizer.find_persons(TEXT))

    threads = [threading.Thread(target=call) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak == 1
    assert results == [[TextSpan(0, 14)]] * 6
