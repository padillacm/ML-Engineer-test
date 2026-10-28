"""``TransformersNameRecognizer`` with the ``transformers`` pipeline replaced by a stub."""

import gc
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
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

    def factory(self, config: TransformersNerConfig) -> list[TokenClassifier]:
        self.configs.append(config)
        return [self.classify] * config.pool_size

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
    return TransformersNerConfig(model_dir, pool_size=1)


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

    def broken(config: TransformersNerConfig) -> list[TokenClassifier]:
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
    def broken(config: TransformersNerConfig) -> list[TokenClassifier]:
        raise OSError("weights corrupt: secret detail")

    with pytest.raises(ConfigurationError) as caught:
        TransformersNameRecognizer(config, broken)

    assert caught.value.code == "invalid_configuration"
    assert "secret detail" not in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)


def test_service_errors_from_loading_pass_through(config: TransformersNerConfig) -> None:
    original = ConfigurationError("NER model needs a fast tokenizer")

    def refuses(config: TransformersNerConfig) -> list[TokenClassifier]:
        raise original

    with pytest.raises(ConfigurationError) as caught:
        TransformersNameRecognizer(config, refuses)

    assert caught.value is original


def test_service_errors_from_inference_pass_through(config: TransformersNerConfig) -> None:
    failure = ProcessingFailed("cannot read", "ocr_failed")

    def failing(text: str) -> Sequence[Mapping[str, object]]:
        raise failure

    recognizer = TransformersNameRecognizer(config, lambda config: [failing])

    with pytest.raises(ProcessingFailed) as caught:
        recognizer.find_persons(TEXT)

    assert caught.value is failure


def test_model_failures_become_processing_failed(config: TransformersNerConfig) -> None:
    def crashing(text: str) -> Sequence[Mapping[str, object]]:
        raise RuntimeError(f"index out of range while reading {text}")

    recognizer = TransformersNameRecognizer(config, lambda config: [crashing])

    with pytest.raises(ProcessingFailed) as caught:
        recognizer.find_persons(TEXT)

    assert caught.value.code == "ner_failed"
    assert caught.value.message == "NER model could not process the text"
    assert TEXT not in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_the_pipeline_is_given_back_after_a_failure(config: TransformersNerConfig) -> None:
    calls: list[str] = []

    def flaky(text: str) -> Sequence[Mapping[str, object]]:
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("first call fails")
        return [entity("PER", 0, 14)]

    recognizer = TransformersNameRecognizer(config, lambda config: [flaky])

    with pytest.raises(ProcessingFailed):
        recognizer.find_persons(TEXT)
    assert recognizer.find_persons(TEXT) == [TextSpan(0, 14)]


class GatedPipeline:
    """One pool slot: blocks every call until ``release`` is set and records concurrency."""

    def __init__(self, pool: "GatedPool", index: int) -> None:
        self.pool = pool
        self.index = index
        self.busy = False
        self.calls = 0

    def __call__(self, text: str) -> Sequence[Mapping[str, object]]:
        with self.pool.guard:
            assert not self.busy, "one pipeline ran two calls at once"
            self.busy = True
            self.calls += 1
            self.pool.active += 1
            self.pool.peak = max(self.pool.peak, self.pool.active)
        try:
            assert self.pool.release.wait(5), "the test never released the calls"
            return [entity("PER", 0, 14)]
        finally:
            with self.pool.guard:
                self.busy = False
                self.pool.active -= 1


class GatedPool:
    def __init__(self) -> None:
        self.guard = threading.Lock()
        self.release = threading.Event()
        self.active = 0
        self.peak = 0
        self.pipelines: list[GatedPipeline] = []

    def factory(self, config: TransformersNerConfig) -> list[TokenClassifier]:
        self.pipelines = [GatedPipeline(self, index) for index in range(config.pool_size)]
        return list(self.pipelines)

    def wait_for_active(self, count: int) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with self.guard:
                if self.active == count:
                    return
            time.sleep(0.005)
        raise AssertionError(f"{count} calls never ran at once (active={self.active})")


def run_concurrently(calls: int, target: Callable[[], object]) -> list[threading.Thread]:
    threads = [threading.Thread(target=target) for _ in range(calls)]
    for thread in threads:
        thread.start()
    return threads


def join_all(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()


def test_a_pool_of_one_serialises_inference(model_dir: Path) -> None:
    pool = GatedPool()
    recognizer = TransformersNameRecognizer(
        TransformersNerConfig(model_dir, pool_size=1), pool.factory
    )
    results: list[list[TextSpan]] = []

    threads = run_concurrently(6, lambda: results.append(recognizer.find_persons(TEXT)))
    pool.wait_for_active(1)
    time.sleep(0.05)
    assert pool.active == 1
    pool.release.set()
    join_all(threads)

    assert pool.peak == 1
    assert results == [[TextSpan(0, 14)]] * 6


def test_at_most_pool_size_calls_run_at_once(model_dir: Path) -> None:
    pool = GatedPool()
    config = TransformersNerConfig(model_dir, pool_size=3)
    recognizer = TransformersNameRecognizer(config, pool.factory)
    results: list[list[TextSpan]] = []

    threads = run_concurrently(8, lambda: results.append(recognizer.find_persons(TEXT)))
    pool.wait_for_active(3)
    time.sleep(0.05)  # a fourth call, were it admitted, would start now
    assert pool.active == 3
    assert all(pipeline.busy for pipeline in pool.pipelines)
    pool.release.set()
    join_all(threads)

    assert pool.peak == 3
    assert results == [[TextSpan(0, 14)]] * 8
    assert sum(pipeline.calls for pipeline in pool.pipelines) == 8


def test_a_free_pipeline_is_reused(model_dir: Path) -> None:
    pool = GatedPool()
    pool.release.set()
    config = TransformersNerConfig(model_dir, pool_size=2)
    recognizer = TransformersNameRecognizer(config, pool.factory)

    for _ in range(5):
        assert recognizer.find_persons(TEXT) == [TextSpan(0, 14)]

    assert sorted(pipeline.calls for pipeline in pool.pipelines) == [0, 5]


def test_every_failure_gives_the_pipeline_back(model_dir: Path) -> None:
    failures: list[Exception] = [
        RuntimeError("vendor failure"),
        ProcessingFailed("cannot read", "ocr_failed"),
    ]

    def failing(text: str) -> Sequence[Mapping[str, object]]:
        if failures:
            raise failures.pop(0)
        return [entity("PER", 0, 14)]

    recognizer = TransformersNameRecognizer(
        TransformersNerConfig(model_dir, pool_size=1), lambda config: [failing]
    )

    with pytest.raises(ProcessingFailed, match="NER model could not process the text"):
        recognizer.find_persons(TEXT)
    with pytest.raises(ProcessingFailed, match="cannot read"):
        recognizer.find_persons(TEXT)
    assert recognizer.find_persons(TEXT) == [TextSpan(0, 14)]


@pytest.mark.parametrize("built", [0, 1, 3])
def test_the_factory_must_build_pool_size_pipelines(model_dir: Path, built: int) -> None:
    pipeline = ScriptedPipeline()

    with pytest.raises(ConfigurationError, match="pool_size=2"):
        TransformersNameRecognizer(
            TransformersNerConfig(model_dir, pool_size=2),
            lambda config: [pipeline.classify] * built,
        )


def test_a_closed_recognizer_refuses_calls(config: TransformersNerConfig) -> None:
    recognizer, pipeline = recognizer_for(config, entity("PER", 0, 14))

    recognizer.close()
    recognizer.close()  # idempotent

    with pytest.raises(ProcessingFailed) as caught:
        recognizer.find_persons(TEXT)
    assert caught.value.code == "ner_failed"
    assert caught.value.message == "NER model is closed"
    assert pipeline.calls == []


def test_close_drops_every_pipeline(model_dir: Path) -> None:
    pool = GatedPool()
    pool.release.set()
    recognizer = TransformersNameRecognizer(
        TransformersNerConfig(model_dir, pool_size=3), pool.factory
    )
    references = [weakref.ref(pipeline) for pipeline in pool.pipelines]
    recognizer.find_persons(TEXT)

    recognizer.close()
    pool.pipelines.clear()
    gc.collect()

    assert [reference() for reference in references] == [None, None, None]


def test_close_lets_running_calls_finish_and_wakes_waiting_ones(model_dir: Path) -> None:
    pool = GatedPool()
    recognizer = TransformersNameRecognizer(
        TransformersNerConfig(model_dir, pool_size=1), pool.factory
    )
    outcomes: list[object] = []

    def call() -> None:
        try:
            outcomes.append(recognizer.find_persons(TEXT))
        except ProcessingFailed as error:
            outcomes.append(error.message)

    threads = run_concurrently(3, call)
    pool.wait_for_active(1)
    time.sleep(0.05)  # the two other calls are now waiting for the only pipeline
    recognizer.close()
    deadline = time.monotonic() + 5
    while len(outcomes) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert outcomes == ["NER model is closed"] * 2
    pool.release.set()
    join_all(threads)

    assert outcomes[2] == [TextSpan(0, 14)]
    with pytest.raises(ProcessingFailed, match="closed"):
        recognizer.find_persons(TEXT)
