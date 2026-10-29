"""``PromptGuardDetector`` with a stubbed classifier: windows, batches, errors, pool, labels."""

import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pytest
import torch

from app.adapters.injection import InjectionDetectorConfig, PromptGuardDetector
from app.adapters.injection.detector import (
    PAGE_LANE,
    QUERY_LANE,
    LoadedClassifier,
    WindowBatch,
    positive_label_indexes,
)
from app.application.ports import InjectionDetector
from app.domain.errors import ConfigurationError, ProcessingFailed

MODEL_PATH = "/models/prompt-guard"


class ScriptedClassifier:
    """A text ``"0.2|0.9"`` is two windows scoring 0.2 and 0.9; records every batch scored."""

    def __init__(self, *, pause: float = 0.0) -> None:
        self.batches: list[tuple[int, ...]] = []
        self.fail_with: Exception | None = None
        self.pause = pause
        self.guard = threading.Lock()
        self.active = 0
        self.peak = 0

    def loaded(self, config: InjectionDetectorConfig) -> LoadedClassifier:
        del config
        return LoadedClassifier(windows=self.windows, score=self.score)

    def windows(self, texts: Sequence[str], batch_size: int) -> list[WindowBatch]:
        rows = [
            (owner, float(part)) for owner, text in enumerate(texts) for part in text.split("|")
        ]
        return [
            WindowBatch(
                {
                    "score": torch.tensor(
                        [score for _, score in rows[start : start + batch_size]],
                        dtype=torch.float64,
                    )
                },
                tuple(owner for owner, _ in rows[start : start + batch_size]),
            )
            for start in range(0, len(rows), batch_size)
        ]

    def score(self, inputs: Mapping[str, torch.Tensor]) -> list[float]:
        with self.guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.pause)
            self.batches.append(tuple(round(float(value) * 100) for value in inputs["score"]))
            if self.fail_with is not None:
                error, self.fail_with = self.fail_with, None
                raise error
            result: list[float] = inputs["score"].tolist()
            return result
        finally:
            with self.guard:
                self.active -= 1


def make_detector(classifier: ScriptedClassifier, **settings: Any) -> PromptGuardDetector:
    config = InjectionDetectorConfig(MODEL_PATH, **settings)
    return PromptGuardDetector(config, model_factory=classifier.loaded)


def test_the_detector_is_an_injection_detector() -> None:
    detector = make_detector(ScriptedClassifier())

    assert isinstance(detector, InjectionDetector)
    assert detector.configured is True


def test_scores_are_in_range_and_in_input_order() -> None:
    detector = make_detector(ScriptedClassifier())

    assert detector.screen(["0.3", "0.9", "0.0", "1.0"]) == [0.3, 0.9, 0.0, 1.0]


def test_a_text_scores_the_highest_of_its_windows() -> None:
    detector = make_detector(ScriptedClassifier())

    assert detector.screen(["0.1|0.2|0.8|0.3", "0.4"]) == [0.8, 0.4]


def test_blank_texts_score_zero_without_running_the_model() -> None:
    classifier = ScriptedClassifier()
    detector = make_detector(classifier)

    assert detector.screen(["", "   ", "\n\t"]) == [0.0, 0.0, 0.0]
    assert detector.screen([]) == []
    assert classifier.batches == []


def test_blank_texts_keep_their_place_among_scored_ones() -> None:
    classifier = ScriptedClassifier()
    detector = make_detector(classifier)

    assert detector.screen(["", "0.7", " ", "0.2|0.5"]) == [0.0, 0.7, 0.0, 0.5]
    assert classifier.batches == [(70, 20, 50)]


def test_windows_run_in_batches_of_batch_size() -> None:
    classifier = ScriptedClassifier()
    detector = make_detector(classifier, batch_size=2)

    scores = detector.screen(["0.1|0.2|0.3", "0.4", "0.5"])

    assert classifier.batches == [(10, 20), (30, 40), (50,)]
    assert scores == [0.3, 0.4, 0.5]


def test_scores_are_clipped_into_the_unit_interval() -> None:
    detector = make_detector(ScriptedClassifier())

    assert detector.screen(["1.5", "-0.5"]) == [1.0, 0.0]


def test_a_model_failure_is_detector_failed_with_the_cause_and_the_next_call_works() -> None:
    classifier = ScriptedClassifier()
    detector = make_detector(classifier)
    cause = RuntimeError("CUDA is a lie")
    classifier.fail_with = cause

    with pytest.raises(ProcessingFailed) as caught:
        detector.screen(["0.9"])

    assert caught.value.code == "detector_failed"
    assert caught.value.message == "the injection detector failed"
    assert caught.value.__cause__ is cause
    assert detector.screen(["0.9"]) == [0.9]


def test_a_tokenizer_failure_is_detector_failed_with_the_cause() -> None:
    classifier = ScriptedClassifier()
    detector = make_detector(classifier)

    with pytest.raises(ProcessingFailed) as caught:
        detector.screen(["not a number"])

    assert caught.value.code == "detector_failed"
    assert isinstance(caught.value.__cause__, ValueError)
    assert "not a number" not in caught.value.message


def test_a_service_error_of_the_model_passes_through_unchanged() -> None:
    classifier = ScriptedClassifier()
    detector = make_detector(classifier)
    classifier.fail_with = ProcessingFailed("custom", "detector_failed")

    with pytest.raises(ProcessingFailed, match="custom"):
        detector.screen(["0.9"])


def test_the_wrong_number_of_scores_is_detector_failed() -> None:
    def windows(texts: Sequence[str], batch_size: int) -> list[WindowBatch]:
        del batch_size
        return [WindowBatch({"score": torch.tensor([0.5])}, (0, 1))]

    def score(inputs: Mapping[str, torch.Tensor]) -> list[float]:
        return [0.5]

    config = InjectionDetectorConfig(MODEL_PATH)
    detector = PromptGuardDetector(config, model_factory=lambda _: LoadedClassifier(windows, score))

    with pytest.raises(ProcessingFailed) as caught:
        detector.screen(["a", "b"])

    assert caught.value.code == "detector_failed"


def test_calls_after_close_fail_with_detector_failed() -> None:
    classifier = ScriptedClassifier()
    detector = make_detector(classifier)
    detector.close()
    detector.close()  # idempotent

    with pytest.raises(ProcessingFailed) as caught:
        detector.screen(["0.9"])

    assert caught.value.code == "detector_failed"
    assert classifier.batches == []


def test_blank_texts_still_score_zero_after_close() -> None:
    detector = make_detector(ScriptedClassifier())
    detector.close()

    assert detector.screen(["", " "]) == [0.0, 0.0]


def test_a_call_waiting_for_a_slot_when_closed_fails_with_detector_failed() -> None:
    classifier = ScriptedClassifier(pause=0.2)
    detector = make_detector(classifier, pool_size=1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        holder = executor.submit(detector.screen, ["0.1"])
        time.sleep(0.05)
        waiter = executor.submit(detector.screen, ["0.2"])
        time.sleep(0.05)
        detector.close()

        assert holder.result() == [0.1]
        with pytest.raises(ProcessingFailed) as caught:
            waiter.result()

    assert caught.value.code == "detector_failed"


def test_at_most_pool_size_batches_run_at_once() -> None:
    classifier = ScriptedClassifier(pause=0.02)
    detector = make_detector(classifier, pool_size=2)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda text: detector.screen([text]), ["0.5"] * 16))

    assert results == [[0.5]] * 16
    assert 1 < classifier.peak <= 2


def wait_until(condition: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


def test_page_batches_never_take_the_slot_kept_for_questions() -> None:
    classifier = ScriptedClassifier(pause=0.02)
    detector = make_detector(classifier, pool_size=2)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda text: detector.screen([text]), ["0.5"] * 16))

    assert results == [[0.5]] * 16
    assert classifier.peak == 2


def test_a_question_takes_the_reserved_slot_while_pages_fill_the_pool() -> None:
    classifier = ScriptedClassifier(pause=0.3)
    detector = make_detector(classifier, pool_size=1)
    with ThreadPoolExecutor(max_workers=4) as executor:
        pages = [executor.submit(detector.screen, ["0.1"]) for _ in range(3)]
        wait_until(lambda: detector._slots.waiting(PAGE_LANE) == 2)

        started = time.monotonic()
        assert detector.screen_query("0.9") == 0.9
        elapsed = time.monotonic() - started

        assert [page.result() for page in pages] == [[0.1]] * 3
    assert classifier.peak == 2
    assert elapsed < 0.55  # one pass of the model, not behind the two waiting pages


def test_a_waiting_question_goes_before_waiting_pages() -> None:
    classifier = ScriptedClassifier(pause=0.15)
    detector = make_detector(classifier, pool_size=1)
    with ThreadPoolExecutor(max_workers=6) as executor:
        running: list[Future[Any]] = [
            executor.submit(detector.screen, ["0.11"]),
            executor.submit(detector.screen_query, "0.12"),
        ]
        wait_until(lambda: classifier.active == 2)
        pages = [executor.submit(detector.screen, ["0.13"]) for _ in range(2)]
        wait_until(lambda: detector._slots.waiting(PAGE_LANE) == 2)
        question = executor.submit(detector.screen_query, "0.14")
        wait_until(lambda: detector._slots.waiting(QUERY_LANE) == 1)

        for future in [*running, *pages]:
            future.result()
        question.result()

    # Batches are recorded as they end. The two freed slots go to the question and one page;
    # the other page runs only after them. Without the priority the question would run last.
    order = [batch[0] for batch in classifier.batches]
    assert sorted(order[2:4]) == [13, 14]
    assert order[4] == 13


def test_a_question_scores_as_screen_does() -> None:
    detector = make_detector(ScriptedClassifier())

    assert detector.screen_query("0.2|0.7") == detector.screen(["0.2|0.7"])[0] == 0.7
    assert detector.screen_query("   ") == 0.0


def test_concurrent_calls_equal_the_single_call_scores() -> None:
    texts = [f"0.{index}|0.{(index * 7) % 10}|0.05" for index in range(1, 25)]
    detector = make_detector(ScriptedClassifier(), pool_size=2, batch_size=3)
    expected = [detector.screen([text]) for text in texts]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda text: detector.screen([text]), texts))

    assert results == expected
    assert detector.screen(texts) == [score for [score] in expected]


def test_loading_failures_are_invalid_configuration() -> None:
    def broken(config: InjectionDetectorConfig) -> LoadedClassifier:
        raise OSError("no such directory")

    with pytest.raises(ConfigurationError) as caught:
        PromptGuardDetector(InjectionDetectorConfig(MODEL_PATH), model_factory=broken)

    assert caught.value.code == "invalid_configuration"
    assert MODEL_PATH in caught.value.message
    assert isinstance(caught.value.__cause__, OSError)


def test_a_service_error_while_loading_passes_through() -> None:
    def refused(config: InjectionDetectorConfig) -> LoadedClassifier:
        raise ConfigurationError("no positive label")

    with pytest.raises(ConfigurationError, match="no positive label"):
        PromptGuardDetector(InjectionDetectorConfig(MODEL_PATH), model_factory=refused)


def test_loading_sets_the_configured_threads() -> None:
    before = torch.get_num_threads()
    target = 1 if before != 1 else 2

    make_detector(ScriptedClassifier(), threads=target)

    assert torch.get_num_threads() == target


def test_a_failed_load_leaves_the_thread_count_unchanged() -> None:
    before = torch.get_num_threads()
    target = 1 if before != 1 else 2

    def broken(config: InjectionDetectorConfig) -> LoadedClassifier:
        raise OSError("no such directory")

    with pytest.raises(ConfigurationError):
        PromptGuardDetector(InjectionDetectorConfig(MODEL_PATH, threads=target), broken)

    assert torch.get_num_threads() == before


@pytest.mark.parametrize(
    ("id2label", "expected"),
    [
        ({0: "LABEL_0", 1: "LABEL_1"}, (1,)),  # Prompt Guard 2 (86M and 22M)
        ({"0": "LABEL_0", "1": "LABEL_1"}, (1,)),  # as read back from a config.json
        ({0: "BENIGN", 1: "INJECTION", 2: "JAILBREAK"}, (1, 2)),  # the first Prompt Guard
        ({0: "SAFE", 1: "UNSAFE"}, (1,)),
        ({0: "malicious", 1: "benign"}, (0,)),  # names win over the position
        ({0: "good", 1: "bad"}, (1,)),  # two unknown names: index 1 is the positive one
    ],
)
def test_positive_labels_are_found_by_name_or_position(
    id2label: dict[object, object], expected: tuple[int, ...]
) -> None:
    assert positive_label_indexes(id2label) == expected


@pytest.mark.parametrize(
    "id2label",
    [
        {0: "a", 1: "b", 2: "c"},  # three unknown names: no position convention
        {0: "benign"},  # one label
        {0: "INJECTION", 1: "JAILBREAK"},  # every label positive: nothing is benign
        {},
    ],
)
def test_unrecognisable_labels_are_invalid_configuration(id2label: dict[object, object]) -> None:
    with pytest.raises(ConfigurationError, match="labels"):
        positive_label_indexes(id2label)
