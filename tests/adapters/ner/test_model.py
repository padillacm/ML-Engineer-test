"""The adapter with the real pinned model (``ner_model``: runs in the container)."""

import logging
import os
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from app.adapters.ner import TransformersNameRecognizer, TransformersNerConfig
from app.application.ports import NameRecognizer
from tests.application.contracts import NameRecognizerContract

pytestmark = pytest.mark.ner_model

MIXED = "Maria Gonzalez of Acme Corporation met in Madrid on 3 March"
_FILLER = (
    "The supplier shall deliver the goods described in the annex to the warehouse in "
    "Rotterdam within thirty days of the order. Invoices are payable within sixty days and "
    "late payments accrue interest at the statutory rate. Either party may terminate this "
    "agreement with ninety days written notice if the other party breaches a material "
    "obligation and fails to remedy it."
)


def _model_path() -> Path:
    return Path(os.environ.get("NER_MODEL_PATH", "/models/xlm-roberta-base-ner-hrl"))


@pytest.fixture(scope="module")
def recognizer() -> Iterator[TransformersNameRecognizer]:
    threads = torch.get_num_threads()
    yield TransformersNameRecognizer(TransformersNerConfig(_model_path()))
    torch.set_num_threads(threads)


def covered(recognizer: NameRecognizer, text: str) -> list[str]:
    return [text[span.start : span.end] for span in recognizer.find_persons(text)]


class TestTransformersNameRecognizerContract(NameRecognizerContract):
    @pytest.fixture
    def recognizer(self, recognizer: TransformersNameRecognizer) -> NameRecognizer:
        return recognizer


def test_mixed_entities_give_only_the_person(recognizer: TransformersNameRecognizer) -> None:
    assert covered(recognizer, MIXED) == ["Maria Gonzalez"]


def test_name_in_the_last_paragraph_of_a_long_page_is_found(
    recognizer: TransformersNameRecognizer,
) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415 - only to measure the page

    paragraphs = [_FILLER] * 11 + ["This agreement was signed by Maria Gonzalez on 3 March."]
    page = "\n\n".join(paragraphs)
    tokenizer = AutoTokenizer.from_pretrained(
        str(_model_path()), local_files_only=True, fix_mistral_regex=False
    )

    assert len(page) > 3500
    assert len(tokenizer(page)["input_ids"]) > tokenizer.model_max_length
    assert covered(recognizer, page) == ["Maria Gonzalez"]


def test_threads_follow_the_configuration(recognizer: TransformersNameRecognizer) -> None:
    assert torch.get_num_threads() == 2


def test_loading_and_inference_work_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[object] = []

    def refuse(self: socket.socket, address: object) -> None:
        attempts.append(address)
        raise OSError("network disabled by the test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    threads = torch.get_num_threads()
    try:
        offline = TransformersNameRecognizer(TransformersNerConfig(_model_path()))
        names = covered(offline, MIXED)
    finally:
        torch.set_num_threads(threads)

    assert names == ["Maria Gonzalez"]
    assert attempts == []


def test_loading_does_not_warn_about_the_tokenizer_regex(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415
    from transformers.utils import logging as hf_logging  # noqa: PLC0415

    hf_logging.enable_propagation()
    try:
        with caplog.at_level(logging.WARNING, logger="transformers"):
            silenced = AutoTokenizer.from_pretrained(
                str(_model_path()), local_files_only=True, fix_mistral_regex=False
            )
        assert not _regex_warnings(caplog)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="transformers"):
            default = AutoTokenizer.from_pretrained(str(_model_path()), local_files_only=True)
        assert _regex_warnings(caplog), "without the flag transformers warns (false positive)"
    finally:
        hf_logging.disable_propagation()

    # The flag only silences the warning: tokenization is the same as without it.
    assert silenced(MIXED)["input_ids"] == default(MIXED)["input_ids"]
    assert silenced.is_fast


def _regex_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if "incorrect regex" in record.getMessage()]


def test_inference_does_not_repeat_the_word_heuristic_warning(
    recwarn: pytest.WarningsRecorder,
) -> None:
    threads = torch.get_num_threads()
    try:
        fresh = TransformersNameRecognizer(TransformersNerConfig(_model_path()))
        names = covered(fresh, MIXED)
    finally:
        torch.set_num_threads(threads)

    assert names == ["Maria Gonzalez"]
    assert not [w for w in recwarn if "does not support real words" in str(w.message)]
