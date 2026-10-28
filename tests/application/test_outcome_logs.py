"""The one outcome line each use case logs: its fields, their types, and no personal data."""

import logging
from collections.abc import Sequence

import pytest

from app.application import ask_question, extract_names, ingest_document
from app.application.ask_question import AskQuestion
from app.application.extract_names import ExtractNames
from app.application.ingest_document import IngestDocument
from app.domain.cancellation import NEVER_CANCELLED, Cancellation
from app.domain.errors import InvalidInput, ProcessingFailed
from app.domain.prompting import REFUSAL, REFUSAL_SENTINEL
from app.domain.types import Box, Chunk, NamePair, OcrPage, PageWord
from app.observability import ALLOWED_FIELDS, JsonFormatter
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm import FakeLLMClient
from tests.fakes.names import FakeNameRecognizer
from tests.fakes.ocr import FakeOcrEngine
from tests.fakes.vector_store import FakeVectorStore

PDF = b"%PDF-1.7 outcome logs"
PERSON = "María González"
CONTRACT = "The contract was signed by María González on 3 March 2024 in Madrid."
QUESTION = "Who signed the contract in Madrid?"
ANSWER = "It was signed by María González."
FORBIDDEN = ("María", "Maria", "González", "Gonzalez", "contract", "Madrid", "signed")
# The fake embedder is a bag of words: the question scores about 0.45 against CONTRACT.
LOW_THRESHOLD = 0.3

EXTRACTION_FIELDS = {
    "size_bytes": int,
    "pages": int,
    "names_found": int,
    "requested_names": int,
    "matches": int,
    "ocr_ms": float,
    "ner_ms": float,
    "duration_ms": float,
}
INGESTION_BASE = {
    "size_bytes": int,
    "document_id": str,
    "already_indexed": bool,
    "chunks_stored": int,
    "duration_ms": float,
}
INGESTION_STAGES = {"pages": int, "ocr_ms": float, "embed_ms": float, "store_ms": float}
QUESTION_FIELDS = {
    "hits": int,
    "hits_kept": int,
    "refused": bool,
    "sources": int,
    "embed_ms": float,
    "search_ms": float,
    "duration_ms": float,
}


def make_page(page: int, text: str) -> OcrPage:
    words: list[PageWord] = []
    x = 50.0
    for word in text.split():
        box = Box(page=page, x=x, y=50.0, width=6.0 * len(word), height=12.0)
        words.append(PageWord(text=word, box=box, line=0, paragraph=0))
        x += 6.0 * (len(word) + 1)
    return OcrPage(page=page, width=612, height=792, words=tuple(words), language=None)


def outcome(caplog: pytest.LogCaptureFixture, logger: str, message: str) -> logging.LogRecord:
    """The only record ``logger`` wrote, which must be its outcome line."""
    records = [record for record in caplog.records if record.name == logger]
    assert len(records) == 1, [record.getMessage() for record in records]
    (record,) = records
    assert record.levelno == logging.INFO
    assert record.getMessage() == message
    return record


def logged_fields(record: logging.LogRecord) -> dict[str, object]:
    return {key: record.__dict__[key] for key in ALLOWED_FIELDS if key in record.__dict__}


def assert_types(fields: dict[str, object], expected: dict[str, type]) -> None:
    for key, kind in expected.items():
        assert type(fields[key]) is kind, key


def assert_no_personal_data(caplog: pytest.LogCaptureFixture) -> None:
    formatter = JsonFormatter()
    text = "\n".join(formatter.format(record) for record in caplog.records)
    assert text, "nothing was logged"
    for word in FORBIDDEN:
        assert word not in text, f"{word!r} reached the logs"


@pytest.fixture(autouse=True)
def info_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)


# --- extraction ---------------------------------------------------------------------------


def test_extraction_logs_one_outcome_line_with_counts_and_durations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pages = [make_page(0, CONTRACT), make_page(1, "Nothing to see here")]
    use_case = ExtractNames(FakeOcrEngine(pages), FakeNameRecognizer([PERSON]))
    requested = [NamePair("María", "González"), NamePair("Ana", "Pérez")]

    result = use_case(PDF, requested)

    record = outcome(caplog, extract_names.__name__, extract_names.OUTCOME_MESSAGE)
    fields = logged_fields(record)
    assert set(fields) == set(EXTRACTION_FIELDS)
    assert_types(fields, EXTRACTION_FIELDS)
    assert (fields["pages"], fields["names_found"], fields["requested_names"]) == (2, 1, 2)
    assert fields["matches"] == len(result.matches) == 1
    assert fields["size_bytes"] == len(PDF)
    assert all(float(fields[key]) >= 0 for key in ("ocr_ms", "ner_ms", "duration_ms"))  # type: ignore[arg-type]


def test_extraction_without_requested_names_logs_zero_matches(
    caplog: pytest.LogCaptureFixture,
) -> None:
    use_case = ExtractNames(FakeOcrEngine([]), FakeNameRecognizer())

    use_case(PDF, [])

    fields = logged_fields(outcome(caplog, extract_names.__name__, "extraction completed"))
    assert (fields["pages"], fields["names_found"], fields["matches"]) == (0, 0, 0)


def test_extraction_logs_no_name_or_document_text(caplog: pytest.LogCaptureFixture) -> None:
    use_case = ExtractNames(FakeOcrEngine([make_page(0, CONTRACT)]), FakeNameRecognizer([PERSON]))

    result = use_case(PDF, [NamePair("María", "González")])

    assert [mention.name for mention in result.names] == [PERSON]
    assert_no_personal_data(caplog)


def test_failed_extraction_logs_no_outcome_line(caplog: pytest.LogCaptureFixture) -> None:
    class Failing(FakeOcrEngine):
        def recognize(
            self,
            pdf: bytes,
            language: str | None = None,
            *,
            cancel: Cancellation = NEVER_CANCELLED,
        ) -> list[OcrPage]:
            raise RuntimeError("ocr failed")

    with pytest.raises(RuntimeError):
        ExtractNames(Failing(), FakeNameRecognizer())(PDF, [])

    assert not [r for r in caplog.records if r.name == extract_names.__name__]


# --- ingestion ----------------------------------------------------------------------------


def ingest_case(pages: Sequence[OcrPage]) -> tuple[IngestDocument, FakeVectorStore]:
    store = FakeVectorStore()
    return IngestDocument(FakeOcrEngine(pages), FakeEmbedder(), store), store


def test_ingestion_logs_one_outcome_line_with_stage_durations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    use_case, _ = ingest_case([make_page(0, CONTRACT), make_page(1, "Payment is due soon")])

    result = use_case(PDF)

    record = outcome(caplog, ingest_document.__name__, ingest_document.OUTCOME_MESSAGE)
    fields = logged_fields(record)
    expected = INGESTION_BASE | INGESTION_STAGES
    assert set(fields) == set(expected)
    assert_types(fields, expected)
    assert fields["document_id"] == result.document_id
    assert (fields["already_indexed"], fields["chunks_stored"]) == (False, result.chunks_stored)
    assert (fields["pages"], fields["size_bytes"]) == (2, len(PDF))


def test_already_indexed_ingestion_logs_no_stage_fields(caplog: pytest.LogCaptureFixture) -> None:
    use_case, _ = ingest_case([make_page(0, CONTRACT)])
    first = use_case(PDF)
    caplog.clear()

    use_case(PDF)

    fields = logged_fields(outcome(caplog, ingest_document.__name__, "ingestion completed"))
    assert set(fields) == set(INGESTION_BASE)
    assert fields["already_indexed"] is True
    assert fields["chunks_stored"] == first.chunks_stored
    assert fields["document_id"] == first.document_id


def test_ingestion_without_text_logs_ocr_but_no_embedding_or_storage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    use_case, _ = ingest_case([make_page(0, "   ")])

    use_case(PDF)

    fields = logged_fields(outcome(caplog, ingest_document.__name__, "ingestion completed"))
    assert set(fields) == set(INGESTION_BASE) | {"pages", "ocr_ms"}
    assert (fields["pages"], fields["chunks_stored"]) == (1, 0)


def test_ingestion_logs_no_document_text(caplog: pytest.LogCaptureFixture) -> None:
    use_case, _ = ingest_case([make_page(0, CONTRACT)])

    use_case(PDF)

    assert_no_personal_data(caplog)


# --- questions ----------------------------------------------------------------------------


def ask_case(
    llm: FakeLLMClient | None = None, **config: int | float
) -> tuple[AskQuestion, FakeLLMClient]:
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    store.ensure_collection(embedder.identity)
    chunks = [Chunk(CONTRACT, 0, 0, 0, len(CONTRACT))]
    store.upsert(
        "doc-a",
        chunks,
        embedder.embed_passages([CONTRACT]),
        page_texts={0: CONTRACT},
        content_hash="hash-a",
    )
    llm = llm or FakeLLMClient(default=ANSWER)
    return AskQuestion(embedder, store, llm, **config), llm  # type: ignore[arg-type]


def question_fields(caplog: pytest.LogCaptureFixture) -> dict[str, object]:
    record = outcome(caplog, ask_question.__name__, ask_question.OUTCOME_MESSAGE)
    fields = logged_fields(record)
    assert_types(fields, QUESTION_FIELDS)
    return fields


def test_answered_question_logs_hits_score_and_model_latency(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ask, _ = ask_case(threshold=LOW_THRESHOLD)

    answer = ask(QUESTION)

    assert answer.text == ANSWER
    fields = question_fields(caplog)
    assert set(fields) == set(QUESTION_FIELDS) | {"top_score", "refusal_reason", "llm_ms"}
    assert (fields["hits"], fields["hits_kept"], fields["sources"]) == (1, 1, 1)
    assert (fields["refused"], fields["refusal_reason"]) == (False, None)
    assert isinstance(fields["top_score"], float)
    assert fields["top_score"] == round(fields["top_score"], 4)
    assert fields["top_score"] >= LOW_THRESHOLD
    assert isinstance(fields["llm_ms"], float)


def test_question_below_the_threshold_is_refused_without_calling_the_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ask, llm = ask_case(threshold=0.99)

    assert ask(QUESTION).text == REFUSAL

    fields = question_fields(caplog)
    assert (fields["refused"], fields["refusal_reason"]) == (True, "below_threshold")
    assert (fields["hits"], fields["hits_kept"], fields["sources"]) == (1, 0, 0)
    assert isinstance(fields["top_score"], float)
    assert "llm_ms" not in fields
    assert not llm.calls


def test_question_over_an_empty_store_logs_a_null_top_score(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ask = AskQuestion(FakeEmbedder(), FakeVectorStore(), FakeLLMClient())

    ask(QUESTION)

    fields = question_fields(caplog)
    assert (fields["hits"], fields["top_score"]) == (0, None)
    assert fields["refusal_reason"] == "below_threshold"
    assert "llm_ms" not in fields


def test_question_whose_context_does_not_fit_is_refused_as_no_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ask, llm = ask_case(threshold=LOW_THRESHOLD, max_context_chars=5)

    assert ask(QUESTION).text == REFUSAL

    fields = question_fields(caplog)
    assert (fields["refused"], fields["refusal_reason"]) == (True, "no_context")
    assert fields["hits_kept"] == 1
    assert "llm_ms" not in fields
    assert not llm.calls


def test_question_the_model_declines_is_refused_as_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ask, llm = ask_case(FakeLLMClient(default=REFUSAL_SENTINEL), threshold=LOW_THRESHOLD)

    assert ask(QUESTION).text == REFUSAL

    fields = question_fields(caplog)
    assert (fields["refused"], fields["refusal_reason"], fields["sources"]) == (True, "model", 0)
    assert isinstance(fields["llm_ms"], float)
    assert len(llm.calls) == 1


def test_blank_question_logs_no_outcome_line(caplog: pytest.LogCaptureFixture) -> None:
    ask, _ = ask_case()

    with pytest.raises(InvalidInput):
        ask("   ")

    assert not [r for r in caplog.records if r.name == ask_question.__name__]


def test_question_logs_no_question_answer_or_source_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ask, _ = ask_case(threshold=LOW_THRESHOLD)

    answer = ask(QUESTION)

    assert answer.sources == (CONTRACT,)
    assert_no_personal_data(caplog)


# --- cancelled jobs -----------------------------------------------------------------------


def cancelled() -> Cancellation:
    cancel = Cancellation()
    cancel.cancel()
    return cancel


def test_cancelled_extraction_logs_one_cancelled_line_with_the_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    use_case = ExtractNames(FakeOcrEngine([make_page(0, CONTRACT)]), FakeNameRecognizer([PERSON]))

    with pytest.raises(ProcessingFailed) as raised:
        use_case(PDF, [NamePair("María", "González")], cancel=cancelled())

    assert raised.value.code == "request_cancelled"
    record = outcome(caplog, extract_names.__name__, extract_names.CANCELLED_MESSAGE)
    fields = logged_fields(record)
    assert fields == {
        "code": "request_cancelled",
        "size_bytes": len(PDF),
        "duration_ms": fields["duration_ms"],
    }
    assert isinstance(fields["duration_ms"], float)
    assert_no_personal_data(caplog)


def test_cancelled_ingestion_logs_one_cancelled_line_with_the_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    use_case, store = ingest_case([make_page(0, CONTRACT)])

    with pytest.raises(ProcessingFailed) as raised:
        use_case(PDF, cancel=cancelled())

    assert raised.value.code == "request_cancelled"
    assert store.count() == 0
    record = outcome(caplog, ingest_document.__name__, ingest_document.CANCELLED_MESSAGE)
    fields = logged_fields(record)
    assert set(fields) == {"code", "size_bytes", "document_id", "duration_ms"}
    assert (fields["code"], fields["size_bytes"]) == ("request_cancelled", len(PDF))
    assert isinstance(fields["document_id"], str)
    assert isinstance(fields["duration_ms"], float)
    assert_no_personal_data(caplog)


@pytest.mark.parametrize("code", ["ocr_failed", "ner_failed"])
def test_other_processing_failures_log_no_use_case_line(
    caplog: pytest.LogCaptureFixture, code: str
) -> None:
    class Failing(FakeOcrEngine):
        def recognize(
            self,
            pdf: bytes,
            language: str | None = None,
            *,
            cancel: Cancellation = NEVER_CANCELLED,
        ) -> list[OcrPage]:
            raise ProcessingFailed("failed", code=code)

    with pytest.raises(ProcessingFailed):
        ExtractNames(Failing(), FakeNameRecognizer())(PDF, [])
    with pytest.raises(ProcessingFailed):
        IngestDocument(Failing(), FakeEmbedder(), FakeVectorStore())(PDF)

    names = {extract_names.__name__, ingest_document.__name__}
    assert not [record for record in caplog.records if record.name in names]
