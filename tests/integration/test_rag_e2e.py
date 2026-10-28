"""Ingestion and grounded answering end to end: real OCR, real e5 embeddings, the Qdrant server.

Only the language model is scripted: it records the messages it receives and returns a fixed
reply. Answerable questions use ``ANSWER_THRESHOLD`` (below the 0.808 default) so that they do
not pass on a razor-thin score margin; one test asks the off-topic question at the default
threshold and records how far below it the best score is. The store uses a collection prefix
unique to this module, dropped at teardown. The two sample PDFs are ingested once per module;
tests that ingest further documents come last.
"""

import os
import textwrap
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path

import pymupdf
import pytest
import torch
from qdrant_client import QdrantClient

from app.adapters.embeddings import E5Embedder, E5EmbedderConfig
from app.adapters.ocr import TesseractOcrEngine
from app.adapters.vector_store import QdrantStoreConfig, QdrantVectorStore
from app.application.ask_question import DEFAULT_THRESHOLD, AskQuestion
from app.application.ingest_document import IngestDocument
from app.domain.cancellation import NEVER_CANCELLED, Cancellation
from app.domain.prompting import REFUSAL, SYSTEM_PROMPT, select_context, shown_sources
from app.domain.types import IngestResult, OcrPage, SearchHit
from tests.adapters.ocr.pdfs import A4
from tests.fakes.llm import FakeLLMClient
from tests.integration.known_words import Line, known_words_page

pytestmark = [pytest.mark.tesseract, pytest.mark.embedding_model, pytest.mark.qdrant_server]

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "sample_pdfs"
MEMO = "company_memo.pdf"
REPORT = "research_report.pdf"
SCRIPTED_REPLY = "Robert Chen (scripted reply)."
TOP_K = 5
ANSWER_THRESHOLD = 0.70

MEMO_QUESTION = "Who was promoted to Vice President of Engineering?"
MEMO_FACT = "Robert Chen has been promoted to Vice President of Engineering"
REPORT_QUESTION = "How many peer-reviewed publications did the research department produce in 2024?"
REPORT_FACT = "23 peer-reviewed publications"
OFF_TOPIC_QUESTION = "What is the boiling point of mercury on Venus?"


class CountingOcr:
    """Delegates to a real OCR engine and counts the calls."""

    def __init__(self, engine: TesseractOcrEngine) -> None:
        self._engine = engine
        self.calls = 0

    def recognize(
        self,
        pdf: bytes,
        language: str | None = None,
        *,
        cancel: Cancellation = NEVER_CANCELLED,
    ) -> list[OcrPage]:
        self.calls += 1
        return self._engine.recognize(pdf, language, cancel=cancel)


@pytest.fixture(scope="module")
def embedder() -> Iterator[E5Embedder]:
    threads = torch.get_num_threads()
    path = os.environ.get("EMBEDDING_MODEL_PATH", "/models/multilingual-e5-base")
    revision = os.environ.get("EMBEDDING_MODEL_REVISION", "unknown")
    yield E5Embedder(E5EmbedderConfig(model_path=path, revision=revision))
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def store() -> Iterator[QdrantVectorStore]:
    url = os.environ["QDRANT_URL"]
    prefix = f"e2e{uuid.uuid4().hex[:10]}"
    store = QdrantVectorStore(QdrantStoreConfig(url=url, collection_prefix=prefix))
    yield store
    store.close()
    admin = QdrantClient(url=url)
    for collection in admin.get_collections().collections:
        if collection.name.startswith(f"{prefix}_"):
            admin.delete_collection(collection.name)
    admin.close()


@pytest.fixture(scope="module")
def ocr() -> CountingOcr:
    return CountingOcr(TesseractOcrEngine())


@pytest.fixture(scope="module")
def ingest(ocr: CountingOcr, embedder: E5Embedder, store: QdrantVectorStore) -> IngestDocument:
    return IngestDocument(ocr, embedder, store)


@pytest.fixture(scope="module")
def ingested(ingest: IngestDocument) -> dict[str, IngestResult]:
    """The first ingestion of each sample, done once for the module."""
    return {name: ingest((SAMPLES_DIR / name).read_bytes()) for name in (MEMO, REPORT)}


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient(default=SCRIPTED_REPLY)


@pytest.fixture
def ask(embedder: E5Embedder, store: QdrantVectorStore, llm: FakeLLMClient) -> AskQuestion:
    return AskQuestion(embedder, store, llm, threshold=ANSWER_THRESHOLD, top_k=TOP_K)


@pytest.fixture
def ask_default(embedder: E5Embedder, store: QdrantVectorStore, llm: FakeLLMClient) -> AskQuestion:
    return AskQuestion(embedder, store, llm, top_k=TOP_K)


def _search(embedder: E5Embedder, store: QdrantVectorStore, question: str) -> list[SearchHit]:
    return store.search(embedder.embed_query(question), TOP_K)


def _describe(hits: Sequence[SearchHit]) -> str:
    return "; ".join(f"{h.score:.4f} {h.document_id[:8]} p{h.page} {h.text[:60]!r}" for h in hits)


def test_both_samples_are_stored_under_distinct_ids(
    ingested: dict[str, IngestResult], store: QdrantVectorStore
) -> None:
    memo, report = ingested[MEMO], ingested[REPORT]

    for result in (memo, report):
        assert result.chunks_stored > 0
        assert result.already_indexed is False
        assert store.document_chunk_count(result.document_id) == result.chunks_stored
    assert memo.document_id != report.document_id
    assert store.count() == memo.chunks_stored + report.chunks_stored


def test_reingesting_the_memo_reports_already_indexed_without_ocr_or_new_points(
    ingested: dict[str, IngestResult],
    ingest: IngestDocument,
    ocr: CountingOcr,
    store: QdrantVectorStore,
) -> None:
    first = ingested[MEMO]
    points_before, ocr_calls_before = store.count(), ocr.calls

    again = ingest((SAMPLES_DIR / MEMO).read_bytes())

    assert again == IngestResult(first.document_id, first.chunks_stored, already_indexed=True)
    assert store.count() == points_before
    assert ocr.calls == ocr_calls_before


def test_an_answerable_question_is_answered_from_the_memo_page(
    ingested: dict[str, IngestResult],
    ask: AskQuestion,
    llm: FakeLLMClient,
    embedder: E5Embedder,
    store: QdrantVectorStore,
) -> None:
    hits = _search(embedder, store, MEMO_QUESTION)
    print(f"\n[top-1 answerable] {hits[0].score:.4f} (threshold {ANSWER_THRESHOLD})")
    print(f"[hits answerable] {_describe(hits)}")
    kept = [hit for hit in hits if hit.score >= ANSWER_THRESHOLD]
    memo_hits = [hit for hit in kept if hit.document_id == ingested[MEMO].document_id]
    assert memo_hits, _describe(hits)
    assert MEMO_FACT in memo_hits[0].page_text

    answer = ask(MEMO_QUESTION)

    assert answer.text == SCRIPTED_REPLY
    assert answer.sources, _describe(hits)
    (messages,) = llm.calls
    system, user = messages
    assert (system.role, system.content) == ("system", SYSTEM_PROMPT)
    assert user.role == "user"
    assert memo_hits[0].page_text in user.content
    assert MEMO_FACT in user.content
    assert user.content.endswith(f"Question: {MEMO_QUESTION}")
    assert answer.sources == shown_sources(kept, select_context(kept))
    assert answer.sources[0] == kept[0].text
    kept_texts = [hit.text for hit in kept]
    positions = [kept_texts.index(source) for source in answer.sources]
    assert positions == sorted(positions)
    for source in answer.sources:
        assert source.strip() in user.content


def test_an_off_topic_question_is_refused_at_the_default_threshold_without_the_model(
    ingested: dict[str, IngestResult],
    ask_default: AskQuestion,
    llm: FakeLLMClient,
    embedder: E5Embedder,
    store: QdrantVectorStore,
) -> None:
    hits = _search(embedder, store, OFF_TOPIC_QUESTION)
    margin = DEFAULT_THRESHOLD - hits[0].score
    print(f"\n[top-1 off-topic] {hits[0].score:.4f} (default {DEFAULT_THRESHOLD})")
    print(f"[margin off-topic] {margin:.4f}; hits {_describe(hits)}")
    assert margin > 0, _describe(hits)

    answer = ask_default(OFF_TOPIC_QUESTION)

    assert answer.text == REFUSAL
    assert answer.sources == ()
    assert llm.calls == []


def test_the_second_document_remains_retrievable(
    ingested: dict[str, IngestResult],
    ask: AskQuestion,
    llm: FakeLLMClient,
    embedder: E5Embedder,
    store: QdrantVectorStore,
) -> None:
    report_id = ingested[REPORT].document_id
    hits = _search(embedder, store, REPORT_QUESTION)
    print(f"\n[top-1 report] {hits[0].score:.4f} (threshold {ANSWER_THRESHOLD})")
    print(f"[hits report] {_describe(hits)}")
    assert hits[0].document_id == report_id, _describe(hits)
    assert hits[0].score >= ANSWER_THRESHOLD, _describe(hits)

    answer = ask(REPORT_QUESTION)

    assert answer.sources
    assert answer.sources[0] == hits[0].text
    (messages,) = llm.calls
    assert REPORT_FACT in messages[1].content
    assert f"document {report_id}" in messages[1].content


# --- A synthetic scanned document whose answer sits late on a long page --------------------

FIRST_PAGE = (
    "Quarterly cafeteria survey. Most staff preferred the vegetable soup on Mondays and asked "
    "for more seating near the windows. The kitchen will open fifteen minutes earlier."
)
LONG_PAGE_OPENING = "Facilities handbook, chapter four: building maintenance and safety."
LONG_PAGE_FILLER = (
    "The main lobby is cleaned every evening after the offices close. Carpets on the upper "
    "floors are replaced every five years, and the stairwells are painted in the summer. "
    "Visitors must sign the register at the reception desk and wear a badge at all times. "
    "Bicycle racks are located behind the north entrance, next to the delivery bay. "
    "Meeting rooms on the third floor can be booked through the internal calendar, and "
    "projectors are kept in the cabinet beside the printer. Lost property is held at the "
    "reception for thirty days before it is donated. The parking garage closes at midnight "
    "and reopens at six in the morning. Plants in the atrium are watered twice a week by an "
    "outside gardening company. The roof terrace is closed during storms and heavy snow."
)
ANSWER_SENTENCE = (
    "The emergency diesel generator in the basement is serviced by Halvorsen Electric "
    "on the first Tuesday of every month."
)
LONG_PAGE_CLOSING = (
    "Fire drills take place every spring, and all staff gather in the car park across the "
    "street. Wardens wear yellow vests and count everyone before the building is reopened."
)
SYNTHETIC_QUESTION = "Which company services the emergency diesel generator?"
ANSWER_WORD = "Halvorsen"


def _scanned_page(text: str) -> bytes:
    wrapped = textwrap.wrap(text, width=80)
    lines = [Line((60.0, 70.0 + 17.0 * index), line) for index, line in enumerate(wrapped)]
    assert 70.0 + 17.0 * len(wrapped) < A4[1] - 40.0, "page text does not fit on A4"
    return known_words_page(A4, lines, (), fontsize=11.0).pdf


def _two_page_pdf() -> bytes:
    long_page = " ".join([LONG_PAGE_OPENING, LONG_PAGE_FILLER, ANSWER_SENTENCE, LONG_PAGE_CLOSING])
    assert long_page.index(ANSWER_WORD) > len(long_page) // 2
    with pymupdf.open() as merged:
        for text in (FIRST_PAGE, long_page):
            with pymupdf.open(stream=_scanned_page(text), filetype="pdf") as single:
                merged.insert_pdf(single)
        data: bytes = merged.tobytes()
    return data


def test_the_answer_late_on_a_long_scanned_page_reaches_the_model_with_its_page(
    ingested: dict[str, IngestResult],
    ingest: IngestDocument,
    ask: AskQuestion,
    llm: FakeLLMClient,
    embedder: E5Embedder,
    store: QdrantVectorStore,
) -> None:
    result = ingest(_two_page_pdf())
    assert result.already_indexed is False
    assert result.document_id not in {r.document_id for r in ingested.values()}
    hits = _search(embedder, store, SYNTHETIC_QUESTION)
    print(f"\n[top-1 synthetic] {hits[0].score:.4f} (threshold {ANSWER_THRESHOLD})")
    print(f"[hits synthetic] {_describe(hits)}")
    long_page_hits = [
        h
        for h in hits
        if h.document_id == result.document_id and h.page == 1 and h.score >= ANSWER_THRESHOLD
    ]
    assert long_page_hits, _describe(hits)
    page_text = long_page_hits[0].page_text
    assert len(page_text) > 800, "the long page must be split into several chunks"
    assert page_text.index(ANSWER_WORD) > len(page_text) // 2, page_text
    assert result.chunks_stored >= 3
    answer_chunks = [h.text for h in long_page_hits if ANSWER_WORD in h.text]
    assert answer_chunks, _describe(hits)
    assert "Facilities handbook" not in answer_chunks[0], "the answer chunk must not be the page"

    answer = ask(SYNTHETIC_QUESTION)

    assert answer.text == SCRIPTED_REPLY
    (messages,) = llm.calls
    user = messages[1].content
    assert f"document {result.document_id}, page 2" in user
    assert page_text in user
    assert ANSWER_WORD in user
    assert "Facilities handbook" in user
