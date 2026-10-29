"""The HTTP API end to end: real OCR, NER, e5 embeddings and the Qdrant server behind FastAPI.

Every application is ``create_app`` on a container from ``build_container`` and ``Settings``
(the image sets the model paths, compose sets ``QDRANT_URL``), driven by ``TestClient`` as a
context manager so the lifespan runs. ``SharedModels`` hands the same NER model and e5 embedder
to every container, so each model loads once per module while the containers that need their
own settings (page limit, unreachable Qdrant) still get their own OCR engine and store. Only the
language model is scripted: the ``ask`` use case of the main container is replaced by one wired
to the same embedder and store and a ``FakeLLMClient``. The collection prefix is unique to this
module and its collections are dropped at teardown.
"""

import dataclasses
import json
import logging
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pymupdf
import pytest
import torch
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.adapters.embeddings import E5Embedder, E5EmbedderConfig
from app.adapters.ner import TransformersNameRecognizer, TransformersNerConfig
from app.adapters.vector_store import QdrantStoreConfig, QdrantVectorStore
from app.application.ask_question import AskQuestion
from app.bootstrap import AdapterFactories, Container, build_container
from app.domain.prompting import REFUSAL
from app.main import create_app
from app.settings import Settings, load_settings
from tests.fakes.llm import FakeLLMClient

pytestmark = [
    pytest.mark.tesseract,
    pytest.mark.ner_model,
    pytest.mark.embedding_model,
    pytest.mark.qdrant_server,
    # qdrant-client's version check warns when it builds a client for the unreachable server.
    pytest.mark.filterwarnings("ignore:Failed to obtain server version:UserWarning"),
]

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "sample_pdfs"
MEMO = "company_memo.pdf"
MINUTES = "meeting_minutes.pdf"
REPORT = "research_report.pdf"
# A person each sample names on its first page (see test_extract_names_e2e's printed output).
PRESENT = {
    MEMO: ("Robert", "Chen"),
    MINUTES: ("Elizabeth", "Park"),
    REPORT: ("Priya", "Sharma"),
}
ABSENT = ("Zebulon", "Quackenbush")
MIN_MATCH_SCORE = 0.9

SCRIPTED_REPLY = "Robert Chen."  # only context tokens: the support check keeps it
ANSWER_THRESHOLD = 0.70  # as in test_rag_e2e: no razor-thin margin for the answerable question
MEMO_QUESTION = "Who was promoted to Vice President of Engineering?"
OFF_TOPIC_QUESTION = "What is the boiling point of mercury on Venus?"

PAGE_LIMIT = 2
UPLOAD_LIMIT_MB = 1
MIB = 1024 * 1024
UNREACHABLE_QDRANT = "http://qdrant-unreachable.invalid:6333"

PREFIX = f"http{uuid.uuid4().hex[:10]}"
UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

# Starlette's TestClient returns responses of ``httpx2``, a transitive dependency only.
type Response = Any


class SharedModels:
    """Adapter factories that build the NER model and the embedder once for every container.

    OCR engines and stores are built per container (cheap, and they carry per-settings
    values); each store built is recorded in ``stores``, in build order.
    """

    def __init__(self) -> None:
        self.recognizer: TransformersNameRecognizer | None = None
        self.embedder: E5Embedder | None = None
        self._configs: dict[str, object] = {}
        self.builds = {"recognizer": 0, "embedder": 0}
        self.stores: list[QdrantVectorStore] = []

    def factories(self) -> AdapterFactories:
        return AdapterFactories(
            recognizer=self._recognizer, embedder=self._embedder, store=self._store
        )

    def _recognizer(self, config: TransformersNerConfig) -> TransformersNameRecognizer:
        if self.recognizer is None:
            self.recognizer = TransformersNameRecognizer(config)
            self.builds["recognizer"] += 1
        # Every container must ask for the same model, or sharing it would be wrong.
        assert self._configs.setdefault("recognizer", config) == config
        return self.recognizer

    def _embedder(self, config: E5EmbedderConfig) -> E5Embedder:
        if self.embedder is None:
            self.embedder = E5Embedder(config)
            self.builds["embedder"] += 1
        assert self._configs.setdefault("embedder", config) == config
        return self.embedder

    def _store(self, config: QdrantStoreConfig) -> QdrantVectorStore:
        store = QdrantVectorStore(config)
        self.stores.append(store)
        return store


def _settings(**overrides: Any) -> Settings:
    return load_settings(_env_file=None, qdrant_collection_prefix=PREFIX, **overrides)


@pytest.fixture(scope="module", autouse=True)
def restore_logging() -> Iterator[None]:
    """Each lifespan configures logging process-wide; put the loggers back after the module."""
    root = logging.getLogger()
    saved_root = (root.handlers[:], root.level)
    saved = {
        name: (lg.handlers[:], lg.level, lg.propagate, lg.disabled)
        for name in UVICORN_LOGGERS
        for lg in [logging.getLogger(name)]
    }
    yield
    for handler in root.handlers[:]:
        if handler not in saved_root[0]:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved_root[1])
    for name, (handlers, level, propagate, disabled) in saved.items():
        lg = logging.getLogger(name)
        lg.handlers, lg.propagate, lg.disabled = handlers, propagate, disabled
        lg.setLevel(level)


@pytest.fixture(scope="module")
def shared() -> Iterator[SharedModels]:
    threads = torch.get_num_threads()
    yield SharedModels()
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def settings() -> Settings:
    return _settings()


@pytest.fixture(scope="module")
def container(shared: SharedModels, settings: Settings) -> Iterator[Container]:
    built = build_container(settings, shared.factories())
    yield built
    built.close()
    url = settings.qdrant_url
    assert url is not None
    admin = QdrantClient(url=url)
    for collection in admin.get_collections().collections:
        if collection.name.startswith(f"{PREFIX}_"):
            admin.delete_collection(collection.name)
    admin.close()


@pytest.fixture(scope="module")
def store(shared: SharedModels, container: Container) -> QdrantVectorStore:
    """The main container's store (the first one built)."""
    return shared.stores[0]


@pytest.fixture(scope="module")
def client(settings: Settings, container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(settings, container)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def limited_container(shared: SharedModels) -> Iterator[tuple[Settings, Container]]:
    """A container whose OCR accepts at most ``PAGE_LIMIT`` pages, with a 1 MiB upload limit."""
    limited = _settings(max_pages=PAGE_LIMIT, max_upload_mb=UPLOAD_LIMIT_MB)
    built = build_container(limited, shared.factories())
    yield limited, built
    built.close()


@pytest.fixture(scope="module")
def limited_client(limited_container: tuple[Settings, Container]) -> Iterator[TestClient]:
    with TestClient(create_app(*limited_container)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def unreachable_container(shared: SharedModels) -> Iterator[tuple[Settings, Container]]:
    """A container whose Qdrant URL names a host that does not exist."""
    unreachable = _settings(
        qdrant_url=UNREACHABLE_QDRANT, qdrant_timeout_s=1.0, qdrant_deadline_s=1.0
    )
    built = build_container(unreachable, shared.factories())
    yield unreachable, built
    built.close()


@pytest.fixture(scope="module")
def ingested(client: TestClient) -> dict[str, Response]:
    """The first ingestion of the memo and the report, done once for the module."""
    return {name: _post_pdf(client, "/api/ingest", name) for name in (MEMO, REPORT)}


def _pdf(name: str) -> bytes:
    return (SAMPLES_DIR / name).read_bytes()


def _post_pdf(
    client: TestClient,
    path: str,
    name: str,
    data: bytes | None = None,
    form: dict[str, str] | None = None,
    request_id: str | None = None,
) -> Response:
    content = _pdf(name) if data is None else data
    headers = {} if request_id is None else {"X-Request-ID": request_id}
    return client.post(
        path,
        files={"pdf_file": (name, content, "application/pdf")},
        data=form or {},
        headers=headers,
    )


def _page_sizes(name: str) -> list[tuple[float, float]]:
    with pymupdf.open(SAMPLES_DIR / name) as document:
        return [(float(page.rect.width), float(page.rect.height)) for page in document]


def _full(pair: tuple[str, str]) -> str:
    return f"{pair[0]} {pair[1]}"


def _names_form(*pairs: tuple[str, str]) -> dict[str, str]:
    return {"names": json.dumps([{"first_name": f, "last_name": last} for f, last in pairs])}


def _assert_error(response: Response, status: int, code: str, request_id: str | None) -> None:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["code"] == code, body
    assert isinstance(body["detail"], str)
    assert body["detail"]
    echoed = response.headers["X-Request-ID"]
    assert body["request_id"] == echoed
    if request_id is not None:
        assert echoed == request_id


def _stored_texts(store: QdrantVectorStore, url: str) -> list[str]:
    collection = store.collection
    assert collection is not None, "nothing was ingested"
    admin = QdrantClient(url=url)
    try:
        points, _ = admin.scroll(collection, limit=10_000, with_payload=["text"])
    finally:
        admin.close()
    return [str((point.payload or {})["text"]) for point in points]


# --- Extraction ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", [MEMO, MINUTES, REPORT])
def test_extract_finds_the_named_person_with_boxes_on_its_pages(
    client: TestClient, name: str
) -> None:
    present = PRESENT[name]
    request_id = f"e2e-extract-{name.split('.', maxsplit=1)[0]}"

    response = _post_pdf(
        client, "/api/extract", name, form=_names_form(present, ABSENT), request_id=request_id
    )

    assert response.status_code == 200, response.text
    assert response.headers["X-Request-ID"] == request_id
    body = response.json()
    extracted = body["extracted_names"]
    print(f"\n{name}: {[(e['name'], e['bounding_box']['page_number']) for e in extracted]}")
    print(f"{name} matches: {body['fuzzy_matches']}")
    sizes = _page_sizes(name)
    assert extracted
    for entry in extracted:
        box = entry["bounding_box"]
        assert 0 <= box["page_number"] < len(sizes), entry
        width, height = sizes[box["page_number"]]
        assert box["width"] > 0, entry
        assert box["height"] > 0, entry
        assert box["x"] >= 0, entry
        assert box["y"] >= 0, entry
        assert box["x"] + box["width"] <= width + 1e-6, entry
        assert box["y"] + box["height"] <= height + 1e-6, entry
    pages = [entry["bounding_box"]["page_number"] for entry in extracted]
    assert pages == sorted(pages)

    matches = body["fuzzy_matches"]
    present_matches = [m for m in matches if m["matched_name"] == _full(present)]
    assert present_matches, matches
    assert all(m["score"] >= MIN_MATCH_SCORE for m in present_matches), matches
    assert all(0.0 <= m["score"] <= 1.0 for m in matches)
    assert not [m for m in matches if m["matched_name"] == _full(ABSENT)], matches
    matched_extracted = {m["extracted_name"] for m in present_matches}
    present_pages = {
        entry["bounding_box"]["page_number"]
        for entry in extracted
        if entry["name"] in matched_extracted
    }
    # Every sample names its person on the first page, including the two-page ones.
    assert 0 in present_pages, (present_pages, extracted)
    if len(sizes) > 1:
        assert set(pages) == set(range(len(sizes))), "names on every page of a multipage sample"


# --- Ingestion and questions -----------------------------------------------------------------


def test_ingest_stores_two_samples_and_reports_the_repeat_as_already_indexed(
    client: TestClient, ingested: dict[str, Response], store: QdrantVectorStore
) -> None:
    bodies = {}
    for name, response in ingested.items():
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "indexed", body
        assert body["chunks_stored"] > 0, body
        assert isinstance(body["document_id"], str)
        assert body["document_id"]
        assert store.document_chunk_count(body["document_id"]) == body["chunks_stored"]
        bodies[name] = body
    assert bodies[MEMO]["document_id"] != bodies[REPORT]["document_id"]
    points_before = store.count()

    again = _post_pdf(client, "/api/ingest", MEMO, request_id="e2e-ingest-again")

    assert again.status_code == 200, again.text
    assert again.headers["X-Request-ID"] == "e2e-ingest-again"
    assert again.json() == {
        "status": "already_indexed",
        "chunks_stored": bodies[MEMO]["chunks_stored"],
        "document_id": bodies[MEMO]["document_id"],
    }
    assert store.count() == points_before


def _ask_client(
    settings: Settings,
    container: Container,
    shared: SharedModels,
    store: QdrantVectorStore,
    llm: FakeLLMClient,
    threshold: float,
) -> TestClient:
    assert shared.embedder is not None
    ask = AskQuestion(
        shared.embedder,
        store,
        llm,
        threshold=threshold,
        top_k=settings.rag_top_k,
        max_chunks=settings.rag_max_chunks,
        max_context_chars=settings.rag_max_context_chars,
    )
    return TestClient(create_app(settings, dataclasses.replace(container, ask=ask)))


def test_an_answerable_question_gets_the_scripted_answer_and_stored_sources(
    settings: Settings,
    container: Container,
    shared: SharedModels,
    store: QdrantVectorStore,
    ingested: dict[str, Response],
) -> None:
    llm = FakeLLMClient(default=SCRIPTED_REPLY)
    with _ask_client(settings, container, shared, store, llm, ANSWER_THRESHOLD) as ask_client:
        response = ask_client.post(
            "/api/ask", json={"question": MEMO_QUESTION}, headers={"X-Request-ID": "e2e-ask-1"}
        )

    assert response.status_code == 200, response.text
    assert response.headers["X-Request-ID"] == "e2e-ask-1"
    body = response.json()
    assert body["answer"] == SCRIPTED_REPLY
    assert body["sources"], body
    assert settings.qdrant_url is not None
    stored = _stored_texts(store, settings.qdrant_url)
    for source in body["sources"]:
        assert any(source in text for text in stored), source
    (messages,) = llm.calls
    assert messages[-1].content.endswith(f"Question: {MEMO_QUESTION}")
    assert "Robert Chen" in messages[-1].content


def test_an_off_topic_question_is_refused_without_calling_the_model(
    settings: Settings,
    container: Container,
    shared: SharedModels,
    store: QdrantVectorStore,
    ingested: dict[str, Response],
) -> None:
    llm = FakeLLMClient(default=SCRIPTED_REPLY)
    threshold = settings.rag_threshold  # the configured default
    with _ask_client(settings, container, shared, store, llm, threshold) as ask_client:
        response = ask_client.post("/api/ask", json={"question": OFF_TOPIC_QUESTION})

    assert response.status_code == 200, response.text
    assert response.json() == {"answer": REFUSAL, "sources": []}
    assert llm.calls == []


# --- Validation through the real stack -------------------------------------------------------


def _png() -> bytes:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), False)
    data: bytes = pixmap.tobytes("png")
    return data


def _pdf_with_pages(count: int) -> bytes:
    with pymupdf.open() as document:
        for index in range(count):
            page = document.new_page()
            page.insert_text((72, 72), f"Page {index + 1}: Maria Gonzalez signed the minutes.")
        data: bytes = document.tobytes()
    return data


@pytest.mark.parametrize("path", ["/api/extract", "/api/ingest"])
def test_a_png_renamed_pdf_is_refused_as_not_a_pdf(client: TestClient, path: str) -> None:
    png = _png()
    assert png.startswith(b"\x89PNG")

    response = _post_pdf(client, path, "scan.pdf", data=png, request_id="e2e-png")

    _assert_error(response, 415, "not_a_pdf", "e2e-png")


def test_a_truncated_pdf_is_unreadable(client: TestClient) -> None:
    whole = _pdf(MEMO)

    response = _post_pdf(client, "/api/extract", "cut.pdf", data=whole[: len(whole) // 3])

    _assert_error(response, 422, "pdf_unreadable", None)


@pytest.mark.parametrize(
    "size", [UPLOAD_LIMIT_MB * MIB + 1, 2 * UPLOAD_LIMIT_MB * MIB], ids=["one-byte-over", "2x"]
)
def test_an_upload_over_the_limit_is_refused_as_too_large(
    limited_client: TestClient, size: int
) -> None:
    header = b"%PDF-1.7\n"
    data = header + b"0" * (size - len(header))

    response = _post_pdf(limited_client, "/api/extract", "big.pdf", data=data, request_id="e2e-big")

    _assert_error(response, 413, "too_large", "e2e-big")


@pytest.mark.parametrize("path", ["/api/extract", "/api/ingest"])
def test_a_pdf_over_the_page_limit_is_refused(limited_client: TestClient, path: str) -> None:
    response = _post_pdf(limited_client, path, "long.pdf", data=_pdf_with_pages(PAGE_LIMIT + 1))

    _assert_error(response, 413, "too_many_pages", None)


def test_a_pdf_at_the_page_limit_is_accepted(limited_client: TestClient) -> None:
    response = _post_pdf(limited_client, "/api/extract", "ok.pdf", data=_pdf_with_pages(PAGE_LIMIT))

    assert response.status_code == 200, response.text


# --- Health and readiness --------------------------------------------------------------------


def test_liveness_and_readiness_with_qdrant_up(client: TestClient) -> None:
    live = client.get("/health", headers={"X-Request-ID": "e2e-live"})
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "ok"}
    assert live.headers["X-Request-ID"] == "e2e-live"
    assert ready.status_code == 200, ready.text
    assert ready.json() == {
        "status": "ready",
        "checks": {"vector_store": "ok", "llm": "not_configured"},
    }


def test_readiness_fails_naming_the_vector_store_when_qdrant_is_unreachable(
    unreachable_container: tuple[Settings, Container],
) -> None:
    with TestClient(create_app(*unreachable_container)) as unreachable:
        begin = time.perf_counter()
        ready = unreachable.get("/health/ready", headers={"X-Request-ID": "e2e-not-ready"})
        elapsed = time.perf_counter() - begin
        live = unreachable.get("/health")

    print(f"\n[readiness unreachable] {ready.status_code} in {elapsed:.2f}s")
    assert ready.status_code == 503, ready.text
    assert ready.json() == {
        "status": "not_ready",
        "checks": {"vector_store": "unavailable", "llm": "not_configured"},
    }
    assert ready.headers["X-Request-ID"] == "e2e-not-ready"
    assert live.status_code == 200
    assert live.json() == {"status": "ok"}


def test_an_unreachable_store_fails_ingestion_with_503(
    unreachable_container: tuple[Settings, Container],
) -> None:
    with TestClient(create_app(*unreachable_container)) as unreachable:
        response = _post_pdf(unreachable, "/api/ingest", MEMO, request_id="e2e-store-down")

    _assert_error(response, 503, "vector_store_unavailable", "e2e-store-down")
    assert "Retry-After" in response.headers


def test_a_request_without_an_id_gets_a_generated_one(client: TestClient) -> None:
    response = client.get("/health")

    request_id = response.headers["X-Request-ID"]
    assert len(request_id) == 32
    int(request_id, 16)


# --- Last: every container shared the models -------------------------------------------------


def test_every_container_shared_one_ner_model_and_one_embedder(
    shared: SharedModels,
    container: Container,
    limited_container: tuple[Settings, Container],
    unreachable_container: tuple[Settings, Container],
) -> None:
    assert shared.builds == {"recognizer": 1, "embedder": 1}
    assert len(shared.stores) == 3
