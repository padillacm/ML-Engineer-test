"""Builders for the HTTP tests: settings, pages, a container of fakes and form bodies."""

import json
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.application.ask_question import AskQuestion
from app.application.extract_names import ExtractNames
from app.application.ingest_document import IngestDocument
from app.application.ports import LLMClient, OcrEngine
from app.bootstrap import Container, ReadinessCheck
from app.domain.cancellation import NEVER_CANCELLED, Cancellation
from app.domain.types import Box, ChatMessage, OcrPage, PageWord
from app.settings import Settings, load_settings
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm import FakeLLMClient
from tests.fakes.names import FakeNameRecognizer
from tests.fakes.ocr import FakeOcrEngine
from tests.fakes.vector_store import FakeVectorStore

MIB = 1024 * 1024
PDF = b"%PDF-1.7\n% a test document\n%%EOF\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

PERSON = "María González"
CONTRACT_PAGE_LINES = (
    "The contract was signed by María González",
    "on 3 March 2024 in Madrid before the notary.",
)


def api_settings(**overrides: Any) -> Settings:
    """Settings for the HTTP tests: no environment or ``.env``, a 1 MiB upload limit."""
    values: dict[str, Any] = {
        "ner_model_path": "unused-ner",
        "embedding_model_path": "unused-e5",
        "embedding_model_revision": "rev-test",
        "qdrant_url": None,
        "qdrant_location": ":memory:",
        "llm_base_url": None,
        "llm_model": None,
        "max_upload_mb": 1,
        "max_names": 5,
        "max_concurrent_jobs": 2,
        "busy_wait_s": 0.2,
        "rag_threshold": 0.3,
        "log_level": "INFO",
        "log_format": "json",
    }
    values.update(overrides)
    return load_settings(_env_file=None, **values)


def make_page(page: int, lines: Sequence[str]) -> OcrPage:
    """A page with one recognised line per entry of ``lines``, words left to right."""
    words: list[PageWord] = []
    for line_index, line in enumerate(lines):
        x = 50.0
        for text in line.split():
            box = Box(page=page, x=x, y=50.0 + 20.0 * line_index, width=6.0 * len(text), height=12)
            words.append(PageWord(text=text, box=box, line=line_index, paragraph=0))
            x += 6.0 * (len(text) + 1)
    return OcrPage(page=page, width=612, height=792, words=tuple(words), language=None)


CONTRACT_PAGES = (make_page(0, CONTRACT_PAGE_LINES),)


@dataclass
class Fakes:
    """The fakes behind one container, kept so tests can inspect and script them."""

    ocr: OcrEngine = field(default_factory=lambda: FakeOcrEngine(CONTRACT_PAGES))
    recognizer: FakeNameRecognizer = field(default_factory=lambda: FakeNameRecognizer([PERSON]))
    embedder: FakeEmbedder = field(default_factory=FakeEmbedder)
    store: FakeVectorStore = field(default_factory=FakeVectorStore)
    llm: LLMClient = field(default_factory=lambda: FakeLLMClient(default="The notary was there."))
    readiness: tuple[tuple[str, ReadinessCheck], ...] = (
        ("vector_store", lambda: "ok"),
        ("llm", lambda: "configured"),
    )
    closed: list[str] = field(default_factory=list)

    def container(self, settings: Settings) -> Container:
        """Real use cases over the fakes, configured from ``settings``."""
        return Container(
            extract=ExtractNames(self.ocr, self.recognizer, settings.match_threshold),
            ingest=IngestDocument(self.ocr, self.embedder, self.store),
            ask=AskQuestion(self.embedder, self.store, self.llm, threshold=settings.rag_threshold),
            readiness=self.readiness,
            llm_configured=True,
            closers=(lambda: self.closed.append("container"),),
        )


def names_json(*pairs: tuple[str, str]) -> str:
    return json.dumps([{"first_name": first, "last_name": last} for first, last in pairs])


def multipart(
    file_bytes: bytes, fields: dict[str, str] | None = None, boundary: str = "testboundary"
) -> tuple[bytes, str]:
    """A ``multipart/form-data`` body with ``pdf_file`` and ``fields``, and its content type."""
    parts: list[bytes] = []
    for name, value in (fields or {}).items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            + value.encode()
            + b"\r\n"
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="pdf_file"; '
        f'filename="doc.pdf"\r\nContent-Type: application/pdf\r\n\r\n'.encode()
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def chunked(body: bytes, size: int = 64 * 1024) -> Callable[[], Any]:
    """A generator factory streaming ``body`` in pieces, so no ``Content-Length`` is sent."""

    def stream() -> Any:
        for start in range(0, len(body), size):
            yield body[start : start + size]

    return stream


BLOCK_S = 10.0
"""Longest a blocking fake waits to be released, so a failing test cannot hang."""


class BlockingOcr:
    """Returns the contract pages, but only once ``release`` is set; counts its callers.

    It ignores ``cancel``, like an engine in the middle of its last page.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.calls = 0

    def recognize(
        self, pdf: bytes, language: str | None = None, *, cancel: Cancellation = NEVER_CANCELLED
    ) -> list[OcrPage]:
        with self._lock:
            self.calls += 1
        self.entered.set()
        assert self.release.wait(BLOCK_S)
        return list(CONTRACT_PAGES)


class PagedOcr:
    """Recognises ``pages`` pages of ``page_s`` seconds each, checking ``cancel`` before each
    page as the real engine does, then returns the contract pages.

    ``recognised`` counts the pages done and ``on_page`` is called after each; ``started`` is
    set when the first page begins and ``finished`` when the call returns or raises.
    """

    def __init__(
        self, pages: int = 10, page_s: float = 0.05, on_page: Callable[[], None] = lambda: None
    ) -> None:
        self.pages = pages
        self.page_s = page_s
        self.on_page = on_page
        self.recognised = 0
        self.started = threading.Event()
        self.finished = threading.Event()

    def recognize(
        self, pdf: bytes, language: str | None = None, *, cancel: Cancellation = NEVER_CANCELLED
    ) -> list[OcrPage]:
        try:
            for _ in range(self.pages):
                cancel.raise_if_cancelled()
                self.started.set()
                time.sleep(self.page_s)
                self.recognised += 1
                self.on_page()
            return list(CONTRACT_PAGES)
        finally:
            self.finished.set()


class BlockingLLM:
    """Answers once ``release`` is set; records how many calls wait at once.

    ``waiting(n)`` blocks until ``n`` calls are waiting (or ``BLOCK_S`` passes).
    """

    def __init__(self, answer: str = "The notary was there.") -> None:
        self.release = threading.Event()
        self._answer = answer
        self._changed = threading.Condition()
        self.in_flight = 0
        self.most_in_flight = 0

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        with self._changed:
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
            self._changed.notify_all()
        try:
            assert self.release.wait(BLOCK_S)
            return self._answer
        finally:
            with self._changed:
                self.in_flight -= 1

    def waiting(self, count: int) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: self.in_flight >= count, BLOCK_S)


class RecordedSpool(tempfile.SpooledTemporaryFile):  # type: ignore[type-arg]
    """The form parser's spool file, recorded so a test can check that it was closed.

    Patch it over ``starlette.formparsers.SpooledTemporaryFile`` and reset ``instances``.
    """

    instances: list["RecordedSpool"] = []  # noqa: RUF012 - shared on purpose, reset per test

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        RecordedSpool.instances.append(self)
