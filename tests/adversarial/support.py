"""Helpers of the adversarial suite: fakes with a detector, documents, outcome lines, reports."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import ModuleType

import pytest

from app.application.ask_question import AskQuestion
from app.application.extract_names import ExtractNames
from app.application.ingest_document import IngestDocument, document_id_for
from app.application.ports import InjectionDetector
from app.bootstrap import Container
from app.domain.guardrails import DEFAULT_INJECTION_THRESHOLD
from app.domain.types import OcrPage
from app.settings import Settings
from tests.adversarial.cases import ALL_MARKERS, LANGUAGES, Benign, Case
from tests.api.support import Fakes, make_page
from tests.fakes.detector import FakeInjectionDetector

CONTRACT_LINES = ("The contract was signed by Maria Gonzalez", "on 3 March 2024 in Madrid.")
CONTRACT_TEXT = " ".join(CONTRACT_LINES)
GROUNDED_REPLY = "Maria Gonzalez signed it."
"""A model reply whose every content token is in the contract page (support share 1.0)."""
QUESTION = "Who signed the contract?"
THRESHOLD = DEFAULT_INJECTION_THRESHOLD


def marker_detector(*extra: str) -> FakeInjectionDetector:
    """A fake detector flagging the default phrases, every case marker and ``extra``."""
    return FakeInjectionDetector((*ALL_MARKERS, *extra))


@dataclass
class GuardedFakes(Fakes):
    """``Fakes`` whose ingestion and answering use cases screen with ``detector``."""

    detector: FakeInjectionDetector = field(default_factory=marker_detector)

    def container(self, settings: Settings) -> Container:
        return Container(
            extract=ExtractNames(self.ocr, self.recognizer, settings.match_threshold),
            ingest=IngestDocument(
                self.ocr,
                self.embedder,
                self.store,
                detector=self.detector,
                injection_threshold=settings.injection_threshold,
            ),
            ask=AskQuestion(
                self.embedder,
                self.store,
                self.llm,
                threshold=settings.rag_threshold,
                detector=self.detector,
                injection_threshold=settings.injection_threshold,
                support_min_share=settings.rag_support_min_share,
            ),
            readiness=self.readiness,
            llm_configured=True,
            closers=(lambda: self.closed.append("container"),),
        )


def page_form(text: str) -> str:
    """``text`` as a page built with ``make_page`` holds it: words separated by one space."""
    return " ".join(text.split())


def injected_document(sentence: str) -> list[OcrPage]:
    """Three pages: the contract, a payment page ending with ``sentence``, a closing page."""
    return [
        make_page(0, CONTRACT_LINES),
        make_page(1, ("Payment is due within thirty days.", sentence)),
        make_page(2, ("Both parties keep one signed copy", "of the agreement.")),
    ]


def pdf_bytes(label: str) -> bytes:
    """Distinct PDF-looking bytes per ``label`` (the fake OCR engine ignores them)."""
    return b"%PDF-1.7 " + label.encode("utf-8")


def document_id_of(pdf: bytes) -> str:
    return document_id_for(hashlib.sha256(pdf).hexdigest())


def outcome_fields(caplog: pytest.LogCaptureFixture, module: ModuleType) -> dict[str, object]:
    """The fields of the one outcome line ``module`` logged (``OUTCOME_MESSAGE``)."""
    records = [
        record
        for record in caplog.records
        if record.name == module.__name__ and record.getMessage() == module.OUTCOME_MESSAGE
    ]
    assert len(records) == 1, [record.getMessage() for record in caplog.records]
    return dict(records[0].__dict__)


@dataclass(frozen=True, slots=True)
class FlagReport:
    """How many texts of each language a detector flagged, and which it got wrong."""

    flagged: dict[str, int]
    total: dict[str, int]
    wrong: tuple[str, ...]

    def summary(self, label: str) -> str:
        counts = ", ".join(
            f"{language} {self.flagged[language]}/{self.total[language]}"
            for language in LANGUAGES
            if self.total[language]
        )
        wrong = "; ".join(self.wrong) if self.wrong else "none"
        return f"{label}: flagged {counts}; unexpected: {wrong}"


def flag_report(
    detector: InjectionDetector, items: Sequence[Case | Benign], *, injections: bool
) -> FlagReport:
    """Score ``items`` in one call and count, per language, those at or above the threshold.

    ``wrong`` lists the injections that were not flagged (``injections=True``) or the benign
    texts that were (``injections=False``), with their scores.
    """
    scores = detector.screen([item.text for item in items])
    flagged = dict.fromkeys(LANGUAGES, 0)
    total = dict.fromkeys(LANGUAGES, 0)
    wrong: list[str] = []
    for item, score in zip(items, scores, strict=True):
        total[item.language] += 1
        hit = score >= THRESHOLD
        flagged[item.language] += hit
        if hit != injections:
            wrong.append(f"{item.id} ({score:.3f})")
    return FlagReport(flagged, total, tuple(wrong))
