"""In-memory OCR engine returning canned pages."""

from collections.abc import Sequence

from app.domain.types import OcrPage


class FakeOcrEngine:
    """Returns the pages it was given, whatever the PDF, and records every call."""

    def __init__(self, pages: Sequence[OcrPage] = ()) -> None:
        self.pages = list(pages)
        self.calls: list[tuple[bytes, str | None]] = []

    def recognize(self, pdf: bytes, language: str | None = None) -> list[OcrPage]:
        self.calls.append((pdf, language))
        return list(self.pages)
