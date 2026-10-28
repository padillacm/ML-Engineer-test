"""In-memory OCR engine returning canned pages."""

import dataclasses
from collections.abc import Sequence

from app.domain.cancellation import NEVER_CANCELLED, Cancellation
from app.domain.types import OcrPage


class FakeOcrEngine:
    """Returns the pages it was given, whatever the PDF, and records every call.

    With a language hint every page reports that language, as a real engine does; without one
    each page keeps the language it was given (the "detected" one, or ``None``).

    An already cancelled ``cancel`` raises ``request_cancelled`` before anything is returned;
    ``cancels`` records the token of every call.
    """

    def __init__(self, pages: Sequence[OcrPage] = ()) -> None:
        self.pages = list(pages)
        self.calls: list[tuple[bytes, str | None]] = []
        self.cancels: list[Cancellation] = []

    def recognize(
        self,
        pdf: bytes,
        language: str | None = None,
        *,
        cancel: Cancellation = NEVER_CANCELLED,
    ) -> list[OcrPage]:
        self.calls.append((pdf, language))
        self.cancels.append(cancel)
        cancel.raise_if_cancelled()
        if language is None:
            return list(self.pages)
        return [dataclasses.replace(page, language=language) for page in self.pages]
