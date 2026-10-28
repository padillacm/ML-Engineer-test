"""Text-line indices of recognised words with the real Tesseract (``page-ocr`` spec)."""

import pytest

from app.adapters.ocr import TesseractOcrEngine
from app.domain.types import OcrPage
from tests.adapters.ocr.pdfs import A4, PageSpec, image_only_pdf
from tests.integration.known_words import normalise, recognised_text

pytestmark = pytest.mark.tesseract


def _line_of(page: OcrPage, word: str) -> int:
    (line,) = {found.line for found in page.words if normalise(found.text) == word}
    return line


def test_two_printed_lines_get_different_indices_in_reading_order() -> None:
    document = image_only_pdf([PageSpec(size=A4, lines=("Maria Gonzalez", "Director"))], dpi=300)

    (page,) = TesseractOcrEngine().recognize(document, language="en")

    print(f"recognised: {[(word.text, word.line) for word in page.words]}")
    assert [normalise(word.text) for word in page.words] == ["Maria", "Gonzalez", "Director"]
    assert _line_of(page, "Maria") == _line_of(page, "Gonzalez") == 0
    assert _line_of(page, "Director") > _line_of(page, "Gonzalez")


def test_line_indices_restart_on_every_page() -> None:
    document = image_only_pdf(
        [
            PageSpec(size=A4, lines=("Quarterly review", "signed by the board")),
            PageSpec(size=A4, lines=("Annual budget", "approved in March")),
        ],
        dpi=300,
    )

    pages = TesseractOcrEngine().recognize(document, language="en")

    assert [page.page for page in pages] == [0, 1]
    for page in pages:
        assert page.words, recognised_text(page)
        lines = [word.line for word in page.words]
        assert lines[0] == 0, recognised_text(page)
        assert lines == sorted(lines)
        assert set(lines) == set(range(max(lines) + 1))
