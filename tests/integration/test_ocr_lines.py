"""Line and paragraph indices of recognised words with the real Tesseract (``page-ocr`` spec)."""

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


def _paragraph_of(page: OcrPage, word: str) -> int:
    (paragraph,) = {found.paragraph for found in page.words if normalise(found.text) == word}
    return paragraph


def test_a_heading_and_a_paragraph_below_a_gap_get_different_paragraph_indices() -> None:
    document = image_only_pdf(
        [
            PageSpec(
                size=A4,
                lines=("Budget", "", "", "The board approved the budget", "for the new office."),
            )
        ],
        dpi=300,
    )

    (page,) = TesseractOcrEngine().recognize(document, language="en")

    print(f"recognised: {[(word.text, word.line, word.paragraph) for word in page.words]}")
    assert _paragraph_of(page, "Budget") == 0
    assert _paragraph_of(page, "board") == _paragraph_of(page, "office") > 0
    assert _line_of(page, "office") > _line_of(page, "board")


def test_paragraph_indices_restart_on_every_page() -> None:
    lines = ("Summary", "", "", "The plan was signed", "by the whole board.")
    document = image_only_pdf([PageSpec(size=A4, lines=lines)] * 2, dpi=300)

    pages = TesseractOcrEngine().recognize(document, language="en")

    assert [page.page for page in pages] == [0, 1]
    for page in pages:
        paragraphs = [word.paragraph for word in page.words]
        assert paragraphs, recognised_text(page)
        assert paragraphs[0] == 0, recognised_text(page)
        assert paragraphs == sorted(paragraphs)
        assert set(paragraphs) == set(range(max(paragraphs) + 1))
        assert max(paragraphs) >= 1, recognised_text(page)
