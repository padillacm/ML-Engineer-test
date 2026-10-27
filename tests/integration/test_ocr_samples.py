"""The scanned sample PDFs through the real Tesseract: every page, in order, boxes on the page."""

from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest

from app.adapters.ocr import TesseractOcrConfig, TesseractOcrEngine
from app.domain.types import OcrPage

pytestmark = pytest.mark.tesseract

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "sample_pdfs"
SAMPLES = sorted(path.name for path in SAMPLES_DIR.glob("*.pdf"))
HINTS = [pytest.param(None, id="no-hint"), pytest.param("en", id="hint-en")]

type Recognize = Callable[[str, str | None, int], list[OcrPage]]


def test_the_three_samples_are_present() -> None:
    assert SAMPLES == ["company_memo.pdf", "meeting_minutes.pdf", "research_report.pdf"]


@pytest.fixture(scope="module")
def recognize() -> Recognize:
    """Recognise a sample with a hint and a worker count, once per combination per module."""
    cache: dict[tuple[str, str | None, int], list[OcrPage]] = {}

    def run(name: str, language: str | None, workers: int) -> list[OcrPage]:
        key = (name, language, workers)
        if key not in cache:
            engine = TesseractOcrEngine(TesseractOcrConfig(workers=workers))
            cache[key] = engine.recognize((SAMPLES_DIR / name).read_bytes(), language)
        return cache[key]

    return run


def _page_sizes(name: str) -> list[tuple[float, float]]:
    with pymupdf.open(SAMPLES_DIR / name) as document:
        return [(float(page.rect.width), float(page.rect.height)) for page in document]


@pytest.mark.parametrize("language", HINTS)
@pytest.mark.parametrize("name", SAMPLES)
def test_every_page_is_recognised_in_order_with_boxes_on_the_page(
    recognize: Recognize, name: str, language: str | None
) -> None:
    sizes = _page_sizes(name)

    pages = recognize(name, language, 4)

    assert [page.page for page in pages] == list(range(len(sizes)))
    for page, size in zip(pages, sizes, strict=True):
        assert (page.width, page.height) == pytest.approx(size)
    assert pages[0].words, "the first page must contain recognised words"
    for page in pages:
        for word in page.words:
            box = word.box
            assert box.page == page.page
            assert box.x >= 0, word
            assert box.y >= 0, word
            assert box.right <= page.width + 1e-6, word
            assert box.bottom <= page.height + 1e-6, word
    print(f"{name} language={language}: words per page {[len(page.words) for page in pages]}")


@pytest.mark.parametrize("name", SAMPLES)
def test_one_and_four_workers_give_identical_results(recognize: Recognize, name: str) -> None:
    assert recognize(name, "en", 1) == recognize(name, "en", 4)
