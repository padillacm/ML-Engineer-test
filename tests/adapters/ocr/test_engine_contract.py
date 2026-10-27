"""The real Tesseract adapter against the ``OcrEngine`` contract (runs where Tesseract exists)."""

import pytest

from app.adapters.ocr import TesseractOcrEngine
from app.application.ports import OcrEngine
from app.domain.types import OcrPage
from tests.adapters.ocr.pdfs import A4, ENGLISH_LINES, PageSpec, image_only_pdf
from tests.application.contracts import OcrEngineContract


@pytest.mark.tesseract
class TestTesseractOcrEngineContract(OcrEngineContract):
    @pytest.fixture
    def engine(self) -> OcrEngine:
        return TesseractOcrEngine()

    @pytest.fixture
    def sample_pdf(self) -> bytes:
        return image_only_pdf(
            [
                PageSpec(size=A4, lines=ENGLISH_LINES),
                PageSpec(size=A4, lines=()),
                PageSpec(size=A4, lines=ENGLISH_LINES[:2], rotation=90),
            ]
        )

    @pytest.fixture
    def expected_pages(self) -> int:
        return 3

    def test_blank_page_has_no_words(self, pages: list[OcrPage]) -> None:
        assert pages[1].words == ()
        assert (pages[1].width, pages[1].height) == A4
