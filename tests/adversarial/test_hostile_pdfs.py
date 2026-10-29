"""Hostile PDFs against the OCR bounds (spec page-ocr: ``A page render has a time budget``,
``Page text is limited``; the page-count and pixel-budget rules of the OCR adapter; the
upload checks of the HTTP edge).

Each generator of ``hostile_pdfs`` is run through the real OCR adapter with a stand-in for the
Tesseract process (the bounds act before any text is read), with a small render budget, and
the test asserts the expected code and how long the refusal took. The same files go through
the HTTP application, where the edge adds its own checks (``not_a_pdf``, the body limit). The
page-text cap needs recognised text, so that test runs the real Tesseract (``tesseract``).
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import pypdfium2.raw as pdfium_c
import pytest

from app.adapters.ocr import TesseractOcrConfig, TesseractOcrEngine
from app.api.errors import STATUS_BY_CODE
from app.domain.errors import InvalidInput
from tests.adapters.ocr.fake_runner import FakeTesseract
from tests.adapters.ocr.pdfs import A4, ENGLISH_LINES, PageSpec, build_pdf
from tests.adversarial import hostile_pdfs as hostile
from tests.api.conftest import ClientFactory
from tests.api.support import Fakes

TINY_BUDGET_S = 0.05
STEP_SLACK_S = 0.5
"""One progress step of PDFium (it polls every hundred objects) plus slack for a slow host."""
INFLATE_SLACK_S = 3.0
"""Inflating and parsing a content stream happens when the page is loaded, before the first
pause poll, so it is outside the budget: about a second for 64 MB on the measuring machine."""
QUICK_S = 1.0
"""A refusal that comes from dictionary lookups, never from drawing or decoding."""
MAX_PAGES = 50


def _engine(**options: object) -> TesseractOcrEngine:
    config = TesseractOcrConfig(dpi=200, workers=1, render_timeout_s=TINY_BUDGET_S, **options)  # type: ignore[arg-type]
    return TesseractOcrEngine(config, runner=FakeTesseract())


def _refusal(engine: TesseractOcrEngine, pdf: bytes) -> tuple[InvalidInput, float]:
    """The ``InvalidInput`` recognising ``pdf`` raises, and the seconds it took."""
    started = time.perf_counter()
    with pytest.raises(InvalidInput) as caught:
        engine.recognize(pdf, "en")
    return caught.value, time.perf_counter() - started


def _no_drawing(*args: object) -> None:
    raise AssertionError("the page was drawn despite the pre-check")


# --- through the OCR adapter ----------------------------------------------------------------


def test_thousands_of_large_images_are_refused_by_the_image_budget_without_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdfium_c, "FPDF_RenderPageBitmap_Start", _no_drawing)

    error, elapsed = _refusal(_engine(), hostile.many_images_pdf())

    assert error.code == "too_large"
    assert error.message == "page 0 holds more than 80000000 decoded image pixels"
    assert elapsed < QUICK_S


def test_thousands_of_large_images_stop_at_the_render_budget_when_the_pixel_budget_is_off() -> None:
    engine = _engine(max_image_pixels=10**15)

    error, elapsed = _refusal(engine, hostile.many_images_pdf())

    assert error.code == "pdf_unreadable"
    assert error.message == f"page 0 exceeded the render budget of {TINY_BUDGET_S} s"
    assert elapsed < TINY_BUDGET_S + STEP_SLACK_S


def test_a_million_tiny_paths_stop_at_the_render_budget() -> None:
    error, elapsed = _refusal(_engine(), hostile.many_paths_pdf())
    print(f"a million paths stopped after {elapsed:.2f} s (budget {TINY_BUDGET_S} s)")

    assert error.code == "pdf_unreadable"
    assert "exceeded the render budget" in error.message
    assert elapsed < TINY_BUDGET_S + INFLATE_SLACK_S  # the stream is inflated before any poll


def test_a_declared_million_pages_over_more_real_pages_than_the_limit_is_too_many_pages() -> None:
    pdf = hostile.declared_million_pages_pdf(real_pages=MAX_PAGES + 10)

    error, elapsed = _refusal(_engine(max_pages=MAX_PAGES), pdf)

    assert error.code == "too_many_pages"
    assert (
        error.message
        == f"the PDF has more than {MAX_PAGES} pages; at most {MAX_PAGES} are accepted"
    )
    assert elapsed < QUICK_S  # one lookup of page index 50, never a million


def test_a_declared_million_pages_over_one_real_page_is_that_one_blank_page() -> None:
    started = time.perf_counter()
    pages = _engine(max_pages=MAX_PAGES).recognize(hostile.declared_million_pages_pdf(), "en")
    elapsed = time.perf_counter() - started

    assert len(pages) == 1
    assert pages[0].words == ()
    assert elapsed < QUICK_S


def test_a_million_kids_in_the_page_tree_are_refused_after_one_lookup() -> None:
    pdf = hostile.million_kids_pdf()
    assert len(pdf) > 6_000_000  # six bytes per reference

    error, elapsed = _refusal(_engine(max_pages=MAX_PAGES), pdf)

    assert error.code == "too_many_pages"
    assert elapsed < QUICK_S


def test_a_hundred_nested_forms_are_refused_before_anything_is_drawn() -> None:
    # PDFium would draw nothing beyond its own nesting limit, but the pre-check refuses the
    # page first: a form nested deeper than the poll depth is never drawn under the budget.
    error, elapsed = _refusal(_engine(), hostile.nested_forms_pdf(depth=100, strokes=1000))

    assert error.code == "pdf_unreadable"
    assert error.message == "page 0 nests form XObjects deeper than 8 levels"
    assert elapsed < TINY_BUDGET_S + STEP_SLACK_S


def test_a_form_at_the_poll_depth_is_stopped_by_the_render_budget() -> None:
    pdf = hostile.nested_forms_pdf(depth=hostile.NESTED_FORM_POLL_DEPTH, strokes=50_000)

    error, elapsed = _refusal(_engine(), pdf)

    assert error.code == "pdf_unreadable"
    assert elapsed < TINY_BUDGET_S + STEP_SLACK_S


def test_a_form_nested_beyond_the_poll_depth_is_stopped_by_the_render_budget() -> None:
    # PDFium would draw such a form as one operation, outside the budget: the pre-check
    # refuses the page instead, before anything is drawn.
    pdf = hostile.nested_forms_pdf(depth=hostile.NESTED_FORM_POLL_DEPTH + 1, strokes=50_000)

    error, elapsed = _refusal(_engine(), pdf)
    print(f"a form at depth {hostile.NESTED_FORM_POLL_DEPTH + 1} stopped after {elapsed:.2f} s")

    assert error.code == "pdf_unreadable"
    assert error.message == "page 0 nests form XObjects deeper than 8 levels"
    assert elapsed < TINY_BUDGET_S + STEP_SLACK_S


def test_a_huge_mediabox_is_refused_by_the_pixel_budget_before_anything_is_drawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdfium_c, "FPDF_RenderPageBitmap_Start", _no_drawing)

    error, elapsed = _refusal(_engine(), hostile.huge_mediabox_pdf())

    assert error.code == "too_large"
    assert error.message == "page 0 would render to more than 40000000 pixels at 200 DPI"
    assert elapsed < QUICK_S


def test_a_compression_bomb_stops_at_the_render_budget_once_inflated() -> None:
    pdf = hostile.compression_bomb_pdf()
    assert len(pdf) < 100_000

    error, elapsed = _refusal(_engine(), pdf)
    print(f"the compression bomb stopped after {elapsed:.2f} s (budget {TINY_BUDGET_S} s)")

    assert error.code == "pdf_unreadable"
    assert "exceeded the render budget" in error.message
    assert elapsed < TINY_BUDGET_S + INFLATE_SLACK_S


def test_an_encrypted_pdf_is_unreadable() -> None:
    error, elapsed = _refusal(_engine(), hostile.encrypted_pdf())

    assert error.code == "pdf_unreadable"
    assert error.message == "the PDF is password protected"
    assert elapsed < QUICK_S


def test_a_truncated_pdf_is_unreadable() -> None:
    error, elapsed = _refusal(_engine(), hostile.truncated_pdf())

    assert error.code == "pdf_unreadable"
    assert error.message == "the file cannot be opened as a PDF"
    assert elapsed < QUICK_S


# --- through the HTTP application -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EdgeCase:
    name: str
    build: Callable[[], bytes]
    code: str
    bound_s: float


EDGE_CASES = [
    EdgeCase("headerless", hostile.headerless_bytes, "not_a_pdf", QUICK_S),
    EdgeCase("truncated", hostile.truncated_pdf, "pdf_unreadable", QUICK_S),
    EdgeCase("encrypted", hostile.encrypted_pdf, "pdf_unreadable", QUICK_S),
    EdgeCase("huge-mediabox", hostile.huge_mediabox_pdf, "too_large", QUICK_S),
    EdgeCase(
        "declared-million-pages",
        lambda: hostile.declared_million_pages_pdf(real_pages=MAX_PAGES + 10),
        "too_many_pages",
        QUICK_S,
    ),
    # Six megabytes: the body limit (1 MiB in the test settings) refuses it before PDFium.
    EdgeCase("million-kids", hostile.million_kids_pdf, "too_large", QUICK_S),
    EdgeCase("many-images", hostile.many_images_pdf, "too_large", QUICK_S),
    EdgeCase(
        "many-paths", hostile.many_paths_pdf, "pdf_unreadable", TINY_BUDGET_S + INFLATE_SLACK_S
    ),
    EdgeCase(
        "compression-bomb",
        hostile.compression_bomb_pdf,
        "pdf_unreadable",
        TINY_BUDGET_S + INFLATE_SLACK_S,
    ),
]


@pytest.mark.parametrize("case", EDGE_CASES, ids=lambda case: case.name)
def test_hostile_uploads_are_refused_with_the_expected_code_within_the_bound(
    case: EdgeCase, make_client: ClientFactory, fakes: Fakes
) -> None:
    fakes.ocr = _engine(max_pages=MAX_PAGES)
    client = make_client()
    pdf = case.build()

    started = time.perf_counter()
    response = client.post(
        "/api/extract",
        files={"pdf_file": ("a.pdf", pdf, "application/pdf")},
        headers={"X-Request-ID": case.name},
    )
    elapsed = time.perf_counter() - started

    assert response.status_code == STATUS_BY_CODE[case.code]
    body = response.json()
    assert (body["code"], body["request_id"]) == (case.code, case.name)
    assert elapsed < case.bound_s
    assert client.post("/api/ask", json={"question": "who signed?"}).status_code == 200


@pytest.mark.tesseract
def test_a_page_with_more_text_than_the_cap_is_413_too_large_once_read(
    make_client: ClientFactory, fakes: Fakes
) -> None:
    # Fifteen lines of English on an A4 page: a few hundred characters, over a cap of 200.
    fakes.ocr = TesseractOcrEngine(TesseractOcrConfig(max_page_chars=200))
    client = make_client()
    pdf = build_pdf([PageSpec(size=A4, lines=ENGLISH_LINES * 3)])

    response = client.post("/api/extract", files={"pdf_file": ("a.pdf", pdf, "application/pdf")})

    assert response.status_code == 413
    assert response.json()["code"] == "too_large"
    assert response.json()["detail"] == "page 0 has more text than the limit of 200 characters"
