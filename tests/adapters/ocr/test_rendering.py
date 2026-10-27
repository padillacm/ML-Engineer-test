import re
from pathlib import Path

import numpy as np
import pymupdf
import pypdfium2 as pdfium
import pytest

from app.adapters.ocr.rendering import RenderedPage, encode_png, open_pdf, render_pages
from app.domain.errors import InvalidInput
from tests.adapters.ocr.fake_runner import decode
from tests.adapters.ocr.pdfs import (
    A4,
    ENGLISH_LINES,
    LETTER,
    PageSpec,
    build_pdf,
    encrypted_pdf,
    image_only_pdf,
)
from tests.architecture.scan import collect_imports

APP = Path(__file__).resolve().parents[3] / "app"
ZERO_PAGE_PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
EMPTY_TREE_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF"
)
# A page tree whose second kid is the integer 42 instead of a page dictionary.
NON_PAGE_KID_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >> endobj
4 0 obj 42 endobj
trailer << /Root 1 0 R >>
%%EOF"""
MISSING_KID_PDF = NON_PAGE_KID_PDF.replace(b"4 0 obj 42 endobj\n", b"")


def _render(pdf: bytes, dpi: int, max_page_pixels: int = 40_000_000) -> list[RenderedPage]:
    with open_pdf(pdf) as document:
        return list(render_pages(document, dpi, max_page_pixels))


@pytest.mark.parametrize(
    ("size", "dpi", "pixels"),
    [(A4, 72, (595, 842)), (LETTER, 200, (1700, 2200)), (A4, 144, (1190, 1684))],
)
def test_pages_render_in_grayscale_at_the_requested_dpi(
    size: tuple[float, float], dpi: int, pixels: tuple[int, int]
) -> None:
    pdf = build_pdf([PageSpec(size=size), PageSpec(size=size, lines=())])

    rendered = _render(pdf, dpi)

    assert [page.index for page in rendered] == [0, 1]
    for page in rendered:
        image = decode(page.png)
        assert (image.shape[1], image.shape[0]) == pixels
        assert (page.raster.width_px, page.raster.height_px) == pixels
        assert (page.raster.page_width_pt, page.raster.page_height_pt) == size
    assert decode(rendered[0].png).min() < 128
    assert decode(rendered[1].png).min() == 255


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_page_is_rendered_and_measured_in_its_displayed_orientation(
    rotation: int,
) -> None:
    pdf = build_pdf([PageSpec(size=A4, rotation=rotation)])
    displayed = A4 if rotation == 180 else (A4[1], A4[0])

    (page,) = _render(pdf, 72)

    assert (page.raster.page_width_pt, page.raster.page_height_pt) == displayed
    assert decode(page.png).shape == (round(displayed[1]), round(displayed[0]))


def test_rotated_page_shows_the_text_where_it_is_displayed() -> None:
    pdf = build_pdf([PageSpec(size=A4, rotation=90)])
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        expected = document[0].get_pixmap(dpi=72, colorspace=pymupdf.csGRAY, alpha=False)
        reference = np.frombuffer(expected.samples, np.uint8).reshape(
            expected.height, expected.width
        )

    (page,) = _render(pdf, 72)

    ink = decode(page.png) < 128
    reference_ink = reference < 128
    assert ink.any()
    overlap = (ink & reference_ink).sum() / (ink | reference_ink).sum()
    assert overlap > 0.5


def test_crop_box_is_the_displayed_page() -> None:
    with pymupdf.open(stream=build_pdf([PageSpec(size=A4)]), filetype="pdf") as document:
        document[0].set_cropbox(pymupdf.Rect(10, 20, 310, 420))
        pdf = document.tobytes()

    (page,) = _render(pdf, 72)

    assert (page.raster.page_width_pt, page.raster.page_height_pt) == (300.0, 400.0)
    assert decode(page.png).shape == (400, 300)
    # The text drawn at (20, 30) on the media box shows near (10, 10) of the crop box.
    assert decode(page.png)[:40, :60].min() < 128


def test_transform_is_applied_to_the_gray_pixels_before_encoding() -> None:
    pdf = build_pdf([PageSpec()])
    with open_pdf(pdf) as document:
        (plain,) = render_pages(document, 72, 40_000_000)
        (inverted,) = render_pages(document, 72, 40_000_000, lambda image: 255 - image)

    assert np.array_equal(decode(inverted.png), 255 - decode(plain.png))


def test_encode_png_round_trips_a_gray_image() -> None:
    image = np.arange(60, dtype=np.uint8).reshape(6, 10)

    assert np.array_equal(decode(encode_png(image)), image)


@pytest.mark.parametrize(
    "pdf",
    [
        b"",
        b"%PDF-1.7\n corrupt garbage",
        b"not a pdf at all",
        ZERO_PAGE_PDF,
        EMPTY_TREE_PDF,
        NON_PAGE_KID_PDF,
        MISSING_KID_PDF,
    ],
    ids=["empty", "corrupt", "not-pdf", "no-pages", "empty-page-tree", "non-page-kid", "no-kid"],
)
def test_unreadable_bytes_are_rejected(pdf: bytes) -> None:
    with pytest.raises(InvalidInput) as caught, open_pdf(pdf):
        pass

    assert caught.value.code == "pdf_unreadable"


@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.6, 0.9, 0.99])
@pytest.mark.parametrize("kind", ["image-only", "vector"])
def test_truncated_pdf_is_unreadable_instead_of_blank_pages(fraction: float, kind: str) -> None:
    pages = [PageSpec(size=A4, lines=ENGLISH_LINES)] * 3
    pdf = image_only_pdf(pages, dpi=72) if kind == "image-only" else build_pdf(pages)

    with pytest.raises(InvalidInput) as caught:
        _render(pdf[: int(len(pdf) * fraction)], 72)

    assert caught.value.code == "pdf_unreadable"


def test_broken_xref_offsets_with_intact_objects_are_still_read() -> None:
    pdf = image_only_pdf([PageSpec(size=A4, lines=ENGLISH_LINES)] * 2, dpi=72)
    shifted = re.sub(
        rb"(\d{10}) 00000 n", lambda match: b"%010d 00000 n" % (int(match[1]) + 7), pdf
    )
    assert shifted != pdf

    assert [decode(page.png).tolist() for page in _render(shifted, 72)] == [
        decode(page.png).tolist() for page in _render(pdf, 72)
    ]


def test_encrypted_pdf_is_rejected_as_password_protected() -> None:
    with pytest.raises(InvalidInput) as caught, open_pdf(encrypted_pdf()):
        pass

    assert caught.value.code == "pdf_unreadable"
    assert "password" in caught.value.message
    assert isinstance(caught.value.__cause__, pdfium.PdfiumError)


def test_document_with_a_damaged_page_is_closed(
    opened_documents: list[pdfium.PdfDocument],
) -> None:
    with pytest.raises(InvalidInput), open_pdf(NON_PAGE_KID_PDF):
        pass

    assert len(opened_documents) == 1
    assert opened_documents[0].raw is None


def test_document_is_closed_when_the_block_fails(
    opened_documents: list[pdfium.PdfDocument],
) -> None:
    with pytest.raises(LookupError), open_pdf(build_pdf([PageSpec()])):
        raise LookupError

    assert len(opened_documents) == 1
    assert opened_documents[0].raw is None


def test_oversized_page_is_rejected_before_any_page_is_rendered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = build_pdf([PageSpec(), PageSpec(size=(10_000.0, 10_000.0))])

    def no_rendering(*args: object, **kwargs: object) -> None:
        raise AssertionError("rendered before the budget check")

    monkeypatch.setattr(pdfium.PdfPage, "render", no_rendering)

    with pytest.raises(InvalidInput) as caught, open_pdf(pdf) as document:
        next(render_pages(document, 200, 40_000_000))

    assert caught.value.code == "too_large"
    assert "page 1" in caught.value.message


def test_page_exactly_at_the_budget_is_rendered() -> None:
    pdf = build_pdf([PageSpec(size=(100.0, 50.0))])

    (page,) = _render(pdf, 72, max_page_pixels=5_000)

    assert decode(page.png).shape == (50, 100)


def test_render_failure_is_reported_as_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise pdfium.PdfiumError("content stream is damaged")

    monkeypatch.setattr(pdfium.PdfPage, "render", broken)

    with pytest.raises(InvalidInput) as caught:
        _render(build_pdf([PageSpec()]), 72)

    assert caught.value.code == "pdf_unreadable"
    assert "damaged" not in str(caught.value)
    assert isinstance(caught.value.__cause__, pdfium.PdfiumError)


def test_production_code_never_imports_pymupdf() -> None:
    """PyMuPDF is AGPL-3.0: it may build test PDFs but must never ship in the service."""
    imports = collect_imports(APP, "app")

    offending = [
        str(found) for found in imports if found.module.split(".")[0] in {"pymupdf", "fitz"}
    ]

    assert imports
    assert offending == []
