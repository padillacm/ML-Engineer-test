import re
import time
import zlib
from pathlib import Path

import numpy as np
import pymupdf
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest

from app.adapters.ocr.rendering import (
    PDFIUM_LOCK,
    RenderedPage,
    encode_png,
    open_pdf,
    render_pages,
)
from app.domain.errors import InvalidInput
from tests.adapters.ocr.fake_runner import decode
from tests.adapters.ocr.pdfs import (
    A4,
    ENGLISH_LINES,
    LETTER,
    PAGE_DICT,
    SMALL,
    PageSpec,
    build_pdf,
    cropped_pdf,
    declare_page_count,
    encrypted_pdf,
    image_only_pdf,
    inherited_boxes_pdf,
    page_tree_pdf,
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
# Three kids, the middle one the integer 42: page 1 does not load while page 2 does.
MIDDLE_DAMAGED_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >> endobj
4 0 obj 42 endobj
5 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] >> endobj
trailer << /Root 1 0 R >>
%%EOF"""
MIDDLE_MISSING_PDF = MIDDLE_DAMAGED_PDF.replace(b"4 0 obj 42 endobj\n", b"")
# The class itself: the ``opened_documents`` fixture replaces ``pdfium.PdfDocument`` by a spy.
PDF_DOCUMENT_CLASS = pdfium.PdfDocument


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
        MIDDLE_DAMAGED_PDF,
        MIDDLE_MISSING_PDF,
    ],
    ids=[
        "empty",
        "corrupt",
        "not-pdf",
        "no-pages",
        "empty-page-tree",
        "non-page-kid-in-the-middle",
        "no-kid-in-the-middle",
    ],
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
    with pytest.raises(InvalidInput) as caught, open_pdf(MIDDLE_DAMAGED_PDF):
        pass

    assert caught.value.message == "page 1 of the PDF is missing or damaged"
    assert len(opened_documents) == 1
    assert opened_documents[0].raw is None


def test_document_is_closed_when_the_block_fails(
    opened_documents: list[pdfium.PdfDocument],
) -> None:
    with pytest.raises(LookupError), open_pdf(build_pdf([PageSpec()])):
        raise LookupError

    assert len(opened_documents) == 1
    assert opened_documents[0].raw is None


@pytest.fixture
def loaded_pages(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The index of every page PDFium is asked to load (``FPDF_LoadPage``), in order."""
    loaded: list[int] = []
    real_getitem = PDF_DOCUMENT_CLASS.__getitem__

    def spy(document: pdfium.PdfDocument, index: int) -> pdfium.PdfPage:
        loaded.append(index)
        return real_getitem(document, index)

    monkeypatch.setattr(PDF_DOCUMENT_CLASS, "__getitem__", spy)
    return loaded


@pytest.fixture
def looked_up_pages(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The index of every page whose size is looked up without loading it, in order."""
    looked_up: list[int] = []
    real_get_page_size = PDF_DOCUMENT_CLASS.get_page_size

    def spy(document: pdfium.PdfDocument, index: int) -> object:
        looked_up.append(index)
        return real_get_page_size(document, index)

    monkeypatch.setattr(PDF_DOCUMENT_CLASS, "get_page_size", spy)
    return looked_up


def test_too_many_pages_is_rejected_after_checking_only_the_first_page_over_the_limit(
    opened_documents: list[pdfium.PdfDocument],
    looked_up_pages: list[int],
    loaded_pages: list[int],
) -> None:
    pdf = build_pdf([PageSpec(lines=())] * 3)

    with pytest.raises(InvalidInput) as caught, open_pdf(pdf, max_pages=2):
        pass

    assert caught.value.code == "too_many_pages"
    assert caught.value.message == "the PDF has more than 2 pages; at most 2 are accepted"
    assert looked_up_pages == [2]
    assert loaded_pages == []
    assert len(opened_documents) == 1
    assert opened_documents[0].raw is None


@pytest.mark.parametrize(
    ("declared", "checks"), [(2, [1, 0]), (51, [1, 50, 0]), (1_000_000, [1, 999_999, 0])]
)
def test_overstated_page_count_is_the_number_of_pages_that_exist(
    declared: int,
    checks: list[int],
    opened_documents: list[pdfium.PdfDocument],
    looked_up_pages: list[int],
    loaded_pages: list[int],
) -> None:
    pdf = declare_page_count(build_pdf([PageSpec(size=SMALL)]), declared)
    with PDFIUM_LOCK:
        probe = PDF_DOCUMENT_CLASS(pdf)
        assert len(probe) == declared  # PDFium trusts the declared count
        probe.close()

    with open_pdf(pdf, max_pages=1) as document:
        assert document.page_count == 1
        assert document.page_sizes == (SMALL,)
        assert loaded_pages == []  # verifying the count loads no page
        pages = list(render_pages(document, 72, 40_000_000))

    assert [page.index for page in pages] == [0]
    # The page over the limit, the last declared page, then the checked range.
    assert looked_up_pages == checks
    assert loaded_pages == [0]  # the one render
    assert opened_documents[-1].raw is None


def test_overstated_page_count_counts_up_to_the_first_page_that_does_not_exist(
    looked_up_pages: list[int], loaded_pages: list[int]
) -> None:
    sizes = [(200.0 + 10 * index, 300.0) for index in range(3)]
    pdf = declare_page_count(build_pdf([PageSpec(size=size) for size in sizes]), 1_000_000)

    with open_pdf(pdf, max_pages=50) as document:
        assert document.page_count == 3
        assert document.page_sizes == tuple(sizes)
        assert [page.index for page in render_pages(document, 72, 40_000_000)] == [0, 1, 2]

    # Page 50, the last declared page, then every page of the checked range: 52 lookups.
    assert looked_up_pages == [50, 999_999, *range(50)]
    assert loaded_pages == [0, 1, 2]  # only the renders load pages


def test_overstated_page_tree_without_any_page_is_unreadable(
    opened_documents: list[pdfium.PdfDocument],
) -> None:
    pdf = EMPTY_TREE_PDF.replace(b"/Count 0", b"/Count 1000000")

    with pytest.raises(InvalidInput) as caught, open_pdf(pdf, max_pages=50):
        pass

    assert caught.value.code == "pdf_unreadable"
    assert opened_documents[-1].raw is None


@pytest.mark.parametrize("max_pages", [50, None])
def test_overstated_page_count_within_the_limit_follows_the_same_rule(
    max_pages: int | None, looked_up_pages: list[int]
) -> None:
    pdf = declare_page_count(build_pdf([PageSpec(size=SMALL)]), 3)

    with open_pdf(pdf, max_pages=max_pages) as document:
        assert document.page_count == 1
        assert document.page_sizes == (SMALL,)

    assert looked_up_pages == [0, 1, 2]


@pytest.mark.parametrize("pdf", [NON_PAGE_KID_PDF, MISSING_KID_PDF], ids=["non-page", "missing"])
def test_entries_missing_at_the_end_of_the_tree_are_dropped(pdf: bytes) -> None:
    # Through PDFium a tree whose last entries do not load cannot be told from an overstated
    # /Count, so both are the run of pages that load.
    with open_pdf(pdf, max_pages=50) as document:
        assert document.page_count == 1


@pytest.mark.parametrize("max_pages", [3, 50, None])
def test_a_damaged_page_with_a_later_page_loading_is_unreadable(max_pages: int | None) -> None:
    with pytest.raises(InvalidInput) as caught, open_pdf(MIDDLE_DAMAGED_PDF, max_pages=max_pages):
        pass

    assert caught.value.code == "pdf_unreadable"
    assert caught.value.message == "page 1 of the PDF is missing or damaged"


def test_a_damaged_page_in_an_overstated_tree_is_unreadable_not_cut() -> None:
    pdf = MIDDLE_DAMAGED_PDF.replace(b"/Count 3", b"/Count 1000000")

    with pytest.raises(InvalidInput) as caught, open_pdf(pdf, max_pages=50):
        pass

    assert caught.value.code == "pdf_unreadable"
    assert "page 1" in caught.value.message


def test_a_damaged_page_at_the_limit_followed_by_a_page_is_unreadable_not_cut(
    looked_up_pages: list[int],
) -> None:
    # max_pages=1: page 1, the one over the limit, does not exist but page 2 does.
    with pytest.raises(InvalidInput) as caught, open_pdf(MIDDLE_DAMAGED_PDF, max_pages=1):
        pass

    assert caught.value.code == "pdf_unreadable"
    assert caught.value.message == "page 1 of the PDF is missing or damaged"
    assert looked_up_pages == [1, 2, 0]


@pytest.mark.parametrize(
    ("damaged", "max_pages", "message"),
    [
        ({50}, 50, "page 50 of the PDF is missing or damaged"),
        ({50}, None, "page 50 of the PDF is missing or damaged"),
        ({3, 50}, 50, "page 3 of the PDF is missing or damaged"),
        ({3, 50}, None, "page 3 of the PDF is missing or damaged"),
    ],
)
def test_a_longer_document_damaged_at_the_limit_fails_as_without_a_limit(
    damaged: set[int], max_pages: int | None, message: str
) -> None:
    pdf = page_tree_pdf([b"42" if index in damaged else PAGE_DICT for index in range(60)])

    with pytest.raises(InvalidInput) as caught, open_pdf(pdf, max_pages=max_pages):
        pass

    assert caught.value.code == "pdf_unreadable"
    assert caught.value.message == message


def test_a_longer_document_missing_its_page_at_the_limit_and_its_last_page_is_cut() -> None:
    # Known limitation: with page 50 and the last declared page both missing, the tree looks
    # like an overstated /Count to the two checks beyond the limit, so the run that exists is
    # kept, as for missing entries at the end of the tree.
    kids = [b"42" if index in {50, 59} else PAGE_DICT for index in range(60)]

    with open_pdf(page_tree_pdf(kids), max_pages=50) as document:
        assert document.page_count == 50


def test_a_full_document_at_the_limit_checks_each_page_once_without_loading_any(
    looked_up_pages: list[int], loaded_pages: list[int]
) -> None:
    pdf = build_pdf([PageSpec(lines=())] * 50)

    with open_pdf(pdf, max_pages=50) as document:
        assert document.page_count == 50

    assert looked_up_pages == list(range(50))
    assert loaded_pages == []


def test_checking_the_pages_does_not_parse_their_content() -> None:
    # One page whose content stream decodes to about 5 MB of drawing operators: loading it
    # parses them (tens of milliseconds or more), looking its size up does not.
    content = zlib.compress(b"0 0 m 100 100 l S\n" * 300_000, 1)
    stream = b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(content)
    pdf = page_tree_pdf(
        [PAGE_DICT.replace(b" >>", b" /Contents 4 0 R >>")],
        extra=[stream + content + b"\nendstream"],
    )
    with PDFIUM_LOCK:
        probe = PDF_DOCUMENT_CLASS(pdf)
        started = time.perf_counter()
        probe[0].close()
        page_load = time.perf_counter() - started
        probe.close()

    started = time.perf_counter()
    with open_pdf(pdf, max_pages=50) as document:
        assert document.page_sizes == ((200.0, 200.0),)
    opening = time.perf_counter() - started

    assert opening * 10 < page_load


@pytest.mark.parametrize(
    "pdf",
    [
        *(build_pdf([PageSpec(size=A4, rotation=rotation)]) for rotation in (0, 90, 180, 270)),
        *(cropped_pdf(rotation) for rotation in (0, 90, 180, 270)),
        *(inherited_boxes_pdf(rotation) for rotation in (0, 90, 180, 270, -90, 450)),
    ],
    ids=[
        *(f"rotate-{rotation}" for rotation in (0, 90, 180, 270)),
        *(f"cropbox-rotate-{rotation}" for rotation in (0, 90, 180, 270)),
        *(f"inherited-boxes-rotate-{rotation}" for rotation in (0, 90, 180, 270, -90, 450)),
    ],
)
def test_checked_page_size_is_the_displayed_size_of_the_loaded_page(pdf: bytes) -> None:
    with pymupdf.open(stream=pdf, filetype="pdf") as reference:
        displayed = (float(reference[0].rect.width), float(reference[0].rect.height))
    with PDFIUM_LOCK:
        probe = PDF_DOCUMENT_CLASS(pdf)
        page = probe[0]
        loaded = page.get_size()
        page.close()
        probe.close()

    with open_pdf(pdf) as document:
        assert document.page_sizes == (displayed,)
        assert document.page_sizes == (loaded,)


def test_a_long_real_document_is_refused_fast_after_one_page_check(
    five_thousand_pages: bytes, looked_up_pages: list[int], loaded_pages: list[int]
) -> None:
    pdf = five_thousand_pages

    started = time.perf_counter()
    with pytest.raises(InvalidInput) as caught, open_pdf(pdf, max_pages=50):
        pass
    elapsed = time.perf_counter() - started

    assert caught.value.code == "too_many_pages"
    assert caught.value.message == "the PDF has more than 50 pages; at most 50 are accepted"
    assert looked_up_pages == [50]
    assert loaded_pages == []
    assert elapsed < 1.0


@pytest.mark.parametrize("max_pages", [3, None])
def test_page_count_at_the_limit_or_unlimited_opens(max_pages: int | None) -> None:
    with open_pdf(build_pdf([PageSpec(lines=())] * 3), max_pages=max_pages) as document:
        assert document.page_count == 3


def test_oversized_page_is_rejected_before_any_page_is_rendered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = build_pdf([PageSpec(), PageSpec(size=(10_000.0, 10_000.0))])

    def no_rendering(*args: object, **kwargs: object) -> None:
        raise AssertionError("rendered before the budget check")

    monkeypatch.setattr(pdfium_c, "FPDF_RenderPageBitmap_Start", no_rendering)

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

    monkeypatch.setattr(pdfium_c, "FPDF_RenderPageBitmap_Start", broken)

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
