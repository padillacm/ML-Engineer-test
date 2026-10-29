"""The browser tests' own helpers, without a browser: the frame PDF and the ink measure.

Runs everywhere (host and images). The frame PDF must show upright text on every page as
PDFium (the service's renderer) displays it, with each expected name inside its recorded box
and the name outside a crop box not shown; the ink measure must give a ring of the box's area
and tell a box on text from a box on paper.
"""

import math

import numpy as np
import pypdfium2 as pdfium
import pytest

from tests.e2e import frames, ink


@pytest.fixture(scope="module")
def document() -> frames.FrameDocument:
    return frames.build()


def _render(pdf: pdfium.PdfDocument, index: int) -> np.ndarray:
    page = pdf[index]
    try:
        bitmap = page.render(scale=frames.PX_PER_PT, grayscale=True)
        array = bitmap.to_numpy()
        return np.array(array[:, :, 0] if array.ndim == 3 else array, dtype=np.uint8)
    finally:
        page.close()


def test_the_pages_have_their_rotation_crop_and_displayed_size(
    document: frames.FrameDocument,
) -> None:
    pdf = pdfium.PdfDocument(document.pdf)
    try:
        assert len(pdf) == len(frames.PAGES)
        for index, spec in enumerate(frames.PAGES):
            page = pdf[index]
            try:
                assert page.get_rotation() == spec.rotate
                assert page.get_size() == pytest.approx(document.displayed_sizes[index], abs=0.01)
                assert (page.get_cropbox() != page.get_mediabox()) == spec.crop
            finally:
                page.close()
    finally:
        pdf.close()
    assert {spec.rotate for spec in frames.PAGES} == set(frames.ROTATIONS)
    assert len({spec.size_pt for spec in frames.PAGES}) == 3


def test_every_name_is_upright_ink_in_its_box_as_pdfium_displays_it(
    document: frames.FrameDocument,
) -> None:
    pdf = pdfium.PdfDocument(document.pdf)
    try:
        for name in document.names:
            gray = _render(pdf, name.page)
            box = tuple(v * frames.PX_PER_PT for v in (name.x, name.y, name.width, name.height))
            measured = ink.measure(gray, box)  # type: ignore[arg-type]
            assert measured.passes(), (name, measured)
            # Upright: the glyph box is wider than tall, and the same box turned upside down
            # (as a /Rotate without compensation would show it) holds no ink.
            assert name.width > 3 * name.height
            flipped = (
                gray.shape[1] - box[0] - box[2],
                gray.shape[0] - box[1] - box[3],
                box[2],
                box[3],
            )
            assert ink.measure(gray, flipped).inside == 0
    finally:
        pdf.close()


def test_the_names_outside_the_crop_boxes_are_not_displayed(
    document: frames.FrameDocument,
) -> None:
    pdf = pdfium.PdfDocument(document.pdf)
    try:
        for index, spec in enumerate(frames.PAGES):
            if not spec.crop:
                continue
            gray = _render(pdf, index)
            top_band = gray[: round(frames.TOP_PT * frames.PX_PER_PT) - 4]
            assert (top_band < 255 - ink.CONTRAST).sum() == 0
    finally:
        pdf.close()
    assert len(document.hidden_names) == sum(spec.crop for spec in frames.PAGES)


@pytest.mark.parametrize(("width", "height"), [(100.0, 14.0), (40.0, 40.0), (3.0, 200.0)])
def test_the_ring_has_the_area_of_the_box(width: float, height: float) -> None:
    margin = ink.ring_margin(width, height)

    ring = (width + 2 * margin) * (height + 2 * margin) - width * height

    assert ring == pytest.approx(width * height)


def test_ink_on_text_passes_and_paper_or_a_shifted_box_fails() -> None:
    gray = np.full((200, 400), 250, dtype=np.uint8)
    gray[90:110:2, 100:300:3] = 10  # a "word": dense dark dots in a 200 x 20 box

    on_text = ink.measure(gray, (100, 90, 200, 20))
    on_paper = ink.measure(gray, (100, 20, 200, 20))
    half_off = ink.measure(gray, (100, 100, 200, 20))

    assert on_text.passes()
    assert on_text.ring == 0
    assert on_text.ratio == math.inf
    assert not on_paper.passes()
    assert not half_off.passes()


def test_a_box_at_the_page_edge_measures_its_clamped_ring() -> None:
    gray = np.full((100, 100), 250, dtype=np.uint8)
    gray[0:10, 0:50] = 0

    measured = ink.measure(gray, (0, 0, 50, 10))

    assert measured.inside == 1.0
    assert measured.ring == 0.0


def test_the_special_pages_carry_their_unit_and_codecs(document: frames.FrameDocument) -> None:
    """One /UserUnit 2 page, one JPEG 2000 and one CCITT G4 image, in the saved file itself."""
    kinds = [(spec.user_unit, spec.codec) for spec in frames.PAGES]

    assert kinds.count((frames.USER_UNIT, "jpeg")) == 1
    assert kinds.count((1.0, "jpx")) == 1
    assert kinds.count((1.0, "ccitt")) == 1
    assert document.pdf.count(b"/UserUnit 2") == 1
    assert document.pdf.count(b"/JPXDecode") == 1
    assert document.pdf.count(b"/CCITTFaxDecode") == 1


def test_a_string_rotate_is_written_as_it_is_and_pdfium_reads_it_as_0() -> None:
    document = frames.build((frames.STRING_ROTATE_PAGE,))

    pdf = pdfium.PdfDocument(document.pdf)
    try:
        assert pdf[0].get_rotation() == 0
    finally:
        pdf.close()
    assert b"/Rotate(90)" in document.pdf.replace(b" ", b"")


def test_repeat_offsets_the_names_of_every_copy(document: frames.FrameDocument) -> None:
    long = frames.repeat(document, 3)

    pdf = pdfium.PdfDocument(long.pdf)
    try:
        assert len(pdf) == 3 * len(frames.PAGES)
    finally:
        pdf.close()
    assert len(long.names) == 3 * len(document.names)
    last = long.names[-1]
    assert last.page == document.names[-1].page + 2 * len(frames.PAGES)
    assert (last.x, last.y) == (document.names[-1].x, document.names[-1].y)


def test_the_canned_response_matches_the_listed_names(document: frames.FrameDocument) -> None:
    body = frames.canned_response(document)

    assert len(body["extracted_names"]) == len(document.names)
    assert {match["extracted_name"] for match in body["fuzzy_matches"]} == set(frames.LISTED_NAMES)
