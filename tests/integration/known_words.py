"""Image-only PDF pages with words drawn at known rectangles, and helpers to compare boxes.

Ground truth comes from the vector version of the page, before it is rasterised: PyMuPDF's
``search_for`` locates each target word. Two PyMuPDF facts shape the computation, both checked
by ``test_ocr_positions``:

* ``search_for`` reports rectangles in the *unrotated* page space (the ``/MediaBox`` frame),
  even on a page with ``/Rotate``. Multiplying by ``page.rotation_matrix`` maps them to the
  displayed (rotated) page, which is the space the OCR engine renders and reports in: origin at
  the top-left of the page as it is shown, size ``page.rect``.
* By default the rectangles span the font's ascender to descender, far taller than the ink that
  Tesseract boxes. With "small glyph heights" they are one font size tall, which is close to the
  ink of a word with an accent, a capital or a descender.

The scanned page keeps the same ``/MediaBox`` and ``/Rotate`` as the vector page and holds only
an image of the unrotated content, so a rotated page is rotated by its flag, as a scanner that
stores a landscape image in a portrait frame would produce it.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import pymupdf

from app.domain.geometry import iou
from app.domain.types import Box, OcrPage

_PUNCTUATION = ".,:;!?\"'()"


@dataclass(frozen=True, slots=True)
class Line:
    """A line of text whose baseline starts at ``origin``, in displayed page points."""

    origin: tuple[float, float]
    text: str


@dataclass(frozen=True, slots=True)
class KnownPage:
    """A one-page image-only PDF and the displayed-space rectangle of each target word."""

    pdf: bytes
    width: float
    height: float
    targets: dict[str, Box]


@contextmanager
def _small_glyph_heights() -> Iterator[None]:
    previous = bool(pymupdf.TOOLS.set_small_glyph_heights())
    pymupdf.TOOLS.set_small_glyph_heights(True)
    try:
        yield
    finally:
        pymupdf.TOOLS.set_small_glyph_heights(previous)


def _locate(page: pymupdf.Page, word: str) -> Box:
    with _small_glyph_heights():
        hits = page.search_for(word)
    assert len(hits) == 1, f"{word!r} must occur once on the page, found {len(hits)}"
    rect = pymupdf.Rect(hits[0]) * page.rotation_matrix
    return Box(page=0, x=rect.x0, y=rect.y0, width=rect.width, height=rect.height)


def known_words_page(
    size: tuple[float, float],
    lines: Sequence[Line],
    targets: Sequence[str],
    *,
    rotation: int = 0,
    fontsize: float = 20.0,
    raster_dpi: int = 300,
) -> KnownPage:
    """Draw ``lines`` upright on a page of ``size`` with ``/Rotate`` ``rotation``, rasterise it.

    ``size`` is the unrotated ``/MediaBox``; line origins are in the displayed page. Returns
    the scanned PDF, the displayed page size and the ground-truth box of every target word.
    """
    with pymupdf.open() as vector, pymupdf.open() as scanned:
        page = vector.new_page(width=size[0], height=size[1])
        page.set_rotation(rotation)
        for line in lines:
            origin = pymupdf.Point(*line.origin) * page.derotation_matrix
            page.insert_text(origin, line.text, fontsize=fontsize, rotate=rotation)
        truth = {word: _locate(page, word) for word in targets}
        width, height = float(page.rect.width), float(page.rect.height)
        page.set_rotation(0)
        pixmap = page.get_pixmap(dpi=raster_dpi, colorspace=pymupdf.csGRAY, alpha=False)
        image_page = scanned.new_page(width=size[0], height=size[1])
        image_page.insert_image(image_page.rect, pixmap=pixmap)
        image_page.set_rotation(rotation)
        assert not image_page.get_text().strip(), "the scanned page must have no text layer"
        data: bytes = scanned.tobytes()
    return KnownPage(data, width, height, truth)


def normalise(text: str) -> str:
    """Return ``text`` without surrounding punctuation, as recognised words are compared."""
    return text.strip(_PUNCTUATION)


def boxes_of(page: OcrPage, word: str) -> list[Box]:
    """Return the boxes of every recognised word of ``page`` that reads ``word``."""
    return [found.box for found in page.words if normalise(found.text) == word]


def best_iou(page: OcrPage, word: str, expected: Box) -> float:
    """Return the best IoU between ``expected`` and a recognised ``word``; 0.0 when absent."""
    return max((iou(box, expected) for box in boxes_of(page, word)), default=0.0)


def recognised_text(page: OcrPage) -> str:
    """Return the recognised words of ``page`` joined by spaces, for assertion messages."""
    return " ".join(found.text for found in page.words)
