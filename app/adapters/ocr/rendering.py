"""Open a PDF from memory and render its pages to grayscale PNG images with PDFium.

PDFium (through ``pypdfium2``, BSD-3-Clause / Apache-2.0) is used instead of PyMuPDF, whose
AGPL-3.0 licence would reach a network service.

Thread safety: PDFium is not thread-safe, not even across different documents, so every PDFium
call made here (opening, looking up page sizes, loading and rendering pages, closing) runs
while holding the module-level ``PDFIUM_LOCK``. Several requests may therefore render from
different threads; only numpy arrays, bytes and ``PageRaster`` values leave the lock. Every
PDFium object (document, page, bitmap) is closed explicitly inside the lock, never left to the
garbage collector, whose finalizers could run on another thread. Code elsewhere in the process
that uses pypdfium2 must take the same lock.

Damaged files: PDFium refuses to load a truncated file (no usable cross-reference data) instead
of inventing pages, and loads a file whose cross-reference offsets are wrong but whose objects
are intact.

Page count: PDFium reports the ``/Count`` the page tree declares, which a damaged or hostile
file can overstate, so the count is verified when the document is opened, with one rule for
every document. A page is checked by reading its displayed size from its page dictionary
(``FPDF_GetPageSizeByIndexF``): the page is found in the tree exactly as ``FPDF_LoadPage``
finds it and its size is computed by the same code (boxes and ``/Rotate``, inherited ones
included), but its content stream is never parsed, so a check costs microseconds however heavy
the page's content is. When the declared count is over ``max_pages``, page index ``max_pages``
is checked first: if it exists, the document is too long. If it does not, the last declared
page is checked too: if that one exists, a missing page is followed by a readable one and the
PDF is ``pdf_unreadable`` (never cut to its first ``max_pages`` pages). Then pages
``0 .. min(declared, max_pages) - 1`` are all checked. The document is the run of pages from 0
up to the first that does not exist, provided no later page in that range exists; a page that
fails with a later one existing (a damaged page in the middle) fails the whole PDF with
``pdf_unreadable``, so pages are never silently dropped from the middle. Pages missing at the
end of the tree cannot be told apart from an overstated ``/Count`` through PDFium and are
dropped; so is a longer document whose page ``max_pages`` and last declared page are both
missing. At most ``max_pages + 2`` pages are checked. Every later step uses that verified count
(``PdfFile.page_count``), never PDFium's.

Coordinates: pages are rendered as displayed, after ``/Rotate`` and relative to the CropBox, and
the page size is the displayed size in points (PDFium reports it already rotated), so
``PageRaster`` and every derived word box are in that displayed frame. Drawing or redacting on
the unrotated page later needs the derotation (in PyMuPDF terms ``page.derotation_matrix``)
and the CropBox offset to map them back.
"""

import io
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pypdfium2 as pdfium
from PIL import Image

from app.domain.errors import GeometryError, InvalidInput
from app.domain.geometry import PageRaster

type GrayImage = npt.NDArray[np.uint8]
type ImageTransform = Callable[[GrayImage], GrayImage]

PDFIUM_LOCK = threading.RLock()
"""Serialises every PDFium call of the process that goes through this module."""

_POINTS_PER_INCH = 72.0
_UNREADABLE = "pdf_unreadable"
_PASSWORD_ERROR = 4  # FPDF_ERR_PASSWORD
_PDFIUM_ERRORS = (pdfium.PdfiumError, RuntimeError, ValueError, OSError)


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """PNG image of one page together with the raster descriptor that maps it to points.

    The raster describes the page as displayed: after ``/Rotate`` and relative to its CropBox.
    """

    index: int
    png: bytes
    raster: PageRaster


@dataclass(frozen=True, slots=True)
class PdfFile:
    """An open PDF: the PDFium document and the displayed size of every page, in points.

    Use it only through ``render_pages`` (or while holding ``PDFIUM_LOCK``).
    """

    document: pdfium.PdfDocument
    page_sizes: tuple[tuple[float, float], ...]

    @property
    def page_count(self) -> int:
        return len(self.page_sizes)


@contextmanager
def open_pdf(pdf: bytes, max_pages: int | None = None) -> Iterator[PdfFile]:
    """Open ``pdf`` from memory and close it on exit, whatever happens inside the block.

    Raises ``InvalidInput`` (code ``pdf_unreadable``) when the bytes are not a readable PDF
    (corrupt or truncated), when it is password protected, when no page exists, or when a page
    is missing or damaged while a later one exists. With ``max_pages``, a document whose page
    index ``max_pages`` exists raises ``InvalidInput`` (code ``too_many_pages``) after that
    single check, before any other page is checked. The page count is verified as described in
    the module notes, without loading any page (at most ``max_pages + 2`` size lookups; without
    ``max_pages`` every declared page is looked up). The document is closed before any error
    leaves.
    """
    with PDFIUM_LOCK:
        document = _load(pdf)
        try:
            sizes = _verified_page_sizes(document, max_pages)
        except BaseException:
            document.close()
            raise
    try:
        yield PdfFile(document, sizes)
    finally:
        with PDFIUM_LOCK:
            document.close()


def _load(pdf: bytes) -> pdfium.PdfDocument:
    try:
        return pdfium.PdfDocument(pdf)
    except _PDFIUM_ERRORS as error:
        if getattr(error, "err_code", None) == _PASSWORD_ERROR:
            raise InvalidInput("the PDF is password protected", _UNREADABLE) from error
        raise InvalidInput("the file cannot be opened as a PDF", _UNREADABLE) from error


def _verified_page_sizes(
    document: pdfium.PdfDocument, max_pages: int | None
) -> tuple[tuple[float, float], ...]:
    """The sizes of the pages the document really holds (see the module notes)."""
    declared = len(document)
    checked = declared if max_pages is None else min(declared, max_pages)
    later_page_exists = checked < declared and _exists_beyond_limit(document, checked, declared)
    sizes: list[tuple[float, float]] = []
    first_missing: int | None = None
    for index in range(checked):
        size = _page_size(document, index)
        if size is None:
            if first_missing is None:
                first_missing = index
        elif first_missing is not None:
            raise _missing_page(first_missing)
        else:
            sizes.append(size)
    if later_page_exists:
        raise _missing_page(checked if first_missing is None else first_missing)
    if not sizes:
        raise InvalidInput("the PDF has no pages", _UNREADABLE)
    return tuple(sizes)


def _exists_beyond_limit(document: pdfium.PdfDocument, limit: int, declared: int) -> bool:
    """Whether the last declared page exists, given ``declared > limit``.

    Raises ``too_many_pages`` when page index ``limit`` exists. When it does not, a later
    existing page means the document goes on after a missing page.
    """
    if _page_size(document, limit) is not None:
        raise InvalidInput(
            f"the PDF has more than {limit} pages; at most {limit} are accepted",
            "too_many_pages",
        )
    last = declared - 1
    return last > limit and _page_size(document, last) is not None


def _missing_page(index: int) -> InvalidInput:
    return InvalidInput(f"page {index} of the PDF is missing or damaged", _UNREADABLE)


def _page_size(document: pdfium.PdfDocument, index: int) -> tuple[float, float] | None:
    """The displayed size of page ``index`` in points, or ``None`` when it does not exist.

    ``FPDF_GetPageSizeByIndexF`` finds the page and computes its size like ``FPDF_LoadPage``
    followed by ``FPDF_GetPageWidthF``/``HeightF`` (same tree lookup, same box and rotation
    handling) but never parses the content stream.
    """
    try:
        width, height = document.get_page_size(index)
    except _PDFIUM_ERRORS:
        return None
    return float(width), float(height)


def render_pages(
    pdf: PdfFile,
    dpi: int,
    max_page_pixels: int,
    transform: ImageTransform | None = None,
) -> Iterator[RenderedPage]:
    """Yield every page of ``pdf`` rendered in grayscale at ``dpi``, in page order.

    The pixel budget of every page is checked before the first page is rendered, so an
    oversized page fails with ``InvalidInput`` (code ``too_large``) without any rendering.
    ``transform``, when given, is applied to the gray pixels before PNG encoding, outside the
    PDFium lock. A page that cannot be rendered fails with ``pdf_unreadable``. Pages are
    rendered lazily, one per ``next()``, so the caller decides how many exist at once.
    """
    for index, size in enumerate(pdf.page_sizes):
        _check_budget(index, size, dpi, max_page_pixels)
    for index in range(pdf.page_count):
        yield _render(pdf, index, dpi, transform)


def _check_budget(index: int, size: tuple[float, float], dpi: int, max_page_pixels: int) -> None:
    scale = dpi / _POINTS_PER_INCH
    expected = (size[0] * scale) * (size[1] * scale)
    if expected > max_page_pixels:
        raise InvalidInput(
            f"page {index} would render to more than {max_page_pixels} pixels at {dpi} DPI",
            "too_large",
        )


def _render(pdf: PdfFile, index: int, dpi: int, transform: ImageTransform | None) -> RenderedPage:
    width_pt, height_pt = pdf.page_sizes[index]
    try:
        with PDFIUM_LOCK:
            pixels = _rasterize(pdf.document, index, dpi / _POINTS_PER_INCH)
        raster = PageRaster(pixels.shape[1], pixels.shape[0], width_pt, height_pt)
        png = encode_png(pixels if transform is None else transform(pixels))
    except (*_PDFIUM_ERRORS, GeometryError) as error:
        raise InvalidInput(f"page {index} of the PDF cannot be rendered", _UNREADABLE) from error
    return RenderedPage(index, png, raster)


def _rasterize(document: pdfium.PdfDocument, index: int, scale: float) -> GrayImage:
    """Render page ``index`` as displayed; return a private copy of its gray pixels."""
    page = document[index]
    try:
        bitmap = page.render(scale=scale, grayscale=True, may_draw_forms=False)
        try:
            return _gray_pixels(bitmap.to_numpy())
        finally:
            bitmap.close()
    finally:
        page.close()


def _gray_pixels(array: npt.NDArray[np.uint8]) -> GrayImage:
    """Copy a PDFium gray bitmap (2-D, or 3-D with one channel) into an owned 2-D array."""
    if array.ndim == 3:  # noqa: PLR2004 - rows, columns, channels
        array = array[:, :, 0]
    if array.ndim != 2 or array.dtype != np.uint8:  # noqa: PLR2004 - rows, columns
        raise ValueError("unexpected bitmap layout")
    return np.array(array, dtype=np.uint8, copy=True)


def encode_png(image: GrayImage) -> bytes:
    """Encode a 2-D ``uint8`` image as an 8-bit grayscale PNG, in memory."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()
