"""Open a PDF from memory and render its pages to grayscale PNG images with PDFium.

PDFium (through ``pypdfium2``, BSD-3-Clause / Apache-2.0) is used instead of PyMuPDF, whose
AGPL-3.0 licence would reach a network service.

Thread safety: PDFium is not thread-safe, not even across different documents, so every PDFium
call made here (opening, loading pages, rendering and closing) runs while holding the
module-level ``PDFIUM_LOCK``. Several requests may therefore render from different threads; only
numpy arrays, bytes and ``PageRaster`` values leave the lock. Every PDFium object (document,
page, bitmap) is closed explicitly inside the lock, never left to the garbage collector, whose
finalizers could run on another thread. Code elsewhere in the process that uses pypdfium2 must
take the same lock.

Damaged files: PDFium refuses to load a truncated file (no usable cross-reference data) instead
of inventing pages, and loads a file whose cross-reference offsets are wrong but whose objects
are intact. Every page is loaded once when the document is opened, so a page-tree entry that is
missing or is not a page dictionary fails the whole PDF with ``pdf_unreadable`` before any OCR.

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
def open_pdf(pdf: bytes) -> Iterator[PdfFile]:
    """Open ``pdf`` from memory and close it on exit, whatever happens inside the block.

    Raises ``InvalidInput`` (code ``pdf_unreadable``) when the bytes are not a readable PDF
    (corrupt or truncated), when it is password protected, when it has no pages or when one of
    its pages cannot be loaded.
    """
    with PDFIUM_LOCK:
        document = _load(pdf)
        try:
            sizes = _page_sizes(document)
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


def _page_sizes(document: pdfium.PdfDocument) -> tuple[tuple[float, float], ...]:
    count = len(document)
    if count == 0:
        raise InvalidInput("the PDF has no pages", _UNREADABLE)
    sizes: list[tuple[float, float]] = []
    for index in range(count):
        try:
            page = document[index]
        except _PDFIUM_ERRORS as error:
            raise InvalidInput(
                f"page {index} of the PDF is missing or damaged", _UNREADABLE
            ) from error
        try:
            width, height = page.get_size()
        finally:
            page.close()
        sizes.append((float(width), float(height)))
    return tuple(sizes)


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
