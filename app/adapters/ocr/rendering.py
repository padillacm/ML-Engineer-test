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

Render bounds (``RenderLimits``): a page is drawn through PDFium's progressive API
(``FPDF_RenderPageBitmap_Start`` / ``FPDF_RenderPage_Continue`` with an ``IFSDK_PAUSE``) whose
pause callback answers "pause" once ``timeout_s`` has elapsed since the page's render started
(the clock starts before the page is loaded, so content parsing counts). PDFium polls the
callback between drawing operations (every hundred page objects, and between the bands of an
image decode), so a page stops within the budget plus one operation and fails the document
with ``pdf_unreadable`` naming the page; the lock is released and the next page or request
renders normally. The bitmap, matrix and flags reproduce ``PdfPage.render(scale=...,
grayscale=True, may_draw_forms=False)`` of pypdfium2 5.14 exactly (an 8-bit gray bitmap filled
white, the whole displayed page at ``ceil(size x scale)`` pixels, no extra rotation,
``FPDF_GRAYSCALE | FPDF_ANNOT``), so a page within the budget renders bit for bit as that call
would. Before a page is drawn, the declared sizes (``FPDFImageObj_GetImageMetadata``, a
dictionary lookup, not a decode) of its image objects, nested form XObjects included, are
summed and a page declaring more than ``max_image_pixels`` pixels fails with ``too_large``
without decoding anything. The walk covers the objects of the page's annotation appearance
streams too (``FPDFPage_GetAnnot`` / ``FPDFAnnot_GetObject``): ``FPDF_ANNOT`` draws them, so
an image placed in a stamp annotation instead of the page content would otherwise be decoded
unchecked (a review found a 400-million-pixel image drawn that way under a one-million limit).
The scan polls the same clock, so a page with millions of objects
cannot spend more than the budget there either. The same walk refuses, as ``pdf_unreadable``,
a page whose form XObjects nest deeper than ``MAX_FORM_DEPTH`` levels: PDFium polls the pause
callback between the objects of a form only up to that depth and draws a deeper form as one
operation, which the budget could not stop (measured: 50,000 strokes at depth 9 drew for
2.2 s against a 0.05 s budget). What the budget does not cover: PDFium parses a page's
content stream when the page is loaded, before the first pause poll (the raw API exposes no
stream lengths to check beforehand), and a single huge image decode is one operation (the
pixel budget and the image pre-check bound those).
"""

import ctypes
import io
import math
import threading
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from app.domain.errors import GeometryError, InvalidInput
from app.domain.geometry import PageRaster

type GrayImage = npt.NDArray[np.uint8]
type ImageTransform = Callable[[GrayImage], GrayImage]
type Clock = Callable[[], float]

PDFIUM_LOCK = threading.RLock()
"""Serialises every PDFium call of the process that goes through this module."""

DEFAULT_RENDER_TIMEOUT_S = 10.0
DEFAULT_MAX_IMAGE_PIXELS = 80_000_000

_POINTS_PER_INCH = 72.0
_UNREADABLE = "pdf_unreadable"
_TOO_LARGE = "too_large"
_PASSWORD_ERROR = 4  # FPDF_ERR_PASSWORD
_PDFIUM_ERRORS = (pdfium.PdfiumError, RuntimeError, ValueError, OSError)
# What ``PdfPage.render(grayscale=True)`` passes: gray output, annotations drawn, no rotation.
_RENDER_FLAGS = pdfium_c.FPDF_GRAYSCALE | pdfium_c.FPDF_ANNOT
_NO_ROTATION = 0
_WHITE = (255, 255, 255, 255)
_OBJECTS_PER_CLOCK_POLL = 256

MAX_FORM_DEPTH = 8
"""Deepest form XObject nesting a page may hold.

PDFium's progressive renderer polls the pause callback between the objects of a form only up
to this depth; a form nested deeper is drawn as one operation, outside the render budget
(measured with pypdfium2 5.14: depth 8 polls, depth 9 does not), so deeper pages are refused.
"""


@dataclass(frozen=True, slots=True)
class RenderLimits:
    """Bounds on the rendering of one page (see the module notes).

    ``timeout_s`` is the time budget of one page's render, ``max_image_pixels`` the largest sum
    of declared image pixels a page may hold; ``clock`` is the monotonic clock the budget is
    measured with (replaceable in tests).
    """

    timeout_s: float = DEFAULT_RENDER_TIMEOUT_S
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS
    clock: Clock = time.monotonic


DEFAULT_RENDER_LIMITS = RenderLimits()


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
    limits: RenderLimits = DEFAULT_RENDER_LIMITS,
) -> Iterator[RenderedPage]:
    """Yield every page of ``pdf`` rendered in grayscale at ``dpi``, in page order.

    The pixel budget of every page is checked before the first page is rendered, so an
    oversized page fails with ``InvalidInput`` (code ``too_large``) without any rendering.
    Before each page is drawn, its declared image pixels are checked against
    ``limits.max_image_pixels`` (``too_large``, nothing decoded), and its drawing is stopped
    once ``limits.timeout_s`` has elapsed (``pdf_unreadable`` naming the page).
    ``transform``, when given, is applied to the gray pixels before PNG encoding, outside the
    PDFium lock. A page that cannot be rendered fails with ``pdf_unreadable``. Pages are
    rendered lazily, one per ``next()``, so the caller decides how many exist at once.
    """
    for index, size in enumerate(pdf.page_sizes):
        _check_budget(index, size, dpi, max_page_pixels)
    for index in range(pdf.page_count):
        yield _render(pdf, index, dpi, transform, limits)


def _check_budget(index: int, size: tuple[float, float], dpi: int, max_page_pixels: int) -> None:
    scale = dpi / _POINTS_PER_INCH
    expected = (size[0] * scale) * (size[1] * scale)
    if expected > max_page_pixels:
        raise InvalidInput(
            f"page {index} would render to more than {max_page_pixels} pixels at {dpi} DPI",
            "too_large",
        )


def _render(
    pdf: PdfFile,
    index: int,
    dpi: int,
    transform: ImageTransform | None,
    limits: RenderLimits = DEFAULT_RENDER_LIMITS,
) -> RenderedPage:
    width_pt, height_pt = pdf.page_sizes[index]
    try:
        with PDFIUM_LOCK:
            pixels = _rasterize(pdf.document, index, dpi / _POINTS_PER_INCH, limits)
        raster = PageRaster(pixels.shape[1], pixels.shape[0], width_pt, height_pt)
        png = encode_png(pixels if transform is None else transform(pixels))
    except (*_PDFIUM_ERRORS, GeometryError) as error:
        raise InvalidInput(f"page {index} of the PDF cannot be rendered", _UNREADABLE) from error
    return RenderedPage(index, png, raster)


def _rasterize(
    document: pdfium.PdfDocument, index: int, scale: float, limits: RenderLimits
) -> GrayImage:
    """Render page ``index`` as displayed; return a private copy of its gray pixels.

    Raises ``InvalidInput`` ``too_large`` when the page declares more image pixels than
    ``limits.max_image_pixels`` (before anything is drawn or decoded) and ``pdf_unreadable``
    when drawing is not finished within ``limits.timeout_s``; a PDFium failure raises
    ``PdfiumError``. Must be called with ``PDFIUM_LOCK`` held.
    """
    deadline = _Deadline(limits.clock, limits.timeout_s)
    page = document[index]
    try:
        _check_image_pixels(page, index, limits.max_image_pixels, deadline)
        width = math.ceil(page.get_width() * scale)
        height = math.ceil(page.get_height() * scale)
        if width < 1 or height < 1:
            raise ValueError("the page has no area")
        bitmap = pdfium.PdfBitmap.new_native(width, height, format=pdfium_c.FPDFBitmap_Gray)
        try:
            bitmap.fill_rect(_WHITE, 0, 0, width, height)
            _draw(page, bitmap, index, (width, height), deadline)
            return _gray_pixels(bitmap.to_numpy())
        finally:
            bitmap.close()
    finally:
        page.close()


class _Deadline:
    """The end of one page's render budget on the injected monotonic clock."""

    __slots__ = ("_clock", "_end", "timeout_s")

    def __init__(self, clock: Clock, timeout_s: float) -> None:
        self._clock = clock
        self.timeout_s = timeout_s
        self._end = clock() + timeout_s

    def passed(self) -> bool:
        return self._clock() >= self._end

    def exceeded(self, index: int) -> InvalidInput:
        return InvalidInput(
            f"page {index} exceeded the render budget of {self.timeout_s:g} s", _UNREADABLE
        )


def _draw(
    page: pdfium.PdfPage,
    bitmap: pdfium.PdfBitmap,
    index: int,
    size: tuple[int, int],
    deadline: _Deadline,
) -> None:
    """Draw the whole page into ``bitmap`` progressively, giving up once ``deadline`` passes.

    PDFium calls ``NeedToPauseNow`` between drawing operations and returns
    ``FPDF_RENDER_TOBECONTINUED`` from ``Start``/``Continue`` once it answers true; the render
    is then closed and the page fails with ``pdf_unreadable``. Any other unfinished status is a
    PDFium failure.
    """
    pause = pdfium_c.IFSDK_PAUSE(version=1)
    need_to_pause = type(pause.NeedToPauseNow)(lambda _pause: int(deadline.passed()))
    pause.NeedToPauseNow = need_to_pause  # ``need_to_pause`` stays referenced while drawing
    width, height = size
    status = pdfium_c.FPDF_RenderPageBitmap_Start(
        bitmap, page, 0, 0, width, height, _NO_ROTATION, _RENDER_FLAGS, pause
    )
    try:
        while status == pdfium_c.FPDF_RENDER_TOBECONTINUED:
            if deadline.passed():
                raise deadline.exceeded(index)
            status = pdfium_c.FPDF_RenderPage_Continue(page, pause)
        if status != pdfium_c.FPDF_RENDER_DONE:
            raise pdfium.PdfiumError(f"progressive rendering ended with status {status}")
    finally:
        pdfium_c.FPDF_RenderPage_Close(page)


def _check_image_pixels(
    page: pdfium.PdfPage, index: int, max_image_pixels: int, deadline: _Deadline
) -> None:
    """Refuse the page when it declares too many image pixels or nests forms too deep.

    Walks every page object, the objects of every annotation's appearance stream (drawn by
    ``FPDF_ANNOT``) and, iteratively, the objects of every form XObject (each form once: a
    visited set guards against a cycle), summing ``width x height`` from each image's metadata
    and stopping at the first image that takes the sum over ``max_image_pixels``
    (``too_large``). A form whose content would lie deeper than ``MAX_FORM_DEPTH`` levels
    fails the page as ``pdf_unreadable`` (see the constant). The walk polls ``deadline`` every
    ``_OBJECTS_PER_CLOCK_POLL`` objects so that a page with millions of objects stays within
    the render budget. Every generator opened is closed before returning, so an annotation
    handle is never left to the garbage collector (whose finalizer could run on another
    thread, outside the lock).
    """
    total = 0
    visited = 0
    seen_forms: set[int] = set()
    frames = _root_frames(page)
    opened = list(frames)
    try:
        while frames:
            objects, depth = frames[-1]
            page_object = next(objects, None)
            if page_object is None:
                frames.pop()
                continue
            visited += 1
            if visited % _OBJECTS_PER_CLOCK_POLL == 0 and deadline.passed():
                raise deadline.exceeded(index)
            kind = pdfium_c.FPDFPageObj_GetType(page_object)
            if kind == pdfium_c.FPDF_PAGEOBJ_IMAGE:
                total += _declared_pixels(page_object, page)
                if total > max_image_pixels:
                    raise InvalidInput(
                        f"page {index} holds more than {max_image_pixels} decoded image pixels",
                        _TOO_LARGE,
                    )
            elif kind == pdfium_c.FPDF_PAGEOBJ_FORM:
                frame = _form_frame(page_object, depth, index, seen_forms)
                if frame is not None:
                    frames.append(frame)
                    opened.append(frame)
    finally:
        for objects, _ in opened:
            if isinstance(objects, Generator):
                objects.close()


type _Frame = tuple[Iterator[object], int]
"""The objects of one container being walked and the form depth of that container."""


def _root_frames(page: pdfium.PdfPage) -> list[_Frame]:
    """The page's own objects (walked first: last in the list) and those of each annotation."""
    return [
        *(
            (_annotation_objects(page, position), 0)
            for position in reversed(range(pdfium_c.FPDFPage_GetAnnotCount(page)))
        ),
        (_page_objects(page), 0),
    ]


def _form_frame(form_object: object, depth: int, index: int, seen_forms: set[int]) -> _Frame | None:
    """The frame of a form XObject met at ``depth``, ``None`` when it was walked already;
    ``pdf_unreadable`` when its content would lie beyond ``MAX_FORM_DEPTH``."""
    if depth >= MAX_FORM_DEPTH:
        raise InvalidInput(
            f"page {index} nests form XObjects deeper than {MAX_FORM_DEPTH} levels", _UNREADABLE
        )
    handle = ctypes.cast(cast(ctypes.c_void_p, form_object), ctypes.c_void_p).value or 0
    if handle in seen_forms:
        return None
    seen_forms.add(handle)
    return _form_objects(form_object), depth + 1


def _page_objects(page: pdfium.PdfPage) -> Iterator[object]:
    for position in range(pdfium_c.FPDFPage_CountObjects(page)):
        yield pdfium_c.FPDFPage_GetObject(page, position)


def _annotation_objects(page: pdfium.PdfPage, position: int) -> Iterator[object]:
    """The page objects of annotation ``position``'s normal appearance stream (none when the
    annotation has no appearance); the annotation handle is closed when the walk ends."""
    annotation = pdfium_c.FPDFPage_GetAnnot(page, position)
    if not annotation:
        return
    try:
        for index in range(pdfium_c.FPDFAnnot_GetObjectCount(annotation)):
            yield pdfium_c.FPDFAnnot_GetObject(annotation, index)
    finally:
        pdfium_c.FPDFPage_CloseAnnot(annotation)


def _form_objects(form_object: object) -> Iterator[object]:
    for position in range(pdfium_c.FPDFFormObj_CountObjects(form_object)):
        yield pdfium_c.FPDFFormObj_GetObject(form_object, position)


def _declared_pixels(image_object: object, page: pdfium.PdfPage) -> int:
    """``width x height`` from the image's metadata (a dictionary lookup, no decoding)."""
    metadata = pdfium_c.FPDF_IMAGEOBJ_METADATA()
    if not pdfium_c.FPDFImageObj_GetImageMetadata(image_object, page, ctypes.byref(metadata)):
        return 0
    return int(metadata.width) * int(metadata.height)


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
