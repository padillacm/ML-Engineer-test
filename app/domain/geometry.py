"""Page geometry: pixel to point conversion, box algebra and text to box location.

Every box produced here is in PDF points with a top-left origin (see ``Box``).
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite

from app.domain.errors import GeometryError
from app.domain.types import Box, PageWord, PixelBox, TextSpan


@dataclass(frozen=True, slots=True)
class PageRaster:
    """Measured size of a rendered page image together with the page rectangle it depicts.

    The scale is derived per axis from these four numbers; no DPI is ever assumed. For a rotated
    page the adapter passes the rotated page rectangle and an image rendered in that orientation.
    """

    width_px: float
    height_px: float
    page_width_pt: float
    page_height_pt: float

    def __post_init__(self) -> None:
        dimensions = (self.width_px, self.height_px, self.page_width_pt, self.page_height_pt)
        if not all(isfinite(value) for value in dimensions):
            raise GeometryError("raster and page dimensions must be finite")
        if min(dimensions) <= 0:
            raise GeometryError("raster and page dimensions must be > 0")

    def to_points(self, page: int, box: PixelBox) -> Box:
        """Convert a pixel box of this raster into a point box on ``page``.

        The edges are converted and the size is derived from them, so a box reaching the raster
        border ends at the page border up to floating-point rounding.
        """
        scale_x = self.page_width_pt / self.width_px
        scale_y = self.page_height_pt / self.height_px
        left = box.x * scale_x
        top = box.y * scale_y
        right = (box.x + box.width) * scale_x
        bottom = (box.y + box.height) * scale_y
        return Box(page=page, x=left, y=top, width=right - left, height=bottom - top)


def union(boxes: Sequence[Box]) -> Box:
    """Return the smallest box containing every box of ``boxes``, which must share one page."""
    if not boxes:
        raise GeometryError("cannot unite an empty set of boxes")
    page = boxes[0].page
    if any(box.page != page for box in boxes):
        raise GeometryError("cannot unite boxes from different pages")
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.right for box in boxes)
    bottom = max(box.bottom for box in boxes)
    return Box(page=page, x=left, y=top, width=right - left, height=bottom - top)


def clamp(box: Box, page_width_pt: float, page_height_pt: float) -> Box:
    """Clip ``box`` to the page rectangle ``[0, page_width_pt] x [0, page_height_pt]``.

    A box that has no extent inside the page is rejected rather than turned into an empty box.
    """
    if not (isfinite(page_width_pt) and isfinite(page_height_pt)):
        raise GeometryError("page dimensions must be finite")
    left, right = _clip_axis(box.x, box.right, page_width_pt)
    top, bottom = _clip_axis(box.y, box.bottom, page_height_pt)
    return Box(page=box.page, x=left, y=top, width=right - left, height=bottom - top)


def _clip_axis(start: float, end: float, limit: float) -> tuple[float, float]:
    clipped_start = max(start, 0.0)
    clipped_end = min(end, limit)
    outside = clipped_end < clipped_start or (clipped_end == clipped_start and end > start)
    if outside:
        raise GeometryError("box lies outside the page")
    return clipped_start, clipped_end


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes; 0.0 on different pages or without overlap."""
    if a.page != b.page:
        return 0.0
    overlap_width = min(a.right, b.right) - max(a.x, b.x)
    overlap_height = min(a.bottom, b.bottom) - max(a.y, b.y)
    if overlap_width <= 0 or overlap_height <= 0:
        return 0.0
    # Lengths are normalised by the largest one so that the areas can neither overflow nor
    # underflow into a 0/0 or inf/inf division.
    unit = max(a.width, a.height, b.width, b.height)
    intersection = (overlap_width / unit) * (overlap_height / unit)
    total = (
        (a.width / unit) * (a.height / unit) + (b.width / unit) * (b.height / unit) - intersection
    )
    return intersection / total if total > 0 else 0.0


@dataclass(frozen=True, slots=True)
class PageText:
    """The text of one page assembled from its words, with the character range of each word.

    ``words`` and ``offsets`` are parallel: ``offsets[i]`` is the half-open ``(start, end)``
    range of ``words[i]`` inside ``text``.
    """

    text: str
    words: tuple[PageWord, ...]
    offsets: tuple[tuple[int, int], ...]

    @classmethod
    def from_words(cls, words: Iterable[PageWord]) -> "PageText":
        """Join the non-blank words with single spaces, in the order given.

        All words, blank ones included, must be on the same page.
        """
        page_words = tuple(words)
        if len({word.box.page for word in page_words}) > 1:
            raise GeometryError("page words must all be on the same page")
        kept = tuple(word for word in page_words if word.text.strip())
        offsets: list[tuple[int, int]] = []
        position = 0
        for word in kept:
            offsets.append((position, position + len(word.text)))
            position += len(word.text) + 1
        return cls(text=" ".join(word.text for word in kept), words=kept, offsets=tuple(offsets))

    def locate(self, span: TextSpan) -> Box | None:
        """Return the union box of every word overlapping ``span``, even partially.

        Returns ``None`` when no word overlaps, for instance for an empty span, a span lying
        beyond the text or one covering only a separating space. The box is not clamped because
        the page size is unknown here; callers apply ``clamp`` with the page rectangle.
        """
        overlapped = [
            word
            for word, (start, end) in zip(self.words, self.offsets, strict=True)
            if span.start < span.end and start < span.end and span.start < end
        ]
        if not overlapped:
            return None
        return union([word.box for word in overlapped])
