"""The ink check of the audit viewer's boxes (spec: audit-viewer; design D7 item 2, D8 item 2).

A box drawn on a name has dark text inside and mostly paper around it. On the canvas that pdf.js
rendered, the check measures the share of dark pixels inside the box and in a ring of the same
area around it (clamped to the page), and passes when the inside share is at least
``MIN_INSIDE`` and at least ``MIN_RATIO`` times the ring share.

"Dark" is relative to the paper around the box: a pixel is ink when it is at least ``CONTRAST``
grey levels below the median of the box's neighbourhood (the box, its ring and a margin of one
box height). A page-wide level missed faint grey text on degraded scans, where black table rules,
dark scan edges or salt-and-pepper noise set the page's darkest levels (docs/BENCHMARKS.md
section 20.3); the local paper level follows uneven illumination too. Dark pixels on table
rules are not ink (see ``measure``), so a name in a ruled cell is measured like one on paper.

The thresholds were calibrated on the 805 boxes the service returned for the 60-document sample
of the synthetic corpus (docs/BENCHMARKS.md section 20.3): the largest ``MIN_INSIDE`` that keeps
99.5 % of the boxes matching a truth mention (IoU >= 0.5) and then the largest ``MIN_RATIO``
that keeps 99 % of them passing both (763 of 770, 99.1 %); a box moved by a wrong frame passes
far less often (pixels taken for points 0.9 %, a y-flip 20 %, one line down 31 %).
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

type Gray = npt.NDArray[np.uint8]

MIN_INSIDE = 0.16
"""Least share of dark pixels inside a box (docs/BENCHMARKS.md section 20.3)."""
MIN_RATIO = 2.1
"""Least ratio of the inside share to the ring share (docs/BENCHMARKS.md section 20.3)."""
CONTRAST = 40.0
"""Grey levels below the local paper at which a pixel counts as ink."""
CONTEXT = 1.0
"""Neighbourhood of the paper level: the ring plus this many box heights on each side."""


@dataclass(frozen=True)
class Ink:
    """Dark-pixel shares of one box: ``inside`` the box, in its ``ring``, and their ratio."""

    inside: float
    ring: float

    @property
    def ratio(self) -> float:
        return self.inside / self.ring if self.ring > 0 else math.inf

    def passes(self, min_inside: float = MIN_INSIDE, min_ratio: float = MIN_RATIO) -> bool:
        return self.inside >= min_inside and self.ratio >= min_ratio


def ring_margin(width: float, height: float) -> float:
    """The margin ``m`` around a ``width`` x ``height`` box whose ring has the box's area.

    ``(w + 2m)(h + 2m) - wh = wh``, i.e. ``4m^2 + 2(w + h)m - wh = 0``.
    """
    if width <= 0 or height <= 0:
        return 0.0
    total = width + height
    return (-total + math.sqrt(total * total + 4 * width * height)) / 4


type Bounds = tuple[int, int, int, int]


def _bounds(gray: Gray, box: tuple[float, float, float, float], pad: float) -> Bounds:
    """``(x0, y0, x1, y1)`` of ``box`` grown by ``pad`` on every side, clamped to the image."""
    rows, columns = gray.shape
    x, y, width, height = box
    return (
        min(max(round(x - pad), 0), columns),
        min(max(round(y - pad), 0), rows),
        min(max(round(x + width + pad), 0), columns),
        min(max(round(y + height + pad), 0), rows),
    )


SPREAD = 2
"""Rows a scanned rule may drift (skew) or break across before its run is cut."""


def _long_runs(mask: npt.NDArray[np.bool_], length: float) -> npt.NDArray[np.bool_]:
    """The pixels of ``mask`` on a run of at least ``length`` along a row.

    Rules of a scanned page are skewed by up to about 2 degrees and broken by noise, so runs are
    looked for after spreading the mask over ``SPREAD`` rows up and down and bridging one-pixel
    gaps; glyphs stay apart, because letters are separated along the row.
    """
    spread = mask.copy()
    for shift in range(1, SPREAD + 1):
        spread[shift:] |= mask[:-shift]
        spread[:-shift] |= mask[shift:]
    spread[:, 1:-1] |= spread[:, :-2] & spread[:, 2:]
    found = np.zeros_like(mask)
    edges = np.diff(np.pad(spread.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    for row in range(mask.shape[0]):
        starts = np.flatnonzero(edges[row] == 1)
        ends = np.flatnonzero(edges[row] == -1)
        for start, end in zip(starts, ends, strict=True):
            if end - start >= length:
                found[row, start:end] = True
    return found & mask


def _share(dark: npt.NDArray[np.bool_], origin: tuple[int, int], bounds: Bounds) -> tuple[int, int]:
    """Dark pixels and all pixels of ``bounds`` (absolute) in ``dark`` (placed at ``origin``)."""
    x0, y0, x1, y1 = bounds
    part = dark[y0 - origin[1] : y1 - origin[1], x0 - origin[0] : x1 - origin[0]]
    return int(part.sum()), int(part.size)


def measure(gray: Gray, box: tuple[float, float, float, float]) -> Ink:
    """The ink of ``box`` (``x, y, width, height`` in pixels of ``gray``).

    Dark pixels on table rules (a horizontal run longer than the box plus one ring margin, or
    a vertical run longer than twice its height: no glyph of the box's line is that long) do
    not count as ink, so a name in a ruled table cell is measured like one on plain paper.
    """
    width, height = box[2], box[3]
    margin = ring_margin(width, height)
    x0, y0, x1, y1 = _bounds(gray, box, margin + CONTEXT * height)
    neighbourhood = gray[y0:y1, x0:x1]
    if neighbourhood.size == 0:
        return Ink(0.0, 0.0)
    dark = neighbourhood < float(np.median(neighbourhood)) - CONTRAST
    rules = _long_runs(dark, width + margin) | _long_runs(dark.T, 2 * height).T
    ink = dark & ~rules
    inner_dark, inner_count = _share(ink, (x0, y0), _bounds(gray, box, 0))
    outer_dark, outer_count = _share(ink, (x0, y0), _bounds(gray, box, margin))
    ring_count = outer_count - inner_count
    inside = inner_dark / inner_count if inner_count else 0.0
    ring = (outer_dark - inner_dark) / ring_count if ring_count > 0 else 0.0
    return Ink(inside, ring)
