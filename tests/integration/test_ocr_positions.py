"""Recognised word boxes land on the rectangles where the words were drawn (real Tesseract).

The page mixes names from four languages. Each single-language model reads its own language's
diacritics reliably (``spa`` -> Ramírez, ``deu`` -> Müller, ``por`` -> São, ``fra`` -> François),
while the combined model often drops them (Ramirez, Sao, Miller...). So every page is recognised
once per hint and each accented word is checked, spelled exactly, in its own language's result;
the unaccented words are checked in every result.
"""

from collections.abc import Callable

import numpy as np
import pymupdf
import pytest

from app.adapters.ocr import TesseractOcrConfig, TesseractOcrEngine
from app.domain.geometry import iou
from app.domain.types import Box, OcrPage
from tests.adapters.ocr.pdfs import A4, LETTER
from tests.integration.known_words import (
    KnownPage,
    Line,
    best_iou,
    boxes_of,
    known_words_page,
    recognised_text,
)

pytestmark = pytest.mark.tesseract

MIN_IOU = 0.5

LINES = (
    Line((72.0, 120.0), "Contract signed by Ramírez"),
    Line((72.0, 180.0), "Witness Müller from São Paulo"),
    Line((72.0, 240.0), "Approved by François Dupont"),
    Line((72.0, 300.0), "Total amount 4500 euros"),
)
ACCENTED = {"Ramírez": "es", "Müller": "de", "São": "pt", "François": "fr"}
PLAIN = ("Contract", "Approved", "Paulo", "Dupont")
TARGETS = (*ACCENTED, *PLAIN)
HINTS = tuple(sorted(set(ACCENTED.values())))

LAYOUTS = [
    pytest.param(A4, 0, id="a4"),
    pytest.param(LETTER, 0, id="letter"),
    pytest.param(A4, 90, id="a4-rotated-90"),
    pytest.param(LETTER, 90, id="letter-rotated-90"),
]

type Layout = tuple[tuple[float, float], int]
type Recognize = Callable[[tuple[float, float], int, int], dict[str, OcrPage]]


@pytest.fixture(scope="module")
def pages() -> Callable[[tuple[float, float], int], KnownPage]:
    """Build each known-words page once per module."""
    cache: dict[Layout, KnownPage] = {}

    def build(size: tuple[float, float], rotation: int) -> KnownPage:
        if (size, rotation) not in cache:
            cache[size, rotation] = known_words_page(size, LINES, TARGETS, rotation=rotation)
        return cache[size, rotation]

    return build


@pytest.fixture(scope="module")
def recognize(pages: Callable[[tuple[float, float], int], KnownPage]) -> Recognize:
    """Recognise a layout at a render DPI once per hint, cached per module."""
    cache: dict[tuple[tuple[float, float], int, int], dict[str, OcrPage]] = {}

    def run(size: tuple[float, float], rotation: int, dpi: int) -> dict[str, OcrPage]:
        key = (size, rotation, dpi)
        if key not in cache:
            engine = TesseractOcrEngine(TesseractOcrConfig(dpi=dpi))
            pdf = pages(size, rotation).pdf
            cache[key] = {hint: engine.recognize(pdf, hint)[0] for hint in HINTS}
        return cache[key]

    return run


def _ious(results: dict[str, OcrPage], page: KnownPage) -> dict[str, float]:
    """IoU of every target against its ground truth: accented words in their language only."""
    ious: dict[str, float] = {}
    for word, box in page.targets.items():
        hints = (ACCENTED[word],) if word in ACCENTED else HINTS
        for hint in hints:
            ious[f"{word}/{hint}"] = round(best_iou(results[hint], word, box), 3)
    return ious


def _report(label: str, ious: dict[str, float]) -> None:
    values = list(ious.values())
    print(f"{label}: IoU min={min(values):.3f} mean={sum(values) / len(values):.3f} {ious}")


def _assert_all_found(ious: dict[str, float], results: dict[str, OcrPage]) -> None:
    low = {key: value for key, value in ious.items() if value < MIN_IOU}
    read = {hint: recognised_text(page) for hint, page in results.items()}
    assert not low, f"IoU below {MIN_IOU}: {low}; recognised: {read}"


def _ink_box(gray: np.ndarray, around: Box, scale: float, margin: float) -> Box | None:
    """Return the box (points) of the dark pixels within ``margin`` points of ``around``."""
    left = max(int((around.x - margin) * scale), 0)
    top = max(int((around.y - margin) * scale), 0)
    right = int((around.right + margin) * scale)
    bottom = int((around.bottom + margin) * scale)
    ys, xs = np.nonzero(gray[top:bottom, left:right] < 128)
    if xs.size == 0:
        return None
    return Box(
        page=0,
        x=(left + xs.min()) / scale,
        y=(top + ys.min()) / scale,
        width=(xs.max() - xs.min() + 1) / scale,
        height=(ys.max() - ys.min() + 1) / scale,
    )


@pytest.mark.parametrize(("size", "rotation"), LAYOUTS)
def test_ground_truth_is_in_the_displayed_page_space(
    pages: Callable[[tuple[float, float], int], KnownPage], size: tuple[float, float], rotation: int
) -> None:
    """The ground truth sits on the ink of the page as it is displayed (rotation applied).

    This pins the PyMuPDF behaviour the ground truth relies on: ``search_for`` answers in the
    unrotated frame and ``rotation_matrix`` maps that to the displayed page.
    """
    page = pages(size, rotation)
    dpi = 150
    scale = dpi / 72.0
    with pymupdf.open(stream=page.pdf, filetype="pdf") as document:
        shown = document[0]
        assert shown.rotation == rotation
        assert (shown.rect.width, shown.rect.height) == pytest.approx((page.width, page.height))
        pixmap = shown.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
        rows = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(-1, pixmap.stride)
        gray = rows[:, : pixmap.width]
    for word, truth in page.targets.items():
        ink = _ink_box(gray, truth, scale, margin=2.0)
        assert ink is not None, f"no ink around the ground truth of {word!r}"
        assert iou(ink, truth) >= MIN_IOU, (word, ink, truth)


@pytest.mark.parametrize(("size", "rotation"), LAYOUTS)
def test_words_are_found_at_the_rectangles_where_they_were_drawn(
    pages: Callable[[tuple[float, float], int], KnownPage],
    recognize: Recognize,
    size: tuple[float, float],
    rotation: int,
) -> None:
    page = pages(size, rotation)

    results = recognize(size, rotation, 200)

    for result in results.values():
        assert (result.width, result.height) == pytest.approx((page.width, page.height))
    ious = _ious(results, page)
    _report(f"{size} rotation={rotation} dpi=200", ious)
    _assert_all_found(ious, results)


@pytest.mark.parametrize("dpi", [150, 300])
def test_positions_hold_at_each_render_resolution(
    pages: Callable[[tuple[float, float], int], KnownPage], recognize: Recognize, dpi: int
) -> None:
    results = recognize(LETTER, 0, dpi)

    ious = _ious(results, pages(LETTER, 0))

    _report(f"letter dpi={dpi}", ious)
    _assert_all_found(ious, results)


def test_the_same_word_overlaps_across_render_resolutions(recognize: Recognize) -> None:
    low, high = recognize(LETTER, 0, 150), recognize(LETTER, 0, 300)
    ious: dict[str, float] = {}
    for word in TARGETS:
        for hint in (ACCENTED[word],) if word in ACCENTED else HINTS:
            [box_low] = boxes_of(low[hint], word)
            [box_high] = boxes_of(high[hint], word)
            ious[f"{word}/{hint}"] = round(iou(box_low, box_high), 3)

    _report("150 vs 300 DPI", ious)
    assert all(value >= MIN_IOU for value in ious.values()), ious
