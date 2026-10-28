import dataclasses
import math
import random

import pytest

from app.domain.errors import GeometryError
from app.domain.geometry import PageRaster, PageText, clamp, iou, union
from app.domain.types import Box, PageWord, PixelBox, TextSpan

A4_WIDTH = 595.0
A4_HEIGHT = 842.0


def _word(text: str, x: float, y: float, width: float, height: float, page: int = 0) -> PageWord:
    return PageWord(text=text, box=Box(page=page, x=x, y=y, width=width, height=height), line=0)


def _signed_by_maria_gonzalez_today() -> PageText:
    return PageText.from_words(
        [
            _word("Signed", 10, 20, 40, 10),
            _word("by", 55, 20, 15, 10),
            _word("Maria", 75, 21, 35, 10),
            _word("Gonzalez", 115, 20, 55, 12),
            _word("today", 175, 22, 30, 9),
        ]
    )


def _assert_box(box: Box, page: int, x: float, y: float, width: float, height: float) -> None:
    assert box.page == page
    assert (box.x, box.y, box.width, box.height) == pytest.approx((x, y, width, height), abs=0.01)


# --- PageRaster.to_points -------------------------------------------------------------------


def test_image_rendered_at_an_unexpected_resolution_is_converted_per_axis() -> None:
    raster = PageRaster(
        width_px=1653, height_px=2339, page_width_pt=A4_WIDTH, page_height_pt=A4_HEIGHT
    )

    box = raster.to_points(0, PixelBox(x=165.3, y=233.9, width=330.6, height=46.78))

    _assert_box(box, 0, 59.5, 84.2, 119.0, 16.84)


def test_rounded_image_width_uses_the_measured_pixel_width() -> None:
    raster = PageRaster(width_px=1654, height_px=2339, page_width_pt=A4_WIDTH, page_height_pt=842)

    right_edge = raster.to_points(0, PixelBox(x=1554, y=0, width=100, height=10))

    assert right_edge.x == pytest.approx(1554 * 595 / 1654)
    assert right_edge.right <= 595.0


def test_axes_are_scaled_independently() -> None:
    raster = PageRaster(width_px=200, height_px=100, page_width_pt=400, page_height_pt=100)

    box = raster.to_points(3, PixelBox(x=10, y=10, width=20, height=30))

    _assert_box(box, 3, 20, 10, 40, 30)


@pytest.mark.parametrize(
    ("width_px", "height_px", "page_width_pt", "page_height_pt"),
    [(0, 10, 10, 10), (10, 0, 10, 10), (10, 10, 0, 10), (10, 10, 10, -1)],
)
def test_raster_rejects_non_positive_dimensions(
    width_px: float, height_px: float, page_width_pt: float, page_height_pt: float
) -> None:
    with pytest.raises(GeometryError):
        PageRaster(width_px, height_px, page_width_pt, page_height_pt)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("position", range(4))
def test_raster_rejects_non_finite_dimensions(position: int, bad: float) -> None:
    dimensions = [10.0, 10.0, 10.0, 10.0]
    dimensions[position] = bad

    with pytest.raises(GeometryError):
        PageRaster(*dimensions)


@pytest.mark.parametrize(
    "raster_size", [(1654, 2339), (1653, 2339), (1, 1), (999, 1234), (4961, 7016)]
)
def test_boxes_reaching_the_raster_border_end_at_the_page_border(
    raster_size: tuple[int, int],
) -> None:
    width_px, height_px = raster_size
    raster = PageRaster(width_px, height_px, A4_WIDTH, A4_HEIGHT)

    box = raster.to_points(0, PixelBox(x=0, y=0, width=width_px, height=height_px))

    assert box.right == pytest.approx(A4_WIDTH, abs=1e-9)
    assert box.bottom == pytest.approx(A4_HEIGHT, abs=1e-9)


def test_converted_boxes_never_leave_the_page_by_more_than_rounding() -> None:
    rng = random.Random(0)
    for _ in range(100_000):
        width_px = rng.randint(1, 8000)
        height_px = rng.randint(1, 8000)
        raster = PageRaster(width_px, height_px, rng.uniform(100, 1500), rng.uniform(100, 1500))
        left = rng.uniform(0, width_px)
        top = rng.uniform(0, height_px)
        # Half of the boxes touch the raster border exactly.
        right = width_px if rng.random() < 0.5 else rng.uniform(left, width_px)
        bottom = height_px if rng.random() < 0.5 else rng.uniform(top, height_px)

        box = raster.to_points(0, PixelBox(left, top, right - left, bottom - top))

        assert box.right <= raster.page_width_pt + 1e-9
        assert box.bottom <= raster.page_height_pt + 1e-9


def test_raster_is_immutable() -> None:
    raster = PageRaster(width_px=10, height_px=10, page_width_pt=10, page_height_pt=10)

    with pytest.raises(dataclasses.FrozenInstanceError):
        raster.width_px = 5  # type: ignore[misc]  # frozen on purpose


# --- union ----------------------------------------------------------------------------------


def test_two_words_on_one_line_are_united() -> None:
    result = union([Box(0, 10, 20, 30, 10), Box(0, 45, 21, 40, 10)])

    assert result == Box(page=0, x=10, y=20, width=75, height=11)


def test_union_of_one_box_is_that_box() -> None:
    box = Box(2, 1, 2, 3, 4)

    assert union([box]) == box


def test_union_of_boxes_on_different_pages_fails() -> None:
    with pytest.raises(GeometryError):
        union([Box(0, 10, 20, 30, 10), Box(1, 10, 20, 30, 10)])


def test_union_of_nothing_fails() -> None:
    with pytest.raises(GeometryError):
        union([])


# --- clamp ----------------------------------------------------------------------------------


def test_box_slightly_beyond_the_right_edge_is_clipped_and_keeps_its_left_edge() -> None:
    box = Box(page=0, x=500, y=100, width=97, height=14)

    result = clamp(box, A4_WIDTH, A4_HEIGHT)

    assert result == Box(page=0, x=500, y=100, width=95, height=14)
    assert result.right == 595


def test_box_beyond_the_top_left_corner_is_clipped() -> None:
    result = clamp(Box(0, -5, -3, 20, 10), A4_WIDTH, A4_HEIGHT)

    assert result == Box(page=0, x=0, y=0, width=15, height=7)


def test_box_inside_the_page_is_unchanged() -> None:
    box = Box(1, 72, 100, 120, 14)

    assert clamp(box, A4_WIDTH, A4_HEIGHT) == box


@pytest.mark.parametrize(
    "box",
    [
        Box(0, 700, 100, 20, 10),
        Box(0, 100, 900, 20, 10),
        Box(0, -30, 100, 20, 10),
        Box(0, 100, -30, 20, 10),
        Box(0, 595, 100, 20, 10),
    ],
)
def test_box_fully_outside_the_page_is_rejected(box: Box) -> None:
    with pytest.raises(GeometryError):
        clamp(box, A4_WIDTH, A4_HEIGHT)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("axis", ["page_width_pt", "page_height_pt"])
def test_clamp_rejects_non_finite_page_size(axis: str, bad: float) -> None:
    sizes = {"page_width_pt": A4_WIDTH, "page_height_pt": A4_HEIGHT, axis: bad}

    with pytest.raises(GeometryError):
        clamp(Box(0, 1, 1, 2, 2), **sizes)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_boxes_cannot_reach_clamp(bad: float) -> None:
    with pytest.raises(GeometryError):
        clamp(Box(0, bad, 1, 2, 2), A4_WIDTH, A4_HEIGHT)


# --- iou ------------------------------------------------------------------------------------


def test_identical_boxes_overlap_completely() -> None:
    box = Box(0, 10, 20, 30, 10)

    assert iou(box, box) == 1.0


def test_disjoint_boxes_do_not_overlap() -> None:
    assert iou(Box(0, 0, 0, 10, 10), Box(0, 50, 50, 10, 10)) == 0.0


def test_touching_boxes_do_not_overlap() -> None:
    assert iou(Box(0, 0, 0, 10, 10), Box(0, 10, 0, 10, 10)) == 0.0


def test_same_position_on_different_pages_does_not_overlap() -> None:
    assert iou(Box(0, 10, 20, 30, 10), Box(1, 10, 20, 30, 10)) == 0.0


def test_partial_overlap_is_intersection_over_union() -> None:
    assert iou(Box(0, 0, 0, 10, 10), Box(0, 5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_degenerate_boxes_have_no_overlap() -> None:
    zero = Box(0, 5, 5, 0, 0)

    assert iou(zero, zero) == 0.0


def test_iou_of_extreme_valid_boxes_is_never_nan() -> None:
    huge = Box(0, 0, 0, 1e200, 1e200)
    tiny = Box(0, 0, 0, 1e-200, 1e-200)

    assert iou(huge, huge) == 1.0
    assert iou(tiny, tiny) == 1.0
    assert iou(huge, tiny) == pytest.approx(0.0, abs=1e-300)


# --- PageText -------------------------------------------------------------------------------


def test_page_text_joins_words_with_single_spaces_and_tracks_offsets() -> None:
    page_text = _signed_by_maria_gonzalez_today()

    assert page_text.text == "Signed by Maria Gonzalez today"
    assert page_text.offsets == ((0, 6), (7, 9), (10, 15), (16, 24), (25, 30))
    assert [word.text for word in page_text.words] == ["Signed", "by", "Maria", "Gonzalez", "today"]


def test_page_text_accepts_a_generator_of_words() -> None:
    words = (_word(text, 10 * i, 0, 8, 10) for i, text in enumerate(["Maria", "", "Gonzalez"]))

    page_text = PageText.from_words(words)

    assert page_text.text == "Maria Gonzalez"
    assert len(page_text.words) == len(page_text.offsets) == 2


def test_page_text_of_no_words_is_empty() -> None:
    page_text = PageText.from_words([])

    assert page_text.text == ""
    assert page_text.words == ()
    assert page_text.offsets == ()
    assert page_text.locate(TextSpan(0, 1)) is None


def test_page_text_rejects_words_from_different_pages() -> None:
    with pytest.raises(GeometryError):
        PageText.from_words([_word("a", 0, 0, 1, 1, page=0), _word("b", 0, 0, 1, 1, page=1)])


def test_span_covering_a_full_name_yields_the_union_of_those_words_only() -> None:
    page_text = _signed_by_maria_gonzalez_today()
    start = page_text.text.index("Maria Gonzalez")

    box = page_text.locate(TextSpan(start, start + len("Maria Gonzalez")))

    assert box == union([page_text.words[2].box, page_text.words[3].box])
    assert box == Box(page=0, x=75, y=20, width=95, height=12)


def test_span_starting_inside_a_word_includes_the_whole_word() -> None:
    page_text = _signed_by_maria_gonzalez_today()
    start = page_text.text.index("Maria") + 2
    end = page_text.text.index("Gonzalez") + len("Gonzalez")

    box = page_text.locate(TextSpan(start, end))

    assert box is not None
    assert box.x == 75
    assert box == union([page_text.words[2].box, page_text.words[3].box])


def test_span_ending_inside_a_word_includes_the_whole_word() -> None:
    page_text = _signed_by_maria_gonzalez_today()

    box = page_text.locate(TextSpan(0, 2))

    assert box == page_text.words[0].box


def test_blank_words_contribute_no_characters_and_no_box() -> None:
    page_text = PageText.from_words(
        [
            _word("Maria", 10, 10, 30, 10),
            _word("  \t", 500, 700, 50, 10),
            _word("", 400, 600, 5, 10),
            _word("Gonzalez", 45, 10, 50, 10),
        ]
    )

    assert page_text.text == "Maria Gonzalez"
    assert len(page_text.words) == len(page_text.offsets) == 2
    assert page_text.locate(TextSpan(0, len(page_text.text))) == Box(0, 10, 10, 85, 10)


def test_page_text_of_only_blank_words_is_empty() -> None:
    page_text = PageText.from_words([_word(" ", 0, 0, 1, 1)])

    assert page_text.text == ""
    assert page_text.locate(TextSpan(0, 5)) is None


def test_span_after_the_last_character_yields_no_box() -> None:
    page_text = _signed_by_maria_gonzalez_today()
    end = len(page_text.text)

    assert page_text.locate(TextSpan(end, end + 5)) is None
    assert page_text.locate(TextSpan(end + 3, end + 8)) is None


def test_span_covering_only_the_separating_space_yields_no_box() -> None:
    page_text = _signed_by_maria_gonzalez_today()

    assert page_text.locate(TextSpan(6, 7)) is None


def test_empty_span_yields_no_box() -> None:
    page_text = _signed_by_maria_gonzalez_today()

    assert page_text.locate(TextSpan(12, 12)) is None


def test_locate_does_not_clamp_to_the_page() -> None:
    page_text = PageText.from_words([_word("edge", 590, 10, 20, 10)])

    box = page_text.locate(TextSpan(0, 4))

    assert box is not None
    assert box.right == 610
    assert clamp(box, A4_WIDTH, A4_HEIGHT).right == 595


def test_located_box_contains_every_overlapped_word_for_generated_pages() -> None:
    rng = random.Random(0)
    alphabet = "abcdefghij"
    for _ in range(300):
        words = [
            _word(
                text="".join(rng.choices(alphabet, k=rng.randint(1, 8)))
                if rng.random() > 0.2
                else " ",
                x=rng.uniform(0, 500),
                y=rng.uniform(0, 800),
                width=rng.uniform(1, 90),
                height=rng.uniform(1, 30),
            )
            for _ in range(rng.randint(0, 12))
        ]
        page_text = PageText.from_words(words)
        start = rng.randint(0, len(page_text.text) + 3)
        span = TextSpan(start, rng.randint(start, len(page_text.text) + 6))

        located = page_text.locate(span)

        covered = set(range(span.start, span.end))
        overlapped = [
            word
            for word, (word_start, word_end) in zip(page_text.words, page_text.offsets, strict=True)
            if covered & set(range(word_start, word_end))
        ]
        for word, (word_start, word_end) in zip(page_text.words, page_text.offsets, strict=True):
            assert page_text.text[word_start:word_end] == word.text
        if not overlapped:
            assert located is None
            continue
        assert located is not None
        for word in overlapped:
            # right/bottom are recomputed as x + width, which may lose one ulp.
            assert located.x <= word.box.x
            assert located.y <= word.box.y
            assert located.right >= word.box.right - 1e-9
            assert located.bottom >= word.box.bottom - 1e-9
        assert located == union([word.box for word in overlapped])
