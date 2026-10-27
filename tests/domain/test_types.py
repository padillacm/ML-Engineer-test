import dataclasses
import math

import pytest

from app.domain.errors import GeometryError
from app.domain.types import (
    Box,
    ChatMessage,
    Chunk,
    EmbeddingIdentity,
    FuzzyMatch,
    NamePair,
    OcrPage,
    PageWord,
    PixelBox,
    TextSpan,
)


def test_box_edges_are_measured_from_the_top_left_origin() -> None:
    box = Box(page=0, x=72, y=100, width=120, height=14)

    assert box.right == 192
    assert box.bottom == 114


def test_box_allows_zero_size() -> None:
    box = Box(page=2, x=5, y=5, width=0, height=0)

    assert (box.right, box.bottom) == (5, 5)


@pytest.mark.parametrize(
    ("page", "width", "height"),
    [(-1, 1, 1), (0, -0.1, 1), (0, 1, -0.1)],
)
def test_box_rejects_invalid_values(page: int, width: float, height: float) -> None:
    with pytest.raises(GeometryError):
        Box(page=page, x=0, y=0, width=width, height=height)


def test_equal_values_compare_equal_and_hash_equal() -> None:
    assert Box(0, 1, 2, 3, 4) == Box(0, 1, 2, 3, 4)
    assert hash(Box(0, 1, 2, 3, 4)) == hash(Box(0, 1, 2, 3, 4))
    assert Box(0, 1, 2, 3, 4) != Box(1, 1, 2, 3, 4)
    assert NamePair("Maria", "Gonzalez") == NamePair("Maria", "Gonzalez")
    assert TextSpan(1, 4) != TextSpan(1, 5)


def test_nested_values_compare_by_content() -> None:
    assert PixelBox(1, 2, 3, 4) == PixelBox(1, 2, 3, 4)
    assert PageWord("a", Box(0, 1, 2, 3, 4)) == PageWord("a", Box(0, 1, 2, 3, 4))


@pytest.mark.parametrize(
    ("instance", "field"),
    [
        (Box(0, 1, 2, 3, 4), "x"),
        (PixelBox(1, 2, 3, 4), "x"),
        (PageWord("a", Box(0, 1, 2, 3, 4)), "text"),
        (TextSpan(0, 1), "end"),
        (Chunk("t", 0, 0, 0, 1), "text"),
        (NamePair("a", "b"), "first_name"),
        (FuzzyMatch("a", "b", 0.5), "score"),
        (ChatMessage("user", "hi"), "content"),
        (EmbeddingIdentity("m", 3, "none", True, "cosine"), "dimension"),
    ],
)
def test_values_are_immutable(instance: object, field: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, None)


def test_text_span_accepts_empty_and_rejects_reversed_or_negative() -> None:
    assert TextSpan(3, 3).end == 3
    with pytest.raises(ValueError, match="invalid span"):
        TextSpan(4, 3)
    with pytest.raises(ValueError, match="invalid span"):
        TextSpan(-1, 3)


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_fuzzy_match_rejects_score_outside_unit_interval(score: float) -> None:
    with pytest.raises(ValueError, match="score must be within"):
        FuzzyMatch("a", "b", score)


@pytest.mark.parametrize("score", [0.0, 0.9, 1.0])
def test_fuzzy_match_accepts_scores_in_unit_interval(score: float) -> None:
    assert FuzzyMatch("a", "b", score).score == score


@pytest.mark.parametrize("dimension", [0, -384])
def test_embedding_identity_rejects_non_positive_dimension(dimension: int) -> None:
    with pytest.raises(ValueError, match="dimension must be"):
        EmbeddingIdentity("m", dimension, "none", True, "cosine")


def test_embedding_identities_differing_in_one_field_are_unequal() -> None:
    base = EmbeddingIdentity("m", 384, "e5", True, "cosine")

    assert base == EmbeddingIdentity("m", 384, "e5", True, "cosine")
    assert base != dataclasses.replace(base, prefix_scheme="none")
    assert base != dataclasses.replace(base, dimension=768)


_NON_FINITE = [math.nan, math.inf, -math.inf]


@pytest.mark.parametrize("bad", _NON_FINITE)
@pytest.mark.parametrize("field", ["x", "y", "width", "height"])
def test_box_rejects_non_finite_values(field: str, bad: float) -> None:
    values = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0, field: bad}

    with pytest.raises(GeometryError):
        Box(page=0, **values)


def test_box_rejects_edges_that_overflow() -> None:
    with pytest.raises(GeometryError):
        Box(page=0, x=1e308, y=0, width=1e308, height=1)


@pytest.mark.parametrize("bad", _NON_FINITE)
@pytest.mark.parametrize("field", ["x", "y", "width", "height"])
def test_pixel_box_rejects_non_finite_values(field: str, bad: float) -> None:
    values = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0, field: bad}

    with pytest.raises(GeometryError):
        PixelBox(**values)


def test_ocr_page_rejects_a_word_from_another_page() -> None:
    with pytest.raises(ValueError, match="every word must be on page 1"):
        OcrPage(page=1, width=595, height=842, words=(PageWord("a", Box(0, 1, 2, 3, 4)),))


def test_ocr_page_accepts_words_of_its_own_page() -> None:
    word = PageWord("a", Box(1, 1, 2, 3, 4))

    assert OcrPage(page=1, width=595, height=842, words=(word,)).words == (word,)
