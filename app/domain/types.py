"""Immutable value objects shared by the domain, the ports and the adapters."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from app.domain.errors import GeometryError


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned rectangle in PDF page space.

    Units are points (1/72 inch), the origin is the top-left corner of the page, x grows right
    and y grows down. ``page`` is the 0-based page index.
    """

    page: int
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.page < 0:
            raise GeometryError(f"page index must be >= 0, got {self.page}")
        values = (
            self.x,
            self.y,
            self.width,
            self.height,
            self.x + self.width,
            self.y + self.height,
        )
        if not all(isfinite(value) for value in values):
            raise GeometryError("box coordinates and edges must be finite")
        if self.width < 0 or self.height < 0:
            raise GeometryError("box width and height must be >= 0")

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class PixelBox:
    """Axis-aligned rectangle in raster pixels (top-left origin) with no page attached."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y, self.width, self.height)):
            raise GeometryError("pixel box coordinates must be finite")


@dataclass(frozen=True, slots=True)
class PageWord:
    """A word already converted to page space."""

    text: str
    box: Box


@dataclass(frozen=True, slots=True)
class TextSpan:
    """Half-open character range ``[start, end)`` into some text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if not 0 <= self.start <= self.end:
            raise ValueError(f"invalid span: start={self.start}, end={self.end}")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A piece of one page's cleaned text; ``start``/``end`` are offsets into that page text."""

    text: str
    page: int
    sequence: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class NamePair:
    first_name: str
    last_name: str


@dataclass(frozen=True, slots=True)
class FuzzyMatch:
    extracted_name: str
    matched_name: str
    score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be within 0..1, got {self.score}")


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """What makes stored vectors interchangeable between two embedding providers."""

    model_id: str
    dimension: int
    prefix_scheme: str
    normalized: bool
    metric: str

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError(f"dimension must be > 0, got {self.dimension}")


@dataclass(frozen=True, slots=True)
class OcrPage:
    """OCR result of one PDF page, with words already converted to page space.

    ``width`` and ``height`` are the page size in points. Every word must lie on ``page``.
    """

    page: int
    width: float
    height: float
    words: tuple[PageWord, ...]

    def __post_init__(self) -> None:
        if self.page < 0:
            raise ValueError(f"page index must be >= 0, got {self.page}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("page width and height must be > 0")
        if any(word.box.page != self.page for word in self.words):
            raise ValueError(f"every word must be on page {self.page}")


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A stored chunk returned by a similarity search; ``score`` is cosine similarity in 0..1."""

    document_id: str
    page: int
    sequence: int
    text: str
    score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be within 0..1, got {self.score}")
