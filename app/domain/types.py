"""Immutable value objects shared by the domain, the ports and the adapters."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from app.domain.errors import GeometryError
from app.domain.language import SUPPORTED_LANGUAGES

_MAX_CONFIDENCE = 100


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
    """A word already converted to page space.

    ``line`` is the 0-based index of the text line the word belongs to on its page, as
    segmented by the recognition engine, with lines numbered in reading order. Words on the same
    printed line share an index. ``paragraph`` is the 0-based index of the paragraph the word
    belongs to on its page, numbered the same way; words of the same printed paragraph share it.
    ``confidence`` is the recognition engine's confidence in the word, from 0 to 100, or
    ``None`` when the engine reports none.
    """

    text: str
    box: Box
    line: int
    paragraph: int
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.line < 0:
            raise ValueError(f"line index must be >= 0, got {self.line}")
        if self.paragraph < 0:
            raise ValueError(f"paragraph index must be >= 0, got {self.paragraph}")
        if self.confidence is not None and not (
            isfinite(self.confidence) and 0 <= self.confidence <= _MAX_CONFIDENCE
        ):
            raise ValueError(f"confidence must be None or in 0..100, got {self.confidence}")


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
    """A piece of one page's cleaned text; ``start``/``end`` are offsets into that page text.

    ``flagged`` marks a chunk the injection detector scored at or above the threshold: it is
    stored with the document but never returned by a search.
    """

    text: str
    page: int
    sequence: int
    start: int
    end: int
    flagged: bool = False


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
class NameMention:
    """A person name found in a document and the page-space box that covers it."""

    name: str
    box: Box

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be blank")


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """The names found in a document, in reading order, and their matches to requested names."""

    names: tuple[NameMention, ...]
    matches: tuple[FuzzyMatch, ...]


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
    ``language`` is the code (from ``SUPPORTED_LANGUAGES``) of the language the page was read
    with, or ``None`` when it was read with several languages together.
    """

    page: int
    width: float
    height: float
    words: tuple[PageWord, ...]
    language: str | None

    def __post_init__(self) -> None:
        if self.page < 0:
            raise ValueError(f"page index must be >= 0, got {self.page}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("page width and height must be > 0")
        if any(word.box.page != self.page for word in self.words):
            raise ValueError(f"every word must be on page {self.page}")
        if self.language is not None and self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"language must be None or one of: {', '.join(SUPPORTED_LANGUAGES)}, "
                f"got {self.language!r}"
            )


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A stored chunk returned by a similarity search; ``score`` is cosine similarity in 0..1.

    ``page_text`` is the cleaned text of the whole page the chunk comes from (its parent page).
    """

    document_id: str
    page: int
    sequence: int
    text: str
    page_text: str
    score: float

    def __post_init__(self) -> None:
        if self.page < 0:
            raise ValueError(f"page index must be >= 0, got {self.page}")
        if self.sequence < 0:
            raise ValueError(f"sequence must be >= 0, got {self.sequence}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be within 0..1, got {self.score}")


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of ingesting one PDF.

    ``chunks_stored`` is the number of chunks held for ``document_id`` after the call. When
    ``already_indexed`` is true the file's content was found in the store and nothing was
    recognised, embedded or written, so at least one chunk is stored for it.
    ``chunks_flagged`` is how many of the chunks written by this call the injection detector
    flagged (0 when the document was already indexed: nothing was screened).
    """

    document_id: str
    chunks_stored: int
    already_indexed: bool
    chunks_flagged: int = 0

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be blank")
        if self.chunks_stored < 0:
            raise ValueError(f"chunks_stored must be >= 0, got {self.chunks_stored}")
        if self.already_indexed and self.chunks_stored == 0:
            raise ValueError("an already indexed document must have stored chunks")
        if not 0 <= self.chunks_flagged <= self.chunks_stored:
            raise ValueError(
                f"chunks_flagged must be within 0..chunks_stored, got {self.chunks_flagged}"
            )


@dataclass(frozen=True, slots=True)
class Answer:
    """A grounded answer and the exact texts of the retrieved chunks it was given, best first."""

    text: str
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("answer text must not be blank")
        if not isinstance(self.sources, tuple):
            raise ValueError("sources must be a tuple")
