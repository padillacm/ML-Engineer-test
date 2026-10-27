"""Cleaning of recognised page text and its splitting into overlapping, word-safe chunks.

Cleaning joins words hyphenated across a line break, turns single line breaks into spaces and
keeps blank lines as paragraph separators. Splitting packs words greedily into chunks of at most
``max_chars`` characters, ending each chunk at the strongest boundary that fits: a paragraph,
then a sentence, then a word. A period ends a sentence unless the word before it is a known
title or abbreviation ("Dr.", "Sra.", "etc."), a single capital initial ("J.") or a dotted
acronym ("U.S.", "e.g."); ordinary words, however short ("it.", "so."), do end it.
"""

import itertools
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.errors import ConfigurationError
from app.domain.types import Chunk

DEFAULT_MAX_CHARS = 800
DEFAULT_OVERLAP_RATIO = 0.12
_OVERLAP_RATIO_LIMIT = 0.5  # exclusive upper bound

_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"\S+")
# A word ending a sentence, optionally followed by closing quotes or brackets.
_SENTENCE_END = re.compile(r"[.?!…][\"'\u201d\u2019»)\]]*$")
_CLOSERS = "\"'\u201d\u2019»)]"
_OPENERS = "\"'“\u2018«([¿¡"
# Letters separated by single dots: "U.S", "e.g", "S.L" (the final period is stripped first).
_DOTTED_ACRONYM = re.compile(r"[^\W\d_](?:\.[^\W\d_])+$")
# Titles and abbreviations (case-insensitive, without their period) that never end a sentence.
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "mme", "mlle", "dr", "dra", "prof", "profa", "sr", "sra", "srta",
        "sres", "dña", "don", "ing", "lic", "m", "hr", "hrn", "fr", "jr", "st", "vs", "etc",
    }
)  # fmt: skip
# Hyphens that may end a line inside a split word; the soft hyphen is invisible.
_LINE_END_HYPHENS = frozenset("-\N{HYPHEN}\N{SOFT HYPHEN}")

_WORD_GAP = 0
_SENTENCE_GAP = 1
_PARAGRAPH_GAP = 2
_PARAGRAPH_NEWLINES = 2


def clean_page_text(text: str) -> str:
    """Normalise recognised text into paragraphs joined by a blank line.

    A hyphen (``-``, U+2010 or the invisible soft hyphen U+00AD) ending a line after a letter is
    dropped when the next line starts with a lowercase letter, since the word was split by the
    line break; before an uppercase letter ("Karl-Heinz") it is kept. Any other line break
    inside a paragraph becomes a space, and one or more blank lines separate paragraphs. Runs in
    time linear in the length of the text.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in lines:
        line = _WHITESPACE.sub(" ", raw_line).strip()
        if line:
            current.append(line)
        elif current:
            paragraphs.append(_join_lines(current))
            current = []
    if current:
        paragraphs.append(_join_lines(current))
    return "\n\n".join(paragraphs)


def _join_lines(lines: list[str]) -> str:
    parts = [lines[0]]
    previous = lines[0]
    for line in lines[1:]:
        if (
            len(previous) > 1
            and previous[-1] in _LINE_END_HYPHENS
            and previous[-2].isalpha()
            and line[0].islower()
        ):
            parts[-1] = previous[:-1]
        else:
            parts.append(" ")
        parts.append(line)
        previous = line
    return "".join(parts)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Chunk size limit and the share of it that consecutive chunks may repeat.

    ``max_chars`` (default 800) bounds the length of a chunk; only a single word longer than it
    is emitted whole. ``overlap_ratio`` (default 0.12, valid within [0, 0.5)) bounds the text a
    chunk repeats from the end of the previous one to ``floor(overlap_ratio * max_chars)``
    characters, starting at a word.
    """

    max_chars: int = DEFAULT_MAX_CHARS
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ConfigurationError(f"max_chars must be > 0, got {self.max_chars}")
        if not 0 <= self.overlap_ratio < _OVERLAP_RATIO_LIMIT:
            raise ConfigurationError(
                f"overlap_ratio must be within [0, 0.5), got {self.overlap_ratio}"
            )

    @property
    def max_overlap_chars(self) -> int:
        return math.floor(self.overlap_ratio * self.max_chars)


def chunk_page(
    cleaned_text: str,
    page: int,
    config: ChunkingConfig,
    first_sequence: int = 0,
) -> list[Chunk]:
    """Split one page of already cleaned text into chunks.

    Each chunk is the contiguous slice ``cleaned_text[start:end]``. A chunk ends at the
    strongest natural boundary (paragraph, then sentence, then word) that keeps it within
    ``config.max_chars``; only a single word longer than that limit exceeds it.
    """
    words = [(m.start(), m.end()) for m in _WORD.finditer(cleaned_text)]
    if not words:
        return []
    gaps = _gap_levels(cleaned_text, words)

    chunks: list[Chunk] = []
    first = 0
    previous: tuple[int, int] | None = None
    while first < len(words):
        start = _overlap_start(words, previous, first, config)
        last = _last_fitting_word(words, start, first, config.max_chars)
        if last < first:
            last = first  # a word longer than the limit is emitted whole
        elif last < len(words) - 1:
            last = _best_boundary(gaps, first, last)
        end = words[last][1]
        chunks.append(
            Chunk(
                text=cleaned_text[start:end],
                page=page,
                sequence=first_sequence + len(chunks),
                start=start,
                end=end,
            )
        )
        previous = (start, end)
        first = last + 1
    return chunks


def chunk_document(
    pages: Sequence[str],
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Clean and chunk every page independently, numbering chunks across the document."""
    config = config if config is not None else ChunkingConfig()
    chunks: list[Chunk] = []
    for page, raw_text in enumerate(pages):
        chunks.extend(chunk_page(clean_page_text(raw_text), page, config, len(chunks)))
    return chunks


def _gap_levels(text: str, words: list[tuple[int, int]]) -> list[int]:
    """Strength of the boundary after each word except the last: paragraph > sentence > word."""
    levels: list[int] = []
    for (start, end), (next_start, _) in itertools.pairwise(words):
        if text.count("\n", end, next_start) >= _PARAGRAPH_NEWLINES:
            levels.append(_PARAGRAPH_GAP)
        elif _ends_sentence(text[start:end]):
            levels.append(_SENTENCE_GAP)
        else:
            levels.append(_WORD_GAP)
    return levels


def _ends_sentence(word: str) -> bool:
    if not _SENTENCE_END.search(word):
        return False
    stripped = word.rstrip(_CLOSERS)
    return not (stripped.endswith(".") and _is_abbreviation(stripped[:-1]))


def _is_abbreviation(token: str) -> bool:
    """Whether the text before a period is a title, an initial or a dotted acronym."""
    token = token.lstrip(_OPENERS)
    if token.casefold() in _ABBREVIATIONS:
        return True
    if len(token) == 1 and token.isupper():
        return True
    return bool(_DOTTED_ACRONYM.match(token))


def _overlap_start(
    words: list[tuple[int, int]],
    previous: tuple[int, int] | None,
    first: int,
    config: ChunkingConfig,
) -> int:
    """Offset where the chunk starting with word ``first`` begins, repeating previous text.

    The repeated text is the largest proper suffix of the previous chunk that starts at a word
    and is at most ``max_overlap_chars`` long; it is shortened (possibly to nothing) until the
    chunk's first new word still fits in ``max_chars``.
    """
    own_start, own_end = words[first]
    if previous is None:
        return own_start
    previous_start, previous_end = previous
    earliest = max(
        previous_start + 1,
        previous_end - config.max_overlap_chars,
        own_end - config.max_chars,
    )
    start = own_start
    for index in range(first - 1, -1, -1):
        word_start = words[index][0]
        if word_start < earliest:
            break
        start = word_start
    return start


def _last_fitting_word(words: list[tuple[int, int]], start: int, first: int, max_chars: int) -> int:
    """Index of the last word whose end keeps the chunk from ``start`` within ``max_chars``."""
    last = first - 1
    for index in range(first, len(words)):
        if words[index][1] - start > max_chars:
            break
        last = index
    return last


def _best_boundary(gaps: list[int], first: int, last: int) -> int:
    """Pick the last word of a chunk: the furthest strongest boundary within ``first..last``."""
    best = last
    best_level = gaps[last]
    for index in range(last - 1, first - 1, -1):
        if gaps[index] > best_level:
            best, best_level = index, gaps[index]
    return best
