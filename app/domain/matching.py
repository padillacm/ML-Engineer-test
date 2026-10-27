"""Fuzzy comparison of person names found in documents with the names a caller asks for."""

import math
import unicodedata
from collections.abc import Sequence

from rapidfuzz import fuzz

from app.domain.errors import ConfigurationError
from app.domain.types import FuzzyMatch, NamePair

DEFAULT_THRESHOLD = 0.90

# Absorbs the rounding noise of ``ratio / 100`` (about 1e-15) so a score "exactly at" the
# threshold is reported.
_EPSILON = 1e-12

# Letters without a Unicode decomposition, apostrophe variants and hyphen or dash variants.
_FOLDING = str.maketrans(
    {
        **{"ø": "o", "ł": "l", "đ": "d", "ð": "d", "þ": "th", "æ": "ae", "œ": "oe", "\u0131": "i"},
        **dict.fromkeys("\u2019\u2018\u02bc`\u00b4", "'"),
        **dict.fromkeys("-\u2010\u2011\u2012\u2013\u2014\u2212", " "),
    }
)


def normalize(text: str) -> str:
    """Canonical form used for comparison.

    Removes accents, ignores case (``ß`` becomes ``ss``), folds letters without a
    decomposition (``ø``, ``ł``, ``đ``, ``æ``, ``œ``, dotless i), unifies apostrophe variants,
    treats hyphens and dashes as spaces and collapses whitespace.
    """
    # Folding runs before NFKD (which turns the standalone acute accent into a space plus an
    # accent) and after it (which can produce apostrophes, dashes and, after casefolding,
    # ``ø``-like letters).
    decomposed = unicodedata.normalize("NFKD", text.translate(_FOLDING))
    unmarked = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(unmarked.casefold().translate(_FOLDING).split())


def _similarity(extracted: str, requested: str) -> float:
    """Word-order-insensitive similarity of two already normalised texts, in ``0.0..1.0``."""
    return min(fuzz.token_sort_ratio(extracted, requested) / 100, 1.0)


def _requested_text(pair: NamePair) -> str:
    return f"{pair.first_name} {pair.last_name}"


def score(extracted: str, pair: NamePair) -> float:
    """Full-string similarity in ``0.0..1.0`` between an extracted name and a requested pair.

    Word order is ignored; a name that merely contains the requested one is not rewarded.
    """
    return _similarity(normalize(extracted), normalize(_requested_text(pair)))


def _validated_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise ConfigurationError(f"threshold must be a number within 0..1, got {threshold!r}")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ConfigurationError(f"threshold must be within 0..1, got {threshold}")
    return float(threshold)


def match_names(
    extracted_names: Sequence[str],
    pairs: Sequence[NamePair],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[FuzzyMatch]:
    """Report, for each extracted name, the best requested pair scoring at or above ``threshold``.

    Ties go to the pair that appears first in ``pairs``. Names that are blank after
    normalisation are skipped, as are requested pairs whose names are both blank; repeated
    names are reported once per occurrence.
    """
    limit = _validated_threshold(threshold) - _EPSILON

    requested = [
        (pair, normalized) for pair in pairs if (normalized := normalize(_requested_text(pair)))
    ]

    matches: list[FuzzyMatch] = []
    for extracted in extracted_names:
        normalized_name = normalize(extracted)
        if not normalized_name:
            continue
        best_pair: NamePair | None = None
        best_score = 0.0
        for pair, normalized_pair in requested:
            candidate = _similarity(normalized_name, normalized_pair)
            if candidate >= limit and (best_pair is None or candidate > best_score):
                best_pair, best_score = pair, candidate
        if best_pair is not None:
            matches.append(
                FuzzyMatch(
                    extracted_name=extracted,
                    matched_name=_requested_text(best_pair),
                    score=best_score,
                )
            )
    return matches
