"""Document language: the supported codes and a stop-word vote over recognised text.

Each supported language has about forty frequent function words. The lists are disjoint, so a
word votes for one language only; words common to several of them (``de``, ``que``, ``la``,
``en``, ``a``, ``in``, ``des``...) are left out because they cannot tell those languages apart.
Words are compared lowercased and without accents, which also absorbs OCR accent errors.

The vote is decided only with a clear winner: at least ``_MIN_HITS`` hits and ``min_margin``
times the hits of the runner-up. Anything closer returns ``None`` so the caller can fall back to
reading with every supported language instead of committing to a wrong single model.
"""

import re
import unicodedata

from app.domain.errors import InvalidInput

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "es", "pt", "fr", "de")

DEFAULT_MIN_WORDS = 20
DEFAULT_MIN_MARGIN = 1.5

_MIN_HITS = 3


def _words(listing: str) -> frozenset[str]:
    return frozenset(listing.split())


STOP_WORDS: dict[str, frozenset[str]] = {
    "en": _words(
        "the and of to is are were be been this that these those with for on it its by from "
        "have has had shall would should which who not or at their they we our you your any "
        "but than into about"
    ),
    "es": _words(
        "el los las del al y lo con una unos unas su sus pero fue han hay muy sin hasta cuando "
        "donde tambien ya esto estos dicho dicha segun otro otra puede pueden debe deben usted "
        "nuestro nuestra cual mismo"
    ),
    "pt": _words(
        "e os do da na nas um uma ao aos pelo pela pelos pelas com nao sao foi mais tambem "
        "ja seu sua seus suas isso isto esse essa quando onde muito sem ate pode devem deve "
        "nosso nossa ainda"
    ),
    "fr": _words(
        "le les des du et est une qui dans pour par sur avec ce cette ces il ils elle nous vous "
        "leur leurs sont ont au aux pas ne sa ses tout tous etre ete selon afin ainsi notre "
        "votre dont doit peut"
    ),
    "de": _words(
        "der die das und ist nicht mit von zu den dem ein eine einen einem einer auf fur sich "
        "auch werden wird wurde sind oder aber bei nach uber aus durch wie dass wir ihre zum "
        "zur vom im unter diese dieser kann muss"
    ),
}

_WORD = re.compile(r"[^\W\d_]+")


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def detect_language(
    text: str, *, min_words: int = DEFAULT_MIN_WORDS, min_margin: float = DEFAULT_MIN_MARGIN
) -> str | None:
    """Return the language code of ``text``, or ``None`` when the vote is inconclusive.

    ``None`` is returned for text with fewer than ``min_words`` words, when the leading language
    has fewer than three stop-word hits, or when it does not reach ``min_margin`` times the hits
    of the runner-up.
    """
    words = _WORD.findall(_fold(text))
    if len(words) < min_words:
        return None
    hits = {code: sum(word in STOP_WORDS[code] for word in words) for code in SUPPORTED_LANGUAGES}
    ranked = sorted(hits.items(), key=lambda item: item[1], reverse=True)
    (best, best_hits), (_, runner_up_hits) = ranked[0], ranked[1]
    if best_hits < _MIN_HITS or best_hits < min_margin * runner_up_hits:
        return None
    return best


def validate_language(code: str) -> str:
    """Return ``code`` lowercased and stripped; raise ``unsupported_language`` if unsupported."""
    normalized = code.strip().lower()
    if normalized not in SUPPORTED_LANGUAGES:
        raise InvalidInput(
            f"language must be one of: {', '.join(SUPPORTED_LANGUAGES)}", "unsupported_language"
        )
    return normalized
