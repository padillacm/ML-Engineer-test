"""Guardrails of grounded answering: the shared defaults, the support check and the output filter.

The model that answers a question sees the passages as data (``prompting`` delimits them), but a
model can still be talked into inventing a name, quoting its instructions or pointing at a
document that was never retrieved. Two pure checks run over the reply before it is returned:

- ``support_share`` measures how much of the answer's content comes from the context: the share
  of its content tokens (``content_tokens``: maximal runs of letters and digits, accent-folded
  and casefolded, of length 3 or more, without the function words of the five supported
  languages and without the refusal sentinel) that appear among the context's tokens. A short
  reply without content tokens ("Yes.", "42") has no share and counts as supported: it carries
  nothing to fabricate and the output filter still runs over it.
- ``output_filter`` names the rule an answer breaks when it leaks the instructions (any
  40-character window of the normalised system prompt), gives a URL that no passage holds, or
  cites a document that was not in the context (or a page of a context document that was not).

Both take plain strings so the checks can be run anywhere (the benchmarks calibrate the share on
the evaluation sample). The defaults live here so the settings, the composition root and the use
cases agree on them. The module stays pure Python: no vendor library, as for all of ``domain``.
"""

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence

from app.domain.errors import ConfigurationError
from app.domain.language import STOP_WORDS

DEFAULT_INJECTION_THRESHOLD = 0.5
"""Detector score at or above which a chunk is flagged or a question refused."""

DEFAULT_SUPPORT_MIN_SHARE = 0.6
"""Share of an answer's content tokens that must appear in the context."""

DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 2
"""Context chunks one document may give when chunks of other documents also matched."""

REFUSAL_SENTINEL = "NOT_IN_DOCUMENTS"
"""The exact reply the model is asked to give when the context does not contain the answer.

It is removed from an answer before the support check, so a model that writes it next to an
explanation is judged on the explanation alone (``prompting.is_refusal`` recognises the replies
that start with it).
"""

MIN_TOKEN_LENGTH = 3
"""Shorter runs of letters and digits (articles, "a", "ok", two-digit numbers) are not content."""

PROMPT_WINDOW = 40
"""Characters of the normalised system prompt that, found in an answer, make it a leak."""

_STOP_WORDS: frozenset[str] = frozenset().union(*STOP_WORDS.values())
_TOKEN = re.compile(r"[^\W_]+")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"'()\[\]{}]+", re.IGNORECASE)
_URL_TRAILING = ".,;:!?*_`"
# "document" in the five languages (English and French share it, "documento" is Spanish and
# Portuguese, "Dokument" German) over the folded answer; the id is matched hyphenated or not
# and compared without hyphens, so an answer cannot rewrite an id past the check.
_DOCUMENT_REFERENCE = re.compile(
    r"\b(?:document|documento|dokument)\s+"
    r"(?P<id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32})"
    r"(?:\s*,?\s*(?:page|pagina|seite)\s+(?P<page>\d+))?",
    re.IGNORECASE,
)


def validate_threshold(name: str, value: float) -> float:
    """``value`` as a float when it is a finite number within 0..1, else ``ConfigurationError``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{name} must be a number within 0..1, got {value!r}")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{name} must be within 0..1, got {value}")
    return float(value)


def fold(text: str) -> str:
    """``text`` NFKD-normalised, without combining marks, casefolded.

    So ``González`` and ``GONZALEZ`` fold alike, as do fullwidth and ligature forms; it also
    absorbs the accent errors OCR makes.
    """
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def content_tokens(text: str) -> list[str]:
    """The content tokens of ``text``, in order, repeats included.

    Tokens are the maximal runs of letters and digits of the folded text (``fold``) that are at
    least ``MIN_TOKEN_LENGTH`` long and are neither a stop word of a supported language nor part
    of the refusal sentinel.
    """
    folded = fold(text).replace(fold(REFUSAL_SENTINEL), " ")
    return [
        token
        for token in _TOKEN.findall(folded)
        if len(token) >= MIN_TOKEN_LENGTH and token not in _STOP_WORDS
    ]


def support_share(answer: str, context_texts: Iterable[str]) -> float | None:
    """The fraction of ``answer``'s content tokens found among the tokens of ``context_texts``.

    Repeated tokens count each time they occur. ``None`` when the answer has no content token:
    there is nothing to support, so the caller treats it as supported.
    """
    tokens = content_tokens(answer)
    if not tokens:
        return None
    known = {token for text in context_texts for token in content_tokens(text)}
    return sum(token in known for token in tokens) / len(tokens)


def output_filter(
    answer: str,
    context_texts: Sequence[str],
    system_prompt: str,
    *,
    shown_pages: Collection[tuple[str, int]] = (),
) -> str | None:
    """The rule ``answer`` breaks, or ``None`` when it may be returned.

    ``shown_pages`` are the ``(document_id, page)`` labels of the passages the model saw, with
    the page numbered as the labels show it (1-based). The rules, each named by the string
    returned so the outcome line can carry it without the answer:

    - ``system_prompt``: the answer contains a ``PROMPT_WINDOW``-character window of the
      system prompt (both normalised: folded, whitespace collapsed).
    - ``foreign_url``: a URL (``http://``, ``https://`` or ``www.``) in the answer is in no
      context text.
    - ``unknown_document``: a ``document <uuid>`` reference names an id of no shown page.
    - ``unknown_page``: a ``document <uuid>, page N`` reference names a shown document but a
      page of it that was not shown. Page numbers on their own are not checked: a passage can
      legitimately mention "page 3" of something else.

    The reference words are those of the five languages ("documento", "Dokument", "página",
    "Seite"; "document" and "page" serve English and French), case and accents folded, and an
    id is compared without its hyphens.
    """
    normalised = _collapse(fold(answer))
    prompt = _collapse(fold(system_prompt))
    if _contains_window(normalised, prompt):
        return "system_prompt"
    folded_contexts = [fold(text) for text in context_texts]
    for url in _URL.findall(answer):
        needle = fold(url.rstrip(_URL_TRAILING))
        if not any(needle in text for text in folded_contexts):
            return "foreign_url"
    pages_by_document: defaultdict[str, set[int]] = defaultdict(set)
    for document_id, page in shown_pages:
        pages_by_document[_canonical_id(document_id)].add(page)
    for reference in _DOCUMENT_REFERENCE.finditer(fold(answer)):
        pages = pages_by_document.get(_canonical_id(reference.group("id")))
        if pages is None:
            return "unknown_document"
        if reference.group("page") is not None and int(reference.group("page")) not in pages:
            return "unknown_page"
    return None


def _canonical_id(document_id: str) -> str:
    """A document id as compared: casefolded, without hyphens."""
    return document_id.casefold().replace("-", "")


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _contains_window(text: str, prompt: str) -> bool:
    """Whether any ``PROMPT_WINDOW``-character substring of ``prompt`` occurs in ``text``."""
    if len(prompt) < PROMPT_WINDOW:
        return bool(prompt) and prompt in text
    return any(
        prompt[start : start + PROMPT_WINDOW] in text
        for start in range(len(prompt) - PROMPT_WINDOW + 1)
    )
