"""Person-name input preparation and post-processing of recogniser spans on OCR words.

``title_case_caps`` prepares the text the recogniser reads. The spans it returns are snapped to
whole words and refined by small pure rules working on word ranges, in the order of the
benchmark's recommended chain (``docs/BENCHMARKS.md`` section 4.3, "REC"):

1. Xsplit ``split_lines``: split a range where consecutive words lie on different lines.
2. M3 ``extend_initials``: extend a range to the initials before it, such as "F." or "J. R.".
3. M1 + M2 ``merge_names``: merge ranges joined by a hyphen or apostrophe, and "Last, First"
   unless both sides read as full names of a list ("Dr. Priya Sharma, Lucas Zimmermann") or
   the left side already holds a "Last, First" join.
4. T ``strip_titles``: strip leading titles and mark ranges a title stripped or preceded.
5. Dsingle, Dlower, Ddigit, Dshort: drop one-word ranges unless titled, ranges without an
   upper-case letter, ranges with digits or "@", and ranges with fewer than three letters.
6. Dconf ``drop_unreadable``: drop ranges whose every word was recognised with a confidence
   below ``MIN_NAME_CONFIDENCE`` (signatures and noise read as letters; measured in
   ``docs/BENCHMARKS.md`` section 21, not part of the benchmark's chain).

``clean_mentions`` runs the whole chain on one page.
"""

import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial

from app.domain.geometry import PageText
from app.domain.types import PageWord, TextSpan

MIN_NAME_CONFIDENCE = 20.0
"""A name whose every word has a lower OCR confidence (0-100) is not reported."""

_RIGHT_QUOTE = "\N{RIGHT SINGLE QUOTATION MARK}"
_EN_DASH = "\N{EN DASH}"

# Combining marks, which follow their letter in decomposed (NFD) text ("U" + diaeresis).
_MARKS = "̀-ͯ᪰-᫿᷀-᷿⃐-⃿︠-︯"
_LETTER = rf"[^\W\d_][{_MARKS}]*"
# A word made of letters, optionally joined by hyphens or apostrophes, with edge punctuation.
_CAPS_WORD = re.compile(rf"^([\W_]*)((?:{_LETTER})+(?:[-'{_RIGHT_QUOTE}](?:{_LETTER})+)*)([\W_]*)$")
_NON_SPACE = re.compile(r"\S+")
# Leading non-word characters, and trailing ones other than combining marks.
_EDGE = re.compile(rf"^[\W_]+|(?:(?![{_MARKS}])[\W_])+$")
_INITIAL = re.compile(rf"^{_LETTER}\.$")

# The union of the project's titles and the benchmark's TITLES list (ner_corpus/scripts/pipe.py).
_TITLES = frozenset(
    {
        # English
        "dr", "prof", "professor", "mr", "mrs", "ms", "miss", "mx", "sir", "dame", "madam",
        # Spanish
        "dra", "profa", "sr", "sra", "srta", "sres", "doña", "dña", "señor", "señora",
        "ing", "lic",
        # Portuguese
        "dona", "senhor", "senhora", "doutor", "doutora",
        # French ("M." is handled apart, see _is_title)
        "pr", "mme", "mlle", "mm", "monsieur", "madame", "mademoiselle", "docteur",
        # German
        "herr", "frau", "hr", "fr", "fräulein", "fraulein",
    }
)  # fmt: skip
# Titles that are also common first names ("Don Johnson", "Dom Phillips"): titles only in the
# document language that uses them.
_LANGUAGE_TITLES = {"don": "es", "dom": "pt"}
_JOIN_EDGES = ("-", "'", _RIGHT_QUOTE)
_JOIN_WORDS = frozenset({"-", _EN_DASH, "'", _RIGHT_QUOTE})
_SURNAME_MAX_WORDS = 4
_FIRST_NAME_MAX_WORDS = 2
_FULL_NAME_MIN_WORDS = 2
# Languages whose "Last, First" forms often have two words on each side of the comma.
_DOUBLE_NAME_LANGUAGES = frozenset({"es", "pt"})
_MIN_LETTERS = 3

_MIN_CAPS_LETTERS = 2
_QUOTES = "\"'«»\N{LEFT DOUBLE QUOTATION MARK}\N{RIGHT DOUBLE QUOTATION MARK}" + (
    "\N{LEFT SINGLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}\N{DOUBLE LOW-9 QUOTATION MARK}"
)
_TRAILING_PUNCTUATION = ",;:!?()[]{}-" + _EN_DASH + _QUOTES
_EDGE_PUNCTUATION = _TRAILING_PUNCTUATION + ". \t"


@dataclass(frozen=True, slots=True)
class WordRange:
    """Half-open range ``[start_word, end_word)`` of word indices on one page.

    ``titled`` records that a title was stripped from the range or immediately precedes it.
    """

    start_word: int
    end_word: int
    titled: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.start_word < self.end_word:
            raise ValueError(
                f"invalid word range: start_word={self.start_word}, end_word={self.end_word}"
            )


JoinRule = Callable[[Sequence[PageWord], WordRange, WordRange], bool]


# --- input preparation ------------------------------------------------------------------------


def title_case_caps(text: str) -> str:
    """Return ``text`` with every ALL-CAPS word of two or more letters title-cased.

    The result has the same length as ``text`` and differs only inside changed words, so
    character offsets found on it index ``text``. The first letter of each hyphen or
    apostrophe part is kept and the others are lowered one by one, only when lowering yields a
    single character. Words with a lower-case letter or a digit and single letters are kept.
    Combining marks of decomposed text count as part of their letter ("MU" + diaeresis + "LLER"
    becomes "Mu" + diaeresis + "ller").
    """
    return _NON_SPACE.sub(_title_case_word, text)


def _title_case_word(match: re.Match[str]) -> str:
    word = match.group(0)
    parts = _CAPS_WORD.match(word)
    if parts is None:
        return word
    lead, core, trail = parts.groups()
    if sum(char.isalpha() for char in core) < _MIN_CAPS_LETTERS or not core.isupper():
        return word
    return lead + _title_case_core(core) + trail


def _is_mark(char: str) -> bool:
    return unicodedata.category(char).startswith("M")


def _title_case_core(core: str) -> str:
    chars: list[str] = []
    after_letter = False
    for char in core:
        lowered = char.lower()
        chars.append(lowered if after_letter and len(lowered) == 1 else char)
        after_letter = char.isalpha() or (after_letter and _is_mark(char))
    return "".join(chars)


# --- snapping ---------------------------------------------------------------------------------


def snap_spans(page_text: PageText, spans: Sequence[TextSpan]) -> list[WordRange]:
    """Map character spans of ``page_text.text`` to the words they overlap, even partially.

    Spans touching no word are ignored; ranges sharing a word are merged. The result is sorted.
    """
    starts = [start for start, _ in page_text.offsets]
    ends = [end for _, end in page_text.offsets]
    ranges: list[WordRange] = []
    for span in spans:
        first = bisect_right(ends, span.start)
        stop = bisect_left(starts, span.end)
        if span.start < span.end and first < stop:
            ranges.append(WordRange(first, stop))
    return _merge_overlaps(ranges)


def _merge_overlaps(ranges: Sequence[WordRange]) -> list[WordRange]:
    merged: list[WordRange] = []
    for current in sorted(ranges, key=lambda r: (r.start_word, r.end_word)):
        if merged and current.start_word < merged[-1].end_word:
            last = merged[-1]
            merged[-1] = WordRange(
                last.start_word,
                max(last.end_word, current.end_word),
                last.titled or current.titled,
            )
        else:
            merged.append(current)
    return merged


# --- shared word predicates -------------------------------------------------------------------


def _core(text: str) -> str:
    """The word without leading and trailing non-word characters."""
    return _EDGE.sub("", text)


def _is_title(text: str, language: str | None) -> bool:
    """Whether a word is a title.

    "M." counts only in French, elsewhere it is an initial; "Don" only in Spanish and "Dom"
    only in Portuguese, elsewhere they are first names.
    """
    core = _core(text)
    if core == "M":
        return language == "fr" and text.endswith(".")
    key = unicodedata.normalize("NFC", core).casefold()
    if key in _LANGUAGE_TITLES:
        return language == _LANGUAGE_TITLES[key]
    return key in _TITLES


def _is_initial(text: str) -> bool:
    """Whether a word is one capital letter and a period ("F.", "Ł.")."""
    return _INITIAL.match(text) is not None and text[0].isupper()


def _tokens(words: Sequence[PageWord], word_range: WordRange) -> list[str]:
    cores = (_core(word.text) for word in words[word_range.start_word : word_range.end_word])
    return [core for core in cores if core]


# --- Xsplit -----------------------------------------------------------------------------------


def split_lines(words: Sequence[PageWord], ranges: Sequence[WordRange]) -> list[WordRange]:
    """Xsplit: split each range wherever two consecutive words lie on different lines."""
    split: list[WordRange] = []
    for word_range in ranges:
        start = word_range.start_word
        for index in range(start + 1, word_range.end_word):
            if words[index].line != words[index - 1].line:
                split.append(replace(word_range, start_word=start, end_word=index))
                start = index
        split.append(replace(word_range, start_word=start))
    return split


# --- M3 ---------------------------------------------------------------------------------------


def extend_initials(
    words: Sequence[PageWord], ranges: Sequence[WordRange], *, language: str | None = None
) -> list[WordRange]:
    """M3: extend a capitalised range left over the initials ("J. R. R.") before it on its line.

    "M." is not an initial in French, where it is a title. Ranges that come to share a word
    are merged.
    """
    return _merge_overlaps([_extend_initial(words, r, language) for r in ranges])


def _extend_initial(
    words: Sequence[PageWord], word_range: WordRange, language: str | None
) -> WordRange:
    first = words[word_range.start_word]
    if not first.text[:1].isupper():
        return word_range
    start = word_range.start_word
    while start > 0:
        previous = words[start - 1]
        if (
            previous.line != first.line
            or not _is_initial(previous.text)
            or _is_title(previous.text, language)
        ):
            break
        start -= 1
    return replace(word_range, start_word=start)


# --- M1 and M2 --------------------------------------------------------------------------------


def _hyphen_join(words: Sequence[PageWord], left: WordRange, right: WordRange) -> bool:
    """M1: same line, and the boundary words carry a hyphen or apostrophe, or one lies between."""
    last, first = words[left.end_word - 1], words[right.start_word]
    if last.line != first.line:
        return False
    gap = right.start_word - left.end_word
    if gap == 0:
        return last.text.endswith(_JOIN_EDGES) or first.text.startswith(_JOIN_EDGES)
    return gap == 1 and words[left.end_word].text in _JOIN_WORDS


def _last_first_join(
    words: Sequence[PageWord],
    left: WordRange,
    right: WordRange,
    language: str | None = None,
) -> bool:
    """M2 relaxed: "Last, First" with a surname of up to four words and a first name of two.

    A left range that already holds a comma before its last word (an earlier join, as in
    "Weber, Hans, Smith, John") is not joined again. Two ranges of two or more words each are
    not joined when they read as a list of two full names (see ``_two_full_names``).
    """
    last, first = words[left.end_word - 1], words[right.start_word]
    return (
        right.start_word == left.end_word
        and last.line == first.line
        and last.text.endswith(",")
        and not _has_inner_comma(words, left)
        and left.end_word - left.start_word <= _SURNAME_MAX_WORDS
        and right.end_word - right.start_word <= _FIRST_NAME_MAX_WORDS
        and words[left.start_word].text[:1].isupper()
        and first.text[:1].isupper()
        and not _two_full_names(words, left, right, language)
    )


def _has_inner_comma(words: Sequence[PageWord], word_range: WordRange) -> bool:
    inner = words[word_range.start_word : word_range.end_word - 1]
    return any(word.text.endswith(",") for word in inner)


def _two_full_names(
    words: Sequence[PageWord], left: WordRange, right: WordRange, language: str | None
) -> bool:
    """Whether "A B, C D" is a list of two names in "First Last" order, not "Last, First".

    Both sides must have two or more words (leading titles not counted). They are two names
    when a title starts the left side or stands just before it ("Dr. Priya Sharma, Lucas
    Zimmermann"), or when only the right side is in capitals ("Luisa Ribeiro, JOÃO
    FERNANDES"). Surnames in capitals before given names that are not ("GONZÁLEZ RUIZ, María
    José", "DUPONT MARTIN, Jean Pierre") are one name in any language. Otherwise they are two
    names when the document language is known and is not Spanish or Portuguese, whose double
    surnames and compound first names are where such "Last, First" forms occur ("Muñoz Ibáñez,
    Juan Carlos"); an unknown language keeps the join.
    """
    start = left.start_word
    while start < left.end_word - 1 and _is_title(words[start].text, language):
        start += 1
    surname = _tokens(words, WordRange(start, left.end_word))
    first_name = _tokens(words, right)
    if len(surname) < _FULL_NAME_MIN_WORDS or len(first_name) < _FULL_NAME_MIN_WORDS:
        return False
    if start > left.start_word or _title_precedes(words, start, language):
        return True
    surname_caps, first_name_caps = _all_caps(surname), _all_caps(first_name)
    if surname_caps != first_name_caps:
        return first_name_caps
    return language is not None and language not in _DOUBLE_NAME_LANGUAGES


def _all_caps(tokens: Sequence[str]) -> bool:
    return all(token.isupper() for token in tokens)


def _merge_adjacent(
    words: Sequence[PageWord], ranges: Sequence[WordRange], rules: Sequence[JoinRule]
) -> list[WordRange]:
    """Merge neighbouring ranges, left to right, while any rule joins a pair."""
    merged = list(ranges)
    index = 0
    while index < len(merged) - 1:
        left, right = merged[index], merged[index + 1]
        if any(rule(words, left, right) for rule in rules):
            merged[index : index + 2] = [
                WordRange(left.start_word, right.end_word, left.titled or right.titled)
            ]
            # Only the pair ending at the merged range has changed.
            index = max(index - 1, 0)
        else:
            index += 1
    return merged


def merge_hyphenated(words: Sequence[PageWord], ranges: Sequence[WordRange]) -> list[WordRange]:
    """M1: merge ranges split around a hyphen or apostrophe on the same line."""
    return _merge_adjacent(words, ranges, (_hyphen_join,))


def merge_last_first(
    words: Sequence[PageWord], ranges: Sequence[WordRange], *, language: str | None = None
) -> list[WordRange]:
    """M2 relaxed: merge a range ending in "," with the range that directly follows it.

    ``language`` (ISO 639-1, or None when unknown) decides titles and whether two-word sides
    read as a list of full names.
    """
    return _merge_adjacent(words, ranges, (partial(_last_first_join, language=language),))


def merge_names(
    words: Sequence[PageWord], ranges: Sequence[WordRange], *, language: str | None = None
) -> list[WordRange]:
    """M1 and M2 in one pass, M1 tried first on each pair, as in the benchmark."""
    last_first = partial(_last_first_join, language=language)
    return _merge_adjacent(words, ranges, (_hyphen_join, last_first))


# --- T ----------------------------------------------------------------------------------------


def strip_titles(
    words: Sequence[PageWord], ranges: Sequence[WordRange], *, language: str | None = None
) -> list[WordRange]:
    """T: strip leading titles, keeping at least one word; drop ranges made only of titles.

    A range is marked ``titled`` when a title was stripped from it or the word just before it,
    on the same line, is a title.
    """
    stripped = (_strip_title(words, r, language) for r in ranges)
    return [r for r in stripped if r is not None]


def _strip_title(
    words: Sequence[PageWord], word_range: WordRange, language: str | None
) -> WordRange | None:
    start, last = word_range.start_word, word_range.end_word - 1
    while start < last and _is_title(words[start].text, language):
        start += 1
    if start == last and _is_title(words[start].text, language):
        return None
    titled = (
        word_range.titled
        or start > word_range.start_word
        or _title_precedes(words, start, language)
    )
    return WordRange(start, word_range.end_word, titled)


def _title_precedes(words: Sequence[PageWord], start: int, language: str | None) -> bool:
    previous = start - 1
    return (
        previous >= 0
        and words[previous].line == words[start].line
        and _is_title(words[previous].text, language)
    )


# --- drops ------------------------------------------------------------------------------------


def _keep(
    words: Sequence[PageWord],
    ranges: Sequence[WordRange],
    accept: Callable[[list[str], WordRange], bool],
) -> list[WordRange]:
    return [r for r in ranges if accept(_tokens(words, r), r)]


def drop_single_words(words: Sequence[PageWord], ranges: Sequence[WordRange]) -> list[WordRange]:
    """Dsingle: drop ranges of one word (ignoring punctuation-only words) unless titled."""
    return _keep(words, ranges, lambda tokens, r: len(tokens) != 1 or r.titled)


def drop_lowercase(words: Sequence[PageWord], ranges: Sequence[WordRange]) -> list[WordRange]:
    """Dlower: drop ranges without any upper-case letter."""
    return _keep(words, ranges, lambda tokens, _: any(c.isupper() for t in tokens for c in t))


def drop_digits(words: Sequence[PageWord], ranges: Sequence[WordRange]) -> list[WordRange]:
    """Ddigit: drop ranges containing a digit or "@" (edge punctuation of each word ignored)."""
    return _keep(words, ranges, lambda tokens, _: not any(_has_digit_or_at(t) for t in tokens))


def _has_digit_or_at(token: str) -> bool:
    return "@" in token or any(char.isdigit() for char in token)


def drop_short(words: Sequence[PageWord], ranges: Sequence[WordRange]) -> list[WordRange]:
    """Dshort: drop ranges with fewer than three letters."""
    return _keep(
        words,
        ranges,
        lambda tokens, _: sum(c.isalpha() for t in tokens for c in t) >= _MIN_LETTERS,
    )


def drop_unreadable(words: Sequence[PageWord], ranges: Sequence[WordRange]) -> list[WordRange]:
    """Dconf: drop ranges whose every word has a confidence below ``MIN_NAME_CONFIDENCE``.

    A word without a confidence (``None``) counts as readable, so the rule never drops a range
    from an engine that reports none.
    """
    return [r for r in ranges if not all(_unreadable(w) for w in words[r.start_word : r.end_word])]


def _unreadable(word: PageWord) -> bool:
    return word.confidence is not None and word.confidence < MIN_NAME_CONFIDENCE


# --- reported text and pipeline ---------------------------------------------------------------


def mention_text(words: Sequence[PageWord], word_range: WordRange) -> str:
    """The words of the range joined by single spaces, without edge punctuation.

    Commas, periods, hyphens, quotes, brackets and similar marks are removed at both ends; the
    period of a final initial ("Gonzalez, M.") is kept.
    """
    selected = words[word_range.start_word : word_range.end_word]
    text = " ".join(word.text for word in selected).strip(_EDGE_PUNCTUATION)
    ends_with_initial = _is_initial(selected[-1].text.rstrip(_TRAILING_PUNCTUATION))
    return f"{text}." if ends_with_initial and text else text


def clean_mentions(
    page_text: PageText, spans: Sequence[TextSpan], *, language: str | None = None
) -> list[tuple[str, WordRange]]:
    """Apply the recommended span rules to recogniser spans on one page.

    ``spans`` index ``page_text.text`` (or its ``title_case_caps`` copy); the returned ranges
    index ``page_text.words`` and are sorted and disjoint. ``language`` is the document's
    ISO 639-1 code, used for the French title "M." and to tell a list of full names from
    "Last, First". Each result pairs the reported name (see ``mention_text``) with its word
    range, whose boxes the caller unites.
    """
    words = page_text.words
    ranges = snap_spans(page_text, spans)
    ranges = split_lines(words, ranges)
    ranges = extend_initials(words, ranges, language=language)
    ranges = merge_names(words, ranges, language=language)
    ranges = strip_titles(words, ranges, language=language)
    for drop in (drop_single_words, drop_lowercase, drop_digits, drop_short, drop_unreadable):
        ranges = drop(words, ranges)
    return [(mention_text(words, r), r) for r in ranges]
