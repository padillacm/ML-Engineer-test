import random
import unicodedata
from collections.abc import Sequence
from dataclasses import replace

import pytest

from app.domain.geometry import PageText
from app.domain.names import (
    MIN_NAME_CONFIDENCE,
    WordRange,
    clean_mentions,
    drop_digits,
    drop_lowercase,
    drop_short,
    drop_single_words,
    drop_unreadable,
    extend_initials,
    mention_text,
    merge_hyphenated,
    merge_last_first,
    merge_names,
    snap_spans,
    split_lines,
    strip_titles,
    title_case_caps,
)
from app.domain.types import Box, PageWord, TextSpan

_RQ = "\N{RIGHT SINGLE QUOTATION MARK}"
_EN_DASH = "\N{EN DASH}"


def _page(*lines: str) -> PageText:
    """Page text whose words are the whitespace-separated words of ``lines``, one line each."""
    words = [
        PageWord(
            text=text,
            box=Box(page=0, x=10.0 * column, y=12.0 * line, width=8, height=10),
            line=line,
            paragraph=0,
        )
        for line, content in enumerate(lines)
        for column, text in enumerate(content.split())
    ]
    return PageText.from_words(words)


def _words(*lines: str) -> tuple[PageWord, ...]:
    return _page(*lines).words


def _with_confidences(page: PageText, *confidences: float | None) -> PageText:
    """``page`` with its words given ``confidences`` in order."""
    assert len(confidences) == len(page.words)
    return PageText.from_words(
        [replace(word, confidence=c) for word, c in zip(page.words, confidences, strict=True)]
    )


def _span(page: PageText, phrase: str, occurrence: int = 0) -> TextSpan:
    start = -1
    for _ in range(occurrence + 1):
        start = page.text.index(phrase, start + 1)
    return TextSpan(start, start + len(phrase))


def _names(page: PageText, phrases: Sequence[str], language: str | None = None) -> list[str]:
    spans = [_span(page, phrase) for phrase in phrases]
    return [name for name, _ in clean_mentions(page, spans, language=language)]


def _r(start: int, end: int, titled: bool = False) -> WordRange:
    return WordRange(start, end, titled)


# --- WordRange --------------------------------------------------------------------------------


@pytest.mark.parametrize(("start", "end"), [(-1, 2), (2, 2), (3, 1)])
def test_word_range_rejects_empty_or_negative_ranges(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="invalid word range"):
        WordRange(start, end)


# --- snapping ---------------------------------------------------------------------------------


class TestSnapSpans:
    def test_partial_overlap_selects_whole_words(self) -> None:
        page = _page("Signed by Maria Gonzalez today")
        assert snap_spans(page, [TextSpan(11, 20)]) == [_r(2, 4)]

    def test_spans_sharing_a_word_are_merged_and_sorted(self) -> None:
        page = _page("Maria Gonzalez Ruiz and Ana")
        spans = [_span(page, "Ana"), _span(page, "Maria Gon"), _span(page, "zalez Ruiz")]
        assert snap_spans(page, spans) == [_r(0, 3), _r(4, 5)]

    def test_spans_on_spaces_empty_or_beyond_the_text_are_ignored(self) -> None:
        page = _page("Maria Gonzalez")
        spans = [TextSpan(5, 6), TextSpan(2, 2), TextSpan(40, 50)]
        assert snap_spans(page, spans) == []

    def test_adjacent_spans_stay_separate(self) -> None:
        page = _page("Maria Gonzalez")
        assert snap_spans(page, [_span(page, "Maria"), _span(page, "Gonzalez")]) == [
            _r(0, 1),
            _r(1, 2),
        ]


# --- Xsplit -----------------------------------------------------------------------------------


class TestSplitLines:
    def test_range_crossing_line_breaks_is_split_per_line(self) -> None:
        words = _words("Maria Gonzalez", "Director", "of Sales")
        assert split_lines(words, [_r(0, 5)]) == [_r(0, 2), _r(2, 3), _r(3, 5)]

    def test_range_on_one_line_is_unchanged(self) -> None:
        words = _words("Maria Gonzalez Ruiz")
        assert split_lines(words, [_r(0, 3, titled=True)]) == [_r(0, 3, titled=True)]


# --- M3 ---------------------------------------------------------------------------------------


class TestExtendInitials:
    def test_preceding_initial_is_included(self) -> None:
        words = _words("signed J. Smith")
        assert extend_initials(words, [_r(2, 3)]) == [_r(1, 3)]

    @pytest.mark.parametrize("before", ["j.", "JR", "J", "Jr.", "(J.)"])
    def test_non_initials_are_not_included(self, before: str) -> None:
        words = _words(f"signed {before} Smith")
        assert extend_initials(words, [_r(2, 3)]) == [_r(2, 3)]

    def test_initial_on_the_previous_line_is_not_included(self) -> None:
        words = _words("J.", "Smith")
        assert extend_initials(words, [_r(1, 2)]) == [_r(1, 2)]

    def test_lowercase_range_is_not_extended(self) -> None:
        words = _words("J. smith")
        assert extend_initials(words, [_r(1, 2)]) == [_r(1, 2)]

    def test_french_m_is_a_title_not_an_initial(self) -> None:
        words = _words("M. Dupont")
        assert extend_initials(words, [_r(1, 2)], language="fr") == [_r(1, 2)]
        assert extend_initials(words, [_r(1, 2)], language="en") == [_r(0, 2)]
        assert extend_initials(words, [_r(1, 2)]) == [_r(0, 2)]

    def test_extension_into_the_previous_range_merges_them(self) -> None:
        words = _words("Ana F. Lopez")
        assert extend_initials(words, [_r(0, 2), _r(2, 3)]) == [_r(0, 3)]

    @pytest.mark.parametrize(
        "initial", ["Ł.", "Č.", "Ş.", "Ø.", "Ž.", unicodedata.normalize("NFD", "É.")]
    )
    def test_any_capital_letter_is_an_initial(self, initial: str) -> None:
        words = _words(f"by {initial} Kowalski")
        assert extend_initials(words, [_r(2, 3)]) == [_r(1, 3)]

    @pytest.mark.parametrize("before", ["ł.", "ß.", "1.", "_."])
    def test_lowercase_letters_digits_and_underscores_are_not_initials(self, before: str) -> None:
        words = _words(f"by {before} Kowalski")
        assert extend_initials(words, [_r(2, 3)]) == [_r(2, 3)]

    def test_several_consecutive_initials_are_included(self) -> None:
        words = _words("by J. R. R. Tolkien today")
        assert extend_initials(words, [_r(4, 5)]) == [_r(1, 5)]

    def test_extension_stops_at_a_title_or_a_line_break(self) -> None:
        words = _words("par M. J. Dupont", "R.", "S. Weber")
        ranges = [_r(3, 4), _r(6, 7)]
        assert extend_initials(words, ranges, language="fr") == [_r(2, 4), _r(5, 7)]


# --- M1 ---------------------------------------------------------------------------------------


class TestMergeHyphenated:
    @pytest.mark.parametrize(
        "line",
        ["Anne- Marie Dupont", "Anne -Marie Dupont", "O' Brien Kelly", f"O{_RQ} Brien Kelly"],
    )
    def test_hyphen_or_apostrophe_on_the_boundary_words_joins(self, line: str) -> None:
        words = _words(line)
        assert merge_hyphenated(words, [_r(0, 1), _r(1, 3)]) == [_r(0, 3)]

    @pytest.mark.parametrize("joiner", ["-", _EN_DASH, "'", _RQ])
    def test_lone_joiner_word_between_ranges_joins(self, joiner: str) -> None:
        words = _words(f"Anne {joiner} Marie")
        assert merge_hyphenated(words, [_r(0, 1), _r(2, 3)]) == [_r(0, 3)]

    def test_plain_neighbours_and_wider_gaps_do_not_join(self) -> None:
        words = _words("Anne Marie - - Paul")
        ranges = [_r(0, 1), _r(1, 2), _r(4, 5)]
        assert merge_hyphenated(words, ranges) == ranges

    def test_hyphen_at_a_line_end_does_not_join(self) -> None:
        words = _words("Anne-", "Marie")
        assert merge_hyphenated(words, [_r(0, 1), _r(1, 2)]) == [_r(0, 1), _r(1, 2)]

    def test_chains_merge_into_one_range(self) -> None:
        words = _words("Anne- Marie- Claire")
        assert merge_hyphenated(words, [_r(0, 1), _r(1, 2), _r(2, 3)]) == [_r(0, 3)]


# --- M2 ---------------------------------------------------------------------------------------


class TestMergeLastFirst:
    def test_surname_with_comma_followed_by_first_name_joins(self) -> None:
        words = _words("Gonzalez Ruiz, Maria")
        assert merge_last_first(words, [_r(0, 2), _r(2, 3)]) == [_r(0, 3)]

    def test_surname_of_four_words_and_first_name_of_two_join(self) -> None:
        words = _words("De La Fuente Ruiz, Ana Maria")
        assert merge_last_first(words, [_r(0, 4), _r(4, 6)]) == [_r(0, 6)]

    def test_surname_of_five_words_does_not_join(self) -> None:
        words = _words("A De La Fuente Ruiz, Ana")
        assert merge_last_first(words, [_r(0, 5), _r(5, 6)]) == [_r(0, 5), _r(5, 6)]

    def test_first_name_of_three_words_does_not_join(self) -> None:
        words = _words("Ruiz, Ana Maria Luisa")
        assert merge_last_first(words, [_r(0, 1), _r(1, 4)]) == [_r(0, 1), _r(1, 4)]

    @pytest.mark.parametrize(
        ("line", "ranges"),
        [
            ("Gonzalez Maria", [_r(0, 1), _r(1, 2)]),
            ("Gonzalez, maria", [_r(0, 1), _r(1, 2)]),
            ("gonzalez, Maria", [_r(0, 1), _r(1, 2)]),
            ("Gonzalez, and Maria", [_r(0, 1), _r(2, 3)]),
        ],
    )
    def test_no_comma_lowercase_or_gap_does_not_join(
        self, line: str, ranges: list[WordRange]
    ) -> None:
        assert merge_last_first(_words(line), ranges) == ranges

    def test_comma_at_a_line_end_does_not_join(self) -> None:
        words = _words("Gonzalez,", "Maria")
        assert merge_last_first(words, [_r(0, 1), _r(1, 2)]) == [_r(0, 1), _r(1, 2)]

    def test_merged_range_keeps_the_titled_mark(self) -> None:
        words = _words("Weber, Hans")
        assert merge_last_first(words, [_r(0, 1, titled=True), _r(1, 2)]) == [_r(0, 2, titled=True)]


class TestListOfFullNames:
    """Two ranges of two or more words around a comma: a list of names or "Last, First"."""

    @pytest.mark.parametrize("language", ["en", "fr", "de"])
    def test_two_full_names_are_not_joined_in_a_known_non_iberian_language(
        self, language: str
    ) -> None:
        words = _words("Christopher Wong, Anna Kowalski")
        ranges = [_r(0, 2), _r(2, 4)]
        assert merge_last_first(words, ranges, language=language) == ranges

    @pytest.mark.parametrize("language", ["es", "pt", None])
    def test_double_surname_and_compound_first_name_join_in_spanish_portuguese_or_unknown(
        self, language: str | None
    ) -> None:
        words = _words("Muñoz Ibáñez, Juan Carlos")
        assert merge_last_first(words, [_r(0, 2), _r(2, 4)], language=language) == [_r(0, 4)]

    def test_edge_case_unknown_language_keeps_an_untitled_list_joined(self) -> None:
        # Without a language nothing tells "Christopher Wong, Anna Kowalski" from
        # "Muñoz Ibáñez, Juan Carlos"; the benchmark's join is kept.
        words = _words("Christopher Wong, Anna Kowalski")
        assert merge_last_first(words, [_r(0, 2), _r(2, 4)]) == [_r(0, 4)]

    @pytest.mark.parametrize(
        ("line", "ranges"),
        [
            ("Dr. Priya Sharma, Lucas Zimmermann", [_r(1, 3), _r(3, 5)]),
            ("Dr. Priya Sharma, Lucas Zimmermann", [_r(0, 3), _r(3, 5)]),
            ("Sr. Muñoz Ibáñez, Juan Carlos", [_r(1, 3), _r(3, 5)]),
        ],
    )
    def test_a_title_before_the_left_side_means_first_last_order(
        self, line: str, ranges: list[WordRange]
    ) -> None:
        assert merge_last_first(_words(line), ranges, language="es") == ranges
        assert merge_last_first(_words(line), ranges) == ranges

    @pytest.mark.parametrize("language", ["pt", "en", None])
    def test_only_the_right_side_in_capitals_means_two_names(self, language: str | None) -> None:
        words = _words("Luisa Ribeiro, JOÃO FERNANDES")
        ranges = [_r(0, 2), _r(2, 4)]
        assert merge_last_first(words, ranges, language=language) == ranges

    @pytest.mark.parametrize(
        ("line", "language"),
        [
            ("GONZÁLEZ RUIZ, María José", "es"),
            ("GONZÁLEZ RUIZ, María José", None),
            ("DUPONT MARTIN, Jean Pierre", "fr"),
            ("SMITH JONES, John Paul", "en"),
        ],
    )
    def test_surnames_in_capitals_before_given_names_join_in_any_language(
        self, line: str, language: str | None
    ) -> None:
        words = _words(line)
        assert merge_last_first(words, [_r(0, 2), _r(2, 4)], language=language) == [_r(0, 4)]

    def test_edge_case_a_full_name_in_capitals_before_a_full_name_is_joined(self) -> None:
        # "Cc: JOÃO FERNANDES, Luisa Ribeiro" (pt_111_memo of the corpus) reads like
        # "GONZÁLEZ RUIZ, María José"; nothing on the line tells them apart.
        words = _words("JOÃO FERNANDES, Luisa Ribeiro")
        assert merge_last_first(words, [_r(0, 2), _r(2, 4)], language="pt") == [_r(0, 4)]

    def test_a_title_before_surnames_in_capitals_still_means_two_names(self) -> None:
        words = _words("Dr. GONZÁLEZ RUIZ, María José")
        ranges = [_r(1, 3), _r(3, 5)]
        assert merge_last_first(words, ranges, language="es") == ranges

    def test_both_sides_in_capitals_join(self) -> None:
        words = _words("MUÑOZ IBÁÑEZ, JUAN CARLOS")
        assert merge_last_first(words, [_r(0, 2), _r(2, 4)], language="es") == [_r(0, 4)]

    @pytest.mark.parametrize(
        ("line", "ranges"),
        [
            ("Gonzalez Ruiz, Maria", [_r(0, 2), _r(2, 3)]),
            ("Weber, Hans Peter", [_r(0, 1), _r(1, 3)]),
            ("Dr. Weber, Hans Peter", [_r(0, 2), _r(2, 4)]),
        ],
    )
    def test_a_side_of_one_word_still_joins_in_any_language(
        self, line: str, ranges: list[WordRange]
    ) -> None:
        merged = merge_last_first(_words(line), ranges, language="en")
        assert merged == [_r(ranges[0].start_word, ranges[-1].end_word)]


class TestChainedLastFirst:
    """A range that already joined "Last, First" is not joined to the next name."""

    def test_list_of_one_word_pairs_gives_one_range_per_pair(self) -> None:
        words = _words("Weber, Hans, Smith, John, Rossi, Marco")
        ranges = [_r(i, i + 1) for i in range(6)]
        assert merge_last_first(words, ranges, language="en") == [_r(0, 2), _r(2, 4), _r(4, 6)]

    def test_list_of_two_word_pairs_gives_one_range_per_pair(self) -> None:
        words = _words("Gonzalez Ruiz, Maria, Lopez Diaz, Ana, Perez Gil, Juan")
        ranges = [_r(0, 2), _r(2, 3), _r(3, 5), _r(5, 6), _r(6, 8), _r(8, 9)]
        assert merge_last_first(words, ranges, language="es") == [_r(0, 3), _r(3, 6), _r(6, 9)]

    def test_a_span_holding_a_comma_inside_is_not_joined(self) -> None:
        words = _words("Weber, Hans, Smith")
        assert merge_last_first(words, [_r(0, 2), _r(2, 3)]) == [_r(0, 2), _r(2, 3)]

    def test_a_trailing_comma_alone_still_joins(self) -> None:
        words = _words("Gonzalez Ruiz, Maria")
        assert merge_last_first(words, [_r(0, 2), _r(2, 3)]) == [_r(0, 3)]


class TestMergeNames:
    def test_hyphen_and_last_first_combine(self) -> None:
        words = _words("Garcia- Lopez, Ana")
        assert merge_names(words, [_r(0, 1), _r(1, 2), _r(2, 3)]) == [_r(0, 3)]


# --- T ----------------------------------------------------------------------------------------


class TestStripTitles:
    @pytest.mark.parametrize(
        "title",
        ["Dr.", "dr", "DRA.", "Prof.", "Profa.", "Sr.", "Sra.", "Srta.", "Sres.", "Doña",
         "Dña.", "Mr.", "Mrs", "Ms.", "Mme", "Mlle.", "Herr", "Frau", "Hr.", "Fr.", "Ing.",
         "Lic.", "Sir", "Dame", "Pr", "Pr.", "Professor", "Miss", "Mx.", "Madam", "Señor",
         "Señora", "Dona", "Senhor", "Senhora", "Doutor", "Doutora", "Monsieur",
         "Madame", "Mademoiselle", "MM.", "Docteur", "Fräulein", "Fraulein"],
    )  # fmt: skip
    def test_leading_title_is_stripped_and_marks_the_range(self, title: str) -> None:
        words = _words(f"{title} Weber")
        assert strip_titles(words, [_r(0, 2)]) == [_r(1, 2, titled=True)]

    def test_several_leading_titles_are_stripped(self) -> None:
        words = _words("Prof. Dr. Hans Weber")
        assert strip_titles(words, [_r(0, 4)]) == [_r(2, 4, titled=True)]

    def test_title_just_before_the_range_marks_it(self) -> None:
        words = _words("Dr. Weber")
        assert strip_titles(words, [_r(1, 2)]) == [_r(1, 2, titled=True)]

    def test_title_on_the_previous_line_does_not_mark(self) -> None:
        words = _words("Dr.", "Weber")
        assert strip_titles(words, [_r(1, 2)]) == [_r(1, 2)]

    def test_range_made_only_of_titles_is_dropped(self) -> None:
        words = _words("Dr. Prof. Weber")
        assert strip_titles(words, [_r(0, 2), _r(2, 3)]) == [_r(2, 3, titled=True)]

    def test_untitled_range_is_unchanged(self) -> None:
        words = _words("met Maria Gonzalez")
        assert strip_titles(words, [_r(1, 3)]) == [_r(1, 3)]

    def test_french_m_is_a_title_only_in_french(self) -> None:
        words = _words("M. Dupont")
        assert strip_titles(words, [_r(0, 2)], language="fr") == [_r(1, 2, titled=True)]
        assert strip_titles(words, [_r(0, 2)], language="de") == [_r(0, 2)]
        assert strip_titles(words, [_r(0, 2)]) == [_r(0, 2)]

    @pytest.mark.parametrize(("title", "language"), [("Don", "es"), ("don", "es"), ("Dom", "pt")])
    def test_don_and_dom_are_titles_in_their_language(self, title: str, language: str) -> None:
        words = _words(f"{title} Weber")
        assert strip_titles(words, [_r(0, 2)], language=language) == [_r(1, 2, titled=True)]

    @pytest.mark.parametrize(
        ("title", "language"),
        [("Don", "en"), ("Don", "pt"), ("Don", None), ("Dom", "es"), ("Dom", None)],
    )
    def test_don_and_dom_are_first_names_elsewhere(self, title: str, language: str | None) -> None:
        words = _words(f"{title} Weber")
        assert strip_titles(words, [_r(0, 2)], language=language) == [_r(0, 2)]

    def test_french_m_without_period_is_not_a_title(self) -> None:
        words = _words("M Dupont")
        assert strip_titles(words, [_r(0, 2)], language="fr") == [_r(0, 2)]


# --- drops ------------------------------------------------------------------------------------


class TestDropSingleWords:
    def test_single_word_is_dropped_unless_titled(self) -> None:
        words = _words("Weber Weber Hans Weber")
        ranges = [_r(0, 1), _r(1, 2, titled=True), _r(2, 4)]
        assert drop_single_words(words, ranges) == [_r(1, 2, titled=True), _r(2, 4)]

    def test_punctuation_only_words_do_not_count(self) -> None:
        words = _words("Weber , Hans")
        assert drop_single_words(words, [_r(0, 2), _r(0, 3)]) == [_r(0, 3)]


class TestDropLowercase:
    def test_ranges_without_capitals_are_dropped(self) -> None:
        words = _words("maria gonzalez Maria Gonzalez")
        assert drop_lowercase(words, [_r(0, 2), _r(2, 4)]) == [_r(2, 4)]

    def test_one_capital_letter_is_enough(self) -> None:
        words = _words("maria Gonzalez")
        assert drop_lowercase(words, [_r(0, 2)]) == [_r(0, 2)]


class TestDropDigits:
    @pytest.mark.parametrize("line", ["Maria Gonzalez2", "Room 12 Maria", "maria@acme.com Maria"])
    def test_digits_or_at_sign_drop_the_range(self, line: str) -> None:
        words = _words(line)
        assert drop_digits(words, [_r(0, len(words))]) == []

    def test_letters_only_are_kept(self) -> None:
        words = _words("Maria Gonzalez,")
        assert drop_digits(words, [_r(0, 2)]) == [_r(0, 2)]


class TestDropShort:
    @pytest.mark.parametrize("line", ["Al", "J. B.", "A-B", ", ."])
    def test_fewer_than_three_letters_drop_the_range(self, line: str) -> None:
        words = _words(line)
        assert drop_short(words, [_r(0, len(words))]) == []

    def test_three_letters_are_kept(self) -> None:
        words = _words("Ana")
        assert drop_short(words, [_r(0, 1, titled=True)]) == [_r(0, 1, titled=True)]


class TestDropUnreadable:
    @pytest.mark.parametrize("confidences", [(0.0, 9.0), (19.9, 0.0)])
    def test_every_word_below_the_minimum_drops_the_range(
        self, confidences: tuple[float, float]
    ) -> None:
        words = _with_confidences(_page("Co Jrrao"), *confidences).words
        assert drop_unreadable(words, [_r(0, 2)]) == []

    @pytest.mark.parametrize(
        "confidences",
        [(0.0, MIN_NAME_CONFIDENCE), (0.0, 63.0), (None, 0.0), (None, None)],
        ids=["one-at-the-minimum", "one-readable", "one-without-confidence", "no-confidences"],
    )
    def test_one_readable_word_or_no_confidence_keeps_the_range(
        self, confidences: tuple[float | None, float | None]
    ) -> None:
        words = _with_confidences(_page("Lay Nal"), *confidences).words
        assert drop_unreadable(words, [_r(0, 2)]) == [_r(0, 2)]

    def test_only_the_words_of_each_range_count(self) -> None:
        words = _with_confidences(_page("Ana Lopez Co Jrrao"), 96.0, 91.0, 18.0, 9.0).words
        assert drop_unreadable(words, [_r(0, 2), _r(2, 4)]) == [_r(0, 2)]


# --- reported text ----------------------------------------------------------------------------


class TestMentionText:
    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("Maria Gonzalez,", "Maria Gonzalez"),
            ("(Maria Gonzalez).", "Maria Gonzalez"),
            ("“Maria Gonzalez”", "Maria Gonzalez"),
            ("Gonzalez Ruiz, Maria", "Gonzalez Ruiz, Maria"),
            ("J. Smith.", "J. Smith"),
            ("Gonzalez, M.", "Gonzalez, M."),
            ("Gonzalez, M.,", "Gonzalez, M."),
            ("Anne-Marie O'Brien", "Anne-Marie O'Brien"),
            ("Maria Gonzalez-", "Maria Gonzalez"),
            ("-Maria Gonzalez-,", "Maria Gonzalez"),
            (f"{_EN_DASH}Maria Gonzalez{_EN_DASH}", "Maria Gonzalez"),
            ("Gonzalez, M.-", "Gonzalez, M."),
            ("by Ł.", "by Ł."),
        ],
    )
    def test_edge_punctuation_is_removed_except_a_final_initial_period(
        self, line: str, expected: str
    ) -> None:
        words = _words(line)
        assert mention_text(words, _r(0, len(words))) == expected


# --- pipeline: name-extraction spec scenarios -------------------------------------------------


class TestCleanMentionsScenarios:
    def test_name_read_with_low_confidence_throughout_is_not_reported(self) -> None:
        # A signature scribble read as two capitalised words, next to a real name.
        page = _with_confidences(_page("Signed Pivo Wviy", "Maria Gonzalez"), 90, 0, 12, 95, 18)
        spans = [_span(page, "Pivo Wviy"), _span(page, "Maria Gonzalez")]
        assert [name for name, _ in clean_mentions(page, spans)] == ["Maria Gonzalez"]

    def test_box_covers_the_name_only(self) -> None:
        page = _page("Dr. Maria Gonzalez, CEO")
        mentions = clean_mentions(page, [_span(page, "Maria Gonzalez")])
        assert mentions == [("Maria Gonzalez", _r(1, 3, titled=True))]

    def test_title_inside_the_span_is_stripped(self) -> None:
        page = _page("Dr. Maria Gonzalez, CEO")
        assert clean_mentions(page, [_span(page, "Dr. Maria Gonzalez")]) == [
            ("Maria Gonzalez", _r(1, 3, titled=True))
        ]

    def test_title_stripped(self) -> None:
        page = _page("Sra. Ana López firmó")
        assert _names(page, ["Sra. Ana López"]) == ["Ana López"]

    def test_last_first(self) -> None:
        page = _page("Gonzalez Ruiz, Maria")
        mentions = clean_mentions(page, [_span(page, "Gonzalez Ruiz"), _span(page, "Maria")])
        assert mentions == [("Gonzalez Ruiz, Maria", _r(0, 3))]

    def test_french_pr_title_keeps_a_single_surname(self) -> None:
        page = _page("Le Pr Rogner a signé", "Pr. Rogner")
        spans = [_span(page, "Rogner"), _span(page, "Pr. Rogner")]
        assert clean_mentions(page, spans, language="fr") == [
            ("Rogner", _r(2, 3, titled=True)),
            ("Rogner", _r(6, 7, titled=True)),
        ]

    @pytest.mark.parametrize("language", ["en", None])
    def test_list_of_full_names(self, language: str | None) -> None:
        # The OCR words of research_report.pdf, one line per printed line.
        page = _page(
            "Lead: Dr. Benjamin Foster",
            "Senior Researchers: Dr. Priya Sharma, Lucas Zimmermann",
            "Notable: Published groundbreaking work on protein folding",
        )
        phrases = ["Benjamin Foster", "Priya Sharma", "Lucas Zimmermann"]
        mentions = clean_mentions(page, [_span(page, p) for p in phrases], language=language)
        assert mentions == [
            ("Benjamin Foster", _r(2, 4, titled=True)),
            ("Priya Sharma", _r(7, 9, titled=True)),
            ("Lucas Zimmermann", _r(9, 11)),
        ]

    def test_untitled_list_of_full_names_in_a_detected_language(self) -> None:
        page = _page(
            "Senior Researchers: Christopher Wong, Anna Kowalski",
            "Notable: Developed novel transformer architecture",
        )
        names = ["Christopher Wong", "Anna Kowalski"]
        assert _names(page, names, language="en") == names

    def test_titled_surname_kept_bare_surname_dropped(self) -> None:
        page = _page("We met Dr. Weber today.", "Later Weber left.")
        spans = [_span(page, "Weber"), _span(page, "Weber", occurrence=1)]
        assert clean_mentions(page, spans) == [("Weber", _r(3, 4, titled=True))]

    def test_span_across_a_line_break(self) -> None:
        page = _page("signed by Maria Gonzalez", "Director of Sales")
        assert clean_mentions(page, [_span(page, "Maria Gonzalez Director")]) == [
            ("Maria Gonzalez", _r(2, 4))
        ]

    def test_name_in_capitals_found_on_the_title_cased_copy(self) -> None:
        page = _page("SIGNED BY MARIA GONZALEZ")
        prepared = title_case_caps(page.text)
        assert prepared == "Signed By Maria Gonzalez"
        span = _span(PageText(prepared, page.words, page.offsets), "Maria Gonzalez")
        assert clean_mentions(page, [span]) == [("MARIA GONZALEZ", _r(2, 4))]

    @pytest.mark.parametrize(
        "phrase", ["maria gonzalez", "Maria Gonzalez2", "maria@acme.com Maria", "J. B.", "Dr. Al"]
    )
    def test_lowercase_digit_at_and_short_candidates_are_dropped(self, phrase: str) -> None:
        page = _page(f"text {phrase} text")
        assert _names(page, [phrase]) == []

    def test_french_m_is_a_title_in_french_and_an_initial_otherwise(self) -> None:
        page = _page("Signé par M. Dupont")
        assert _names(page, ["Dupont"], language="fr") == ["Dupont"]
        assert _names(page, ["Dupont"], language="en") == ["M. Dupont"]
        assert _names(page, ["M. Dupont"], language="fr") == ["Dupont"]

    def test_initial_extension_and_hyphen_merge(self) -> None:
        page = _page("Signed J. Garcia- Lopez and Jean -Pierre Martin")
        spans = [
            _span(page, "Garcia"),
            _span(page, "Lopez"),
            _span(page, "Jean"),
            _span(page, "-Pierre Martin"),
        ]
        assert _names(page, ["Garcia", "Lopez", "Jean", "-Pierre Martin"]) == [
            "J. Garcia- Lopez",
            "Jean -Pierre Martin",
        ]
        assert [r for _, r in clean_mentions(page, spans)] == [_r(1, 4), _r(5, 8)]

    def test_longer_comma_separated_list(self) -> None:
        page = _page("Authors: Weber, Hans, Smith, John, Rossi, Marco")
        phrases = ["Weber", "Hans", "Smith", "John", "Rossi", "Marco"]
        expected = ["Weber, Hans", "Smith, John", "Rossi, Marco"]
        assert _names(page, phrases, language="en") == expected
        assert _names(page, phrases) == expected

    def test_list_of_full_names_never_chains_last_first_joins(self) -> None:
        page = _page("Asistentes: Maria Gonzalez, Ana Lopez, Pedro Ruiz y Carmen Diaz")
        phrases = ["Maria Gonzalez", "Ana Lopez", "Pedro Ruiz", "Carmen Diaz"]
        names = _names(page, phrases, language="es")
        assert all(name.count(",") <= 1 for name in names), names
        assert names == ["Maria Gonzalez, Ana Lopez", "Pedro Ruiz", "Carmen Diaz"]

    @pytest.mark.parametrize("language", ["es", None])
    def test_capitalised_surnames_before_the_given_names(self, language: str | None) -> None:
        page = _page("Titular: GONZÁLEZ RUIZ, María José, con domicilio en Madrid")
        phrases = ["GONZÁLEZ RUIZ", "María José"]
        assert _names(page, phrases, language=language) == ["GONZÁLEZ RUIZ, María José"]

    def test_french_capitalised_surnames_before_the_given_names(self) -> None:
        page = _page("Titulaire : DUPONT MARTIN, Jean Pierre")
        phrases = ["DUPONT MARTIN", "Jean Pierre"]
        assert _names(page, phrases, language="fr") == ["DUPONT MARTIN, Jean Pierre"]

    @pytest.mark.parametrize(("language", "expected"), [("en", "Don Johnson"), ("es", "Johnson")])
    def test_don_is_a_first_name_outside_spanish(self, language: str, expected: str) -> None:
        page = _page("Signed by Don Johnson today")
        assert _names(page, ["Don Johnson"], language=language) == [expected]

    def test_dom_is_a_first_name_outside_portuguese(self) -> None:
        page = _page("Signed by Dom Phillips today")
        assert _names(page, ["Dom Phillips"], language="en") == ["Dom Phillips"]
        assert _names(page, ["Dom Phillips"], language="pt") == ["Phillips"]

    def test_initials_of_any_alphabet_are_included(self) -> None:
        page = _page("by Ł. Kowalski and J. R. R. Tolkien today")
        assert _names(page, ["Kowalski", "Tolkien"], language="en") == [
            "Ł. Kowalski",
            "J. R. R. Tolkien",
        ]

    def test_hyphen_at_the_edge_of_a_name_is_not_reported(self) -> None:
        page = _page("by Maria Gonzalez- today")
        assert _names(page, ["Maria Gonzalez-"]) == ["Maria Gonzalez"]

    def test_no_spans_give_no_mentions(self) -> None:
        assert clean_mentions(_page("Maria Gonzalez"), []) == []
        assert clean_mentions(PageText.from_words([]), [TextSpan(0, 3)]) == []


# --- pipeline: robustness ---------------------------------------------------------------------

_VOCABULARY = [
    "Maria", "GONZALEZ", "Ruiz,", "Dr.", "M.", "J.", "-", "'", "Anne-", "-Marie", "de", "la",
    "Weber", "weber", "12", "a@b.c", "Al", ",", "(Ana)", "Doña", "López.", _RQ, "Sir",
]  # fmt: skip


def _random_page(rng: random.Random) -> PageText:
    lines = [
        " ".join(rng.choice(_VOCABULARY) for _ in range(rng.randint(1, 6)))
        for _ in range(rng.randint(0, 5))
    ]
    return _page(*lines)


def _random_spans(rng: random.Random, length: int) -> list[TextSpan]:
    spans: list[TextSpan] = []
    for _ in range(rng.randint(0, 6)):
        start = rng.randint(0, length + 2)
        spans.append(TextSpan(start, start + rng.randint(0, 25)))
    return spans


def test_random_pages_give_sorted_disjoint_ranges_within_bounds() -> None:
    rng = random.Random(2026)
    for _ in range(2000):
        page = _random_page(rng)
        language = rng.choice([None, "fr", "en"])
        mentions = clean_mentions(page, _random_spans(rng, len(page.text)), language=language)
        previous_end = 0
        for name, word_range in mentions:
            assert previous_end <= word_range.start_word < word_range.end_word <= len(page.words)
            assert name
            assert name == name.strip()
            assert (
                len({page.words[i].line for i in range(word_range.start_word, word_range.end_word)})
                == 1
            )
            previous_end = word_range.end_word
