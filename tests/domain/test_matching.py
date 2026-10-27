"""Behaviour of name normalisation, scoring and best-match selection."""

import time

import pytest

from app.domain.errors import ConfigurationError
from app.domain.matching import DEFAULT_THRESHOLD, match_names, normalize, score
from app.domain.types import FuzzyMatch, NamePair

SAMPLES = [
    "JOSÉ GARCÍA",
    "María González",
    "Weiß",
    "  Maria\n  Gonzalez ",
    "Karl-Heinz Müller",
    "Garcia, Jose",
    "",
    "   ",
    "Ｊｏｓé",
    "Zoë\tBrontë",
    "Ana\u00a0Lopez",
]


class TestNormalize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("JOSÉ GARCÍA", "jose garcia"),
            ("María González", "maria gonzalez"),
            ("Zoë Brontë", "zoe bronte"),
            ("MARIA", "maria"),
            ("Weiß", "weiss"),
            ("WEISS", "weiss"),
            ("  Maria\n  Gonzalez ", "maria gonzalez"),
            ("Ana\tLopez", "ana lopez"),
            ("Ana\u00a0Lopez", "ana lopez"),
            ("Ｊｏｓé", "jose"),
            ("", ""),
            (" \n\t ", ""),
        ],
    )
    def test_expected_text(self, raw: str, expected: str) -> None:
        assert normalize(raw) == expected

    def test_sharp_s_and_uppercase_give_same_text(self) -> None:
        assert normalize("Weiß") == normalize("WEISS")

    @pytest.mark.parametrize("sample", SAMPLES)
    def test_is_idempotent(self, sample: str) -> None:
        assert normalize(normalize(sample)) == normalize(sample)

    def test_is_idempotent_for_every_code_point(self) -> None:
        not_stable = [
            hex(code_point)
            for code_point in range(0x30000)
            if not 0xD800 <= code_point <= 0xDFFF
            and normalize(normalize(chr(code_point))) != normalize(chr(code_point))
        ]

        assert not_stable == []


class TestScore:
    @pytest.mark.parametrize(
        ("extracted", "pair"),
        [
            ("JOSÉ GARCÍA", NamePair("Jose", "Garcia")),
            ("María González", NamePair("Maria", "Gonzalez")),
            ("Jose Garcia", NamePair("José", "García")),
            ("Garcia Jose", NamePair("Jose", "Garcia")),
        ],
    )
    def test_identical_after_normalisation_scores_one(self, extracted: str, pair: NamePair) -> None:
        assert score(extracted, pair) == 1.0

    def test_word_order_differs(self) -> None:
        assert score("Garcia, Jose", NamePair("Jose", "Garcia")) >= 0.90

    def test_letters_without_a_decomposition_are_folded(self) -> None:
        assert score("Søren Łukasz", NamePair("Soren", "Lukasz")) >= 0.90

    def test_typographic_apostrophe_matches_the_plain_one(self) -> None:
        assert score("Sean O\u2019Brien", NamePair("Sean", "O'Brien")) == 1.0

    def test_hyphenated_name_matches_its_spaced_form(self) -> None:
        assert score("Jean-Luc Picard", NamePair("Jean Luc", "Picard")) == 1.0

    def test_extra_given_name_is_not_a_full_match(self) -> None:
        assert score("Karl-Heinz Müller", NamePair("Heinz", "Müller")) < 0.90

    def test_same_surname_different_given_name(self) -> None:
        assert score("Pedro Martinez", NamePair("Maria", "Martinez")) < 0.90

    def test_recognition_error_stays_within_tolerance(self) -> None:
        assert score("Maria Gonza1ez", NamePair("Maria", "Gonzalez")) >= 0.90

    def test_unrelated_names_score_low(self) -> None:
        assert score("Carlos Ruiz", NamePair("Ana", "Lopez")) < 0.5

    @pytest.mark.parametrize(
        ("extracted", "pair"),
        [
            ("", NamePair("Ana", "Lopez")),
            ("Ana Lopez", NamePair("", "")),
            ("Ana Lopez", NamePair("Ana", "Lopez")),
            ("zzzz", NamePair("Ana", "Lopez")),
        ],
    )
    def test_is_within_unit_interval(self, extracted: str, pair: NamePair) -> None:
        assert 0.0 <= score(extracted, pair) <= 1.0


class TestThreshold:
    def test_default_is_ninety_percent(self) -> None:
        assert DEFAULT_THRESHOLD == 0.90

    def test_recognition_error_is_reported(self) -> None:
        result = match_names(["Maria Gonza1ez"], [NamePair("Maria", "Gonzalez")])

        assert len(result) == 1
        assert result[0].score >= 0.90

    def test_score_exactly_at_threshold_is_reported(self) -> None:
        pair = NamePair("Eva", "Gomezs")
        assert score("Eva Gomezz", pair) == pytest.approx(0.90)

        result = match_names(["Eva Gomezz"], [pair], threshold=0.90)

        assert [m.matched_name for m in result] == ["Eva Gomezs"]

    def test_score_within_float_noise_of_threshold_counts(self) -> None:
        pair = NamePair("Eva", "Gomezs")
        exact = score("Eva Gomezz", pair)

        assert len(match_names(["Eva Gomezz"], [pair], threshold=exact + 1e-14)) == 1

    def test_score_clearly_below_threshold_is_not_reported(self) -> None:
        pair = NamePair("Eva", "Gomezs")
        exact = score("Eva Gomezz", pair)

        assert match_names(["Eva Gomezz"], [pair], threshold=exact + 1e-6) == []

    def test_stricter_threshold_rejects_a_lower_score(self) -> None:
        pair = NamePair("Maria", "Gonzalez")
        assert score("Maria Gonza1ez", pair) == pytest.approx(0.93, abs=0.005)

        assert match_names(["Maria Gonza1ez"], [pair], threshold=0.98) == []

    @pytest.mark.parametrize("threshold", [1.5, -0.1, float("nan"), float("inf")])
    def test_invalid_threshold_is_rejected(self, threshold: float) -> None:
        with pytest.raises(ConfigurationError) as raised:
            match_names(["Ana Lopez"], [NamePair("Ana", "Lopez")], threshold=threshold)

        assert raised.value.code == "invalid_configuration"

    @pytest.mark.parametrize("threshold", [True, False, "0.9", None, [0.9], 1 + 0j])
    def test_non_numeric_threshold_is_a_configuration_error(self, threshold: object) -> None:
        with pytest.raises(ConfigurationError) as raised:
            match_names(["Ana Lopez"], [NamePair("Ana", "Lopez")], threshold=threshold)  # type: ignore[arg-type]  # invalid on purpose

        assert raised.value.code == "invalid_configuration"

    def test_integer_threshold_within_range_is_accepted(self) -> None:
        assert len(match_names(["Ana Lopez"], [NamePair("Ana", "Lopez")], threshold=1)) == 1

    def test_invalid_threshold_is_rejected_even_without_inputs(self) -> None:
        with pytest.raises(ConfigurationError):
            match_names([], [], threshold=1.5)

    @pytest.mark.parametrize("threshold", [0.0, 1.0])
    def test_bounds_are_valid(self, threshold: float) -> None:
        result = match_names(["Ana Lopez"], [NamePair("Ana", "Lopez")], threshold=threshold)

        assert len(result) == 1


class TestBestMatch:
    def test_two_close_candidates_yield_the_better_one(self) -> None:
        pairs = [NamePair("Ana", "Lopes"), NamePair("Ana", "Lopez")]

        result = match_names(["Ana Lopez"], pairs, threshold=0.80)

        assert result == [FuzzyMatch("Ana Lopez", "Ana Lopez", 1.0)]

    def test_tie_goes_to_the_first_pair_in_the_request(self) -> None:
        pairs = [NamePair("Jose", "Garcia"), NamePair("José", "García")]

        result = match_names(["Jose Garcia"], pairs)

        assert [m.matched_name for m in result] == ["Jose Garcia"]

    def test_tie_between_identical_pairs_reports_one_entry(self) -> None:
        pairs = [NamePair("Ana", "Lopez"), NamePair("Ana", "Lopez")]

        assert len(match_names(["Ana Lopez"], pairs)) == 1

    def test_no_candidate_reaches_the_threshold(self) -> None:
        assert match_names(["Carlos Ruiz"], [NamePair("Ana", "Lopez")]) == []

    def test_matched_name_keeps_the_requested_spelling(self) -> None:
        result = match_names(["JOSE GARCIA"], [NamePair("José", "García")])

        assert result == [FuzzyMatch("JOSE GARCIA", "José García", 1.0)]

    def test_extracted_name_keeps_its_original_form(self) -> None:
        result = match_names(["  María\n González "], [NamePair("Maria", "Gonzalez")])

        assert result[0].extracted_name == "  María\n González "

    def test_output_follows_the_order_of_the_extracted_names(self) -> None:
        pairs = [NamePair("Ana", "Lopez"), NamePair("Jose", "Garcia")]

        result = match_names(["Garcia Jose", "Carlos Ruiz", "Ana Lopez"], pairs)

        assert [m.extracted_name for m in result] == ["Garcia Jose", "Ana Lopez"]

    def test_repeated_extracted_names_are_not_deduplicated(self) -> None:
        result = match_names(["Ana Lopez", "Ana Lopez"], [NamePair("Ana", "Lopez")])

        assert len(result) == 2

    def test_several_extracted_names_can_match_the_same_pair(self) -> None:
        result = match_names(["Ana Lopez", "Lopez Ana"], [NamePair("Ana", "Lopez")])

        assert [m.matched_name for m in result] == ["Ana Lopez", "Ana Lopez"]


class TestDegenerateInputs:
    def test_nothing_extracted(self) -> None:
        assert match_names([], [NamePair("Ana", "Lopez")]) == []

    def test_nothing_requested(self) -> None:
        assert match_names(["Ana Lopez"], []) == []

    def test_both_empty(self) -> None:
        assert match_names([], []) == []

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t ", "\u00a0"])
    def test_blank_extracted_name_produces_no_entry(self, blank: str) -> None:
        assert match_names([blank], [NamePair("Ana", "Lopez")]) == []

    def test_blank_name_does_not_hide_its_neighbours(self) -> None:
        result = match_names(["  ", "Ana Lopez"], [NamePair("Ana", "Lopez")])

        assert [m.extracted_name for m in result] == ["Ana Lopez"]

    def test_blank_requested_pair_never_matches_a_real_name(self) -> None:
        assert match_names(["Ana Lopez"], [NamePair("", "")]) == []

    @pytest.mark.parametrize("pair", [NamePair(" ", "\n"), NamePair("-", "\u2013")])
    def test_pair_blank_after_normalisation_never_matches(self, pair: NamePair) -> None:
        assert match_names(["Ana Lopez"], [pair], threshold=0.0) == []

    def test_blank_pair_does_not_hide_the_other_pairs(self) -> None:
        pairs = [NamePair("", ""), NamePair("Ana", "Lopez")]

        assert [m.matched_name for m in match_names(["Ana Lopez"], pairs)] == ["Ana Lopez"]

    def test_blank_extracted_name_is_skipped_even_with_a_blank_pair(self) -> None:
        assert match_names([" "], [NamePair("", "")], threshold=0.0) == []


class TestScale:
    def test_many_names_against_many_pairs_finish_quickly(self) -> None:
        names = [f"Nombre{i} Apellido{i}" for i in range(2000)]
        pairs = [NamePair(f"Nombre{i}", f"Apellido{i}") for i in range(0, 2000, 4)]

        started = time.perf_counter()
        result = match_names(names, pairs)
        elapsed = time.perf_counter() - started

        assert len(pairs) == 500
        assert len(result) >= 500
        assert elapsed < 5.0
