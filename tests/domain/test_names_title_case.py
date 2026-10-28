import random
import unicodedata

import pytest

from app.domain.names import title_case_caps


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SIGNED BY MARIA GONZALEZ", "Signed By Maria Gonzalez"),
        ("JOSÉ ÁLVAREZ", "José Álvarez"),
        ("JEAN-PIERRE O'BRIEN", "Jean-Pierre O'Brien"),
        ("(GONZALEZ),", "(Gonzalez),"),
        ("ÉMILE ZOLA.", "Émile Zola."),
    ],
)
def test_all_caps_words_are_title_cased(text: str, expected: str) -> None:
    assert title_case_caps(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Maria Gonzalez",
        "McDONALD",
        "A B C",
        "2024 12345",
        "AB12 X9",
        "STRAßE",
        "maria",
        "",
        "  ",
    ],
)
def test_words_that_are_not_all_caps_are_unchanged(text: str) -> None:
    assert title_case_caps(text) == text


def test_capital_sharp_s_is_lowered_without_changing_the_length() -> None:
    assert title_case_caps("GROẞ") == "Groß"


def test_dotted_capital_i_is_kept_because_its_lowercase_has_two_characters() -> None:
    assert len("İ".lower()) == 2
    assert title_case_caps("İSTANBUL KİM") == "İstanbul Kİm"


def test_decomposed_capitals_with_combining_marks_are_title_cased() -> None:
    text = unicodedata.normalize("NFD", "JOSÉ MÜLLER, ÑANDÚ")
    result = title_case_caps(text)
    assert result == unicodedata.normalize("NFD", "José Müller, Ñandú")
    assert len(result) == len(text)


def test_a_single_decomposed_capital_is_kept() -> None:
    text = unicodedata.normalize("NFD", "É")
    assert title_case_caps(text) == text


def test_whitespace_and_digits_between_words_are_preserved() -> None:
    assert title_case_caps("PAGE 2\tOF  3 MARIA") == "Page 2\tOf  3 Maria"


_ALPHABET = (
    list("ABCDEZÁÉÍÓÚÑÄÖÜÇÀÈÂẞßİI\N{LATIN SMALL LETTER DOTLESS I}abcz")
    + list("019-'.,()@\N{RIGHT SINGLE QUOTATION MARK}")
    + ["\N{COMBINING DIAERESIS}", "\N{COMBINING ACUTE ACCENT}"]
    + [" ", " ", "\n"]
)


def _random_text(rng: random.Random) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 40)))


def test_random_texts_keep_length_whitespace_and_only_lower_characters() -> None:
    rng = random.Random(1234)
    for _ in range(3000):
        text = _random_text(rng)
        result = title_case_caps(text)
        assert len(result) == len(text)
        for original, changed in zip(text, result, strict=True):
            assert changed == original or changed == original.lower()
            assert changed.isspace() == original.isspace()
        assert title_case_caps(result) == result


def test_random_all_caps_words_keep_first_letter_and_lower_the_rest() -> None:
    rng = random.Random(99)
    capitals = "ABCDEZÁÉÍÓÚÑÄÖÜÇẞ"
    for _ in range(500):
        word = "".join(rng.choice(capitals) for _ in range(rng.randint(2, 12)))
        result = title_case_caps(word)
        assert result[0] == word[0]
        assert result[1:] == word[1:].lower()
