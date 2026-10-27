"""Regression table of look-alike names at the default threshold.

Each row records whether ``match_names`` accepts the extracted name for the requested pair at
``DEFAULT_THRESHOLD``. A change of threshold or scorer that flips a row fails here, so the effect
on precision and recall is visible. Rows marked "known" document behaviour that is accepted
today but is not an endorsement: see the risks section of the design.
"""

import pytest

from app.domain.matching import DEFAULT_THRESHOLD, match_names
from app.domain.types import NamePair

# (case id, extracted name, requested first name, requested last name, expected to match)
DECOY_TABLE = [
    # Same surname, clearly different given name.
    ("surname-pedro-vs-maria", "Pedro Martinez", "Maria", "Martinez", False),
    ("surname-daniel-vs-david", "Daniel Perez", "David", "Perez", False),
    ("surname-jose-vs-jorge", "Jose Garcia", "Jorge", "Garcia", False),
    # Same given name, different surname.
    ("given-maria-rodriguez-vs-gonzalez", "Maria Rodriguez", "Maria", "Gonzalez", False),
    # Extra or missing name parts.
    ("extra-given-name", "Karl-Heinz Müller", "Heinz", "Müller", False),
    ("extra-surname", "Maria Gonzalez Perez", "Maria", "Gonzalez", False),
    ("missing-second-given-name", "Jose Garcia", "Jose Luis", "Garcia", False),
    # Near-miss given names: known to be accepted at 0.90.
    ("near-miss-known-mario-vs-maria", "Maria Gonzalez", "Mario", "Gonzalez", True),
    ("near-miss-known-marta-vs-maria", "Maria Gonzalez", "Marta", "Gonzalez", True),
    ("near-miss-known-jon-vs-john", "Jon Smith", "John", "Smith", True),
    ("near-miss-known-anna-vs-ana", "Anna Lopez", "Ana", "Lopez", True),
    ("near-miss-known-luis-vs-luisa", "Luis Fernandez", "Luisa", "Fernandez", True),
    ("near-miss-surname-spelling", "Clara Sanchez", "Clara", "Sanches", True),
    # Word order.
    ("reversed-order", "Gonzalez Maria", "Maria", "Gonzalez", True),
    ("reversed-order-with-comma", "Garcia, Jose", "Jose", "Garcia", True),
    # Accent and case only.
    ("accent-only-extracted", "José García", "Jose", "Garcia", True),
    ("accent-only-requested", "Jose Garcia", "José", "García", True),
    ("case-only", "JOSE GARCIA", "Jose", "Garcia", True),
    ("letters-without-decomposition", "Søren Łukasz", "Soren", "Lukasz", True),
    ("typographic-apostrophe", "Sean O\u2019Brien", "Sean", "O'Brien", True),
    ("hyphen-as-space", "Jean-Luc Picard", "Jean Luc", "Picard", True),
    # OCR-style character swaps.
    ("ocr-l-as-1", "Maria Gonza1ez", "Maria", "Gonzalez", True),
    ("ocr-i-as-1", "Mar1a Gonzalez", "Maria", "Gonzalez", True),
    ("ocr-o-as-0", "Pedr0 Martinez", "Pedro", "Martinez", True),
    ("ocr-l-as-1-in-given-name", "Ol1via Ruiz", "Olivia", "Ruiz", True),
    ("ocr-i-as-l", "Marla Gonzalez", "Maria", "Gonzalez", True),
    # "rn" read for "m" is a known miss: it costs two edits and drops under 0.90.
    ("ocr-rn-as-m-known-miss", "Rnaria Gonzalez", "Maria", "Gonzalez", False),
    ("ocr-rn-as-m-short-name-known-miss", "Mark Rnartin", "Mark", "Martin", False),
]


@pytest.mark.parametrize(
    ("extracted", "first_name", "last_name", "expected"),
    [case[1:] for case in DECOY_TABLE],
    ids=[case[0] for case in DECOY_TABLE],
)
def test_decoy_regression_at_default_threshold(
    extracted: str, first_name: str, last_name: str, expected: bool
) -> None:
    result = match_names([extracted], [NamePair(first_name, last_name)], DEFAULT_THRESHOLD)

    assert bool(result) is expected


def test_table_covers_both_outcomes_and_unique_ids() -> None:
    ids = [case[0] for case in DECOY_TABLE]

    assert len(ids) == len(set(ids))
    assert {case[4] for case in DECOY_TABLE} == {True, False}


def test_best_pair_is_chosen_among_look_alikes() -> None:
    pairs = [NamePair("Mario", "Gonzalez"), NamePair("Maria", "Gonzalez")]

    result = match_names(["Maria Gonzalez"], pairs)

    assert [m.matched_name for m in result] == ["Maria Gonzalez"]
