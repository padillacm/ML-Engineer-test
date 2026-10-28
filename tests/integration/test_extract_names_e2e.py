"""``ExtractNames`` end to end: real Tesseract OCR and the real pinned NER model.

Pages are image-only, drawn at 300 DPI with ``known_words_page`` and recognised at the engine's
default 200 DPI, so every reported box can be compared with the rectangle where its words were
drawn. The recogniser is loaded once per module.
"""

import dataclasses
import os
import socket
from collections.abc import Iterator, Sequence
from pathlib import Path

import pymupdf
import pytest
import torch

from app.adapters.ner import TransformersNameRecognizer, TransformersNerConfig
from app.adapters.ocr import TesseractOcrEngine
from app.application.extract_names import ExtractNames
from app.domain.geometry import iou, union
from app.domain.types import Box, ExtractionResult, NameMention, NamePair
from tests.adapters.ocr.pdfs import A4
from tests.integration.known_words import KnownPage, Line, known_words_page

pytestmark = [pytest.mark.tesseract, pytest.mark.ner_model]

MIN_IOU = 0.5
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "sample_pdfs"
SAMPLES = ("company_memo.pdf", "meeting_minutes.pdf", "research_report.pdf")


def _model_path() -> Path:
    return Path(os.environ.get("NER_MODEL_PATH", "/models/xlm-roberta-base-ner-hrl"))


@pytest.fixture(scope="module")
def recognizer() -> Iterator[TransformersNameRecognizer]:
    threads = torch.get_num_threads()
    yield TransformersNameRecognizer(TransformersNerConfig(_model_path()))
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def extract(recognizer: TransformersNameRecognizer) -> ExtractNames:
    return ExtractNames(TesseractOcrEngine(), recognizer)


def _page(lines: Sequence[str], targets: Sequence[str] = ()) -> KnownPage:
    drawn = [Line((72.0, 120.0 + 60.0 * index), text) for index, text in enumerate(lines)]
    return known_words_page(A4, drawn, targets)


def _truth(page: KnownPage, words: Sequence[str], page_index: int = 0) -> Box:
    box = union([page.targets[word] for word in words])
    return dataclasses.replace(box, page=page_index)


def _merge(pages: Sequence[KnownPage]) -> bytes:
    with pymupdf.open() as merged:
        for page in pages:
            with pymupdf.open(stream=page.pdf, filetype="pdf") as single:
                merged.insert_pdf(single)
        data: bytes = merged.tobytes()
    return data


def _names(result: ExtractionResult) -> list[str]:
    return [mention.name for mention in result.names]


def _describe(result: ExtractionResult) -> str:
    return "; ".join(f"{m.name!r} p{m.box.page} {m.box}" for m in result.names)


def test_only_the_person_is_reported_with_a_box_on_its_words(extract: ExtractNames) -> None:
    page = _page(
        ["Maria Gonzalez of Acme Corporation met in Madrid on 3 March"],
        ["Maria", "Gonzalez"],
    )

    result = extract(page.pdf, [], "en")

    assert _names(result) == ["Maria Gonzalez"], _describe(result)
    (mention,) = result.names
    assert iou(mention.box, _truth(page, ["Maria", "Gonzalez"])) >= MIN_IOU, _describe(result)
    assert result.matches == ()


def test_a_name_on_two_pages_gives_one_mention_per_page(extract: ExtractNames) -> None:
    first = _page(["The report was written by Maria Gonzalez"], ["Maria", "Gonzalez"])
    second = _page(
        ["Budget review", "", "Approved later by Maria Gonzalez in person"],
        ["Maria", "Gonzalez"],
    )

    result = extract(_merge([first, second]), [], "en")

    assert _names(result) == ["Maria Gonzalez", "Maria Gonzalez"], _describe(result)
    assert [m.box.page for m in result.names] == [0, 1]
    expected = [_truth(first, ["Maria", "Gonzalez"], 0), _truth(second, ["Maria", "Gonzalez"], 1)]
    for mention, truth in zip(result.names, expected, strict=True):
        assert iou(mention.box, truth) >= MIN_IOU, _describe(result)
    # Each box is on its own occurrence: the two occurrences sit at different heights.
    assert abs(result.names[0].box.y - result.names[1].box.y) > 50


def test_the_box_covers_the_name_without_title_or_role(extract: ExtractNames) -> None:
    page = _page(["Dr. Maria Gonzalez, CEO"], ["Dr.", "Maria", "Gonzalez", "CEO"])

    result = extract(page.pdf, [], "en")

    assert _names(result) == ["Maria Gonzalez"], _describe(result)
    (mention,) = result.names
    name_truth = _truth(page, ["Maria", "Gonzalez"])
    line_truth = _truth(page, ["Dr.", "Maria", "Gonzalez", "CEO"])
    assert iou(mention.box, name_truth) >= MIN_IOU
    # "Maria Gonzalez" is about 63 % of this short line, so even a perfect name box has an IoU
    # above 0.5 with the whole line: an IoU below 0.5 cannot tell the two apart here. The box
    # must be closer to the name than to the line, and its edges must leave out "Dr." and "CEO".
    assert iou(name_truth, line_truth) > MIN_IOU, "geometry of the scenario"
    assert iou(mention.box, name_truth) > iou(mention.box, line_truth)
    assert mention.box.x >= page.targets["Dr."].right - 1.0, "the box must exclude 'Dr.'"
    assert mention.box.right <= page.targets["CEO"].x + 1.0, "the box must exclude 'CEO'"


def test_a_name_in_capitals_is_reported(extract: ExtractNames) -> None:
    page = _page(["SIGNED BY MARIA GONZALEZ"], ["MARIA", "GONZALEZ"])

    result = extract(page.pdf, [], "en")

    truth = _truth(page, ["MARIA", "GONZALEZ"])
    covering = [m for m in result.names if iou(m.box, truth) >= MIN_IOU]
    assert len(covering) == 1, _describe(result)
    assert covering[0].name.replace(" ", "").upper() == "MARIAGONZALEZ", _describe(result)


def test_a_spanish_title_is_stripped(extract: ExtractNames) -> None:
    page = _page(["La carta fue firmada por la Sra. Ana López en Sevilla."], ["Ana", "López"])

    result = extract(page.pdf, [], "es")

    assert _names(result) == ["Ana López"], _describe(result)
    assert iou(result.names[0].box, _truth(page, ["Ana", "López"])) >= MIN_IOU


def test_a_french_title_is_stripped_with_the_detected_language(extract: ExtractNames) -> None:
    page = _page(
        [
            "Le contrat a été revu par le comité et les parties sont convenues que",
            "le paiement sera effectué dès que le service est livré sans défauts.",
            "M. Guyon a signé le contrat pour la société avec leur directeur.",
        ],
        ["Guyon"],
    )

    result = extract(page.pdf, [], None)

    assert _names(result) == ["Guyon"], _describe(result)
    assert iou(result.names[0].box, _truth(page, ["Guyon"])) >= MIN_IOU


def test_last_comma_first_is_one_name(extract: ExtractNames) -> None:
    page = _page(["Signatory: Gonzalez Ruiz, Maria"], ["Gonzalez", "Ruiz,", "Maria"])

    result = extract(page.pdf, [], "en")

    assert _names(result) == ["Gonzalez Ruiz, Maria"], _describe(result)
    (mention,) = result.names
    assert iou(mention.box, _truth(page, ["Gonzalez", "Ruiz,", "Maria"])) >= MIN_IOU


def test_a_longer_comma_separated_list_gives_one_name_per_pair(extract: ExtractNames) -> None:
    page = _page(["Authors: Weber, Hans, Smith, John, Rossi, Marco"])

    result = extract(page.pdf, [], "en")

    assert _names(result) == ["Weber, Hans", "Smith, John", "Rossi, Marco"], _describe(result)


@pytest.mark.parametrize(
    ("line", "language", "expected"),
    [
        (
            "Titular: GONZÁLEZ RUIZ, María José, con domicilio en Madrid",
            "es",
            "GONZÁLEZ RUIZ, María José",
        ),
        ("Titulaire : DUPONT MARTIN, Jean Pierre", "fr", "DUPONT MARTIN, Jean Pierre"),
    ],
)
def test_capitalised_surnames_before_the_given_names_are_one_name(
    extract: ExtractNames, line: str, language: str, expected: str
) -> None:
    page = _page([line])

    result = extract(page.pdf, [], language)

    assert _names(result) == [expected], _describe(result)


def test_a_titled_surname_is_kept_and_a_bare_one_dropped(extract: ExtractNames) -> None:
    page = _page(
        ["The audit was led by Dr. Weber last year.", "", "The new plan was drafted by Weber."],
        ["Dr."],
    )

    result = extract(page.pdf, [], "en")

    assert _names(result) == ["Weber"], _describe(result)
    title = page.targets["Dr."]
    (mention,) = result.names
    assert abs(mention.box.y - title.y) < title.height, "only the titled line's Weber"
    assert mention.box.x > title.right - 1.0


def test_a_name_followed_by_a_role_on_the_next_line_is_only_the_name(
    extract: ExtractNames,
) -> None:
    page = _page(
        ["The contract was signed by Maria Gonzalez", "Director of the regional office"],
        ["Maria", "Gonzalez", "Director"],
    )

    result = extract(page.pdf, [], "en")

    assert _names(result) == ["Maria Gonzalez"], _describe(result)
    assert iou(result.names[0].box, _truth(page, ["Maria", "Gonzalez"])) >= MIN_IOU
    assert result.names[0].box.bottom <= page.targets["Director"].y + 1.0


def test_a_repeated_name_is_matched_once(extract: ExtractNames) -> None:
    page = _page(
        [
            "The memo was sent by Maria Gonzalez today.",
            "Questions go to Maria Gonzalez as well.",
            "Approved by Maria Gonzalez on Friday.",
        ]
    )
    requested = [NamePair("María", "González")]

    result = extract(page.pdf, requested, "en")

    assert _names(result) == ["Maria Gonzalez"] * 3, _describe(result)
    (match,) = result.matches
    assert match.extracted_name == "Maria Gonzalez"
    assert match.score == pytest.approx(1.0)

    without_request = extract(page.pdf, [], "en")
    assert without_request.matches == ()
    assert without_request.names == result.names


@pytest.fixture(scope="module")
def samples(extract: ExtractNames) -> dict[str, ExtractionResult]:
    return {name: extract((SAMPLES_DIR / name).read_bytes(), [], None) for name in SAMPLES}


def _page_sizes(name: str) -> list[tuple[float, float]]:
    with pymupdf.open(SAMPLES_DIR / name) as document:
        return [(float(page.rect.width), float(page.rect.height)) for page in document]


def _person_like(mention: NameMention) -> bool:
    return any(word[:1].isupper() for word in mention.name.split())


@pytest.mark.parametrize("name", SAMPLES)
def test_sample_pdfs_give_person_names_with_boxes_on_their_pages(
    samples: dict[str, ExtractionResult], name: str
) -> None:
    result = samples[name]
    sizes = _page_sizes(name)
    print(f"\n{name}: {[(m.name, m.box.page) for m in result.names]}")

    if name == "company_memo.pdf":
        assert result.names, "company_memo.pdf must contain names"
    pages = [m.box.page for m in result.names]
    assert pages == sorted(pages)
    for mention in result.names:
        assert 0 <= mention.box.page < len(sizes), mention
        width, height = sizes[mention.box.page]
        box = mention.box
        assert box.x >= 0, mention
        assert box.y >= 0, mention
        assert box.right <= width + 1e-6, mention
        assert box.bottom <= height + 1e-6, mention
        assert box.width > 0, mention
        assert box.height > 0, mention
        assert _person_like(mention), mention
    assert result.matches == ()


def test_a_comma_separated_list_of_full_names_is_not_joined(
    samples: dict[str, ExtractionResult],
) -> None:
    joined = [
        m.name
        for m in samples["research_report.pdf"].names
        if "," in m.name and all(len(part.split()) >= 2 for part in m.name.split(","))
    ]
    assert joined == []


@pytest.mark.xfail(
    strict=True,
    reason="known model error: 'ETH Zurich' (an organisation) is tagged PER in research_report",
)
def test_an_organisation_in_a_sample_is_not_reported(
    samples: dict[str, ExtractionResult],
) -> None:
    assert "ETH Zurich" not in [m.name for m in samples["research_report.pdf"].names]


def test_extraction_works_with_the_network_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _page(["Maria Gonzalez of Acme Corporation met in Madrid on 3 March"])
    attempts: list[object] = []

    def refuse(self: socket.socket, address: object) -> None:
        attempts.append(address)
        raise OSError("network disabled by the test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    threads = torch.get_num_threads()
    try:
        offline = TransformersNameRecognizer(TransformersNerConfig(_model_path()))
        result = ExtractNames(TesseractOcrEngine(), offline)(
            page.pdf, [NamePair("Maria", "Gonzalez")], "en"
        )
    finally:
        torch.set_num_threads(threads)

    assert _names(result) == ["Maria Gonzalez"], _describe(result)
    assert len(result.matches) == 1
    assert attempts == []
