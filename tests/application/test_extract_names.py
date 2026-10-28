"""Use case ``ExtractNames`` with in-memory OCR and recogniser fakes."""

import math
from collections.abc import Sequence

import pytest

from app.application.extract_names import ExtractNames
from app.domain.errors import ConfigurationError, ProcessingFailed
from app.domain.geometry import union
from app.domain.types import Box, NamePair, OcrPage, PageWord, TextSpan
from tests.fakes.names import FakeNameRecognizer
from tests.fakes.ocr import FakeOcrEngine

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0
WORD_HEIGHT = 12.0
CHAR_WIDTH = 6.0
LINE_STEP = 20.0


def make_page(page: int, lines: Sequence[str], language: str | None = None) -> OcrPage:
    """A page whose words are laid out left to right, one printed line per entry of ``lines``.

    ``language`` is the language the fake OCR reports having read the page with.
    """
    words: list[PageWord] = []
    for line_index, line in enumerate(lines):
        x = 50.0
        y = 50.0 + line_index * LINE_STEP
        for text in line.split():
            width = CHAR_WIDTH * len(text)
            box = Box(page=page, x=x, y=y, width=width, height=WORD_HEIGHT)
            words.append(PageWord(text=text, box=box, line=line_index, paragraph=0))
            x += width + CHAR_WIDTH
    return OcrPage(
        page=page, width=PAGE_WIDTH, height=PAGE_HEIGHT, words=tuple(words), language=language
    )


def word_boxes(page: OcrPage, *texts: str) -> list[Box]:
    return [word.box for word in page.words if word.text in texts]


def extract(
    pages: Sequence[OcrPage], persons: Sequence[str]
) -> tuple[ExtractNames, FakeOcrEngine, FakeNameRecognizer]:
    ocr = FakeOcrEngine(pages)
    recognizer = FakeNameRecognizer(persons)
    return ExtractNames(ocr, recognizer), ocr, recognizer


def test_only_person_names_are_reported() -> None:
    page = make_page(0, ["Maria Gonzalez of Acme Corporation met in Madrid on 3 March"])
    use_case, _, _ = extract([page], ["Maria Gonzalez"])

    result = use_case(b"%PDF", [])

    assert [mention.name for mention in result.names] == ["Maria Gonzalez"]
    assert result.names[0].box == union(word_boxes(page, "Maria", "Gonzalez"))


def test_repeated_name_on_two_pages_gives_one_mention_per_page_with_its_own_box() -> None:
    first = make_page(0, ["Signed by Maria Gonzalez"])
    second = make_page(1, ["", "Approved: Maria Gonzalez today"])
    use_case, _, _ = extract([first, second], ["Maria Gonzalez"])

    result = use_case(b"%PDF", [])

    assert [(m.name, m.box.page) for m in result.names] == [
        ("Maria Gonzalez", 0),
        ("Maria Gonzalez", 1),
    ]
    assert result.names[0].box == union(word_boxes(first, "Maria", "Gonzalez"))
    assert result.names[1].box == union(word_boxes(second, "Maria", "Gonzalez"))
    assert result.names[0].box != result.names[1].box


def test_box_covers_the_name_words_only() -> None:
    page = make_page(0, ["Dr. Maria Gonzalez, CEO"])
    use_case, _, _ = extract([page], ["Dr. Maria Gonzalez"])

    result = use_case(b"%PDF", [])

    assert [mention.name for mention in result.names] == ["Maria Gonzalez"]
    assert result.names[0].box == union(word_boxes(page, "Maria", "Gonzalez,"))


def test_capitals_are_title_cased_for_the_recognizer_and_reported_as_printed() -> None:
    page = make_page(0, ["SIGNED BY MARIA GONZALEZ"])
    use_case, _, recognizer = extract([page], ["Maria Gonzalez"])

    result = use_case(b"%PDF", [])

    assert recognizer.calls == ["Signed By Maria Gonzalez"]
    assert [mention.name for mention in result.names] == ["MARIA GONZALEZ"]
    assert result.names[0].box == union(word_boxes(page, "MARIA", "GONZALEZ"))


def test_span_across_a_line_break_keeps_the_name_line_only() -> None:
    page = make_page(0, ["Signed by Maria Gonzalez", "Director of Sales"])
    use_case, _, _ = extract([page], ["Maria Gonzalez Director"])

    result = use_case(b"%PDF", [])

    assert [mention.name for mention in result.names] == ["Maria Gonzalez"]
    assert result.names[0].box == union(word_boxes(page, "Maria", "Gonzalez"))


def test_names_follow_page_order_then_reading_order() -> None:
    pages = [
        make_page(0, ["Ana Lopez and Juan Perez", "then Maria Gonzalez"]),
        make_page(1, ["Pedro Ruiz signed"]),
    ]
    use_case, _, _ = extract(pages, ["Maria Gonzalez", "Juan Perez", "Ana Lopez", "Pedro Ruiz"])

    result = use_case(b"%PDF", [])

    assert [(m.name, m.box.page) for m in result.names] == [
        ("Ana Lopez", 0),
        ("Juan Perez", 0),
        ("Maria Gonzalez", 0),
        ("Pedro Ruiz", 1),
    ]


def test_repeated_name_is_matched_once_with_full_score_despite_accents() -> None:
    pages = [
        make_page(0, ["Maria Gonzalez wrote to Maria Gonzalez"]),
        make_page(1, ["Copy to MARIA GONZALEZ"]),
    ]
    use_case, _, _ = extract(pages, ["Maria Gonzalez"])

    result = use_case(b"%PDF", [NamePair("María", "González")])

    assert [m.name for m in result.names] == ["Maria Gonzalez"] * 2 + ["MARIA GONZALEZ"]
    assert len(result.matches) == 1
    match = result.matches[0]
    assert match.extracted_name == "Maria Gonzalez"
    assert match.matched_name == "María González"
    assert match.score == pytest.approx(1.0)


def test_distinct_names_are_matched_in_first_occurrence_order() -> None:
    page = make_page(0, ["Juan Perez met Maria Gonzalez and Juan Perez"])
    use_case, _, _ = extract([page], ["Maria Gonzalez", "Juan Perez"])

    result = use_case(b"%PDF", [NamePair("Maria", "Gonzalez"), NamePair("Juan", "Perez")])

    assert [m.extracted_name for m in result.matches] == ["Juan Perez", "Maria Gonzalez"]


def test_empty_request_reports_names_and_no_matches() -> None:
    page = make_page(0, ["Signed by Maria Gonzalez"])
    use_case, _, _ = extract([page], ["Maria Gonzalez"])

    result = use_case(b"%PDF", [])

    assert [m.name for m in result.names] == ["Maria Gonzalez"]
    assert result.matches == ()


def test_blank_pages_are_skipped_without_calling_the_recognizer() -> None:
    blank = OcrPage(page=0, width=PAGE_WIDTH, height=PAGE_HEIGHT, words=(), language=None)
    whitespace = OcrPage(
        page=1,
        width=PAGE_WIDTH,
        height=PAGE_HEIGHT,
        words=(PageWord(text="  ", box=Box(1, 10.0, 10.0, 5.0, 5.0), line=0, paragraph=0),),
        language=None,
    )
    text_page = make_page(2, ["Signed by Maria Gonzalez"])
    use_case, _, recognizer = extract([blank, whitespace, text_page], ["Maria Gonzalez"])

    result = use_case(b"%PDF", [])

    assert recognizer.calls == ["Signed by Maria Gonzalez"]
    assert [(m.name, m.box.page) for m in result.names] == [("Maria Gonzalez", 2)]


def test_recognizer_is_called_once_per_non_empty_page() -> None:
    pages = [make_page(0, ["Page one text"]), make_page(1, ["Page two text"])]
    use_case, _, recognizer = extract(pages, [])

    result = use_case(b"%PDF", [])

    assert recognizer.calls == ["Page one text", "Page two text"]
    assert result.names == ()


def test_language_hint_is_passed_to_ocr_and_the_hinted_language_makes_m_a_french_title() -> None:
    page = make_page(0, ["Signé par M. Dupont"])
    use_case, ocr, _ = extract([page], ["Dupont"])

    french = use_case(b"%PDF", [], language="fr")

    assert ocr.calls == [(b"%PDF", "fr")]
    assert [m.name for m in french.names] == ["Dupont"]
    assert french.names[0].box == union(word_boxes(page, "Dupont"))


def test_title_follows_the_language_the_ocr_detected_without_a_hint() -> None:
    page = make_page(0, ["M. Guyon a signé"], language="fr")
    use_case, ocr, _ = extract([page], ["M. Guyon"])

    result = use_case(b"%PDF", [])

    assert ocr.calls == [(b"%PDF", None)]
    assert [m.name for m in result.names] == ["Guyon"]
    assert result.names[0].box == union(word_boxes(page, "Guyon"))


def test_without_a_hint_or_a_detected_language_m_stays_an_initial() -> None:
    page = make_page(0, ["Signé par M. Dupont"], language=None)
    use_case, _, _ = extract([page], ["M. Dupont"])

    result = use_case(b"%PDF", [])

    assert [m.name for m in result.names] == ["M. Dupont"]


def test_box_is_clamped_to_the_page() -> None:
    words = (
        PageWord(text="Maria", box=Box(0, 560.0, 50.0, 40.0, 12.0), line=0, paragraph=0),
        PageWord(text="Gonzalez", box=Box(0, 604.0, 50.0, 30.0, 12.0), line=0, paragraph=0),
    )
    page = OcrPage(page=0, width=PAGE_WIDTH, height=PAGE_HEIGHT, words=words, language=None)
    use_case, _, _ = extract([page], ["Maria Gonzalez"])

    result = use_case(b"%PDF", [])

    assert result.names[0].box == Box(0, 560.0, 50.0, PAGE_WIDTH - 560.0, 12.0)


def test_custom_threshold_is_used_for_matching() -> None:
    page = make_page(0, ["Signed by Maria Gonzales"])
    ocr = FakeOcrEngine([page])
    requested = [NamePair("Maria", "Gonzalez")]

    strict = ExtractNames(ocr, FakeNameRecognizer(["Maria Gonzales"]), threshold=1.0)
    lenient = ExtractNames(ocr, FakeNameRecognizer(["Maria Gonzales"]), threshold=0.5)

    assert strict(b"%PDF", requested).matches == ()
    assert [m.extracted_name for m in lenient(b"%PDF", requested).matches] == ["Maria Gonzales"]


@pytest.mark.parametrize("threshold", [-0.1, 1.5, math.nan, math.inf, True])
def test_invalid_threshold_is_rejected_at_construction(threshold: float) -> None:
    with pytest.raises(ConfigurationError):
        ExtractNames(FakeOcrEngine(), FakeNameRecognizer(), threshold=threshold)


class _FailingOcr:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def recognize(self, pdf: bytes, language: str | None = None) -> list[OcrPage]:
        raise self.error


class _FailingRecognizer:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def find_persons(self, text: str) -> list[TextSpan]:
        raise self.error


def test_ocr_errors_propagate_unchanged() -> None:
    error = ProcessingFailed("page cannot be read", code="ocr_failed")
    use_case = ExtractNames(_FailingOcr(error), FakeNameRecognizer())

    with pytest.raises(ProcessingFailed) as raised:
        use_case(b"%PDF", [])

    assert raised.value is error


def test_recognizer_errors_propagate_unchanged() -> None:
    error = RuntimeError("model failed")
    use_case = ExtractNames(FakeOcrEngine([make_page(0, ["Some text"])]), _FailingRecognizer(error))

    with pytest.raises(RuntimeError) as raised:
        use_case(b"%PDF", [])

    assert raised.value is error
