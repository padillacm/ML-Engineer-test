import itertools
import random
import time
from pathlib import Path
from typing import ClassVar

import pytest

from app.domain.chunking import ChunkingConfig, chunk_document, chunk_page, clean_page_text
from app.domain.errors import ConfigurationError
from app.domain.types import Chunk

FIXTURE = Path(__file__).parent / "fixtures" / "ocr_style_pages.txt"
PAGE_MARKER = "=== PAGE ==="


def no_overlap(max_chars: int = 800) -> ChunkingConfig:
    return ChunkingConfig(max_chars=max_chars, overlap_ratio=0.0)


def sentence(letter: str, length: int) -> str:
    """A sentence of exactly ``length`` characters made of words of one letter."""
    body = " ".join(letter * 9 for _ in range(length // 10 + 1))
    return body[: length - 1].rstrip() + "."


def paragraph(letter: str, length: int) -> str:
    """A single-sentence paragraph of exactly ``length`` characters."""
    text = sentence(letter, length)
    assert len(text) == length
    return text


def ending_sentence(letter: str, length: int, ending: str) -> str:
    """A sentence of exactly ``length`` characters whose last word is ``ending``."""
    body = " ".join(letter * 9 for _ in range(length // 10 + 1))
    text = body[: length - len(ending) - 1].rstrip() + " " + ending
    assert len(text) == length
    return text


def strip_whitespace(text: str) -> str:
    return "".join(text.split())


def assert_chunk_invariants(cleaned: str, chunks: list[Chunk], config: ChunkingConfig) -> None:
    """Size limit, word safety, offsets, coverage and overlap bound of one page's chunks."""
    for chunk in chunks:
        assert cleaned[chunk.start : chunk.end] == chunk.text
        assert chunk.text == chunk.text.strip()
        assert chunk.start == 0 or cleaned[chunk.start - 1].isspace()
        assert chunk.end == len(cleaned) or cleaned[chunk.end].isspace()
        if len(chunk.text) > config.max_chars:
            assert len(chunk.text.split()) == 1
    if not chunks:
        assert cleaned.strip() == ""
        return
    assert chunks[0].start == len(cleaned) - len(cleaned.lstrip())
    assert chunks[-1].end == len(cleaned.rstrip())
    for previous, current in itertools.pairwise(chunks):
        assert current.start > previous.start
        assert current.end > previous.end
        if current.start >= previous.end:
            assert cleaned[previous.end : current.start].strip() == ""
        else:
            assert previous.end - current.start <= config.max_overlap_chars


class TestCleanPageText:
    def test_word_split_across_lines_is_joined(self) -> None:
        assert clean_page_text("the docu-\nment was signed") == "the document was signed"

    def test_hyphenated_name_is_preserved(self) -> None:
        assert clean_page_text("Karl-\nHeinz Müller") == "Karl- Heinz Müller"

    def test_lines_of_one_paragraph(self) -> None:
        cleaned = clean_page_text("first line\nsecond line\n\nnext paragraph")
        assert cleaned == "first line second line\n\nnext paragraph"
        assert cleaned.split("\n\n") == ["first line second line", "next paragraph"]

    def test_many_blank_lines_and_whitespace_only_lines_separate_paragraphs(self) -> None:
        assert clean_page_text("one\n\n\n  \t \ntwo") == "one\n\ntwo"

    def test_windows_line_endings_are_normalised(self) -> None:
        assert clean_page_text("a docu-\r\nment\r\n\r\nnext\rline") == "a document\n\nnext line"

    def test_runs_of_spaces_and_tabs_collapse_and_text_is_stripped(self) -> None:
        assert clean_page_text("  a \t\t b   c \n\n") == "a b c"

    def test_other_hyphens_are_kept(self) -> None:
        assert clean_page_text("well-known and\nco-op - x") == "well-known and co-op - x"

    def test_hyphen_before_blank_line_is_kept(self) -> None:
        assert clean_page_text("end-\n\nstart") == "end-\n\nstart"

    def test_blank_text_cleans_to_empty(self) -> None:
        assert clean_page_text(" \n\t\n ") == ""

    def test_soft_hyphen_before_lowercase_is_dropped_with_the_line_break(self) -> None:
        assert clean_page_text("the docu\N{SOFT HYPHEN}\nment") == "the document"

    def test_unicode_hyphen_is_treated_like_a_hyphen(self) -> None:
        assert clean_page_text("the docu\N{HYPHEN}\nment") == "the document"
        assert clean_page_text("Karl\N{HYPHEN}\nHeinz") == "Karl\N{HYPHEN} Heinz"

    def test_soft_hyphen_before_uppercase_is_not_joined(self) -> None:
        assert clean_page_text("Karl\N{SOFT HYPHEN}\nHeinz") == "Karl\N{SOFT HYPHEN} Heinz"

    def test_hyphen_after_a_non_letter_is_kept(self) -> None:
        assert clean_page_text("2020-\nment and a -\nb") == "2020- ment and a - b"

    def test_consecutive_split_words_are_all_joined(self) -> None:
        assert clean_page_text("a docu-\nment-\nation here") == "a documentation here"


class TestChunkingConfig:
    def test_defaults(self) -> None:
        config = ChunkingConfig()
        assert config.max_chars == 800
        assert config.overlap_ratio == 0.12
        assert config.max_overlap_chars == 96

    @pytest.mark.parametrize("ratio", [0.6, 0.5, -0.01, float("nan")])
    def test_invalid_overlap_is_rejected(self, ratio: float) -> None:
        with pytest.raises(ConfigurationError) as error:
            ChunkingConfig(overlap_ratio=ratio)
        assert error.value.code == "invalid_configuration"

    @pytest.mark.parametrize("max_chars", [0, -5])
    def test_invalid_max_chars_is_rejected(self, max_chars: int) -> None:
        with pytest.raises(ConfigurationError):
            ChunkingConfig(max_chars=max_chars)

    @pytest.mark.parametrize("ratio", [0.0, 0.12, 0.499])
    def test_valid_overlap_is_accepted(self, ratio: float) -> None:
        assert ChunkingConfig(overlap_ratio=ratio).overlap_ratio == ratio


class TestMaximumSize:
    def test_short_page_is_one_chunk(self) -> None:
        text = clean_page_text(paragraph("a", 250) + "\n\n" + paragraph("b", 248))
        assert len(text) == 500
        chunks = chunk_page(text, 0, ChunkingConfig())
        assert [c.text for c in chunks] == [text]
        assert (chunks[0].start, chunks[0].end) == (0, 500)

    def test_page_exactly_at_the_limit_is_one_chunk(self) -> None:
        text = paragraph("a", 800)
        assert len(chunk_page(text, 0, ChunkingConfig())) == 1

    def test_long_page_respects_limit_and_covers_text(self) -> None:
        text = clean_page_text(" ".join(paragraph(chr(97 + i), 300) for i in range(10)))
        assert len(text) >= 3000
        config = no_overlap()
        chunks = chunk_page(text, 0, config)
        assert len(chunks) > 1
        assert all(len(c.text) <= 800 for c in chunks)
        assert strip_whitespace("".join(c.text for c in chunks)) == strip_whitespace(text)
        assert_chunk_invariants(text, chunks, config)

    @pytest.mark.parametrize("raw", ["", "   ", "\n\n \t\n"])
    def test_blank_page_has_no_chunk(self, raw: str) -> None:
        assert chunk_page(clean_page_text(raw), 0, ChunkingConfig()) == []
        assert chunk_document([raw]) == []


class TestWordsAreNeverSplit:
    def test_boundary_inside_a_word_moves_the_whole_word(self) -> None:
        head = "a" * 9 + " "
        text = head * 79 + "x" * 20 + " tail"  # the 800th character lies inside the long word
        assert text[799] == "x"
        chunks = chunk_page(text, 0, no_overlap())
        assert chunks[0].text.endswith("a" * 9)
        assert chunks[1].text == "x" * 20 + " tail"

    def test_boundary_inside_a_word_keeps_a_word_ending_exactly_at_the_limit(self) -> None:
        text = ("a" * 9 + " ") * 79 + "x" * 10 + " tail"
        assert text[799] == "x"
        chunks = chunk_page(text, 0, no_overlap())
        assert chunks[0].text.endswith("x" * 10)
        assert len(chunks[0].text) == 800

    def test_oversized_token_is_emitted_whole_in_its_own_chunk(self) -> None:
        token = "z" * 1200
        text = f"before text {token} after text"
        for config in (no_overlap(), ChunkingConfig()):
            chunks = chunk_page(text, 0, config)
            assert [c.text for c in chunks if token in c.text] == [token]
            assert_chunk_invariants(text, chunks, config)

    def test_oversized_token_alone(self) -> None:
        token = "z" * 1200
        chunks = chunk_page(token, 0, ChunkingConfig())
        assert [c.text for c in chunks] == [token]


class TestPagesAreIndependent:
    def test_two_short_pages(self) -> None:
        pages = ["a" * 100, "b" * 100]
        chunks = chunk_document(pages)
        assert [(c.page, c.text) for c in chunks] == [(0, "a" * 100), (1, "b" * 100)]

    def test_blank_pages_are_skipped_but_keep_page_numbers(self) -> None:
        chunks = chunk_document(["first", "  \n ", "third"])
        assert [(c.page, c.sequence) for c in chunks] == [(0, 0), (2, 1)]

    def test_no_pages(self) -> None:
        assert chunk_document([]) == []


class TestNaturalBoundaries:
    def test_three_paragraphs_first_chunk_holds_two(self) -> None:
        paragraphs = [paragraph(letter, 300) for letter in "abc"]
        text = clean_page_text("\n\n".join(paragraphs))
        for config in (no_overlap(), ChunkingConfig()):
            chunks = chunk_page(text, 0, config)
            assert chunks[0].text == paragraphs[0] + "\n\n" + paragraphs[1]
            assert chunks[1].text.endswith(paragraphs[2])
            assert all(c.text.endswith(".") for c in chunks)

    def test_long_paragraph_ends_chunks_at_sentence_boundaries(self) -> None:
        sentences = [sentence(letter, 230) for letter in "abcdefgh"]
        text = " ".join(sentences)
        chunks = chunk_page(text, 0, no_overlap())
        assert len(chunks) > 1
        assert all(c.text.endswith(".") for c in chunks)
        assert chunks[0].text == " ".join(sentences[:3])

    def test_sentence_boundary_is_preferred_over_a_later_word_boundary(self) -> None:
        text = sentence("a", 400) + " " + "b" * 9 + " " + "c" * 9 + " " + "d" * 500
        chunks = chunk_page(text, 0, no_overlap())
        assert chunks[0].text == sentence("a", 400)
        assert chunks[1].text == "b" * 9 + " " + "c" * 9 + " " + "d" * 500

    @pytest.mark.parametrize(
        "abbreviated",
        [
            "Dr. Smith",
            "dr. smith",
            "Sr. López",
            "Mr. Brown",
            "Mrs. Brown",
            "Sra. García",
            "Dra. Ruiz",
            "Prof. Lee",
            "Ing. Soto",
            "Lic. Vega",
            "Dña. Ana",
            "J. Pérez",
            "F. Last",
            "U.S. Embassy",
            "e.g. this",
            "S.L. Madrid",
            "etc. and",
            "(Dr.) Smith",
        ],
    )
    def test_titles_initials_and_acronyms_do_not_end_a_sentence(self, abbreviated: str) -> None:
        # Were the abbreviation a boundary, it would be the furthest one inside the limit.
        text = f"{'a' * 100}. {'b' * 100} {abbreviated} {'c' * 700}"
        chunks = chunk_page(text, 0, no_overlap())
        assert chunks[0].text == "a" * 100 + "."
        assert not any(c.text.endswith(abbreviated.split(maxsplit=1)[0]) for c in chunks)

    @pytest.mark.parametrize(
        "word", ["it.", "me.", "so.", "sí.", "ok.", "no.", "He.", "yo.", "Fin."]
    )
    def test_short_ordinary_words_end_a_sentence(self, word: str) -> None:
        text = f"{'a' * 100}. {'b' * 100} {word} {'c' * 700}"
        chunks = chunk_page(text, 0, no_overlap())
        assert chunks[0].text == f"{'a' * 100}. {'b' * 100} {word}"

    def test_sentences_ending_in_short_words_split_at_sentence_ends(self) -> None:
        sentences = [ending_sentence(letter, 213, "it.") for letter in "abcdef"]
        for config in (no_overlap(400), ChunkingConfig(max_chars=400)):
            chunks = chunk_page(" ".join(sentences), 0, config)
            assert len(chunks) == 6
            assert all(c.text.endswith("it.") for c in chunks)
            assert_chunk_invariants(" ".join(sentences), chunks, config)
        assert [c.text for c in chunk_page(" ".join(sentences), 0, no_overlap(400))] == sentences

    def test_closing_quote_after_an_ordinary_word_still_ends_a_sentence(self) -> None:
        text = f'{"a" * 100}. {"b" * 100} "it." {"c" * 700}'
        assert chunk_page(text, 0, no_overlap())[0].text.endswith('it."')

    def test_question_and_exclamation_marks_end_sentences(self) -> None:
        text = "a" * 400 + "? " + "b" * 300 + "! " + "c" * 300
        chunks = chunk_page(text, 0, no_overlap())
        assert chunks[0].text == "a" * 400 + "? " + "b" * 300 + "!"
        assert chunks[1].text == "c" * 300


class TestOverlap:
    def long_text(self) -> str:
        return clean_page_text("\n\n".join(paragraph(chr(97 + i % 26), 700) for i in range(8)))

    def test_default_overlap_is_at_most_96_characters(self) -> None:
        words = [f"w{i:03d}" for i in range(600)]
        text = " ".join(words)
        chunks = chunk_page(text, 0, ChunkingConfig())
        assert len(chunks) > 3
        repeated = [p.end - c.start for p, c in itertools.pairwise(chunks)]
        assert all(0 < size <= 96 for size in repeated)
        assert max(repeated) > 80  # the overlap is actually used, not just bounded
        assert_chunk_invariants(text, chunks, ChunkingConfig())

    def test_overlap_text_is_the_end_of_the_previous_chunk(self) -> None:
        text = " ".join(f"w{i:03d}" for i in range(600))
        chunks = chunk_page(text, 0, ChunkingConfig())
        for previous, current in itertools.pairwise(chunks):
            repeated = text[current.start : previous.end]
            assert previous.text.endswith(repeated)
            assert current.text.startswith(repeated)

    def test_no_overlap_reproduces_the_cleaned_text(self) -> None:
        text = self.long_text()
        chunks = chunk_page(text, 0, no_overlap())
        assert len(chunks) > 1
        assert all(p.end <= c.start for p, c in itertools.pairwise(chunks))
        assert strip_whitespace("".join(c.text for c in chunks)) == strip_whitespace(text)

    def test_overlap_never_pushes_a_chunk_over_the_limit(self) -> None:
        text = self.long_text()
        config = ChunkingConfig(max_chars=800, overlap_ratio=0.49)
        chunks = chunk_page(text, 0, config)
        assert all(len(c.text) <= 800 for c in chunks)
        assert_chunk_invariants(text, chunks, config)

    def test_overlap_is_dropped_when_it_cannot_fit(self) -> None:
        # The word after the first chunk fills the limit on its own, so nothing can be repeated.
        text = "aaa bbb " + "c" * 20
        config = ChunkingConfig(max_chars=20, overlap_ratio=0.4)
        chunks = chunk_page(text, 0, config)
        assert [c.text for c in chunks] == ["aaa bbb", "c" * 20]

    def test_overlap_is_a_proper_suffix_of_a_short_previous_chunk(self) -> None:
        text = "one two. " + "x" * 35
        config = ChunkingConfig(max_chars=40, overlap_ratio=0.45)
        chunks = chunk_page(text, 0, config)
        assert [c.text for c in chunks] == ["one two.", "two. " + "x" * 35]
        assert_chunk_invariants(text, chunks, config)


class TestProvenance:
    def test_offsets_equal_slices_of_the_cleaned_page(self) -> None:
        raw_pages = [f"{paragraph('a', 500)}\n\n{paragraph('b', 500)}", "page two\n" * 80]
        for page, raw in enumerate(raw_pages):
            cleaned = clean_page_text(raw)
            for chunk in chunk_document(raw_pages, ChunkingConfig(300, 0.12)):
                if chunk.page == page:
                    assert cleaned[chunk.start : chunk.end] == chunk.text

    def test_two_pages_with_five_chunks_are_numbered_zero_to_four(self) -> None:
        pages = [
            "\n\n".join(paragraph(letter, 500) for letter in "abc"),
            "\n\n".join(paragraph(letter, 500) for letter in "de"),
        ]
        chunks = chunk_document(pages, no_overlap())
        assert len(chunks) == 5
        assert [c.sequence for c in chunks] == [0, 1, 2, 3, 4]
        assert [c.page for c in chunks] == [0, 0, 0, 1, 1]

    def test_first_sequence_offsets_numbering(self) -> None:
        chunks = chunk_page("a b c", 3, ChunkingConfig(), first_sequence=7)
        assert [(c.page, c.sequence) for c in chunks] == [(3, 7)]


class TestDeterminism:
    def test_same_document_gives_equal_chunks(self) -> None:
        pages = _fixture_pages()
        config = ChunkingConfig(max_chars=300, overlap_ratio=0.12)
        assert chunk_document(pages, config) == chunk_document(pages, config)


def _fixture_pages() -> list[str]:
    raw = FIXTURE.read_text(encoding="utf-8")
    return [part.strip("\n") for part in raw.split(PAGE_MARKER + "\n")[1:]]


class TestOcrStyleFixture:
    @pytest.mark.parametrize("max_chars", [200, 300, 800])
    @pytest.mark.parametrize("overlap_ratio", [0.0, 0.12, 0.3])
    def test_chunks_respect_limit_words_and_offsets(
        self, max_chars: int, overlap_ratio: float
    ) -> None:
        pages = _fixture_pages()
        assert len(pages) == 3
        config = ChunkingConfig(max_chars, overlap_ratio)
        chunks = chunk_document(pages, config)
        assert [c.sequence for c in chunks] == list(range(len(chunks)))
        for page, raw in enumerate(pages):
            cleaned = clean_page_text(raw)
            page_chunks = [c for c in chunks if c.page == page]
            assert page_chunks
            assert_chunk_invariants(cleaned, page_chunks, config)

    def test_fixture_cleanup(self) -> None:
        first, second, _ = (clean_page_text(page) for page in _fixture_pages())
        assert "representación" in first
        assert "análisis" in first
        assert "personalmente" in first
        assert "bancaria" in first
        assert "\n\n\n" not in first
        assert "Karl- Heinz" in second
        assert "Supercalifragilisticoespialidosamente_largo" in second

    def test_long_fixture_token_stays_whole(self) -> None:
        second = clean_page_text(_fixture_pages()[1])
        token = second.split()[-1]
        assert len(token) > 90
        chunks = chunk_page(second, 1, ChunkingConfig(max_chars=60, overlap_ratio=0.1))
        assert [c.text for c in chunks if token in c.text] == [token]


class TestPerformance:
    """Generous wall-clock bounds that quadratic behaviour exceeds by an order of magnitude."""

    def test_cleaning_a_huge_hyphenated_paragraph_is_linear(self) -> None:
        repetitions = 200_000
        raw = "docu-\nment " * repetitions  # about 2 MB in one paragraph of 200 000 lines
        started = time.perf_counter()
        cleaned = clean_page_text(raw)
        elapsed = time.perf_counter() - started
        assert cleaned == ("document " * repetitions).rstrip()
        assert elapsed < 3.0

    def test_cleaning_time_grows_roughly_linearly(self) -> None:
        def clean_time(repetitions: int) -> float:
            raw = "docu-\nment " * repetitions
            started = time.perf_counter()
            clean_page_text(raw)
            return time.perf_counter() - started

        small = min(clean_time(25_000) for _ in range(3))
        large = min(clean_time(100_000) for _ in range(3))
        assert large < max(small, 0.02) * 12  # 4x the input; quadratic would be about 16x

    def test_chunking_a_megabyte_of_ordinary_text(self) -> None:
        page = "The quick brown fox jumps over the lazy dog. Dr. Smith said so.\n" * 16_000
        assert len(page) > 1_000_000
        started = time.perf_counter()
        chunks = chunk_document([page])
        elapsed = time.perf_counter() - started
        assert len(chunks) > 1000
        assert elapsed < 5.0


class TestRandomised:
    ALPHABET: ClassVar[str] = "abcdefghijklmnopqrstuvwxyzáéíóúñ"
    SEPARATORS: ClassVar[list[str]] = [
        " ",
        " ",
        " ",
        "  ",
        "\n",
        "\t",
        "\n\n",
        " \n \n ",
        ". ",
        "? ",
        "! ",
    ]

    def random_text(self, rng: random.Random) -> str:
        pieces: list[str] = []
        for _ in range(rng.randint(0, 120)):
            length = rng.choice([1, 2, 3, 5, 8, 12, 25, 70])
            word = "".join(rng.choice(self.ALPHABET) for _ in range(length))
            if rng.random() < 0.15:
                word = word.capitalize() + "-"
            pieces.append(word + rng.choice(self.SEPARATORS))
        return "".join(pieces)

    @pytest.mark.parametrize(
        ("max_chars", "overlap_ratio"),
        [(20, 0.0), (20, 0.12), (20, 0.45), (50, 0.12), (100, 0.3), (800, 0.12), (800, 0.499)],
    )
    def test_invariants_hold_for_generated_texts(
        self, max_chars: int, overlap_ratio: float
    ) -> None:
        rng = random.Random(0)
        config = ChunkingConfig(max_chars, overlap_ratio)
        for _ in range(150):
            cleaned = clean_page_text(self.random_text(rng))
            chunks = chunk_page(cleaned, 0, config)
            assert_chunk_invariants(cleaned, chunks, config)
            assert chunk_page(cleaned, 0, config) == chunks
            if overlap_ratio == 0.0:
                assert strip_whitespace("".join(c.text for c in chunks)) == strip_whitespace(
                    cleaned
                )
