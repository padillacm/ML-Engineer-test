import random

import pytest

from app.domain.errors import ConfigurationError
from app.domain.prompting import (
    MIN_PARTIAL_CHARS,
    NO_CONTEXT,
    REFUSAL,
    REFUSAL_SENTINEL,
    SYSTEM_PROMPT,
    ContextPage,
    build_messages,
    is_refusal,
    render_context,
    select_context,
    shown_sources,
)
from app.domain.types import SearchHit


def hit(
    document_id: str,
    page: int,
    sequence: int,
    page_text: str,
    score: float = 0.9,
    text: str | None = None,
) -> SearchHit:
    chunk_text = f"chunk {sequence}" if text is None else text
    return SearchHit(document_id, page, sequence, chunk_text, page_text, score)


def words(letter: str, length: int) -> str:
    """About ``length`` characters of 9-letter words separated by spaces."""
    return " ".join(letter * 9 for _ in range(length // 10 + 1))[:length].strip()


def label(document_id: str, page: int) -> int:
    """Length of the label of a single-digit-numbered page with empty text."""
    return len(render_context([ContextPage(document_id, page, "")]))


# select_context


def test_empty_hits_give_no_pages() -> None:
    assert select_context([]) == []


def test_pages_are_whole_and_in_rank_order() -> None:
    hits = [hit("b", 4, 7, "page b4"), hit("a", 0, 0, "page a0"), hit("a", 2, 3, "page a2")]

    assert select_context(hits) == [
        ContextPage("b", 4, "page b4"),
        ContextPage("a", 0, "page a0"),
        ContextPage("a", 2, "page a2"),
    ]


def test_chunks_of_one_page_give_the_page_once() -> None:
    hits = [hit("a", 0, 0, "page a0"), hit("a", 0, 1, "page a0"), hit("a", 1, 2, "page a1")]

    assert select_context(hits) == [ContextPage("a", 0, "page a0"), ContextPage("a", 1, "page a1")]


def test_same_page_number_in_two_documents_is_kept_twice() -> None:
    hits = [hit("a", 0, 0, "page of a"), hit("b", 0, 0, "page of b")]

    assert [(page.document_id, page.text) for page in select_context(hits)] == [
        ("a", "page of a"),
        ("b", "page of b"),
    ]


def test_only_the_first_max_chunks_distinct_chunks_count() -> None:
    hits = [
        hit("a", 0, 0, "p0"),
        hit("a", 0, 0, "p0"),  # the same chunk twice is one chunk
        hit("a", 0, 1, "p0"),
        hit("a", 1, 2, "p1"),
        hit("a", 2, 3, "p2"),
    ]

    pages = select_context(hits, max_chunks=3)

    assert [page.page for page in pages] == [0, 1]


def test_budget_counts_labels_and_separators() -> None:
    first, second = "x" * 100, "y" * 100
    hits = [hit("a", 0, 0, first), hit("a", 1, 1, second)]
    both = len(render_context([ContextPage("a", 0, first), ContextPage("a", 1, second)]))

    assert both == label("a", 0) + label("a", 1) + 200 + len("\n\n")
    assert len(select_context(hits, max_chars=both)) == 2
    assert select_context(hits, max_chars=both - 1) == [ContextPage("a", 0, first)]


def test_page_that_does_not_fit_is_cut_at_a_word_boundary() -> None:
    first = "x" * 700
    second = "alpha beta gamma delta " * 40
    hits = [hit("a", 0, 0, first), hit("a", 1, 1, second)]

    pages = select_context(hits, max_chars=1000)

    assert pages[0].text == first
    cut = pages[1].text
    assert len(render_context(pages)) <= 1000
    assert second.startswith(cut)
    assert second[len(cut)] == " "
    assert not cut.endswith(" ")
    room = 1000 - label("a", 0) - 700 - 2 - label("a", 1)
    assert len(cut) > room - len("delta ")


def test_selection_stops_after_a_cut_page() -> None:
    hits = [hit("a", 0, 0, words("a", 900)), hit("a", 1, 1, "short page")]

    pages = select_context(hits, max_chars=500)

    assert len(pages) == 1
    assert len(render_context(pages)) <= 500


def test_tiny_remainder_is_skipped() -> None:
    first = "x" * 500
    used = label("a", 0) + len(first) + 2 + label("a", 1)
    hits = [hit("a", 0, 0, first), hit("a", 1, 1, words("b", 900))]

    assert select_context(hits, max_chars=used + MIN_PARTIAL_CHARS - 1) == [
        ContextPage("a", 0, first)
    ]
    assert len(select_context(hits, max_chars=used + MIN_PARTIAL_CHARS)) == 2


def test_single_word_longer_than_the_room_is_cut_hard() -> None:
    pages = select_context([hit("a", 0, 0, "y" * 500)], max_chars=label("a", 0) + 300)

    assert pages == [ContextPage("a", 0, "y" * 300)]


def test_small_budget_still_gives_the_first_page_when_its_label_fits() -> None:
    pages = select_context([hit("a", 0, 0, words("c", 400))], max_chars=label("a", 0) + 50)

    assert len(pages) == 1
    assert 0 < len(pages[0].text) <= 50


def test_budget_below_one_label_gives_no_pages() -> None:
    assert select_context([hit("a", 0, 0, "text")], max_chars=label("a", 0)) == []


def test_blank_pages_are_skipped() -> None:
    hits = [hit("a", 0, 0, "   \n "), hit("a", 1, 1, " page one ")]

    assert select_context(hits) == [ContextPage("a", 1, "page one")]


def test_rendered_context_never_exceeds_the_budget_on_random_inputs() -> None:
    rng = random.Random(20261027)
    vocabulary = ["a", "bb", "contract", "signed", "Gonzalez", "x" * 40, "\n", "2024.", "[2]"]
    documents = ["a", "doc-b", "0f3c2a5e-1111-4222-8333-944445555666"]
    for _ in range(500):
        hits = [
            hit(
                rng.choice(documents),
                rng.randrange(12),
                sequence,
                " ".join(rng.choice(vocabulary) for _ in range(rng.randrange(0, 400))),
            )
            for sequence in range(rng.randrange(0, 8))
        ]
        max_chars = rng.randrange(1, 4000)
        max_chunks = rng.randrange(1, 6)

        pages = select_context(hits, max_chunks=max_chunks, max_chars=max_chars)

        assert len(render_context(pages)) <= max_chars
        keys = [(page.document_id, page.page) for page in pages]
        assert len(keys) == len(set(keys))
        assert len(pages) <= max_chunks
        expected_order = [(h.document_id, h.page) for h in hits]
        positions = [expected_order.index(key) for key in keys]
        assert positions == sorted(positions)
        for page in pages:
            source = next(
                h.page_text.strip()
                for h in hits
                if (h.document_id, h.page) == (page.document_id, page.page)
            )
            assert source.startswith(page.text)


@pytest.mark.parametrize(("max_chunks", "max_chars"), [(0, 6000), (-1, 6000), (3, 0), (3, -5)])
def test_invalid_limits_are_configuration_errors(max_chunks: int, max_chars: int) -> None:
    with pytest.raises(ConfigurationError):
        select_context([hit("a", 0, 0, "text")], max_chunks=max_chunks, max_chars=max_chars)


# ContextPage


def test_context_page_validates_its_fields() -> None:
    with pytest.raises(ValueError, match="document_id"):
        ContextPage("", 0, "t")
    with pytest.raises(ValueError, match="page index"):
        ContextPage("a", -1, "t")


# render_context and build_messages


def test_messages_are_a_system_and_a_user_message() -> None:
    messages = build_messages("Who signed?", [ContextPage("doc-1", 0, "Signed by Maria.")])

    assert [message.role for message in messages] == ["system", "user"]
    assert messages[0].content == SYSTEM_PROMPT


@pytest.mark.parametrize(
    "phrase",
    [
        "only from the numbered context",
        "language of the question",
        f"reply with exactly: {REFUSAL_SENTINEL}",
        "outside knowledge",
        "Do not invent document names, page numbers or quotes",
        "concise",
    ],
)
def test_system_message_states_the_grounding_rules(phrase: str) -> None:
    assert phrase in SYSTEM_PROMPT


def test_refusal_is_the_fixed_sentence() -> None:
    assert REFUSAL == "The provided documents do not contain this information."
    assert REFUSAL_SENTINEL == "NOT_IN_DOCUMENTS"


def test_user_message_numbers_and_labels_pages_one_based() -> None:
    pages = [ContextPage("doc-b", 4, "Text of b."), ContextPage("doc-a", 0, "Text of a.")]

    user = build_messages("Who signed the contract?", pages)[1].content

    assert user == (
        "Context:\n\n"
        "[1] document doc-b, page 5\nText of b.\n\n"
        "[2] document doc-a, page 1\nText of a.\n\n"
        "Question: Who signed the contract?"
    )
    assert user == f"Context:\n\n{render_context(pages)}\n\nQuestion: Who signed the contract?"


def test_no_pages_give_a_placeholder_context() -> None:
    assert build_messages("q?", [])[1].content == f"Context:\n\n{NO_CONTEXT}\n\nQuestion: q?"


def test_question_comes_after_the_context_and_is_stripped() -> None:
    pages = [ContextPage("d", 0, "Firmado por Maria.")]

    user = build_messages("  ¿Quién firmó?\n", pages)[1].content

    assert user.index("Firmado por Maria.") < user.index("Question: ¿Quién firmó?")
    assert user.endswith("Question: ¿Quién firmó?")


def test_blank_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="question"):
        build_messages("  ", [ContextPage("d", 0, "t")])


def test_selected_context_flows_into_the_messages() -> None:
    page_text = "The contract was signed. It was signed by Maria Gonzalez on 3 March 2024."
    hits = [SearchHit("doc", 0, 0, "The contract was signed.", page_text, 0.95)]

    user = build_messages("Who signed the contract?", select_context(hits))[1].content

    assert "Maria Gonzalez" in user


# shown_sources


def test_sources_are_the_shown_chunks_in_rank_order_once_each() -> None:
    hits = [
        hit("a", 1, 5, "beta page", text="beta page"),
        hit("a", 0, 0, "alpha one. alpha two.", text="alpha two."),
        hit("a", 1, 5, "beta page", text="beta page"),
        hit("a", 0, 1, "alpha one. alpha two.", text="alpha one."),
    ]

    sources = shown_sources(hits, select_context(hits), max_chunks=3)

    assert sources == ("beta page", "alpha two.", "alpha one.")


def test_chunks_beyond_max_chunks_are_not_sources() -> None:
    hits = [hit("a", 0, 0, "one two", text="one"), hit("a", 0, 1, "one two", text="two")]

    assert shown_sources(hits, select_context(hits, max_chunks=1), max_chunks=1) == ("one",)


def test_chunks_of_pages_left_out_are_not_sources() -> None:
    first = "x" * 300
    hits = [hit("a", 0, 0, first, text=first), hit("b", 0, 0, "other page", text="other page")]
    pages = select_context(hits, max_chars=label("a", 0) + 300 + 10)

    assert [page.document_id for page in pages] == ["a"]
    assert shown_sources(hits, pages) == (first,)


def test_chunks_in_the_cut_part_of_a_page_are_not_sources() -> None:
    page_text = words("a", 400) + " The answer is Halvorsen."
    hits = [
        hit("a", 0, 1, page_text, text="The answer is Halvorsen."),
        hit("a", 0, 0, page_text, text=words("a", 100)),
    ]
    pages = select_context(hits, max_chars=label("a", 0) + 250)

    assert "Halvorsen" not in pages[0].text
    assert shown_sources(hits, pages) == (words("a", 100),)


# is_refusal


@pytest.mark.parametrize(
    "reply",
    [
        "NOT_IN_DOCUMENTS",
        "  not_in_documents.\n",
        '"NOT_IN_DOCUMENTS"',
        "'not_in_documents'",
        "**NOT_IN_DOCUMENTS**",
        "`NOT_IN_DOCUMENTS`",
        "NOT_IN_DOCUMENTS - the context only covers the cafeteria survey.",
        "NOT_IN_DOCUMENTS\nThe documents do not say.",
        "",
        "   ",
        REFUSAL,
        f"  {REFUSAL.lower()}\n",
    ],
)
def test_refusal_replies_are_recognised(reply: str) -> None:
    assert is_refusal(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "The documents say Maria Gonzalez signed it.",
        "According to the documents, NOT_IN_DOCUMENTS is not a word they use.",
        "Robert Chen.",
        "Not in the documents from 2023, but the 2024 memo names Robert Chen.",
    ],
)
def test_normal_answers_are_not_refusals(reply: str) -> None:
    assert not is_refusal(reply)
