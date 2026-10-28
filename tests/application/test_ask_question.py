"""Use case ``AskQuestion`` with in-memory embedder, vector store and LLM fakes."""

from collections.abc import Sequence

import pytest

from app.application.ask_question import AskQuestion
from app.domain.errors import ConfigurationError, DependencyTimeout, DependencyUnavailable
from app.domain.prompting import REFUSAL, REFUSAL_SENTINEL, SYSTEM_PROMPT
from app.domain.types import Chunk
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm import FakeLLMClient
from tests.fakes.vector_store import FakeVectorStore

CONTRACT = "The contract was signed by Maria Gonzalez on 3 March 2024."
# The fake embedder is a bag of words, so real-model thresholds are too strict for it; the
# questions below score about 0.45 against their page and 0.0 against unrelated text.
LOW_THRESHOLD = 0.3


def add_document(
    store: FakeVectorStore,
    embedder: FakeEmbedder,
    document_id: str,
    pages: Sequence[Sequence[str]],
) -> list[Chunk]:
    """Store pre-chunked pages: each page is a list of chunk texts joined by spaces."""
    store.ensure_collection(embedder.identity)
    chunks: list[Chunk] = []
    page_texts: dict[int, str] = {}
    for page, texts in enumerate(pages):
        page_text = " ".join(texts)
        page_texts[page] = page_text
        offset = 0
        for text in texts:
            chunks.append(Chunk(text, page, len(chunks), offset, offset + len(text)))
            offset += len(text) + 1
    vectors = embedder.embed_passages([chunk.text for chunk in chunks])
    store.upsert(
        document_id, chunks, vectors, page_texts=page_texts, content_hash="h" + document_id
    )
    return chunks


def build(
    llm: FakeLLMClient | None = None, **config: int | float
) -> tuple[AskQuestion, FakeEmbedder, FakeVectorStore, FakeLLMClient]:
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    llm = llm or FakeLLMClient(default="Maria Gonzalez signed it.")
    return AskQuestion(embedder, store, llm, **config), embedder, store, llm  # type: ignore[arg-type]


def user_message(llm: FakeLLMClient) -> str:
    assert len(llm.calls) == 1
    return llm.calls[0][1].content


def test_answerable_question_sends_the_retrieved_text_and_returns_the_answer() -> None:
    ask, embedder, store, llm = build(threshold=LOW_THRESHOLD)
    add_document(store, embedder, "doc-a", [[CONTRACT]])

    answer = ask("Who signed the contract?")

    assert answer.text == "Maria Gonzalez signed it."
    assert answer.sources == (CONTRACT,)
    assert CONTRACT in user_message(llm)
    assert "Who signed the contract?" in user_message(llm)


def test_prompt_forbids_outside_knowledge() -> None:
    ask, embedder, store, llm = build(threshold=LOW_THRESHOLD)
    add_document(store, embedder, "doc-a", [[CONTRACT]])

    ask("Who signed the contract?")

    system = llm.calls[0][0]
    assert system.role == "system"
    assert system.content == SYSTEM_PROMPT
    assert "only from the numbered context" in system.content
    assert REFUSAL_SENTINEL in system.content


def test_whole_parent_page_is_sent_when_the_answer_is_in_another_chunk() -> None:
    ask, embedder, store, llm = build(threshold=LOW_THRESHOLD, max_chunks=1)
    first_half = "Clause seven covers the termination of the contract."
    second_half = "Either party may end it with ninety days notice."
    add_document(store, embedder, "doc-a", [[first_half, second_half]])

    answer = ask("termination of the contract")

    assert answer.sources[0] == first_half
    assert f"{first_half} {second_half}" in user_message(llm)
    assert "[1] document doc-a, page 1" in user_message(llm)


def test_context_respects_the_character_budget_in_rank_order() -> None:
    ask, embedder, store, llm = build(threshold=0.0, max_context_chars=120)
    page_a = "invoice " * 10  # 80 characters, the best match
    page_b = "invoice total " * 8  # does not fit in the 40 characters left
    add_document(store, embedder, "doc-a", [[page_a.strip()]])
    add_document(store, embedder, "doc-b", [[page_b.strip()]])

    ask("invoice")

    message = user_message(llm)
    assert page_a.strip() in message
    assert "doc-b" not in message
    context = message.removeprefix("Context:\n\n").removesuffix("\n\nQuestion: invoice")
    assert context.startswith("[1] document doc-a, page 1\n")
    assert len(context) <= 120


def test_budget_too_small_for_any_page_refuses_without_calling_the_model() -> None:
    ask, embedder, store, llm = build(threshold=LOW_THRESHOLD, max_context_chars=20)
    add_document(store, embedder, "doc-a", [[CONTRACT]])

    answer = ask("Who signed the contract?")

    assert (answer.text, answer.sources) == (REFUSAL, ())
    assert llm.calls == []


def test_below_threshold_refuses_without_sources_or_calling_the_model() -> None:
    ask, embedder, store, llm = build()
    add_document(store, embedder, "doc-a", [[CONTRACT]])

    answer = ask("What is the weather in Lisbon tomorrow?")

    assert answer.text == REFUSAL
    assert answer.sources == ()
    assert llm.calls == []


def test_best_hit_just_below_the_threshold_is_refused_and_at_it_is_answered() -> None:
    ask_strict, embedder, store, llm = build(threshold=1.0)
    add_document(store, embedder, "doc-a", [[CONTRACT]])

    assert ask_strict(CONTRACT).text != REFUSAL  # identical text scores exactly 1.0
    assert ask_strict("Who signed the contract?").text == REFUSAL
    assert len(llm.calls) == 1


def test_empty_store_refuses_without_calling_the_model() -> None:
    ask, _, _, llm = build(threshold=0.0)

    answer = ask("Who signed the contract?")

    assert answer.text == REFUSAL
    assert answer.sources == ()
    assert llm.calls == []


def test_sources_are_the_retrieved_texts_above_the_threshold_in_rank_order() -> None:
    ask, embedder, store, _ = build(threshold=0.2, top_k=5)
    add_document(
        store,
        embedder,
        "doc-a",
        [["contract signed by Maria", "contract renewal terms"], ["weather report"]],
    )
    add_document(store, embedder, "doc-b", [["contract signed by Maria Gonzalez in Madrid"]])
    question = "contract signed by Maria"

    answer = ask(question)

    hits = store.search(embedder.embed_query(question), 5)
    expected = tuple(hit.text for hit in hits if hit.score >= 0.2)
    assert len(expected) == 3  # all within the default three chunks, so all are shown
    assert answer.sources == expected
    assert answer.sources[0] == "contract signed by Maria"
    assert "weather report" not in answer.sources
    stored_texts = {hit.text for hit in store.search(embedder.embed_query("x"), 100)}
    assert set(answer.sources) <= stored_texts


@pytest.mark.parametrize(
    "reply",
    [
        REFUSAL_SENTINEL,
        "  not_in_documents.\n",
        '"NOT_IN_DOCUMENTS"',
        "NOT_IN_DOCUMENTS - the context is about something else.",
        f"  {REFUSAL}\n",
    ],
)
def test_model_replying_with_the_refusal_gives_the_fixed_refusal_and_no_sources(
    reply: str,
) -> None:
    llm = FakeLLMClient(answers=[reply])
    ask, embedder, store, _ = build(llm, threshold=LOW_THRESHOLD)
    add_document(store, embedder, "doc-a", [[CONTRACT]])

    answer = ask("Who signed the contract?")

    assert answer.text == REFUSAL
    assert answer.sources == ()


def test_answer_mentioning_documents_is_not_a_refusal() -> None:
    reply = "The documents say Maria Gonzalez signed it."
    ask, embedder, store, _ = build(FakeLLMClient(answers=[reply]), threshold=LOW_THRESHOLD)
    add_document(store, embedder, "doc-a", [[CONTRACT]])

    answer = ask("Who signed the contract?")

    assert answer.text == reply
    assert answer.sources == (CONTRACT,)


def test_retrieved_chunks_left_out_of_the_context_are_not_sources() -> None:
    ask, embedder, store, llm = build(threshold=0.0, max_chunks=1)
    add_document(store, embedder, "doc-a", [["invoice total due"]])
    add_document(store, embedder, "doc-b", [["invoice number"]])

    answer = ask("invoice total due")

    assert answer.sources == ("invoice total due",)
    assert "doc-b" not in user_message(llm)


def test_chunks_in_the_cut_part_of_a_page_are_not_sources() -> None:
    first = "termination of the contract " * 4
    late = "Either party may end it with ninety days notice."
    ask, embedder, store, llm = build(threshold=0.0, max_context_chars=150)
    add_document(store, embedder, "doc-a", [[first.strip(), late]])

    answer = ask("termination of the contract notice")

    assert late not in user_message(llm)
    assert answer.sources == (first.strip(),)


def test_blank_model_reply_is_treated_as_a_refusal() -> None:
    llm = FakeLLMClient(answers=["   "])
    ask, embedder, store, _ = build(llm, threshold=LOW_THRESHOLD)
    add_document(store, embedder, "doc-a", [[CONTRACT]])

    answer = ask("Who signed the contract?")

    assert (answer.text, answer.sources) == (REFUSAL, ())


def test_model_reply_is_stripped() -> None:
    llm = FakeLLMClient(answers=["  Maria Gonzalez.\n"])
    ask, embedder, store, _ = build(llm, threshold=LOW_THRESHOLD)
    add_document(store, embedder, "doc-a", [[CONTRACT]])

    assert ask("Who signed the contract?").text == "Maria Gonzalez."


@pytest.mark.parametrize(
    "error",
    [
        DependencyUnavailable("language model unavailable", code="llm_unavailable"),
        DependencyTimeout("language model timed out", code="llm_timeout"),
    ],
)
def test_model_failures_propagate(error: DependencyUnavailable | DependencyTimeout) -> None:
    ask, embedder, store, _ = build(FakeLLMClient(answers=[error]), threshold=LOW_THRESHOLD)
    add_document(store, embedder, "doc-a", [[CONTRACT]])

    with pytest.raises(type(error)) as raised:
        ask("Who signed the contract?")

    assert raised.value.code == error.code


def test_store_failure_propagates() -> None:
    ask, embedder, store, _ = build()
    add_document(store, embedder, "doc-a", [[CONTRACT]])
    store.fail_next(1, DependencyUnavailable("store down", code="vector_store_unavailable"))

    with pytest.raises(DependencyUnavailable):
        ask("Who signed the contract?")


@pytest.mark.parametrize("question", ["", "   \n"])
def test_blank_question_is_rejected_before_any_port_call(question: str) -> None:
    ask, embedder, _, llm = build()

    with pytest.raises(ValueError, match="blank"):
        ask(question)

    assert embedder.query_calls == []
    assert llm.calls == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("threshold", -0.1),
        ("threshold", 1.1),
        ("threshold", float("nan")),
        ("threshold", True),
        ("top_k", 0),
        ("top_k", 1.5),
        ("max_chunks", 0),
        ("max_context_chars", 0),
        ("max_context_chars", False),
    ],
)
def test_invalid_configuration_is_rejected(name: str, value: float) -> None:
    with pytest.raises(ConfigurationError, match=name):
        AskQuestion(FakeEmbedder(), FakeVectorStore(), FakeLLMClient(), **{name: value})  # type: ignore[arg-type]
