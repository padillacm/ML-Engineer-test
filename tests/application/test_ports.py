"""The fakes satisfy the ports, and a use case written against the ports runs on the fakes."""

import dataclasses
from typing import Any

import pytest

from app.application.ports import (
    Embedder,
    LLMClient,
    NameRecognizer,
    OcrEngine,
    VectorStore,
)
from app.domain.errors import DependencyTimeout, DependencyUnavailable, IncompatibleEmbedding
from app.domain.types import (
    Box,
    ChatMessage,
    Chunk,
    EmbeddingIdentity,
    OcrPage,
    PageWord,
    SearchHit,
    TextSpan,
)
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm import FakeLLMClient
from tests.fakes.names import FakeNameRecognizer
from tests.fakes.ocr import FakeOcrEngine
from tests.fakes.vector_store import FakeVectorStore


def ingest_and_answer(
    store: VectorStore,
    embedder: Embedder,
    llm: LLMClient,
    document_id: str,
    chunks: list[Chunk],
    question: str,
) -> str:
    """A use-case-shaped function: index the chunks, retrieve context, ask the model."""
    store.ensure_collection(embedder.identity)
    store.upsert(
        document_id,
        chunks,
        embedder.embed_passages([chunk.text for chunk in chunks]),
        page_texts={chunk.page: chunk.text for chunk in chunks},
        content_hash=f"hash-{document_id}",
    )
    hits = store.search(embedder.embed_query(question), top_k=2)
    context = "\n".join(hit.text for hit in hits)
    return llm.complete(
        [
            ChatMessage("system", "Answer from the context only."),
            ChatMessage("user", f"Context:\n{context}\n\nQuestion: {question}"),
        ]
    )


def test_fakes_satisfy_their_ports() -> None:
    assert isinstance(FakeOcrEngine(), OcrEngine)
    assert isinstance(FakeNameRecognizer(), NameRecognizer)
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeVectorStore(), VectorStore)
    assert isinstance(FakeLLMClient(), LLMClient)


def test_fakes_conform_to_their_ports_statically() -> None:
    """The annotation is the check: mypy rejects a fake whose signatures drift from its port."""
    ports: tuple[OcrEngine, NameRecognizer, Embedder, VectorStore, LLMClient] = (
        FakeOcrEngine(),
        FakeNameRecognizer(),
        FakeEmbedder(),
        FakeVectorStore(),
        FakeLLMClient(),
    )

    assert len(ports) == 5


def test_object_without_the_methods_does_not_satisfy_a_port() -> None:
    assert not isinstance(object(), VectorStore)


def test_use_case_runs_entirely_on_fakes() -> None:
    chunks = [
        Chunk("Paris is the capital of France.", 0, 0, 0, 31),
        Chunk("Bananas grow in tropical climates.", 0, 1, 32, 66),
    ]
    store, llm = FakeVectorStore(), FakeLLMClient(["Paris"])

    answer = ingest_and_answer(
        store, FakeEmbedder(), llm, "doc-1", chunks, "What is the capital of France?"
    )

    assert answer == "Paris"
    assert store.count() == 2
    user_message = llm.calls[0][1]
    assert user_message.role == "user"
    assert user_message.content.index("Paris") < user_message.content.index("Bananas")


def test_use_case_propagates_llm_failure_after_indexing() -> None:
    store = FakeVectorStore()
    llm = FakeLLMClient([DependencyTimeout("slow", code="llm_timeout")])

    with pytest.raises(DependencyTimeout):
        ingest_and_answer(store, FakeEmbedder(), llm, "doc-1", [Chunk("text", 0, 0, 0, 4)], "q?")

    assert store.count() == 1


def test_use_case_with_mismatched_embedder_fails_on_second_run() -> None:
    store = FakeVectorStore()
    chunk = [Chunk("text", 0, 0, 0, 4)]
    ingest_and_answer(store, FakeEmbedder(), FakeLLMClient(), "doc-1", chunk, "q?")
    other = FakeEmbedder(dataclasses.replace(FakeEmbedder().identity, dimension=32))

    with pytest.raises(IncompatibleEmbedding):
        ingest_and_answer(store, other, FakeLLMClient(), "doc-2", chunk, "q?")

    assert store.count() == 1


def test_identities_with_same_fields_are_equal() -> None:
    a = EmbeddingIdentity("model", 384, "e5", True, "cosine")
    b = EmbeddingIdentity("model", 384, "e5", True, "cosine")

    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.parametrize(
    "change",
    [
        {"prefix_scheme": "none"},
        {"model_id": "other"},
        {"dimension": 768},
        {"normalized": False},
        {"metric": "dot"},
    ],
)
def test_identities_differing_in_any_field_are_unequal(change: dict[str, Any]) -> None:
    base = EmbeddingIdentity("model", 384, "e5", True, "cosine")

    assert dataclasses.replace(base, **change) != base


def test_ocr_fake_returns_canned_pages_and_records_calls() -> None:
    page = OcrPage(0, 595.0, 842.0, (PageWord("Hello", Box(0, 10, 10, 30, 12), 0, 0),), None)
    engine = FakeOcrEngine([page])

    assert engine.recognize(b"%PDF", language="en") == [dataclasses.replace(page, language="en")]
    assert engine.recognize(b"%PDF-2") == [page]
    assert engine.calls == [(b"%PDF", "en"), (b"%PDF-2", None)]


def test_name_fake_returns_spans_of_configured_names() -> None:
    recognizer = FakeNameRecognizer(["Maria Gonzalez", "Maria", "Karl-Heinz Müller"])
    text = "Maria Gonzalez met Karl-Heinz Müller; Maria left. Mariana stayed."

    spans = recognizer.find_persons(text)

    assert [text[span.start : span.end] for span in spans] == [
        "Maria Gonzalez",
        "Karl-Heinz Müller",
        "Maria",
    ]
    assert all(isinstance(span, TextSpan) for span in spans)
    assert recognizer.calls == [text]


def test_name_fake_finds_nothing_when_unconfigured() -> None:
    assert FakeNameRecognizer().find_persons("Anyone at all") == []


def test_embedder_fake_is_deterministic_across_instances() -> None:
    assert FakeEmbedder().embed_query("same words") == FakeEmbedder().embed_query("same words")


def test_embedder_fake_scores_shared_words_higher() -> None:
    embedder = FakeEmbedder()
    query = embedder.embed_query("capital of France")
    related, unrelated = embedder.embed_passages(["the capital of France", "banana bread recipe"])

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert dot(query, related) > dot(query, unrelated)


def test_vector_store_fake_requires_collection_first() -> None:
    with pytest.raises(RuntimeError):
        FakeVectorStore().search([1.0], top_k=1)


def test_vector_store_fake_fail_next_raises_then_recovers_without_changes() -> None:
    store = FakeVectorStore()
    store.ensure_collection(EmbeddingIdentity("m", 2, "none", True, "cosine"))
    store.fail_next(3, DependencyUnavailable("down", code="vector_store_unavailable"))

    for _ in range(2):
        with pytest.raises(DependencyUnavailable):
            store.upsert(
                "doc", [Chunk("t", 0, 0, 0, 1)], [[1.0, 0.0]], page_texts={0: "t"}, content_hash="h"
            )
    with pytest.raises(DependencyUnavailable):
        store.document_chunk_count("doc")

    assert store.count() == 0
    store.upsert(
        "doc", [Chunk("t", 0, 0, 0, 1)], [[1.0, 0.0]], page_texts={0: "t"}, content_hash="h"
    )
    assert store.count() == 1
    assert store.document_chunk_count("doc") == 1
    assert store.content_hashes("doc") == {"h"}


def test_llm_fake_uses_queue_then_default_and_records_messages() -> None:
    llm = FakeLLMClient(["first"], default="fallback")
    llm.enqueue("second")
    messages = [ChatMessage("user", "hi")]

    assert [llm.complete(messages) for _ in range(3)] == ["first", "second", "fallback"]
    assert llm.calls == [messages] * 3


def test_llm_fake_raises_queued_errors_in_order() -> None:
    llm = FakeLLMClient(
        [
            DependencyUnavailable("down", code="llm_unavailable"),
            DependencyTimeout("slow", code="llm_timeout"),
            "ok",
        ]
    )
    messages = [ChatMessage("user", "hi")]

    with pytest.raises(DependencyUnavailable):
        llm.complete(messages)
    with pytest.raises(DependencyTimeout):
        llm.complete(messages)
    assert llm.complete(messages) == "ok"


def test_new_domain_types_validate_their_fields() -> None:
    with pytest.raises(ValueError, match="page index"):
        OcrPage(-1, 10.0, 10.0, (), None)
    with pytest.raises(ValueError, match="page width and height"):
        OcrPage(0, 0.0, 10.0, (), None)
    with pytest.raises(ValueError, match="score must be within"):
        SearchHit("d", 0, 0, "t", "page t", 1.5)
    with pytest.raises(ValueError, match="page index"):
        SearchHit("d", -1, 0, "t", "page t", 0.5)
    with pytest.raises(ValueError, match="sequence"):
        SearchHit("d", 0, -1, "t", "page t", 0.5)
    hit = SearchHit("d", 0, 0, "t", "page t", 0.0)
    assert (hit.page_text, hit.score) == ("page t", 0.0)
