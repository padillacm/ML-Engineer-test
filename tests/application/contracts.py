"""Reusable contract tests for the ``OcrEngine``, ``NameRecognizer``, ``Embedder``,
``VectorStore`` and ``LLMClient`` ports.

A test module subclasses a contract and overrides its factory fixtures; every real adapter added
in later blocks runs the same test bodies as the fakes. The classes are not named ``Test*`` so
pytest only collects the concrete subclasses.
"""

import dataclasses
import itertools
import math
import threading
from collections.abc import Sequence

import pytest

from app.application.ports import Embedder, LLMClient, NameRecognizer, OcrEngine, VectorStore
from app.domain.cancellation import Cancellation
from app.domain.errors import (
    DependencyTimeout,
    DependencyUnavailable,
    IncompatibleEmbedding,
    ProcessingFailed,
)
from app.domain.language import SUPPORTED_LANGUAGES
from app.domain.types import ChatMessage, Chunk, EmbeddingIdentity, OcrPage, TextSpan
from tests.fakes.vector_store import FakeVectorStore

_IDENTITY_384 = EmbeddingIdentity("contract-model", 384, "none", True, "cosine")
_IDENTITY_768 = dataclasses.replace(_IDENTITY_384, dimension=768)


def one_hot(dimension: int, index: int) -> list[float]:
    vector = [0.0] * dimension
    vector[index] = 1.0
    return vector


def make_chunks(*texts: str) -> list[Chunk]:
    """Chunks of one page, numbered from 0 in the order given."""
    return [Chunk(text, 0, sequence, 0, len(text)) for sequence, text in enumerate(texts)]


def page_texts_of(chunks: Sequence[Chunk]) -> dict[int, str]:
    """Page texts that contain every chunk: each page is its chunks' texts joined by spaces."""
    pages: dict[int, list[str]] = {}
    for chunk in chunks:
        pages.setdefault(chunk.page, []).append(chunk.text)
    return {page: " ".join(texts) for page, texts in pages.items()}


def upsert(
    store: VectorStore,
    document_id: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
) -> None:
    """``store.upsert`` with page texts built from the chunks and a hash derived from the id."""
    store.upsert(
        document_id,
        chunks,
        vectors,
        page_texts=page_texts_of(chunks),
        content_hash=f"sha256-of-{document_id}",
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.hypot(*a) * math.hypot(*b))


class OcrEngineContract:
    """Behaviour every ``OcrEngine`` must show.

    Override ``engine``, ``sample_pdf`` (a PDF whose first page holds text) and
    ``expected_pages`` (its page count). Every page of one result reports the same language: the
    hint when one is given, otherwise a supported code or ``None``.
    """

    @pytest.fixture
    def engine(self) -> OcrEngine:
        raise NotImplementedError

    @pytest.fixture
    def sample_pdf(self) -> bytes:
        raise NotImplementedError

    @pytest.fixture
    def expected_pages(self) -> int:
        raise NotImplementedError

    @pytest.fixture
    def pages(self, engine: OcrEngine, sample_pdf: bytes) -> list[OcrPage]:
        return engine.recognize(sample_pdf, "en")

    def test_one_result_per_page_in_page_order(
        self, pages: list[OcrPage], expected_pages: int
    ) -> None:
        assert [page.page for page in pages] == list(range(expected_pages))

    def test_pages_have_a_positive_size(self, pages: list[OcrPage]) -> None:
        assert all(page.width > 0 and page.height > 0 for page in pages)

    def test_first_page_has_words(self, pages: list[OcrPage]) -> None:
        assert pages[0].words

    def test_word_boxes_lie_inside_their_page(self, pages: list[OcrPage]) -> None:
        for page in pages:
            for word in page.words:
                box = word.box
                assert box.page == page.page
                assert word.text.strip()
                assert 0.0 <= box.x <= box.right <= page.width
                assert 0.0 <= box.y <= box.bottom <= page.height

    def test_without_a_hint_the_page_count_is_the_same(
        self, engine: OcrEngine, sample_pdf: bytes, expected_pages: int
    ) -> None:
        assert len(engine.recognize(sample_pdf)) == expected_pages

    def test_with_a_hint_every_page_reports_the_hint(self, pages: list[OcrPage]) -> None:
        assert {page.language for page in pages} == {"en"}

    def test_without_a_hint_every_page_reports_one_language_or_none(
        self, engine: OcrEngine, sample_pdf: bytes
    ) -> None:
        languages = {page.language for page in engine.recognize(sample_pdf)}

        assert len(languages) == 1
        assert languages <= {None, *SUPPORTED_LANGUAGES}

    @pytest.mark.parametrize("language", [None, "en"])
    def test_a_cancelled_token_raises_request_cancelled(
        self, engine: OcrEngine, sample_pdf: bytes, language: str | None
    ) -> None:
        cancel = Cancellation()
        cancel.cancel()

        with pytest.raises(ProcessingFailed) as raised:
            engine.recognize(sample_pdf, language, cancel=cancel)

        assert raised.value.code == "request_cancelled"

    def test_a_live_token_changes_nothing(
        self, engine: OcrEngine, sample_pdf: bytes, pages: list[OcrPage]
    ) -> None:
        assert engine.recognize(sample_pdf, "en", cancel=Cancellation()) == pages


class NameRecognizerContract:
    """Behaviour every ``NameRecognizer`` must show.

    Override ``recognizer`` and, when the recogniser cannot find the default names,
    ``sample_text`` (a text in which it finds at least one person).
    """

    @pytest.fixture
    def recognizer(self) -> NameRecognizer:
        raise NotImplementedError

    @pytest.fixture
    def sample_text(self) -> str:
        return (
            "Maria Gonzalez of Acme Corporation met in Madrid on 3 March. "
            "The agreement was signed by Peter Smith."
        )

    @pytest.fixture
    def spans(self, recognizer: NameRecognizer, sample_text: str) -> list[TextSpan]:
        return recognizer.find_persons(sample_text)

    def test_finds_a_person_in_the_sample(self, spans: list[TextSpan]) -> None:
        assert spans

    def test_spans_lie_inside_the_text(self, spans: list[TextSpan], sample_text: str) -> None:
        assert all(0 <= span.start < span.end <= len(sample_text) for span in spans)

    def test_spans_are_in_order_and_do_not_overlap(self, spans: list[TextSpan]) -> None:
        assert all(left.end <= right.start for left, right in itertools.pairwise(spans))

    def test_spans_cover_text_without_surrounding_blanks(
        self, spans: list[TextSpan], sample_text: str
    ) -> None:
        for span in spans:
            covered = sample_text[span.start : span.end]
            assert covered.strip()
            assert covered == covered.strip()

    def test_recognition_is_deterministic(
        self, recognizer: NameRecognizer, sample_text: str, spans: list[TextSpan]
    ) -> None:
        assert recognizer.find_persons(sample_text) == spans

    @pytest.mark.parametrize("text", ["", "   ", "\n\t "])
    def test_blank_text_has_no_names(self, recognizer: NameRecognizer, text: str) -> None:
        assert recognizer.find_persons(text) == []

    def test_text_without_people_has_no_names(self, recognizer: NameRecognizer) -> None:
        assert recognizer.find_persons("the agreement was signed on 3 march .") == []


class EmbedderContract:
    """Behaviour every ``Embedder`` must show. Override the ``embedder`` fixture."""

    @pytest.fixture
    def embedder(self) -> Embedder:
        raise NotImplementedError

    def test_identity_is_stable_and_valid(self, embedder: Embedder) -> None:
        assert isinstance(embedder.identity, EmbeddingIdentity)
        assert embedder.identity == embedder.identity

    def test_passages_have_identity_dimension_and_input_order(self, embedder: Embedder) -> None:
        texts = ["alpha beta", "gamma delta", "epsilon"]

        vectors = embedder.embed_passages(texts)

        assert len(vectors) == len(texts)
        assert all(len(vector) == embedder.identity.dimension for vector in vectors)
        # Batching pads inputs, which changes the shape of the matrix products and so the last
        # float bits of real models (~1e-7); equality is checked to 1e-5.
        for text, vector in zip(texts, vectors, strict=True):
            assert vector == pytest.approx(embedder.embed_passages([text])[0], abs=1e-5)

    def test_no_passages_give_no_vectors(self, embedder: Embedder) -> None:
        assert embedder.embed_passages([]) == []

    def test_embedding_is_deterministic(self, embedder: Embedder) -> None:
        text = "The invoice was paid on Monday."

        assert embedder.embed_passages([text]) == embedder.embed_passages([text])
        assert embedder.embed_query(text) == embedder.embed_query(text)

    def test_query_has_identity_dimension(self, embedder: Embedder) -> None:
        assert len(embedder.embed_query("who signed the contract?")) == embedder.identity.dimension

    def test_vectors_are_unit_length_when_normalized(self, embedder: Embedder) -> None:
        if not embedder.identity.normalized:
            pytest.skip("embedder does not promise normalised vectors")
        vectors = [*embedder.embed_passages(["a short passage"]), embedder.embed_query("a query")]

        assert all(math.hypot(*vector) == pytest.approx(1.0) for vector in vectors)

    def test_related_text_scores_higher_than_unrelated(self, embedder: Embedder) -> None:
        related, unrelated = embedder.embed_passages(
            ["Paris is the capital city of France", "Bananas grow in tropical climates"]
        )
        query = embedder.embed_query("What is the capital of France?")

        assert _cosine(query, related) > _cosine(query, unrelated)


class VectorStoreContract:
    """Behaviour every ``VectorStore`` must show.

    Override the ``store`` fixture so it returns a fresh, empty store with no collection yet.
    The hooks ``drop_chunk``, ``writer_failing_after`` and ``other_writer`` default to what
    ``FakeVectorStore`` offers; adapters override them.
    """

    @pytest.fixture
    def store(self) -> VectorStore:
        raise NotImplementedError

    def drop_chunk(self, store: VectorStore, document_id: str, sequence: int) -> None:
        """Lose one stored chunk behind the store's back (as an interrupted write would)."""
        assert isinstance(store, FakeVectorStore), "override drop_chunk for this store"
        store.drop_chunk(document_id, sequence)

    def writer_failing_after(self, store: VectorStore, stored: int) -> VectorStore:
        """A ready store over ``store``'s data whose next ``upsert`` stores ``stored`` chunks and
        then fails with ``DependencyUnavailable``."""
        assert isinstance(store, FakeVectorStore), "override writer_failing_after for this store"
        store.fail_upsert_after(stored, DependencyUnavailable("down", "vector_store_unavailable"))
        return store

    def other_writer(self, store: VectorStore) -> VectorStore:
        """A ready store over ``store``'s data that another thread may use."""
        assert isinstance(store, FakeVectorStore), "override other_writer for this store"
        return store

    @pytest.fixture
    def identity(self) -> EmbeddingIdentity:
        return _IDENTITY_384

    @pytest.fixture
    def ready(self, store: VectorStore, identity: EmbeddingIdentity) -> VectorStore:
        store.ensure_collection(identity)
        return store

    def test_new_collection_is_empty(self, ready: VectorStore) -> None:
        assert ready.count() == 0
        assert ready.search(one_hot(384, 0), top_k=5) == []

    def test_search_returns_best_first_with_chunk_provenance(self, ready: VectorStore) -> None:
        chunks = [
            Chunk("exact", 2, 0, 0, 5),
            Chunk("close", 3, 1, 0, 5),
            Chunk("far", 4, 2, 0, 3),
        ]
        close = [0.0] * 384
        close[0], close[1] = 1.0, 1.0
        upsert(ready, "doc", chunks, [one_hot(384, 0), close, one_hot(384, 7)])

        hits = ready.search(one_hot(384, 0), top_k=2)

        assert [(hit.document_id, hit.page, hit.sequence, hit.text) for hit in hits] == [
            ("doc", 2, 0, "exact"),
            ("doc", 3, 1, "close"),
        ]
        assert hits[0].score == pytest.approx(1.0)
        assert hits[1].score == pytest.approx(math.sqrt(0.5))

    def test_scores_stay_within_zero_and_one(self, ready: VectorStore) -> None:
        opposite = [-value for value in one_hot(384, 0)]
        upsert(
            ready,
            "doc",
            make_chunks("same", "orthogonal", "opposite"),
            [
                one_hot(384, 0),
                one_hot(384, 1),
                opposite,
            ],
        )

        hits = ready.search(one_hot(384, 0), top_k=10)

        assert len(hits) == 3
        assert all(0.0 <= hit.score <= 1.0 for hit in hits)
        assert hits[-1].score == 0.0
        assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)

    def test_reopening_with_same_identity_keeps_chunks_searchable(
        self, ready: VectorStore, identity: EmbeddingIdentity
    ) -> None:
        upsert(ready, "doc", make_chunks("stored text"), [one_hot(384, 3)])

        ready.ensure_collection(dataclasses.replace(identity))

        assert ready.count() == 1
        assert [hit.text for hit in ready.search(one_hot(384, 3), top_k=1)] == ["stored text"]

    def test_reopening_with_another_dimension_fails_and_changes_nothing(
        self, ready: VectorStore
    ) -> None:
        upsert(ready, "doc", make_chunks("stored text"), [one_hot(384, 3)])

        with pytest.raises(IncompatibleEmbedding):
            ready.ensure_collection(_IDENTITY_768)

        assert ready.count() == 1
        assert [hit.text for hit in ready.search(one_hot(384, 3), top_k=1)] == ["stored text"]

    def test_reopening_with_another_prefix_scheme_fails(
        self, ready: VectorStore, identity: EmbeddingIdentity
    ) -> None:
        with pytest.raises(IncompatibleEmbedding):
            ready.ensure_collection(dataclasses.replace(identity, prefix_scheme="e5"))

    def test_adding_vectors_of_another_dimension_fails_and_changes_nothing(
        self, ready: VectorStore
    ) -> None:
        upsert(ready, "doc", make_chunks("stored text"), [one_hot(384, 3)])

        with pytest.raises(IncompatibleEmbedding):
            upsert(ready, "doc", make_chunks("wrong size"), [one_hot(768, 3)])

        assert [hit.text for hit in ready.search(one_hot(384, 3), top_k=5)] == ["stored text"]

    def test_reingesting_a_document_keeps_only_the_latest_chunks(self, ready: VectorStore) -> None:
        upsert(
            ready,
            "doc",
            make_chunks("old one", "old two", "old three"),
            [
                one_hot(384, 0),
                one_hot(384, 1),
                one_hot(384, 2),
            ],
        )

        upsert(ready, "doc", make_chunks("new one", "new two"), [one_hot(384, 4), one_hot(384, 5)])

        assert ready.count() == 2
        texts = {hit.text for hit in ready.search(one_hot(384, 0), top_k=10)}
        assert texts == {"new one", "new two"}

    def test_two_documents_stay_retrievable(self, ready: VectorStore) -> None:
        upsert(ready, "doc-a", make_chunks("from a"), [one_hot(384, 0)])
        upsert(ready, "doc-b", make_chunks("from b"), [one_hot(384, 1)])

        assert ready.count() == 2
        assert ready.search(one_hot(384, 0), top_k=1)[0].document_id == "doc-a"
        assert ready.search(one_hot(384, 1), top_k=1)[0].document_id == "doc-b"

    def test_reingesting_one_document_leaves_another_untouched(self, ready: VectorStore) -> None:
        upsert(ready, "doc-a", make_chunks("from a"), [one_hot(384, 0)])
        upsert(ready, "doc-b", make_chunks("b before"), [one_hot(384, 1)])

        upsert(ready, "doc-b", make_chunks("b after"), [one_hot(384, 2)])

        hits = {hit.document_id: hit.text for hit in ready.search(one_hot(384, 0), top_k=10)}
        assert hits == {"doc-a": "from a", "doc-b": "b after"}

    def test_same_sequence_in_two_documents_does_not_collide(self, ready: VectorStore) -> None:
        upsert(ready, "doc-a", make_chunks("a first"), [one_hot(384, 0)])
        upsert(ready, "doc-b", make_chunks("b first"), [one_hot(384, 0)])

        assert ready.count() == 2

    def test_searching_with_a_vector_of_another_dimension_fails_and_changes_nothing(
        self, ready: VectorStore
    ) -> None:
        upsert(ready, "doc", make_chunks("stored text"), [one_hot(384, 3)])

        with pytest.raises(IncompatibleEmbedding):
            ready.search(one_hot(768, 3), top_k=5)

        assert ready.count() == 1
        assert [hit.text for hit in ready.search(one_hot(384, 3), top_k=5)] == ["stored text"]

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_top_k_below_one_is_rejected(self, ready: VectorStore, top_k: int) -> None:
        upsert(ready, "doc", make_chunks("stored text"), [one_hot(384, 3)])

        with pytest.raises(ValueError):  # noqa: PT011 - type only, per port contract
            ready.search(one_hot(384, 3), top_k=top_k)

    def test_duplicate_sequences_in_one_upsert_fail_and_keep_existing_data(
        self, ready: VectorStore
    ) -> None:
        upsert(ready, "doc", make_chunks("stored text"), [one_hot(384, 3)])
        duplicated = [Chunk("first", 0, 0, 0, 5), Chunk("second", 0, 0, 6, 12)]

        with pytest.raises(ValueError):  # noqa: PT011 - type only, per port contract
            upsert(ready, "doc", duplicated, [one_hot(384, 0), one_hot(384, 1)])

        assert ready.count() == 1
        assert [hit.text for hit in ready.search(one_hot(384, 3), top_k=5)] == ["stored text"]

    def test_chunks_and_vectors_of_different_length_fail_and_keep_existing_data(
        self, ready: VectorStore
    ) -> None:
        upsert(ready, "doc", make_chunks("stored text"), [one_hot(384, 3)])

        with pytest.raises(ValueError):  # noqa: PT011 - type only, per port contract
            upsert(ready, "doc", make_chunks("one", "two"), [one_hot(384, 0)])
        with pytest.raises(ValueError):  # noqa: PT011 - type only, per port contract
            upsert(ready, "doc", make_chunks("one"), [one_hot(384, 0), one_hot(384, 1)])

        assert ready.count() == 1
        assert [hit.text for hit in ready.search(one_hot(384, 3), top_k=5)] == ["stored text"]

    def test_upsert_before_ensure_collection_is_a_programming_error(
        self, store: VectorStore
    ) -> None:
        with pytest.raises(RuntimeError):
            upsert(store, "doc", make_chunks("text"), [one_hot(384, 0)])

    def test_search_before_ensure_collection_is_a_programming_error(
        self, store: VectorStore
    ) -> None:
        with pytest.raises(RuntimeError):
            store.search(one_hot(384, 0), top_k=1)

    def test_delete_document_removes_only_that_document(self, ready: VectorStore) -> None:
        upsert(ready, "doc-a", make_chunks("from a"), [one_hot(384, 0)])
        upsert(ready, "doc-b", make_chunks("from b"), [one_hot(384, 1)])

        ready.delete_document("doc-a")
        ready.delete_document("never-stored")

        assert [hit.document_id for hit in ready.search(one_hot(384, 0), top_k=10)] == ["doc-b"]
        assert ready.count() == 1

    # --- flagged chunks (D6): stored and counted like any other, never returned by a search ---

    def test_a_flagged_best_match_is_never_returned(self, ready: VectorStore) -> None:
        close = [0.0] * 384
        close[0], close[1] = 1.0, 1.0
        chunks = [
            Chunk("injected", 0, 0, 0, 8, flagged=True),
            Chunk("close", 0, 1, 9, 14),
            Chunk("far", 0, 2, 15, 18),
        ]
        upsert(ready, "doc", chunks, [one_hot(384, 0), close, one_hot(384, 7)])

        best = ready.search(one_hot(384, 0), top_k=1)
        every = ready.search(one_hot(384, 0), top_k=10)

        assert [hit.text for hit in best] == ["close"]
        assert [hit.text for hit in every] == ["close", "far"]
        assert every[0].score == pytest.approx(math.sqrt(0.5))

    def test_a_document_of_flagged_chunks_only_is_searched_as_empty(
        self, ready: VectorStore
    ) -> None:
        chunks = [Chunk("one", 0, 0, 0, 3, flagged=True), Chunk("two", 0, 1, 4, 7, flagged=True)]
        upsert(ready, "doc", chunks, [one_hot(384, 0), one_hot(384, 1)])

        assert ready.search(one_hot(384, 0), top_k=10) == []
        assert ready.count() == 2
        assert ready.document_chunk_count("doc") == 2

    @pytest.fixture
    def flagged_document(self, ready: VectorStore) -> list[Chunk]:
        """``doc``: 10 chunks, the ones with sequence 3 and 7 flagged."""
        chunks = [Chunk(f"chunk {i}", 0, i, 0, 7, flagged=i in {3, 7}) for i in range(10)]
        upsert(ready, "doc", chunks, [one_hot(384, i) for i in range(10)])
        return chunks

    def test_chunk_count_includes_flagged_chunks(
        self, ready: VectorStore, flagged_document: list[Chunk]
    ) -> None:
        assert ready.document_chunk_count("doc") == 10
        assert ready.count() == 10
        texts = {hit.text for hit in ready.search(one_hot(384, 3), top_k=20)}
        assert texts == {f"chunk {i}" for i in range(10) if i not in {3, 7}}

    def test_deleting_a_document_removes_its_flagged_chunks(
        self, ready: VectorStore, flagged_document: list[Chunk]
    ) -> None:
        ready.delete_document("doc")

        assert ready.count() == 0
        assert ready.document_chunk_count("doc") == 0

    def test_a_document_missing_a_flagged_chunk_is_incomplete_and_repairable(
        self, ready: VectorStore, flagged_document: list[Chunk]
    ) -> None:
        self.drop_chunk(ready, "doc", 3)
        assert ready.document_chunk_count("doc") == 0

        upsert(ready, "doc", flagged_document, [one_hot(384, i) for i in range(10)])

        assert ready.document_chunk_count("doc") == 10
        assert ready.count() == 10
        assert [hit.text for hit in ready.search(one_hot(384, 3), top_k=1)] != ["chunk 3"]

    def test_rewriting_a_document_unflagged_makes_its_chunks_searchable_again(
        self, ready: VectorStore, flagged_document: list[Chunk]
    ) -> None:
        cleared = [dataclasses.replace(chunk, flagged=False) for chunk in flagged_document]

        upsert(ready, "doc", cleared, [one_hot(384, i) for i in range(10)])

        assert [hit.text for hit in ready.search(one_hot(384, 3), top_k=1)] == ["chunk 3"]
        assert ready.count() == 10

    def test_hits_carry_the_page_text_of_their_chunk(self, ready: VectorStore) -> None:
        chunks = [Chunk("first half", 0, 0, 0, 10), Chunk("other page", 1, 1, 0, 10)]
        page_texts = {0: "first half and second half", 1: "other page text", 2: "unused page"}
        ready.upsert(
            "doc",
            chunks,
            [one_hot(384, 0), one_hot(384, 1)],
            page_texts=page_texts,
            content_hash="abc123",
        )

        hits = ready.search(one_hot(384, 0), top_k=2)

        assert [(hit.page, hit.text, hit.page_text) for hit in hits] == [
            (0, "first half", "first half and second half"),
            (1, "other page", "other page text"),
        ]

    def test_document_chunk_count_is_zero_for_unknown_documents(self, ready: VectorStore) -> None:
        upsert(ready, "doc-a", make_chunks("from a"), [one_hot(384, 0)])

        assert ready.document_chunk_count("never-stored") == 0

    def test_document_chunk_count_counts_only_that_document(self, ready: VectorStore) -> None:
        upsert(ready, "doc-a", make_chunks("a1", "a2", "a3"), [one_hot(384, i) for i in range(3)])
        upsert(ready, "doc-b", make_chunks("b1"), [one_hot(384, 5)])

        assert ready.document_chunk_count("doc-a") == 3
        assert ready.document_chunk_count("doc-b") == 1

    def test_document_chunk_count_after_reupsert_is_replaced_not_added(
        self, ready: VectorStore
    ) -> None:
        upsert(ready, "doc", make_chunks("a1", "a2", "a3"), [one_hot(384, i) for i in range(3)])
        upsert(ready, "doc", make_chunks("a1", "a2", "a3"), [one_hot(384, i) for i in range(3)])
        assert ready.document_chunk_count("doc") == 3

        upsert(ready, "doc", make_chunks("only"), [one_hot(384, 9)])
        assert ready.document_chunk_count("doc") == 1

    def test_document_chunk_count_after_delete_is_zero(self, ready: VectorStore) -> None:
        upsert(ready, "doc-a", make_chunks("a1", "a2"), [one_hot(384, 0), one_hot(384, 1)])
        upsert(ready, "doc-b", make_chunks("b1"), [one_hot(384, 2)])

        ready.delete_document("doc-a")

        assert ready.document_chunk_count("doc-a") == 0
        assert ready.document_chunk_count("doc-b") == 1

    def test_document_chunk_count_before_ensure_collection_is_a_programming_error(
        self, store: VectorStore
    ) -> None:
        with pytest.raises(RuntimeError):
            store.document_chunk_count("doc")

    def test_missing_page_text_fails_and_keeps_existing_data(self, ready: VectorStore) -> None:
        upsert(ready, "doc", make_chunks("stored text"), [one_hot(384, 3)])
        chunks = [Chunk("on page 0", 0, 0, 0, 9), Chunk("on page 1", 1, 1, 0, 9)]

        with pytest.raises(ValueError):  # noqa: PT011 - type only, per port contract
            ready.upsert(
                "doc",
                chunks,
                [one_hot(384, 0), one_hot(384, 1)],
                page_texts={0: "on page 0"},
                content_hash="abc123",
            )

        assert ready.count() == 1
        assert ready.document_chunk_count("doc") == 1
        hits = ready.search(one_hot(384, 3), top_k=5)
        assert [(hit.text, hit.page_text) for hit in hits] == [("stored text", "stored text")]

    def test_empty_content_hash_fails_and_keeps_existing_data(self, ready: VectorStore) -> None:
        upsert(ready, "doc", make_chunks("stored text"), [one_hot(384, 3)])

        with pytest.raises(ValueError):  # noqa: PT011 - type only, per port contract
            ready.upsert(
                "doc",
                make_chunks("new text"),
                [one_hot(384, 0)],
                page_texts={0: "new text"},
                content_hash="",
            )

        assert [hit.text for hit in ready.search(one_hot(384, 3), top_k=5)] == ["stored text"]

    def test_a_document_missing_a_chunk_counts_as_zero(self, ready: VectorStore) -> None:
        upsert(ready, "doc", make_chunks("a1", "a2", "a3"), [one_hot(384, i) for i in range(3)])
        upsert(ready, "other", make_chunks("b1"), [one_hot(384, 5)])

        self.drop_chunk(ready, "doc", 1)

        assert ready.document_chunk_count("doc") == 0
        assert ready.document_chunk_count("other") == 1

    def test_storing_an_incomplete_document_again_completes_it(self, ready: VectorStore) -> None:
        chunks = make_chunks("a1", "a2", "a3")
        upsert(ready, "doc", chunks, [one_hot(384, i) for i in range(3)])
        self.drop_chunk(ready, "doc", 0)

        upsert(ready, "doc", chunks, [one_hot(384, i) for i in range(3)])

        assert ready.document_chunk_count("doc") == 3
        assert ready.count() == 3

    def test_a_write_failing_part_way_leaves_an_incomplete_document(
        self, ready: VectorStore
    ) -> None:
        chunks = make_chunks("a1", "a2", "a3", "a4")
        vectors = [one_hot(384, i) for i in range(4)]
        failing = self.writer_failing_after(ready, 2)

        with pytest.raises(DependencyUnavailable):
            upsert(failing, "doc", chunks, vectors)

        assert ready.document_chunk_count("doc") == 0
        upsert(ready, "doc", chunks, vectors)
        assert ready.document_chunk_count("doc") == 4
        assert ready.count() == 4

    def test_a_failed_second_writer_leaves_the_first_complete(self, ready: VectorStore) -> None:
        chunks = make_chunks("a1", "a2", "a3", "a4")
        vectors = [one_hot(384, i) for i in range(4)]
        upsert(ready, "doc", chunks, vectors)
        failing = self.writer_failing_after(ready, 2)

        with pytest.raises(DependencyUnavailable):
            upsert(failing, "doc", chunks, vectors)

        assert ready.document_chunk_count("doc") == 4
        texts = {hit.text for hit in ready.search(one_hot(384, 0), top_k=10)}
        assert texts == {"a1", "a2", "a3", "a4"}

    def test_concurrent_writes_of_a_document_keep_it_visible_and_leave_one_copy(
        self, ready: VectorStore
    ) -> None:
        chunks = make_chunks("a1", "a2", "a3", "a4")
        vectors = [one_hot(384, i) for i in range(4)]
        upsert(ready, "doc", chunks, vectors)
        writers = [self.other_writer(ready) for _ in range(2)]
        errors: list[BaseException] = []

        def rewrite(writer: VectorStore) -> None:
            try:
                for _ in range(5):
                    upsert(writer, "doc", chunks, vectors)
            except BaseException as error:  # noqa: BLE001 - reported by the main thread
                errors.append(error)

        threads = [threading.Thread(target=rewrite, args=(writer,)) for writer in writers]
        for thread in threads:
            thread.start()
        seen: set[int] = set()
        while any(thread.is_alive() for thread in threads):
            seen.add(ready.document_chunk_count("doc"))
        for thread in threads:
            thread.join()

        assert errors == []
        assert seen <= {4}
        assert ready.document_chunk_count("doc") == 4
        assert ready.count() == 4


class LLMClientContract:
    """Behaviour every ``LLMClient`` must show.

    Override ``client`` (a working client) and ``failing_client`` (one whose model cannot answer,
    however the adapter experiences that).
    """

    @pytest.fixture
    def client(self) -> LLMClient:
        raise NotImplementedError

    @pytest.fixture
    def failing_client(self) -> LLMClient:
        raise NotImplementedError

    @staticmethod
    def conversation() -> list[ChatMessage]:
        return [
            ChatMessage("system", "Answer briefly."),
            ChatMessage("user", "What is the capital of France?"),
        ]

    def test_satisfies_the_port(self, client: LLMClient) -> None:
        assert isinstance(client, LLMClient)

    def test_simple_conversation_returns_text(self, client: LLMClient) -> None:
        assert isinstance(client.complete(self.conversation()), str)

    def test_messages_are_not_modified(self, client: LLMClient) -> None:
        messages = self.conversation()
        before = list(messages)

        client.complete(messages)

        assert messages == before

    def test_only_dependency_errors_escape(self, failing_client: LLMClient) -> None:
        with pytest.raises((DependencyUnavailable, DependencyTimeout)) as info:
            failing_client.complete(self.conversation())

        assert info.value.code in {"llm_unavailable", "llm_timeout"}
        assert "capital of France" not in str(info.value)
