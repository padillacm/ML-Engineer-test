"""``E5Embedder`` with a stubbed encoder: prefixes, batching, limits, errors, pool, identity."""

import gc
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch

from app.adapters.embeddings import E5Embedder, E5EmbedderConfig
from app.adapters.embeddings.embedder import (
    PASSAGE_LANE,
    QUERY_LANE,
    LoadedEncoder,
    mean_pool,
    normalize,
)
from app.domain.errors import (
    ConfigurationError,
    DependencyUnavailable,
    ProcessingFailed,
)
from app.domain.types import EmbeddingIdentity

REVISION = "rev123"
DIMENSION = 3


class RecordingEncoder:
    """Returns a vector derived from each text's length and records every batch it gets."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text)), 0.0, 1.0] for text in texts]


def make_embedder(
    model_dir: Path, encode: Callable[[Sequence[str]], list[list[float]]], **settings: Any
) -> E5Embedder:
    config = E5EmbedderConfig(model_dir, REVISION, **settings)
    return E5Embedder(config, model_factory=lambda config: LoadedEncoder(DIMENSION, encode))


def test_identity_records_model_revision_and_scheme(model_dir: Path) -> None:
    embedder = make_embedder(model_dir, RecordingEncoder())

    assert embedder.identity == EmbeddingIdentity(
        "intfloat/multilingual-e5-base@rev123", DIMENSION, "e5-query-passage", True, "cosine"
    )


def test_passages_and_queries_get_their_prefixes(model_dir: Path) -> None:
    encoder = RecordingEncoder()
    embedder = make_embedder(model_dir, encoder)

    embedder.embed_passages(["first", "second"])
    embedder.embed_query("what?")

    assert encoder.batches == [["passage: first", "passage: second"], ["query: what?"]]


def test_passages_run_in_batches_and_keep_their_order(model_dir: Path) -> None:
    encoder = RecordingEncoder()
    embedder = make_embedder(model_dir, encoder, batch_size=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    vectors = embedder.embed_passages(texts)

    assert [len(batch) for batch in encoder.batches] == [2, 2, 1]
    assert [vector[0] for vector in vectors] == [len("passage: " + text) for text in texts]


def test_no_passages_do_not_run_the_model(model_dir: Path) -> None:
    encoder = RecordingEncoder()

    assert make_embedder(model_dir, encoder).embed_passages([]) == []
    assert encoder.batches == []


def test_oversized_texts_are_refused_before_the_model_runs(model_dir: Path) -> None:
    encoder = RecordingEncoder()
    embedder = make_embedder(model_dir, encoder, max_chars=10)

    with pytest.raises(ProcessingFailed) as passage:
        embedder.embed_passages(["short", "x" * 11])
    with pytest.raises(ProcessingFailed) as query:
        embedder.embed_query("y" * 11)

    assert passage.value.code == query.value.code == "embedding_failed"
    assert encoder.batches == []
    assert embedder.embed_passages(["x" * 10])  # the limit itself is accepted


def test_model_failures_become_one_fixed_error(model_dir: Path) -> None:
    def broken(texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("secret vendor detail")

    embedder = make_embedder(model_dir, broken)

    with pytest.raises(ProcessingFailed) as caught:
        embedder.embed_query("text")

    assert caught.value.code == "embedding_failed"
    assert "secret" not in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_service_errors_of_the_encoder_pass_through(model_dir: Path) -> None:
    def unavailable(texts: Sequence[str]) -> list[list[float]]:
        raise DependencyUnavailable("down", "llm_unavailable")

    with pytest.raises(DependencyUnavailable):
        make_embedder(model_dir, unavailable).embed_passages(["text"])


@pytest.mark.parametrize("vectors", [[], [[1.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], ids=str)
def test_vectors_of_the_wrong_shape_are_refused(
    model_dir: Path, vectors: list[list[float]]
) -> None:
    embedder = make_embedder(model_dir, lambda texts: vectors)

    with pytest.raises(ProcessingFailed, match="wrong shape"):
        embedder.embed_query("text")


class GatedEncoder:
    """Holds every call until ``release`` is set; records how many run at once, and in order."""

    def __init__(self) -> None:
        self.guard = threading.Lock()
        self.release = threading.Event()
        self.active = 0
        self.peak = 0
        self.started: list[str] = []

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        with self.guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.started.append(texts[0])
        try:
            assert self.release.wait(5), "the test never released the calls"
            return [[1.0, 0.0, 0.0] for _ in texts]
        finally:
            with self.guard:
                self.active -= 1

    def wait_for(self, count: int) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with self.guard:
                if len(self.started) >= count:
                    return
            time.sleep(0.005)
        raise AssertionError(f"{count} calls never started (started={len(self.started)})")


class SteppedEncoder(GatedEncoder):
    """Lets calls finish one at a time: each ``step`` ends one running call (any of them)."""

    def __init__(self) -> None:
        super().__init__()
        self.permits = threading.Semaphore(0)

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        with self.guard:
            self.active += 1
            self.started.append(texts[0])
        try:
            assert self.permits.acquire(timeout=5), "the test never let the call finish"
            return [[1.0, 0.0, 0.0] for _ in texts]
        finally:
            with self.guard:
                self.active -= 1

    def step(self, then_started: int) -> None:
        """End one running call, then wait until ``then_started`` calls have started."""
        self.permits.release()
        self.wait_for(then_started)


def start_all(calls: int, target: Callable[[], object]) -> list[threading.Thread]:
    threads = [threading.Thread(target=target) for _ in range(calls)]
    for thread in threads:
        thread.start()
    return threads


def join_all(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()


def wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline, "the condition never held"
        time.sleep(0.005)


def slot_counts(embedder: E5Embedder) -> tuple[int, int, int, int]:
    """Slots held by queries and passages, then queries and passages waiting."""
    slots = embedder._slots
    return (
        slots.held(QUERY_LANE),
        slots.held(PASSAGE_LANE),
        slots.waiting(QUERY_LANE),
        slots.waiting(PASSAGE_LANE),
    )


def queue_one(embedder: E5Embedder, lane: int, call: Callable[[], object]) -> threading.Thread:
    """Start ``call`` and return once it waits for a slot, so that arrival order is fixed."""
    before = embedder._slots.waiting(lane)
    thread = threading.Thread(target=call)
    thread.start()
    wait_until(lambda: embedder._slots.waiting(lane) == before + 1)
    return thread


def test_a_pool_of_one_serialises_passages(model_dir: Path) -> None:
    encoder = GatedEncoder()
    embedder = make_embedder(model_dir, encoder, pool_size=1)

    threads = start_all(6, lambda: embedder.embed_passages(["p"]))
    encoder.wait_for(1)
    wait_until(lambda: embedder._slots.waiting(PASSAGE_LANE) == 5)  # the others queue
    assert encoder.active == 1
    encoder.release.set()
    join_all(threads)

    assert encoder.peak == 1
    assert len(encoder.started) == 6


def test_at_most_pool_size_calls_run_at_once(model_dir: Path) -> None:
    encoder = GatedEncoder()
    embedder = make_embedder(model_dir, encoder, pool_size=3)
    results: list[list[list[float]]] = []

    threads = start_all(8, lambda: results.append(embedder.embed_passages(["p"])))
    encoder.wait_for(3)
    wait_until(lambda: embedder._slots.waiting(PASSAGE_LANE) == 5)  # the others queue
    assert encoder.active == 3
    encoder.release.set()
    join_all(threads)

    assert encoder.peak == 3
    assert results == [[[1.0, 0.0, 0.0]]] * 8


def test_each_batch_of_a_call_takes_a_slot_of_its_own(model_dir: Path) -> None:
    encoder = GatedEncoder()
    embedder = make_embedder(model_dir, encoder, pool_size=2, batch_size=1)

    threads = start_all(1, lambda: embedder.embed_passages(["a", "b", "c"]))
    encoder.wait_for(1)
    # The call holds one slot for its first batch: a second call takes the other slot, and a
    # third one waits (were the batches run at once, the second call would be the one waiting).
    threads += start_all(1, lambda: embedder.embed_passages(["z"]))
    encoder.wait_for(2)
    threads.append(queue_one(embedder, PASSAGE_LANE, lambda: embedder.embed_passages(["y"])))
    assert sorted(encoder.started) == ["passage: a", "passage: z"]
    assert encoder.active == 2
    encoder.release.set()
    join_all(threads)

    own = [text for text in encoder.started if text in {"passage: a", "passage: b", "passage: c"}]
    assert own == ["passage: a", "passage: b", "passage: c"]


def test_a_query_takes_the_reserved_slot_while_passages_fill_the_pool(model_dir: Path) -> None:
    encoder = GatedEncoder()
    embedder = make_embedder(model_dir, encoder, pool_size=2)
    passages = start_all(3, lambda: embedder.embed_passages(["passage"]))
    encoder.wait_for(2)
    wait_until(lambda: embedder._slots.waiting(PASSAGE_LANE) == 1)  # the third passage waits

    query = start_all(1, lambda: embedder.embed_query("question"))
    encoder.wait_for(3)

    assert encoder.active == 3
    assert encoder.started[-1] == "query: question"
    encoder.release.set()
    join_all(passages + query)
    assert encoder.started.count("passage: passage") == 3


def test_a_query_does_not_wait_while_a_slot_is_free(model_dir: Path) -> None:
    encoder = GatedEncoder()
    embedder = make_embedder(model_dir, encoder, pool_size=2)
    passage = start_all(1, lambda: embedder.embed_passages(["running"]))
    encoder.wait_for(1)

    query = start_all(1, lambda: embedder.embed_query("question"))
    encoder.wait_for(2)

    assert encoder.active == 2
    encoder.release.set()
    join_all(passage + query)


def test_at_most_pool_size_plus_one_calls_run_in_all(model_dir: Path) -> None:
    encoder = GatedEncoder()
    embedder = make_embedder(model_dir, encoder, pool_size=2)

    queries = start_all(5, lambda: embedder.embed_query("question"))
    encoder.wait_for(3)
    passages = start_all(2, lambda: embedder.embed_passages(["passage"]))
    wait_until(lambda: embedder._slots.waiting(QUERY_LANE) == 2)
    wait_until(lambda: embedder._slots.waiting(PASSAGE_LANE) == 2)
    assert encoder.active == 3
    encoder.release.set()
    join_all(queries + passages)

    assert encoder.peak == 3
    assert len(encoder.started) == 7


def test_a_waiting_query_takes_the_next_slot_before_waiting_passages(model_dir: Path) -> None:
    encoder = SteppedEncoder()
    embedder = make_embedder(model_dir, encoder, pool_size=1)
    first = start_all(1, lambda: embedder.embed_passages(["first"]))
    encoder.wait_for(1)
    reserved = start_all(1, lambda: embedder.embed_query("in the reserved slot"))
    encoder.wait_for(2)
    passages = [
        queue_one(embedder, PASSAGE_LANE, partial(embedder.embed_passages, [name]))
        for name in ("later 1", "later 2", "later 3")
    ]
    queries = [
        queue_one(embedder, QUERY_LANE, partial(embedder.embed_query, name))
        for name in ("waiting 1", "waiting 2")
    ]

    # While queries wait, each ending call hands exactly one slot to the next query, so the
    # queries' start order is their hand-off order, free of thread wake-up races. The passages
    # then run one at a time (a pool of one), each starting only after the previous one ended.
    encoder.step(3)
    encoder.step(4)
    for _ in range(5):
        encoder.permits.release()
    join_all(first + reserved + passages + queries)

    assert encoder.started == [
        "passage: first",
        "query: in the reserved slot",
        "query: waiting 1",  # queries first, each kind in arrival order
        "query: waiting 2",
        "passage: later 1",
        "passage: later 2",
        "passage: later 3",
    ]


def test_waiting_passages_get_the_slot_in_arrival_order(model_dir: Path) -> None:
    encoder = GatedEncoder()
    embedder = make_embedder(model_dir, encoder, pool_size=1)
    first = start_all(1, lambda: embedder.embed_passages(["first"]))
    encoder.wait_for(1)
    waiting = [
        queue_one(embedder, PASSAGE_LANE, partial(embedder.embed_passages, [name]))
        for name in "abcd"
    ]

    encoder.release.set()
    join_all(first + waiting)

    assert encoder.started == ["passage: first"] + [f"passage: {name}" for name in "abcd"]


def test_a_call_with_more_batches_does_not_overtake_a_waiting_call(model_dir: Path) -> None:
    encoder = GatedEncoder()
    embedder = make_embedder(model_dir, encoder, pool_size=1, batch_size=1)
    long_call = start_all(1, lambda: embedder.embed_passages(["long 1", "long 2", "long 3"]))
    encoder.wait_for(1)
    short_call = queue_one(embedder, PASSAGE_LANE, lambda: embedder.embed_passages(["short"]))

    encoder.release.set()  # the long call gives its slot back between its batches
    join_all([*long_call, short_call])

    assert encoder.started == [
        "passage: long 1",
        "passage: short",
        "passage: long 2",
        "passage: long 3",
    ]


def test_every_failure_gives_the_slot_back(model_dir: Path) -> None:
    failures: list[Exception] = [
        RuntimeError("vendor failure"),
        DependencyUnavailable("down", "llm_unavailable"),
        RuntimeError("vendor failure"),
    ]

    def failing(texts: Sequence[str]) -> list[list[float]]:
        if failures:
            raise failures.pop(0)
        return [[1.0, 0.0]]  # the wrong shape: refused after the slot is given back

    embedder = make_embedder(model_dir, failing, pool_size=1)

    with pytest.raises(ProcessingFailed, match="could not process"):
        embedder.embed_query("q")
    with pytest.raises(DependencyUnavailable):
        embedder.embed_query("q")
    with pytest.raises(ProcessingFailed, match="could not process"):
        embedder.embed_passages(["p"])
    with pytest.raises(ProcessingFailed, match="wrong shape"):
        embedder.embed_passages(["p"])
    assert slot_counts(embedder) == (0, 0, 0, 0)


def test_a_closed_embedder_refuses_calls(model_dir: Path) -> None:
    encoder = RecordingEncoder()
    embedder = make_embedder(model_dir, encoder)

    embedder.close()
    embedder.close()  # idempotent

    for call in (lambda: embedder.embed_query("q"), lambda: embedder.embed_passages(["p"])):
        with pytest.raises(ProcessingFailed) as caught:
            call()
        assert caught.value.code == "embedding_failed"
        assert caught.value.message == "embedding model is closed"
    assert encoder.batches == []
    assert embedder.embed_passages([]) == []  # nothing to embed: no model needed
    assert embedder.identity.dimension == DIMENSION


def test_close_releases_the_encoder(model_dir: Path) -> None:
    encoder = RecordingEncoder()
    reference = weakref.ref(encoder)
    embedder = make_embedder(model_dir, encoder)
    embedder.embed_query("q")
    del encoder

    embedder.close()
    gc.collect()

    assert reference() is None


def test_close_lets_running_calls_finish_and_wakes_waiting_ones(model_dir: Path) -> None:
    encoder = GatedEncoder()
    embedder = make_embedder(model_dir, encoder, pool_size=1)
    outcomes: list[object] = []
    guard = threading.Lock()

    def call(query: bool) -> None:
        try:
            result: object = embedder.embed_query("q") if query else embedder.embed_passages(["p"])
        except ProcessingFailed as error:
            result = error.message
        with guard:
            outcomes.append(result)

    running = start_all(1, lambda: call(query=False)) + start_all(1, lambda: call(query=True))
    encoder.wait_for(2)  # the pool's slot and the reserved one
    waiting = start_all(1, lambda: call(query=False)) + start_all(1, lambda: call(query=True))
    wait_until(lambda: slot_counts(embedder)[2:] == (1, 1))  # a query and a passage wait
    embedder.close()
    wait_until(lambda: len(outcomes) == 2)
    assert outcomes == ["embedding model is closed"] * 2
    encoder.release.set()
    join_all(running + waiting)

    assert sorted(map(str, outcomes[2:])) == ["[1.0, 0.0, 0.0]", "[[1.0, 0.0, 0.0]]"]
    assert slot_counts(embedder) == (0, 0, 0, 0)


def test_a_close_between_the_slot_and_the_call_refuses_it(model_dir: Path) -> None:
    encoder = RecordingEncoder()
    embedder = make_embedder(model_dir, encoder)
    acquire = embedder._slots.acquire

    def acquire_then_close(lane: int = 0) -> bool:
        granted = acquire(lane)
        embedder.close()  # the close lands after the slot was granted
        return granted

    embedder._slots.acquire = acquire_then_close  # type: ignore[method-assign]

    with pytest.raises(ProcessingFailed, match="embedding model is closed"):
        embedder.embed_query("q")
    assert encoder.batches == []
    assert slot_counts(embedder) == (0, 0, 0, 0)


def test_loading_failures_are_configuration_errors(model_dir: Path) -> None:
    def failing(config: E5EmbedderConfig) -> LoadedEncoder:
        raise OSError("cannot read")

    with pytest.raises(ConfigurationError) as caught:
        E5Embedder(E5EmbedderConfig(model_dir, REVISION), model_factory=failing)

    assert caught.value.code == "invalid_configuration"
    assert isinstance(caught.value.__cause__, OSError)


def test_a_model_without_dimension_is_refused(model_dir: Path) -> None:
    def empty(config: E5EmbedderConfig) -> LoadedEncoder:
        return LoadedEncoder(0, lambda texts: [])

    with pytest.raises(ConfigurationError, match="dimension"):
        E5Embedder(E5EmbedderConfig(model_dir, REVISION), model_factory=empty)


def test_threads_are_set_after_a_successful_load(model_dir: Path) -> None:
    make_embedder(model_dir, RecordingEncoder(), threads=1)

    assert torch.get_num_threads() == 1


def test_a_failed_load_leaves_the_thread_count_unchanged(model_dir: Path) -> None:
    before = torch.get_num_threads()
    target = 1 if before != 1 else 2

    def failing(config: E5EmbedderConfig) -> LoadedEncoder:
        raise RuntimeError("broken")

    with pytest.raises(ConfigurationError):
        E5Embedder(E5EmbedderConfig(model_dir, REVISION, threads=target), model_factory=failing)

    assert torch.get_num_threads() == before


def test_mean_pool_ignores_padding() -> None:
    hidden = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]], [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    pooled = mean_pool(hidden, mask)

    assert pooled.tolist() == [[2.0, 3.0], [7.0, 8.0]]


def test_normalize_gives_unit_rows_and_keeps_zero_rows() -> None:
    vectors = normalize(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))

    assert vectors.tolist() == [pytest.approx([0.6, 0.8]), [0.0, 0.0]]
