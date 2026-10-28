"""``E5Embedder`` with a stubbed encoder: prefixes, batching, limits, errors, lock, identity."""

import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import torch

from app.adapters.embeddings import E5Embedder, E5EmbedderConfig
from app.adapters.embeddings.embedder import LoadedEncoder, mean_pool, normalize
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
    model_dir: Path, encode: Callable[[Sequence[str]], list[list[float]]], **settings: int
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


def test_inference_is_serialised(model_dir: Path) -> None:
    active = 0
    peak = 0
    guard = threading.Lock()

    def slow(texts: Sequence[str]) -> list[list[float]]:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        return [[1.0, 0.0, 0.0] for _ in texts]

    embedder = make_embedder(model_dir, slow)
    workers = [threading.Thread(target=embedder.embed_query, args=("q",)) for _ in range(6)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert peak == 1


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
