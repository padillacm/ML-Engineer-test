"""The real loading and inference path (``load_encoder``) on a tiny local model, on the host."""

import json
import math
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from app.adapters.embeddings import E5Embedder, E5EmbedderConfig
from app.adapters.embeddings.embedder import load_encoder
from app.application.ports import Embedder
from app.domain.errors import ConfigurationError
from tests.adapters.embeddings.tiny_model import HIDDEN, MAX_LENGTH
from tests.application.contracts import EmbedderContract

REVISION = "tiny-rev"


@pytest.fixture(scope="module")
def tiny_embedder(tiny_model_dir: Path) -> E5Embedder:
    threads = torch.get_num_threads()
    try:
        return E5Embedder(E5EmbedderConfig(tiny_model_dir, REVISION, max_length=MAX_LENGTH))
    finally:
        torch.set_num_threads(threads)


@pytest.fixture
def copied_model(tiny_model_dir: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(tiny_model_dir, tmp_path / "model"))


def manual_vector(model_dir: Path, text: str) -> list[float]:
    """The e5 recipe written out by hand for one unpadded text: mean of token states, L2."""
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415 - reference only

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).eval()
    inputs = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        states = model(**inputs).last_hidden_state[0].tolist()
    mean = [sum(column) / len(states) for column in zip(*states, strict=True)]
    norm = math.hypot(*mean)
    return [value / norm for value in mean]


class TestTinyModelContract(EmbedderContract):
    @pytest.fixture
    def embedder(self, tiny_embedder: E5Embedder) -> Embedder:
        return tiny_embedder


def test_identity_uses_the_model_hidden_size(tiny_embedder: E5Embedder) -> None:
    assert tiny_embedder.identity.dimension == HIDDEN
    assert tiny_embedder.identity.model_id == f"intfloat/multilingual-e5-base@{REVISION}"


def test_passage_vector_is_the_prefixed_mean_pool(
    tiny_embedder: E5Embedder, tiny_model_dir: Path
) -> None:
    text = "the invoice was paid on monday ."

    vector = tiny_embedder.embed_passages([text])[0]

    assert vector == pytest.approx(manual_vector(tiny_model_dir, "passage: " + text), abs=1e-5)
    assert vector != pytest.approx(manual_vector(tiny_model_dir, text), abs=1e-3)


def test_query_vector_is_the_prefixed_mean_pool(
    tiny_embedder: E5Embedder, tiny_model_dir: Path
) -> None:
    text = "who signed the contract ?"

    vector = tiny_embedder.embed_query(text)

    assert vector == pytest.approx(manual_vector(tiny_model_dir, "query: " + text), abs=1e-5)


def test_padding_in_a_batch_does_not_change_a_vector(
    tiny_embedder: E5Embedder, tiny_model_dir: Path
) -> None:
    short, long = "alpha", "the invoice was paid on monday by a short contract of delta"

    vectors = tiny_embedder.embed_passages([short, long])

    assert vectors[0] == pytest.approx(manual_vector(tiny_model_dir, "passage: " + short), abs=1e-5)
    assert all(math.hypot(*vector) == pytest.approx(1.0) for vector in vectors)


def test_long_texts_are_truncated_at_max_length(tiny_embedder: E5Embedder) -> None:
    head = " ".join(["alpha beta"] * MAX_LENGTH)

    vectors = tiny_embedder.embed_passages([head, head + " gamma delta epsilon"])

    assert vectors[0] == pytest.approx(vectors[1], abs=1e-6)


def test_loading_sets_the_configured_threads(tiny_model_dir: Path) -> None:
    E5Embedder(E5EmbedderConfig(tiny_model_dir, REVISION, max_length=MAX_LENGTH, threads=1))

    assert torch.get_num_threads() == 1


def test_max_length_must_fit_the_tokenizer(tiny_model_dir: Path) -> None:
    config = E5EmbedderConfig(tiny_model_dir, REVISION, max_length=MAX_LENGTH + 1)

    with pytest.raises(ConfigurationError, match="max_length"):
        E5Embedder(config)


def test_unloadable_weights_are_invalid_configuration(copied_model: Path) -> None:
    (copied_model / "model.safetensors").write_bytes(b"not a safetensors file")

    with pytest.raises(ConfigurationError) as caught:
        E5Embedder(E5EmbedderConfig(copied_model, REVISION, max_length=MAX_LENGTH))

    assert caught.value.code == "invalid_configuration"
    assert caught.value.__cause__ is not None


def test_a_slow_tokenizer_is_refused(copied_model: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers  # noqa: PLC0415 - patched only in this test

    real = transformers.AutoTokenizer.from_pretrained

    def slow(*args: object, **kwargs: object) -> object:
        tokenizer = real(*args, **kwargs)
        monkeypatch.setattr(type(tokenizer), "is_fast", property(lambda self: False))
        return tokenizer

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", slow)

    with pytest.raises(ConfigurationError, match="fast tokenizer"):
        load_encoder(E5EmbedderConfig(copied_model, REVISION, max_length=MAX_LENGTH))


def test_unexpected_tensors_are_refused(copied_model: Path) -> None:
    # Weights of a different architecture leave tensors the declared model does not use.
    weights = copied_model / "model.safetensors"
    tensors = load_file(weights)
    tensors["classifier.weight"] = torch.zeros(2, HIDDEN)
    save_file(tensors, weights, metadata={"format": "pt"})

    with pytest.raises(ConfigurationError, match="unexpected_keys") as caught:
        E5Embedder(E5EmbedderConfig(copied_model, REVISION, max_length=MAX_LENGTH))

    assert caught.value.code == "invalid_configuration"


def test_tensors_of_the_wrong_shape_are_refused(copied_model: Path) -> None:
    config_file = copied_model / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config["intermediate_size"] = 3 * HIDDEN
    config_file.write_text(json.dumps(config), encoding="utf-8")

    # transformers itself refuses the load (no ignore_mismatched_sizes); the adapter would
    # refuse the mismatched_keys of its loading report otherwise.
    with pytest.raises(ConfigurationError) as caught:
        E5Embedder(E5EmbedderConfig(copied_model, REVISION, max_length=MAX_LENGTH))

    assert caught.value.code == "invalid_configuration"


def test_truncated_weights_are_refused(copied_model: Path) -> None:
    weights = copied_model / "model.safetensors"
    tensors = load_file(weights)
    removed = next(name for name in tensors if "encoder.layer.0.output.dense.weight" in name)
    del tensors[removed]
    save_file(tensors, weights, metadata={"format": "pt"})

    with pytest.raises(ConfigurationError, match="missing_keys"):
        E5Embedder(E5EmbedderConfig(copied_model, REVISION, max_length=MAX_LENGTH))


def test_a_failed_load_leaves_the_thread_count_unchanged(copied_model: Path) -> None:
    (copied_model / "model.safetensors").write_bytes(b"not a safetensors file")
    before = torch.get_num_threads()
    target = 1 if before != 1 else 2
    config = E5EmbedderConfig(copied_model, REVISION, max_length=MAX_LENGTH, threads=target)

    with pytest.raises(ConfigurationError):
        E5Embedder(config)

    assert torch.get_num_threads() == before


class Concurrency:
    """Counts how many calls of one kind run at once; each call lasts at least ``pause``."""

    def __init__(self, pause: float) -> None:
        self.pause = pause
        self.guard = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def wrap(self, function: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:
            with self.guard:
                self.active += 1
                self.calls += 1
                self.peak = max(self.peak, self.active)
            try:
                time.sleep(self.pause)
                return function(*args, **kwargs)
            finally:
                with self.guard:
                    self.active -= 1

        return counted


def _loaded(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[Any]]:
    """Record every tokenizer and model ``load_encoder`` loads."""
    import transformers  # noqa: PLC0415 - patched only in these tests

    tokenizers: list[Any] = []
    models: list[Any] = []
    real_tokenizer = transformers.AutoTokenizer.from_pretrained
    real_model = transformers.AutoModel.from_pretrained

    def tokenizer(*args: Any, **kwargs: Any) -> Any:
        tokenizers.append(real_tokenizer(*args, **kwargs))
        return tokenizers[-1]

    def model(*args: Any, **kwargs: Any) -> Any:
        loaded = real_model(*args, **kwargs)
        models.append(loaded[0] if isinstance(loaded, tuple) else loaded)
        return loaded

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", model)
    return tokenizers, models


def test_a_pool_loads_one_model_and_one_tokenizer(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizers, models = _loaded(monkeypatch)

    E5Embedder(E5EmbedderConfig(tiny_model_dir, REVISION, max_length=MAX_LENGTH, pool_size=4))

    assert (len(tokenizers), len(models)) == (1, 1)
    assert models[0].training is False


def test_loading_settles_the_tokenizer_settings(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizers, _ = _loaded(monkeypatch)
    embedder = E5Embedder(E5EmbedderConfig(tiny_model_dir, REVISION, max_length=MAX_LENGTH))
    backend = tokenizers[0]._tokenizer
    settled = (backend.truncation, backend.padding, backend.encode_special_tokens)

    embedder.embed_passages(["alpha beta", "the invoice was paid on monday by a short contract"])
    embedder.embed_query("who signed the contract ?")

    assert settled[0] is not None
    assert settled[0]["max_length"] == MAX_LENGTH
    assert settled[1] is not None
    assert (backend.truncation, backend.padding, backend.encode_special_tokens) == settled


def test_tokenising_is_serialised_while_forward_passes_overlap(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizers, models = _loaded(monkeypatch)
    embedder = E5Embedder(
        E5EmbedderConfig(tiny_model_dir, REVISION, max_length=MAX_LENGTH, pool_size=4)
    )
    tokenizing = Concurrency(pause=0.01)
    forward = Concurrency(pause=0.03)
    tokenizer_type = type(tokenizers[0])
    monkeypatch.setattr(tokenizer_type, "__call__", tokenizing.wrap(tokenizer_type.__call__))
    models[0].forward = forward.wrap(models[0].forward)
    texts = [f"passage number {index} of the contract ." for index in range(12)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda text: embedder.embed_passages([text]), texts))

    assert tokenizing.calls == forward.calls == 12
    assert tokenizing.peak == 1
    assert 1 < forward.peak <= 4


def test_concurrent_calls_on_a_pool_match_a_pool_of_one(tiny_model_dir: Path) -> None:
    words = ["the", "invoice", "was", "paid", "on", "monday"]
    words += ["by", "a", "short", "contract", "of", "delta"]
    texts = [" ".join(words[: 2 + index % 10]) for index in range(40)]
    batches = [texts[start : start + 1 + start % 3] for start in range(0, 40, 3)]
    single = E5Embedder(E5EmbedderConfig(tiny_model_dir, REVISION, max_length=MAX_LENGTH))
    pooled = E5Embedder(
        E5EmbedderConfig(tiny_model_dir, REVISION, max_length=MAX_LENGTH, pool_size=4)
    )
    expected = [single.embed_passages(batch) for batch in batches]
    expected_queries = [single.embed_query(text) for text in texts]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(pooled.embed_passages, batches * 3))
        queries = list(executor.map(pooled.embed_query, texts))

    for got, want in zip(results, expected * 3, strict=True):
        assert torch.allclose(torch.tensor(got), torch.tensor(want), rtol=0, atol=1e-6)
    assert torch.allclose(torch.tensor(queries), torch.tensor(expected_queries), rtol=0, atol=1e-6)
    pooled.close()
    single.close()
