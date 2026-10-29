"""The three torch adapters on the GPU against the CPU (``cuda``: needs a visible CUDA GPU).

Spec inference-device, "Results do not depend on the device": on the GPU the models run in
half precision (the recogniser in float32), so the outputs are compared with the CPU's, not with
fixed values. The tiny
hand-wired models run wherever a GPU is visible; the real models run when their directories are
named too (``NER_MODEL_PATH``, ``EMBEDDING_MODEL_PATH``, ``INJECTION_MODEL_PATH``; the NER and
e5 models are only in the images): the NER spans must be equal on the sample texts, every e5
vector within a cosine of 0.999 of the CPU's, and the detector must flag the same cases of the
adversarial suite.
"""

import math
import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from app.adapters.embeddings import E5Embedder, E5EmbedderConfig
from app.adapters.injection import InjectionDetectorConfig, PromptGuardDetector
from app.adapters.ner import TransformersNameRecognizer, TransformersNerConfig
from app.domain.guardrails import DEFAULT_INJECTION_THRESHOLD
from tests.adapters.embeddings.tiny_model import MAX_LENGTH as TINY_E5_LENGTH
from tests.adapters.embeddings.tiny_model import build_tiny_model as build_tiny_e5
from tests.adapters.injection.tiny_model import MAX_LENGTH as TINY_WINDOW
from tests.adapters.injection.tiny_model import STRIDE as TINY_STRIDE
from tests.adapters.injection.tiny_model import build_tiny_model as build_tiny_detector
from tests.adapters.ner.tiny_model import build_tiny_model as build_tiny_ner
from tests.adversarial.cases import (
    BENIGN,
    BENIGN_QUESTIONS,
    DIRECT,
    INDIRECT,
    INDIRECT_NOISY,
    Benign,
    Case,
)

pytestmark = pytest.mark.cuda

GPU = "cuda:0"
COSINE_MIN = 0.999
"""The least cosine between a GPU vector and the CPU vector of the same text (the spec's)."""
TINY_COSINE_MIN = 0.99
"""The tiny seeded encoder's random weights are not trained to be fp16-stable; looser bound."""
TINY_SCORE_TOLERANCE = 0.02
"""The largest difference allowed between a tiny-classifier score on the GPU and on the CPU."""

TINY_NER_TEXTS = (
    "Maria Gonzalez of Acme Corporation met in Madrid on 3 March. " * 6
    + "The agreement was signed by Peter Smith.",
    "the agreement was signed by maria gonzalez and peter smith in madrid",
    "Acme Corporation signed on 3 March.",
)
NAMED_TEXTS = (
    "Maria Gonzalez of Acme Corporation met in Madrid on 3 March.",
    "El contrato fue firmado por Juan Pérez y Ana Martínez en Sevilla el 4 de mayo.",
    "O contrato foi assinado por João Silva e Maria Santos em Lisboa.",
    "Le contrat a été signé par Jean Dupont et Marie Curie à Lyon.",
    "Der Vertrag wurde von Hans Müller und Anna Schmidt in Berlin unterschrieben.",
)
"""Sentences with names in each of the five languages, next to the adversarial benign texts."""
SAMPLE_TEXTS = NAMED_TEXTS + tuple(item.text for item in BENIGN)
_ADVERSARIAL: tuple[Case | Benign, ...] = (
    *DIRECT,
    *INDIRECT,
    *INDIRECT_NOISY,
    *BENIGN,
    *BENIGN_QUESTIONS,
)
ADVERSARIAL_TEXTS = tuple(item.text for item in _ADVERSARIAL)


@pytest.fixture(autouse=True)
def _restore_torch_threads() -> Iterator[None]:
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


def _env_directory(name: str) -> Path:
    value = os.environ.get(name, "")
    if not value or not Path(value).is_dir():
        pytest.skip(f"{name} unset or not a directory (the real model is only in the images)")
    return Path(value)


def _spans(recognizer: TransformersNameRecognizer, texts: Sequence[str]) -> list[list[str]]:
    return [[text[s.start : s.end] for s in recognizer.find_persons(text)] for text in texts]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (math.hypot(*left) * math.hypot(*right))


def _both[T](build: Callable[[str], T]) -> tuple[T, T]:
    return build("cpu"), build(GPU)


def _recognizers(path: Path, **settings: Any) -> tuple[TransformersNameRecognizer, ...]:
    return _both(
        lambda device: TransformersNameRecognizer(
            TransformersNerConfig(path, pool_size=1, device=device, **settings)
        )
    )


def _embedders(path: Path, **settings: Any) -> tuple[E5Embedder, ...]:
    return _both(
        lambda device: E5Embedder(
            E5EmbedderConfig(path, "parity", pool_size=1, device=device, **settings)
        )
    )


def _detectors(path: Path, **settings: Any) -> tuple[PromptGuardDetector, ...]:
    return _both(
        lambda device: PromptGuardDetector(
            InjectionDetectorConfig(str(path), pool_size=1, device=device, **settings)
        )
    )


def _assert_vectors_close(cpu: E5Embedder, gpu: E5Embedder, minimum: float) -> None:
    texts = list(SAMPLE_TEXTS)
    pairs = [
        *zip(cpu.embed_passages(texts), gpu.embed_passages(texts), strict=True),
        *((cpu.embed_query(text), gpu.embed_query(text)) for text in texts[:10]),
    ]
    cosines = [_cosine(left, right) for left, right in pairs]
    assert min(cosines) >= minimum, f"lowest cosine {min(cosines):.6f}"
    assert cpu.identity == gpu.identity


def test_the_gpu_is_visible() -> None:
    assert torch.cuda.is_available()


def test_tiny_ner_spans_are_equal_on_the_gpu(tmp_path: Path) -> None:
    cpu, gpu = _recognizers(build_tiny_ner(tmp_path), stride=4)

    assert _spans(gpu, TINY_NER_TEXTS) == _spans(cpu, TINY_NER_TEXTS)
    assert _spans(gpu, TINY_NER_TEXTS)[1] == ["maria gonzalez", "peter smith"]


def test_tiny_e5_vectors_are_close_on_the_gpu(tmp_path: Path) -> None:
    cpu, gpu = _embedders(build_tiny_e5(tmp_path), max_length=TINY_E5_LENGTH)

    _assert_vectors_close(cpu, gpu, TINY_COSINE_MIN)
    assert all(math.isclose(math.hypot(*v), 1.0, rel_tol=1e-5) for v in gpu.embed_passages(["x"]))


def test_tiny_detector_scores_are_close_on_the_gpu(tmp_path: Path) -> None:
    cpu, gpu = _detectors(
        build_tiny_detector(tmp_path), window_tokens=TINY_WINDOW, stride_tokens=TINY_STRIDE
    )
    texts = list(ADVERSARIAL_TEXTS)

    differences = [abs(a - b) for a, b in zip(cpu.screen(texts), gpu.screen(texts), strict=True)]

    assert max(differences) <= TINY_SCORE_TOLERANCE


def test_the_gpu_models_hold_half_precision_weights(tmp_path: Path) -> None:
    from transformers import AutoModel  # noqa: PLC0415 - only to inspect the placement

    from app.adapters.device import place  # noqa: PLC0415

    model = AutoModel.from_pretrained(str(build_tiny_e5(tmp_path)), local_files_only=True)
    place(model, GPU)

    parameter = next(model.parameters())
    assert (parameter.device.type, parameter.dtype) == ("cuda", torch.float16)


@pytest.mark.ner_model
def test_real_ner_spans_are_equal_on_the_gpu() -> None:
    cpu, gpu = _recognizers(_env_directory("NER_MODEL_PATH"))

    assert _spans(gpu, SAMPLE_TEXTS) == _spans(cpu, SAMPLE_TEXTS)


@pytest.mark.embedding_model
def test_real_e5_vectors_are_within_the_cosine_bound_on_the_gpu() -> None:
    cpu, gpu = _embedders(_env_directory("EMBEDDING_MODEL_PATH"))

    _assert_vectors_close(cpu, gpu, COSINE_MIN)


@pytest.mark.injection_model
def test_real_detector_flags_the_same_adversarial_cases_on_the_gpu() -> None:
    cpu, gpu = _detectors(_env_directory("INJECTION_MODEL_PATH"))
    texts = list(ADVERSARIAL_TEXTS)

    cpu_scores, gpu_scores = cpu.screen(texts), gpu.screen(texts)

    flipped = [
        (text[:60], round(a, 4), round(b, 4))
        for text, a, b in zip(texts, cpu_scores, gpu_scores, strict=True)
        if (a >= DEFAULT_INJECTION_THRESHOLD) != (b >= DEFAULT_INJECTION_THRESHOLD)
    ]
    largest = max(abs(a - b) for a, b in zip(cpu_scores, gpu_scores, strict=True))
    assert flipped == [], f"largest score difference {largest:.4f}"
