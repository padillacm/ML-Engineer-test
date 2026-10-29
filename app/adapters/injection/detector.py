"""``InjectionDetector`` backed by a local Llama Prompt Guard 2 classifier.

Prompt Guard 2 is a sequence classifier (mDeBERTa) trained to tell an instruction aimed at a
language model (a prompt injection or jailbreak) from ordinary text. The adapter runs it with
``transformers`` alone: ``AutoModelForSequenceClassification`` and its fast tokenizer, loaded
once from ``InjectionDetectorConfig.model_path`` with ``local_files_only=True`` (nothing is
fetched from the network; a missing or unusable model fails creation with ``ConfigurationError``
instead of failing the first request), in ``eval`` mode under ``torch.inference_mode``.

Scoring: the model reads at most ``window_tokens`` tokens, so each text is tokenised into
windows of that size that overlap by ``stride_tokens`` (the tokenizer's overflow with stride: a
chunk of a scanned page fits one window, a long question or page a few), every window of a call
is scored in batches of ``batch_size`` and a text's score is the highest positive probability of
its windows (an injection in the last part of a long text is not diluted by the rest). The
positive probability is the softmax mass of the model's positive label(s): Prompt Guard 2's
``config.json`` names them ``LABEL_0`` (benign) and ``LABEL_1`` (malicious), the first Prompt
Guard named ``BENIGN``/``INJECTION``/``JAILBREAK``; ``positive_label_indexes`` accepts both
namings and falls back to index 1 of a two-label model (the convention of binary classifiers).
An empty or whitespace-only text scores 0 without running the model.

Thread safety: ``screen`` and ``screen_query`` may be called from several threads. Up to
``config.pool_size`` page batches run at once on the one loaded model, plus one more slot
reserved for questions (``screen_query``), so a question never waits for a page batch while that
slot is free; a waiting question takes the next free slot before any waiting page batch, as the
embedder's queries do (before the reserved slot, a question waited about 11.7 s behind 12-client
ingestion on the CPU, BENCHMARKS 23). Waiting calls of each kind get the slots in arrival order
(``FifoSlots``), and the model only runs forward under ``torch.inference_mode``, which torch
allows from several threads at once. The fast tokenizer is not thread-safe (a call sets its
truncation state in place; a concurrent use then fails), so tokenising and padding run under one
lock, and one warm-up call at load settles the truncation settings, which are the same on every
call afterwards; it is a small part of a call. ``close`` releases the model.

Threads: a successful creation calls ``torch.set_num_threads(config.threads)``, which is
process-wide: every torch computation in the process (the recogniser's and the embedder's
included) then uses that many intra-op threads, and the last torch adapter created wins, so the
service gives the three the same value (``TORCH_THREADS``). A failed creation leaves it
unchanged.

Device: the model runs on ``config.device``. On a GPU it is moved there once in half precision
(``app.adapters.device.place``), each batch of windows is moved to it, and the logits are cast to
float32 before the softmax, so the scores are computed in full precision on either device and
come back to the CPU. Tokenising and padding stay on the CPU, as do the threads above. On the
CPU the model is used as loaded, as before. A CUDA out-of-memory error during a batch fails the
call with ``detector_failed`` after the cached GPU memory is released, and the next call runs
normally.

Failures: any exception of the tokenizer or the model becomes ``ProcessingFailed`` (code
``detector_failed``) with a fixed message (never the text) and the original exception kept as
``__cause__``; a call on a closed detector, or still waiting for a slot when it is closed, raises
the same code.
"""

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from app.adapters.device import is_cuda, place, release_after_failure
from app.adapters.injection.config import InjectionDetectorConfig
from app.adapters.slots import FifoSlots
from app.domain.errors import ConfigurationError, ProcessingFailed, ServiceError

FAILURE_CODE = "detector_failed"
POSITIVE_LABELS = frozenset({"MALICIOUS", "INJECTION", "JAILBREAK", "UNSAFE", "LABEL_1"})
"""Label names (compared upper-cased) that count as the positive, injection, class."""
_BINARY = 2
"""A model with this many labels and no recognised name takes index 1 as positive."""


QUERY_LANE = 0
"""``FifoSlots`` lane of questions (``screen_query``): highest priority, may use every slot."""
PAGE_LANE = 1
"""``FifoSlots`` lane of page batches (``screen``): every slot but the one kept for questions."""


@dataclass(frozen=True, slots=True)
class WindowBatch:
    """One batch of tokenised windows, ready for a forward pass.

    ``inputs`` are the padded model inputs (``input_ids``, ``attention_mask``, ...) and
    ``owners[i]`` the index, in the texts given to ``windows``, of the text row ``i`` came from.
    """

    inputs: Mapping[str, torch.Tensor]
    owners: tuple[int, ...]


type Windows = Callable[[Sequence[str], int], Sequence[WindowBatch]]
"""Tokenises texts into windows, grouped in batches of at most the given size, in text order."""
type Score = Callable[[Mapping[str, torch.Tensor]], Sequence[float]]
"""Runs one batch of windows; returns the positive probability of each row, in order."""


@dataclass(frozen=True, slots=True)
class LoadedClassifier:
    """A loaded model: ``windows`` tokenises, ``score`` runs one batch of windows.

    Both may be called from several threads at once.
    """

    windows: Windows
    score: Score


type ClassifierFactory = Callable[[InjectionDetectorConfig], LoadedClassifier]


class PromptGuardDetector:
    """Scores texts for prompt injection with a local Prompt Guard 2 model.

    ``model_factory`` builds the classifier from the configuration; the default loads the local
    model with ``transformers``. Tests pass a stub. ``close`` releases the model.
    """

    def __init__(
        self,
        config: InjectionDetectorConfig,
        model_factory: ClassifierFactory | None = None,
    ) -> None:
        factory = load_classifier if model_factory is None else model_factory
        try:
            classifier = factory(config)
        except ServiceError:
            raise
        except Exception as error:  # every other loading failure is a configuration error
            raise ConfigurationError(
                f"injection detector model could not be loaded from {config.model_path}"
            ) from error
        torch.set_num_threads(config.threads)
        self._classifier: LoadedClassifier | None = classifier
        # Lanes in priority order: questions may use every slot, pages all but the reserved one.
        self._slots = FifoSlots(config.pool_size + 1, (config.pool_size + 1, config.pool_size))
        self._batch_size = config.batch_size

    @property
    def configured(self) -> bool:
        """A model backs the scores."""
        return True

    def screen(self, texts: Sequence[str]) -> list[float]:
        """Return the injection score (0..1) of each text, in the order given.

        A text is scored by the highest positive probability of its windows; a blank text scores
        0 without running the model. The windows of all texts run in batches of ``batch_size``,
        each batch taking a slot of the pool behind the batches that were waiting before it. Any
        failure of the tokenizer or the model raises ``ProcessingFailed`` (code
        ``detector_failed``) with a fixed message and the original exception as ``__cause__``;
        so does a call on a closed detector or one still waiting for a slot when it is closed.
        """
        return self._screen(texts, PAGE_LANE)

    def screen_query(self, text: str) -> float:
        """Return the injection score of one question (errors as for ``screen``).

        A question never waits for a page slot while the slot reserved for questions is free;
        waiting, it gets the next free slot before any waiting page batch.
        """
        return self._screen([text], QUERY_LANE)[0]

    def _screen(self, texts: Sequence[str], lane: int) -> list[float]:
        scores = [0.0] * len(texts)
        kept = [index for index, text in enumerate(texts) if text.strip()]
        if not kept:
            return scores
        classifier = self._classifier
        if classifier is None:
            raise _closed()
        try:
            batches = classifier.windows([texts[index] for index in kept], self._batch_size)
        except ServiceError:
            raise
        except Exception as error:  # every vendor failure becomes one fixed error
            raise _failure() from error
        for batch in batches:
            for owner, score in zip(batch.owners, self._run(classifier, batch, lane), strict=True):
                index = kept[owner]
                scores[index] = max(scores[index], score)
        return scores

    def close(self) -> None:
        """Release the model (idempotent); calls already running finish normally."""
        self._classifier = None
        self._slots.close()

    def _run(self, classifier: LoadedClassifier, batch: WindowBatch, lane: int) -> list[float]:
        if not self._slots.acquire(lane):
            raise _closed()
        try:
            if self._classifier is None:  # closed after the slot was granted
                raise _closed()
            scores = list(classifier.score(batch.inputs))
        except ServiceError:
            raise
        except Exception as error:  # every vendor failure becomes one fixed error
            release_after_failure(error)
            raise _failure() from error
        finally:
            self._slots.release(lane)
        if len(scores) != len(batch.owners):
            raise ProcessingFailed(
                "the injection detector returned the wrong number of scores", FAILURE_CODE
            )
        return [min(1.0, max(0.0, float(score))) for score in scores]


def _failure() -> ProcessingFailed:
    return ProcessingFailed("the injection detector failed", FAILURE_CODE)


def _closed() -> ProcessingFailed:
    return ProcessingFailed("the injection detector is closed", FAILURE_CODE)


def positive_label_indexes(id2label: Mapping[object, object]) -> tuple[int, ...]:
    """The label indexes whose softmax mass is the injection score, from ``config.id2label``.

    Labels named as in ``POSITIVE_LABELS`` (case-insensitive) are positive, so Prompt Guard 2's
    ``LABEL_1`` and the first Prompt Guard's ``INJECTION`` and ``JAILBREAK`` both work. A
    two-label model without a recognised name takes index 1 as positive, the usual order of a
    binary classifier. Raises ``ConfigurationError`` when no positive label can be told, or when
    every label would be positive.
    """
    labels = {int(str(index)): str(label).upper() for index, label in id2label.items()}
    positive = tuple(sorted(index for index, label in labels.items() if label in POSITIVE_LABELS))
    if not positive and len(labels) == _BINARY and 1 in labels:
        positive = (1,)
    if not positive or len(positive) == len(labels):
        raise ConfigurationError(
            f"injection detector model has no recognisable benign/malicious labels: {labels}"
        )
    return positive


def load_classifier(config: InjectionDetectorConfig) -> LoadedClassifier:
    """Load tokenizer and model from ``config.model_path`` (local files only).

    The classifier may be called from several threads: tokenising runs under one lock and the
    forward pass outside it (see the module docstring); one warm-up call settles the tokenizer's
    truncation settings before any concurrent use. Raises ``ConfigurationError`` when the
    tokenizer is not a fast one (overflowing windows with a stride need it), when
    ``window_tokens`` exceeds the tokenizer's input length, when the weights do not fit the
    declared architecture (missing, unexpected or mismatched tensors: ``transformers`` would
    otherwise initialise the missing ones at random and only log a warning) or when the labels
    give no positive class. Other loading errors are left to the caller.
    """
    from transformers import (  # noqa: PLC0415 - heavy import, needed only to load a model
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    path = config.model_path
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if not tokenizer.is_fast:
        raise ConfigurationError(f"injection detector model needs a fast tokenizer: {path}")
    if config.window_tokens > tokenizer.model_max_length:
        raise ConfigurationError(
            f"window_tokens must not exceed the model input length "
            f"({tokenizer.model_max_length} tokens)"
        )
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True, output_loading_info=True
    )
    problems = sorted(name for name, found in loading_info.items() if found)
    if problems:
        raise ConfigurationError(
            f"injection detector model weights do not match its configuration ({problems}): {path}"
        )
    model.eval()
    place(model, config.device)
    device = config.device
    on_gpu = is_cuda(device)
    positive = list(positive_label_indexes(model.config.id2label))
    window_tokens = config.window_tokens
    stride_tokens = config.stride_tokens
    tokenizer_lock = threading.Lock()

    def windows(texts: Sequence[str], batch_size: int) -> list[WindowBatch]:
        with tokenizer_lock:
            encoded = tokenizer(
                list(texts),
                truncation=True,
                max_length=window_tokens,
                stride=stride_tokens,
                return_overflowing_tokens=True,
            )
            owners = [int(owner) for owner in encoded["overflow_to_sample_mapping"]]
            rows = {
                key: value for key, value in encoded.items() if key != "overflow_to_sample_mapping"
            }
            batches = []
            for start in range(0, len(owners), batch_size):
                part = {key: value[start : start + batch_size] for key, value in rows.items()}
                padded = tokenizer.pad(part, return_tensors="pt")
                batches.append(
                    WindowBatch(dict(padded.items()), tuple(owners[start : start + batch_size]))
                )
        return batches

    def score(inputs: Mapping[str, torch.Tensor]) -> list[float]:
        if on_gpu:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits.float(), dim=-1)[:, positive].sum(dim=-1)
        result: list[float] = probabilities.cpu().tolist()
        return result

    # Every call passes the same tokenizer settings, so after this first call the tokenizer's
    # truncation settings are only read, never changed again.
    for batch in windows([_WARM_UP_TEXT], 1):
        score(batch.inputs)
    return LoadedClassifier(windows=windows, score=score)


_WARM_UP_TEXT = "The invoice was paid on time."
