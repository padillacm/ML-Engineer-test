"""``NameRecognizer`` backed by a local ``transformers`` token-classification model.

The model is loaded once, when the recogniser is created, from ``TransformersNerConfig.path``
with ``local_files_only=True``: nothing is fetched from the network, and a missing or unusable
model fails creation with ``ConfigurationError`` instead of failing the first request.

Thread safety: ``find_persons`` may be called from several threads. The recogniser keeps a pool
of ``config.pool_size`` pipeline objects (each with its own per-call state) that share one model
and one tokenizer. A call takes a free pipeline, waiting while all are busy, and always gives it
back; waiting calls get the pipelines in arrival order (``FifoSlots``), so a job that calls page
after page does not keep a pipeline from a job that waits. The model only runs forward in
``eval`` mode without gradients, which torch allows from several threads at once, so up to
``pool_size`` forward passes run in parallel. The fast
tokenizer is not thread-safe (it switches its truncation and padding settings in place, and a
concurrent use then fails), so every step of the pipelines that uses it (tokenising a text,
grouping the predicted tokens into entities) runs under one lock shared by the pool; those steps
are a small part of a call. A tokenizer per pipeline would avoid the lock but costs about 300 MB
of memory each for this model's 250,002-token vocabulary.

Threads: a successful creation calls ``torch.set_num_threads(config.threads)``, which is
process-wide. Every torch computation in the process (the embedder's and another recogniser's
included) then uses that many intra-op threads; the last torch adapter created wins, so the
service gives both the same value (``TORCH_THREADS``). A failed creation leaves the thread count
unchanged. ``pool_size`` concurrent calls use up to ``pool_size`` x ``threads``.

Device: the model runs on ``config.device``. On a GPU its weights are moved there once, in
float32 (``app.adapters.device.place`` with ``half=False``: in half precision its spans differed
from the CPU's at name boundaries on 6 of 1,030 corpus pages, design D2), and every pipeline of
the pool is built on that device over the same weights, so the pool holds one copy of them in
VRAM; the pipeline moves each window's tokens to the device and brings the logits back to the
CPU, in float32, before its aggregation (transformers 5 takes a given model's dtype as it is, so
the dtype is set on the model, not passed to the pipeline). Tokenising and grouping stay on the
CPU, as do the threads above. On the CPU the model is used as loaded, as before. A CUDA out-of-
memory error during a call fails it with ``ner_failed`` after the cached GPU memory is released,
and the next call runs normally.

Input size: a text longer than ``config.max_chars`` is refused before the model runs (inference
time grows linearly with the text and holds a pipeline; see ``TransformersNerConfig``).
"""

import threading
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, TypeGuard

import torch

from app.adapters.device import place, release_after_failure
from app.adapters.ner.config import TransformersNerConfig
from app.adapters.slots import FifoSlots
from app.domain.errors import ConfigurationError, ProcessingFailed, ServiceError
from app.domain.types import TextSpan

type Entity = Mapping[str, object]
type TokenClassifier = Callable[[str], Sequence[Entity]]
"""Runs the model on one text; returns aggregated entities with ``entity_group``/``start``/``end``.

``start``/``end`` are character offsets into the text. The ``word`` field is not used: the
pipeline rebuilds it from sub-word tokens and can glue words together ("MariaGonzalez").
"""
type PipelineFactory = Callable[[TransformersNerConfig], Sequence[TokenClassifier]]
"""Builds ``config.pool_size`` classifiers, each safe to run while the others run."""


class TransformersNameRecognizer:
    """Finds the spans of the configured entity groups (``PER``) in a text.

    ``pipeline_factory`` builds the pool of classifiers from the configuration; the default loads
    the local model with ``transformers``. Tests pass a stub. ``close`` releases the pipelines.
    """

    def __init__(
        self,
        config: TransformersNerConfig,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        factory = load_pipelines if pipeline_factory is None else pipeline_factory
        try:
            classifiers = list(factory(config))
        except ServiceError:
            raise
        except Exception as error:  # every other loading failure is a configuration error
            raise ConfigurationError(f"NER model could not be loaded from {config.path}") from error
        if len(classifiers) != config.pool_size:
            raise ConfigurationError(
                f"NER pipeline factory built {len(classifiers)} pipelines "
                f"instead of pool_size={config.pool_size}"
            )
        torch.set_num_threads(config.threads)
        self._max_chars = config.max_chars
        self._labels = frozenset(config.labels)
        self._free = classifiers
        self._slots = FifoSlots(config.pool_size)
        self._guard = threading.Lock()  # the free list and the closed flag
        self._closed = False

    def find_persons(self, text: str) -> list[TextSpan]:
        """Return the person-name spans of ``text`` in order, without overlaps.

        Spans are trimmed of surrounding whitespace and clipped to the text; spans that
        overlap are merged into one. A blank text returns ``[]`` and a text longer than
        ``max_chars`` raises ``ProcessingFailed`` (code ``ner_failed``), both without running
        the model. Any failure of the model raises ``ProcessingFailed`` (code ``ner_failed``)
        with a fixed message; the original exception is kept as ``__cause__``. A call on a
        closed recogniser, or still waiting for a pipeline when it is closed, raises
        ``ProcessingFailed`` (code ``ner_failed``) too.
        """
        if len(text) > self._max_chars:
            raise ProcessingFailed(
                f"text is longer than the NER limit of {self._max_chars} characters", "ner_failed"
            )
        if not text.strip():
            return []
        classify = self._acquire()
        try:
            entities = classify(text)
        except ServiceError:
            raise
        except Exception as error:  # every vendor failure becomes one fixed error
            release_after_failure(error)
            raise ProcessingFailed("NER model could not process the text", "ner_failed") from error
        finally:
            self._release(classify)
        return merge_spans(_label_spans(entities, text, self._labels))

    def close(self) -> None:
        """Release the pipelines (idempotent); calls already running finish normally.

        A pipeline in use when ``close`` runs is dropped as soon as its call returns.
        """
        with self._guard:
            self._closed = True
            self._free.clear()
        self._slots.close()

    def _acquire(self) -> TokenClassifier:
        """Wait in line for a slot of the pool, then take a free pipeline (one exists per slot)."""
        if self._slots.acquire():
            with self._guard:
                if not self._closed:
                    return self._free.pop()
            self._slots.release()  # closed after the slot was granted
        raise ProcessingFailed("NER model is closed", "ner_failed")

    def _release(self, classify: TokenClassifier) -> None:
        with self._guard:
            if not self._closed:
                self._free.append(classify)
        self._slots.release()


def load_pipelines(config: TransformersNerConfig) -> list[TokenClassifier]:
    """Load the model once from ``config.path`` (local files only) into ``pool_size`` pipelines.

    The pipelines share the model and the tokenizer; their tokenizer steps share one lock (see
    the module docstring), and one warm-up call settles the tokenizer's truncation and padding
    settings before any concurrent use. Raises ``ConfigurationError`` when the tokenizer is not a
    fast one (character offsets and ``stride`` need it), when ``stride`` does not fit the model
    input, when the weights do not fit the declared architecture (missing, unexpected or
    mismatched tensors: ``transformers`` would otherwise initialise the missing ones at random
    and only log a warning) or when the model has no entity tag for a configured label. Other
    loading errors are left to the caller.
    """
    from transformers import (  # noqa: PLC0415 - heavy import, needed only to load a model
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )

    path = str(config.path)
    # fix_mistral_regex=False: transformers flags this model's tokenizer as a Mistral one with a
    # broken regex because its config.json has no transformers_version, and warns. The flag is
    # wrong for XLM-R (applying the fix changes the token ids and the benchmark's results);
    # passing False explicitly keeps the tokenizer as it is and silences only that warning.
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, fix_mistral_regex=False)
    if not tokenizer.is_fast:
        raise ConfigurationError(f"NER model needs a fast tokenizer: {path}")
    max_length = tokenizer.model_max_length
    if config.stride >= max_length:
        raise ConfigurationError(
            f"stride must be smaller than the model input length ({max_length} tokens)"
        )
    model, loading_info = AutoModelForTokenClassification.from_pretrained(
        path, local_files_only=True, output_loading_info=True
    )
    problems = sorted(name for name, found in loading_info.items() if found)
    if problems:
        raise ConfigurationError(
            f"NER model weights do not match its configuration ({problems}): {path}"
        )
    model.eval()
    place(model, config.device, half=False)  # fp16 moved name boundaries (BENCHMARKS 23)
    groups = {_entity_group(str(tag)) for tag in model.config.id2label.values()}
    missing = sorted(set(config.labels) - groups)
    if missing:
        raise ConfigurationError(f"NER model has no entity tags for labels {missing}: {path}")
    # SentencePiece tokenizers (XLM-R) have no sub-word prefix, so with "first"/"average"/"max"
    # the pipeline decides word starts from the surrounding spaces and warns on every call.
    # That heuristic is the benchmarked behaviour; this one message is ignored process-wide
    # (``catch_warnings`` per call is not thread-safe). Adding an identical filter replaces the
    # existing one, so loading again does not grow the filter list.
    warnings.filterwarnings(
        "ignore",
        message="Tokenizer does not support real words",
        category=UserWarning,
        module=r"transformers\.pipelines\.token_classification",
    )
    tokenizer_lock = threading.Lock()
    classifiers = []
    for _ in range(config.pool_size):
        classifier = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy=config.aggregation,
            stride=config.stride,
            device=config.device,
        )
        _serialise_tokenizer_steps(classifier, tokenizer_lock)
        classifiers.append(_batched(classifier, config.batch_size))
    # Every call passes the same tokenizer settings, so after this first call the tokenizer's
    # truncation and padding are never changed again (only read), and the pipeline's one-time
    # set-up (the TOKENIZERS_PARALLELISM environment variable) has happened.
    classifiers[0](_WARM_UP_TEXT)
    return classifiers


_WARM_UP_TEXT = "Maria Gonzalez signed the agreement."


def _serialise_tokenizer_steps(classifier: Any, lock: threading.Lock) -> None:
    """Run the pipeline's ``preprocess`` (tokenising) and ``postprocess`` (grouping) under ``lock``.

    The pipeline looks both up on the instance when it is called, so instance attributes take
    the place of the methods; ``forward`` (the model) stays outside the lock. ``preprocess`` is
    a generator over the text's windows: the windows are all produced (one tokenizer call)
    inside the lock and handed out afterwards.
    """
    preprocess = classifier.preprocess
    postprocess = classifier.postprocess

    def locked_preprocess(*args: Any, **kwargs: Any) -> Iterator[Any]:
        with lock:
            windows = list(preprocess(*args, **kwargs))
        return iter(windows)

    def locked_postprocess(*args: Any, **kwargs: Any) -> Any:
        with lock:
            return postprocess(*args, **kwargs)

    classifier.preprocess = locked_preprocess
    classifier.postprocess = locked_postprocess


def _batched(classifier: Callable[..., Sequence[Entity]], batch_size: int) -> TokenClassifier:
    def classify(text: str) -> Sequence[Entity]:
        return classifier(text, batch_size=batch_size)

    return classify


def merge_spans(spans: Iterable[TextSpan]) -> list[TextSpan]:
    """Sort ``spans`` and merge each group of overlapping spans into their union.

    Spans that only touch (one ends where the next starts) stay separate.
    """
    merged: list[TextSpan] = []
    for span in sorted(spans, key=lambda span: (span.start, span.end)):
        if merged and span.start < merged[-1].end:
            last = merged[-1]
            merged[-1] = TextSpan(last.start, max(last.end, span.end))
        else:
            merged.append(span)
    return merged


def _label_spans(
    entities: Iterable[Entity], text: str, labels: frozenset[str]
) -> Iterable[TextSpan]:
    for entity in entities:
        if entity.get("entity_group") not in labels:
            continue
        start, end = entity.get("start"), entity.get("end")
        if _is_int(start) and _is_int(end):
            span = _trimmed(text, max(start, 0), min(end, len(text)))
            if span is not None:
                yield span


def _trimmed(text: str, start: int, end: int) -> TextSpan | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return TextSpan(start, end) if start < end else None


def _entity_group(tag: str) -> str:
    """``"B-PER"``/``"I-PER"`` -> ``"PER"``; tags without a prefix are their own group."""
    prefix, separator, group = tag.partition("-")
    return group if separator and prefix in {"B", "I"} else tag


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)
