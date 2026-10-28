"""``NameRecognizer`` backed by a local ``transformers`` token-classification model on CPU.

The model is loaded once, when the recogniser is created, from ``TransformersNerConfig.path``
with ``local_files_only=True``: nothing is fetched from the network, and a missing or unusable
model fails creation with ``ConfigurationError`` instead of failing the first request.

Thread safety: ``find_persons`` may be called from several threads; inference is serialised by
one lock per recogniser because the ``transformers`` pipeline is not documented as thread-safe.

Threads: a successful creation calls ``torch.set_num_threads(config.threads)``, which is
process-wide. Every torch computation in the process (including another recogniser) then uses
that many intra-op threads; the last recogniser created wins. A failed creation leaves the
thread count unchanged.

Input size: a text longer than ``config.max_chars`` is refused before the model runs (inference
time grows linearly with the text and holds the lock; see ``TransformersNerConfig``).
"""

import threading
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeGuard

import torch

from app.adapters.ner.config import TransformersNerConfig
from app.domain.errors import ConfigurationError, ProcessingFailed, ServiceError
from app.domain.types import TextSpan

type Entity = Mapping[str, object]
type TokenClassifier = Callable[[str], Sequence[Entity]]
"""Runs the model on one text; returns aggregated entities with ``entity_group``/``start``/``end``.

``start``/``end`` are character offsets into the text. The ``word`` field is not used: the
pipeline rebuilds it from sub-word tokens and can glue words together ("MariaGonzalez").
"""
type PipelineFactory = Callable[[TransformersNerConfig], TokenClassifier]


class TransformersNameRecognizer:
    """Finds the spans of the configured entity groups (``PER``) in a text.

    ``pipeline_factory`` builds the classifier from the configuration; the default loads the
    local model with ``transformers``. Tests pass a stub.
    """

    def __init__(
        self,
        config: TransformersNerConfig,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        factory = load_pipeline if pipeline_factory is None else pipeline_factory
        try:
            self._classify = factory(config)
        except ServiceError:
            raise
        except Exception as error:  # every other loading failure is a configuration error
            raise ConfigurationError(f"NER model could not be loaded from {config.path}") from error
        torch.set_num_threads(config.threads)
        self._max_chars = config.max_chars
        self._labels = frozenset(config.labels)
        self._lock = threading.Lock()

    def find_persons(self, text: str) -> list[TextSpan]:
        """Return the person-name spans of ``text`` in order, without overlaps.

        Spans are trimmed of surrounding whitespace and clipped to the text; spans that
        overlap are merged into one. A blank text returns ``[]`` and a text longer than
        ``max_chars`` raises ``ProcessingFailed`` (code ``ner_failed``), both without running
        the model. Any failure of the model raises ``ProcessingFailed`` (code ``ner_failed``)
        with a fixed message; the original exception is kept as ``__cause__``.
        """
        if len(text) > self._max_chars:
            raise ProcessingFailed(
                f"text is longer than the NER limit of {self._max_chars} characters", "ner_failed"
            )
        if not text.strip():
            return []
        with self._lock:
            try:
                entities = self._classify(text)
            except ServiceError:
                raise
            except Exception as error:  # every vendor failure becomes one fixed error
                raise ProcessingFailed(
                    "NER model could not process the text", "ner_failed"
                ) from error
        return merge_spans(_label_spans(entities, text, self._labels))


def load_pipeline(config: TransformersNerConfig) -> TokenClassifier:
    """Load tokenizer and model from ``config.path`` (local files only) into a pipeline.

    Raises ``ConfigurationError`` when the tokenizer is not a fast one (character offsets and
    ``stride`` need it), when ``stride`` does not fit the model input, when the weights do not
    fit the declared architecture (missing, unexpected or mismatched tensors: ``transformers``
    would otherwise initialise the missing ones at random and only log a warning) or when the
    model has no entity tag for a configured label. Other loading errors are left to the caller.
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
    groups = {_entity_group(str(tag)) for tag in model.config.id2label.values()}
    missing = sorted(set(config.labels) - groups)
    if missing:
        raise ConfigurationError(f"NER model has no entity tags for labels {missing}: {path}")
    classifier = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy=config.aggregation,
        stride=config.stride,
        device="cpu",
    )
    # SentencePiece tokenizers (XLM-R) have no sub-word prefix, so with "first"/"average"/"max"
    # the pipeline decides word starts from the surrounding spaces and warns on every call.
    # That heuristic is the benchmarked behaviour; this one message is ignored process-wide
    # (``catch_warnings`` per call is not thread-safe).
    warnings.filterwarnings(
        "ignore",
        message="Tokenizer does not support real words",
        category=UserWarning,
        module=r"transformers\.pipelines\.token_classification",
    )
    batch_size = config.batch_size

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
