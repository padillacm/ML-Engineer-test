"""Composition root: builds every adapter and use case once from ``Settings``.

This module (and ``app.main``, which calls it) is the only place that imports adapters. Every
adapter configuration is built, and so validated, before anything heavy is constructed, so a
bad value fails in milliseconds instead of after the models have loaded. A
``ConfigurationError`` raised by an adapter configuration, an adapter or a use case is raised
again with the environment variable(s) it concerns in front of the message ("NER_MODEL_PATH:
NER model directory not found: ..."); the messages are the fixed ones of the configs, which
never contain a secret.

``TORCH_THREADS`` is given to both the recogniser and the embedder configuration: torch's
intra-op thread count is process-wide, so each adapter's creation sets it for the whole process
and they must agree. When ``NER_POOL_SIZE`` x ``TORCH_THREADS`` (the CPUs concurrent
recognitions can use) exceeds the CPUs available to the process (``available_cpus``), one
warning is logged at start-up; nothing is changed.

Without ``LLM_BASE_URL`` the questions use case gets an ``UnconfiguredLLM`` and readiness
reports the language model as ``not_configured``. Nothing contacts the vector store here, so the
service starts while it is down (readiness then fails and the store recovers per call).
"""

import logging
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.adapters.embeddings.config import E5EmbedderConfig
from app.adapters.embeddings.embedder import E5Embedder
from app.adapters.llm.config import LLMClientConfig
from app.adapters.llm.openai_compatible import OpenAICompatibleClient
from app.adapters.llm.secret import Secret
from app.adapters.ner.config import TransformersNerConfig
from app.adapters.ner.recognizer import TransformersNameRecognizer
from app.adapters.ocr.engine import TesseractOcrConfig, TesseractOcrEngine
from app.adapters.vector_store.config import QdrantStoreConfig
from app.adapters.vector_store.qdrant_store import QdrantVectorStore
from app.application.ask_question import AskQuestion
from app.application.extract_names import ExtractNames
from app.application.ingest_document import IngestDocument
from app.application.ports import Embedder, LLMClient, NameRecognizer, OcrEngine, VectorStore
from app.application.unconfigured_llm import UnconfiguredLLM
from app.domain.chunking import ChunkingConfig
from app.domain.errors import ConfigurationError
from app.settings import Settings

type ReadinessCheck = Callable[[], str]
"""Returns the check's status ("ok", "configured", ...) when it passes and raises otherwise."""

LLM_CONFIGURED = "configured"
LLM_NOT_CONFIGURED = "not_configured"
STORE_OK = "ok"

CGROUP_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
"""The cgroup v2 CPU quota of the process's cgroup (``"<quota> <period>"`` or ``"max ..."``)."""
_CPU_MAX_FIELDS = 2

_logger = logging.getLogger(__name__)


class ManagedVectorStore(VectorStore, Protocol):
    """A ``VectorStore`` that can be probed for readiness and must be closed."""

    def ping(self) -> None: ...

    def close(self) -> None: ...


class ManagedNameRecognizer(NameRecognizer, Protocol):
    """A ``NameRecognizer`` holding a pool of model pipelines that must be released."""

    def close(self) -> None: ...


class ManagedLLMClient(LLMClient, Protocol):
    """An ``LLMClient`` holding connections that must be closed."""

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AdapterFactories:
    """How each adapter is built from its configuration; tests replace them with fakes."""

    ocr: Callable[[TesseractOcrConfig], OcrEngine] = TesseractOcrEngine
    recognizer: Callable[[TransformersNerConfig], ManagedNameRecognizer] = (
        TransformersNameRecognizer
    )
    embedder: Callable[[E5EmbedderConfig], Embedder] = E5Embedder
    store: Callable[[QdrantStoreConfig], ManagedVectorStore] = QdrantVectorStore
    llm: Callable[[LLMClientConfig], ManagedLLMClient] = OpenAICompatibleClient


@dataclass(frozen=True, slots=True)
class AdapterConfigs:
    """Every adapter configuration derived from ``Settings``; ``llm`` is ``None`` without one."""

    ocr: TesseractOcrConfig
    ner: TransformersNerConfig
    embedder: E5EmbedderConfig
    store: QdrantStoreConfig
    llm: LLMClientConfig | None
    chunking: ChunkingConfig


@dataclass(frozen=True, slots=True)
class Container:
    """The use cases the HTTP edge calls, its readiness checks, and what must be closed.

    ``readiness`` pairs a check name with a ``ReadinessCheck``. ``close`` runs ``closers`` in
    order; a closer that fails is logged and the others still run.
    """

    extract: ExtractNames
    ingest: IngestDocument
    ask: AskQuestion
    readiness: tuple[tuple[str, ReadinessCheck], ...]
    llm_configured: bool
    closers: tuple[Callable[[], None], ...] = field(default=())

    def close(self) -> None:
        """Close the recogniser (its pipeline pool), the language-model client and the vector
        store (once per container)."""
        _close_all(self.closers)


# Adapter config field -> environment variable, per configuration.
_OCR_VARIABLES = {
    "dpi": "OCR_DPI",
    "flatten": "OCR_FLATTEN",
    "timeout_s": "OCR_TIMEOUT_S",
    "workers": "OCR_WORKERS",
    "max_pages": "MAX_PAGES",
}
_NER_VARIABLES = {
    "model_path": "NER_MODEL_PATH",
    "threads": "TORCH_THREADS",
    "pool_size": "NER_POOL_SIZE",
}
_EMBEDDER_VARIABLES = {
    "model_path": "EMBEDDING_MODEL_PATH",
    "revision": "EMBEDDING_MODEL_REVISION",
    "batch_size": "EMBEDDING_BATCH_SIZE",
    "threads": "TORCH_THREADS",
}
_STORE_VARIABLES = {
    "url": "QDRANT_URL",
    "location": "QDRANT_LOCATION",
    "collection_prefix": "QDRANT_COLLECTION_PREFIX",
    "timeout_s": "QDRANT_TIMEOUT_S",
    "attempts": "QDRANT_ATTEMPTS",
    "deadline_s": "QDRANT_DEADLINE_S",
    "api_key": "QDRANT_API_KEY",
    "ping_timeout_s": "READINESS_TIMEOUT_S",
}
_LLM_VARIABLES = {
    "base_url": "LLM_BASE_URL",
    "model": "LLM_MODEL",
    "api_key": "LLM_API_KEY",
    "connect_timeout_s": "LLM_CONNECT_TIMEOUT_S",
    "read_timeout_s": "LLM_READ_TIMEOUT_S",
    "total_timeout_s": "LLM_TOTAL_TIMEOUT_S",
    "attempts": "LLM_ATTEMPTS",
    "max_tokens": "LLM_MAX_TOKENS",
}
_CHUNKING_VARIABLES = {"max_chars": "CHUNK_SIZE", "overlap_ratio": "CHUNK_OVERLAP"}
_ASK_VARIABLES = {
    "threshold": "RAG_THRESHOLD",
    "top_k": "RAG_TOP_K",
    "max_chunks": "RAG_MAX_CHUNKS",
    "max_context_chars": "RAG_MAX_CONTEXT_CHARS",
}


def adapter_configs(settings: Settings) -> AdapterConfigs:
    """Map ``settings`` onto the adapter configurations, validating each one.

    Raises ``ConfigurationError`` naming the variable(s) of the first invalid configuration.
    """
    return AdapterConfigs(
        ocr=_named(
            _OCR_VARIABLES,
            lambda: TesseractOcrConfig(
                dpi=settings.ocr_dpi,
                flatten=settings.ocr_flatten,
                timeout_s=settings.ocr_timeout_s,
                workers=settings.ocr_workers,
                max_pages=settings.max_pages,
            ),
        ),
        ner=_named(
            _NER_VARIABLES,
            lambda: TransformersNerConfig(
                model_path=settings.ner_model_path,
                threads=settings.torch_threads,
                pool_size=settings.ner_pool_size,
            ),
            fallback="NER_MODEL_PATH",
        ),
        embedder=_named(
            _EMBEDDER_VARIABLES,
            lambda: E5EmbedderConfig(
                model_path=settings.embedding_model_path,
                revision=settings.embedding_model_revision,
                batch_size=settings.embedding_batch_size,
                threads=settings.torch_threads,
            ),
            fallback="EMBEDDING_MODEL_PATH",
        ),
        store=_named(
            _STORE_VARIABLES, lambda: _store_config(settings), "QDRANT_URL, QDRANT_LOCATION"
        ),
        llm=_llm_config(settings),
        chunking=_named(
            _CHUNKING_VARIABLES,
            lambda: ChunkingConfig(
                max_chars=settings.chunk_size, overlap_ratio=settings.chunk_overlap
            ),
        ),
    )


def build_container(settings: Settings, factories: AdapterFactories | None = None) -> Container:
    """Build every adapter and use case once from ``settings``.

    ``factories`` replaces the adapter constructors (tests). Raises ``ConfigurationError``
    naming the variable(s) at fault; adapters already built are closed first.
    """
    make = factories if factories is not None else AdapterFactories()
    configs = adapter_configs(settings)
    _warn_if_oversubscribed(settings.ner_pool_size, settings.torch_threads, available_cpus())
    closers: list[Callable[[], None]] = []
    try:
        store = _named(_STORE_VARIABLES, lambda: make.store(configs.store))
        closers.append(store.close)
        llm: LLMClient = UnconfiguredLLM()
        if configs.llm is not None:
            llm_config = configs.llm
            client = _named(_LLM_VARIABLES, lambda: make.llm(llm_config), "LLM_BASE_URL")
            closers.insert(0, client.close)
            llm = client
        ocr = _named(_OCR_VARIABLES, lambda: make.ocr(configs.ocr))
        recognizer = _named(
            _NER_VARIABLES, lambda: make.recognizer(configs.ner), fallback="NER_MODEL_PATH"
        )
        closers.insert(0, recognizer.close)
        embedder = _named(
            _EMBEDDER_VARIABLES,
            lambda: make.embedder(configs.embedder),
            fallback="EMBEDDING_MODEL_PATH",
        )
        container = Container(
            extract=_named(
                {"threshold": "MATCH_THRESHOLD"},
                lambda: ExtractNames(ocr, recognizer, settings.match_threshold),
            ),
            ingest=_named(
                {"batch_size": "EMBEDDING_BATCH_SIZE"},
                lambda: IngestDocument(
                    ocr, embedder, store, configs.chunking, settings.embedding_batch_size
                ),
            ),
            ask=_named(_ASK_VARIABLES, lambda: _ask(settings, embedder, store, llm)),
            readiness=(
                ("vector_store", _store_check(store)),
                ("llm", _llm_check(configured=configs.llm is not None)),
            ),
            llm_configured=configs.llm is not None,
            closers=tuple(closers),
        )
    except BaseException:
        _close_all(closers)
        raise
    return container


def available_cpus(
    cpu_max: Path = CGROUP_CPU_MAX,
    affinity: Callable[[], int | None] | None = None,
    cpu_count: Callable[[], int | None] = os.cpu_count,
) -> int:
    """The whole CPUs this process may use, at least 1.

    The CPUs the process may run on (``os.sched_getaffinity`` where the platform has it,
    otherwise ``cpu_count``), lowered to the cgroup v2 quota in ``cpu_max`` when that file sets
    one (a container's ``--cpus``), rounded down. An unreadable or malformed ``cpu_max`` is
    ignored. ``affinity`` and ``cpu_count`` replace the platform calls (tests).
    """
    count = (affinity or _affinity)()
    cpus = count if count is not None else (cpu_count() or 1)
    quota = _cgroup_quota(cpu_max)
    if quota is not None:
        cpus = min(cpus, math.floor(quota))
    return max(1, cpus)


def _affinity() -> int | None:
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is None:
        return None
    try:
        return len(getaffinity(0))
    except OSError:
        return None


def _cgroup_quota(cpu_max: Path) -> float | None:
    """The CPUs allowed by a cgroup v2 ``cpu.max`` file, ``None`` without a (readable) quota."""
    try:
        fields = cpu_max.read_text(encoding="ascii").split()
    except (OSError, UnicodeDecodeError):
        return None
    if len(fields) != _CPU_MAX_FIELDS or fields[0] == "max":
        return None
    try:
        quota, period = int(fields[0]), int(fields[1])
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return quota / period


def _warn_if_oversubscribed(pool_size: int, threads: int, cpus: int) -> None:
    requested = pool_size * threads
    if requested > cpus:
        _logger.warning(
            "NER_POOL_SIZE x TORCH_THREADS exceeds the CPUs available to the process; "
            "concurrent recognitions will compete for CPUs",
            extra={"cpus_requested": requested, "cpus_available": cpus},
        )


def _store_config(settings: Settings) -> QdrantStoreConfig:
    key = settings.qdrant_api_key
    return QdrantStoreConfig(
        url=settings.qdrant_url,
        location=settings.qdrant_location,
        collection_prefix=settings.qdrant_collection_prefix,
        timeout_s=settings.qdrant_timeout_s,
        attempts=settings.qdrant_attempts,
        deadline_s=settings.qdrant_deadline_s,
        api_key=None if key is None else key.get_secret_value(),
        ping_timeout_s=settings.readiness_timeout_s,
    )


def _llm_config(settings: Settings) -> LLMClientConfig | None:
    base_url, model = settings.llm_base_url, settings.llm_model
    if base_url is None or model is None:
        return None
    key = settings.llm_api_key
    api_key = (
        None
        if key is None
        else _named({}, lambda: Secret(key.get_secret_value()), fallback="LLM_API_KEY")
    )
    return _named(
        _LLM_VARIABLES,
        lambda: LLMClientConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            connect_timeout_s=settings.llm_connect_timeout_s,
            read_timeout_s=settings.llm_read_timeout_s,
            total_timeout_s=settings.llm_total_timeout_s,
            attempts=settings.llm_attempts,
            max_tokens=settings.llm_max_tokens,
        ),
        fallback="LLM_BASE_URL",
    )


def _ask(settings: Settings, embedder: Embedder, store: VectorStore, llm: LLMClient) -> AskQuestion:
    return AskQuestion(
        embedder,
        store,
        llm,
        threshold=settings.rag_threshold,
        top_k=settings.rag_top_k,
        max_chunks=settings.rag_max_chunks,
        max_context_chars=settings.rag_max_context_chars,
    )


def _store_check(store: ManagedVectorStore) -> ReadinessCheck:
    def check() -> str:
        store.ping()
        return STORE_OK

    return check


def _llm_check(*, configured: bool) -> ReadinessCheck:
    status = LLM_CONFIGURED if configured else LLM_NOT_CONFIGURED
    return lambda: status


def _named[T](
    variables: Mapping[str, str], build: Callable[[], T], fallback: str | None = None
) -> T:
    """``build()``, with a ``ConfigurationError`` raised again naming its variable(s).

    The variable is the one of the field the message starts with ("dpi must be ...");
    otherwise ``fallback``, or every variable of ``variables``.
    """
    try:
        return build()
    except ConfigurationError as error:
        variable = _variable_for(error.message, variables, fallback)
        raise ConfigurationError(f"{variable}: {error.message}") from error


def _variable_for(message: str, variables: Mapping[str, str], fallback: str | None) -> str:
    for config_field, variable in variables.items():
        if message.startswith(f"{config_field} "):
            return variable
    return fallback if fallback is not None else ", ".join(variables.values())


def _close_all(closers: Sequence[Callable[[], None]]) -> None:
    for close in closers:
        try:
            close()
        except Exception:
            _logger.exception("closing a client failed")
