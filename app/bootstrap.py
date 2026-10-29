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
and they must agree. Unset, it is ``DEFAULT_TORCH_THREADS`` lowered to the CPUs available.

Concurrency: ``MAX_CONCURRENT_JOBS``, ``NER_POOL_SIZE``, ``EMBEDDING_POOL_SIZE`` and
``OCR_WORKERS`` that are not set are derived from the CPUs available to the process
(``available_cpus``: affinity and the cgroup v2 or v1 quota) by ``concurrency_for``; a set value
is used as given. ``resolve_concurrency`` is the one place that does this, and every consumer (the
adapter configurations here, the HTTP job limiter in ``app.main``) uses its result. Building the
container logs the values in use in one line, and one warning per pool whose size x
``TORCH_THREADS`` (the CPUs its concurrent calls can use) exceeds the available CPUs; nothing is
changed. The pools are checked one by one: a job uses one of them at a time, and the job limit
bounds them together.

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
from app.settings import DEFAULT_TORCH_THREADS, Settings

type ReadinessCheck = Callable[[], str]
"""Returns the check's status ("ok", "configured", ...) when it passes and raises otherwise."""

LLM_CONFIGURED = "configured"
LLM_NOT_CONFIGURED = "not_configured"
STORE_OK = "ok"

MAX_DERIVED_JOBS = 12
"""Largest derived ``MAX_CONCURRENT_JOBS``: the measured best on 16 CPUs (BENCHMARKS 15.6)."""
MAX_DERIVED_NER_POOL = 8
"""Largest derived ``NER_POOL_SIZE``: the measured best on 16 CPUs (BENCHMARKS 15.6)."""
MAX_DERIVED_EMBEDDING_POOL = 4
"""Largest derived ``EMBEDDING_POOL_SIZE``: the smallest pool within 5 % of the best ingestion
throughput on 16 CPUs (BENCHMARKS 18.5)."""
MAX_DERIVED_OCR_WORKERS = 4
"""Largest derived ``OCR_WORKERS``: the adapter's own ``min(4, CPUs)`` (BENCHMARKS 15.2)."""

CGROUP_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
"""The cgroup v2 CPU quota of the process's cgroup (``"<quota> <period>"`` or ``"max ..."``)."""
CGROUP_V1_CPU = Path("/sys/fs/cgroup/cpu")
"""The cgroup v1 CPU controller: ``cpu.cfs_quota_us`` (``-1`` without a quota) and
``cpu.cfs_period_us``, read when there is no cgroup v2 ``cpu.max``."""
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


class ManagedEmbedder(Embedder, Protocol):
    """An ``Embedder`` holding a loaded model that must be released."""

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Concurrency:
    """How much work runs at once, resolved from ``Settings`` and the CPUs (``cpus``).

    ``torch_threads`` torch intra-op threads, ``max_concurrent_jobs`` extraction and ingestion
    jobs, ``ner_pool_size`` concurrent recognitions, ``embedding_pool_size`` concurrent passage
    batches and ``ocr_workers`` Tesseract processes per document.
    """

    cpus: int
    max_concurrent_jobs: int
    ner_pool_size: int
    embedding_pool_size: int
    ocr_workers: int
    torch_threads: int = DEFAULT_TORCH_THREADS


@dataclass(frozen=True, slots=True)
class AdapterFactories:
    """How each adapter is built from its configuration; tests replace them with fakes."""

    ocr: Callable[[TesseractOcrConfig], OcrEngine] = TesseractOcrEngine
    recognizer: Callable[[TransformersNerConfig], ManagedNameRecognizer] = (
        TransformersNameRecognizer
    )
    embedder: Callable[[E5EmbedderConfig], ManagedEmbedder] = E5Embedder
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
        """Close the embedder and the recogniser (their models), the language-model client and
        the vector store (once per container)."""
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
    "pool_size": "EMBEDDING_POOL_SIZE",
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
    "reasoning_effort": "LLM_REASONING_EFFORT",
    "max_tokens_field": "LLM_MAX_TOKENS_FIELD",
    "send_temperature": "LLM_SEND_TEMPERATURE",
}
_CHUNKING_VARIABLES = {"max_chars": "CHUNK_SIZE", "overlap_ratio": "CHUNK_OVERLAP"}
_ASK_VARIABLES = {
    "threshold": "RAG_THRESHOLD",
    "top_k": "RAG_TOP_K",
    "max_chunks": "RAG_MAX_CHUNKS",
    "max_context_chars": "RAG_MAX_CONTEXT_CHARS",
}


def adapter_configs(settings: Settings, concurrency: Concurrency | None = None) -> AdapterConfigs:
    """Map ``settings`` onto the adapter configurations, validating each one.

    ``concurrency`` defaults to ``resolve_concurrency(settings)``. Raises ``ConfigurationError``
    naming the variable(s) of the first invalid configuration.
    """
    sizes = resolve_concurrency(settings) if concurrency is None else concurrency
    return AdapterConfigs(
        ocr=_named(
            _OCR_VARIABLES,
            lambda: TesseractOcrConfig(
                dpi=settings.ocr_dpi,
                flatten=settings.ocr_flatten,
                timeout_s=settings.ocr_timeout_s,
                workers=sizes.ocr_workers,
                max_pages=settings.max_pages,
            ),
        ),
        ner=_named(
            _NER_VARIABLES,
            lambda: TransformersNerConfig(
                model_path=settings.ner_model_path,
                threads=sizes.torch_threads,
                pool_size=sizes.ner_pool_size,
            ),
            fallback="NER_MODEL_PATH",
        ),
        embedder=_named(
            _EMBEDDER_VARIABLES,
            lambda: E5EmbedderConfig(
                model_path=settings.embedding_model_path,
                revision=settings.embedding_model_revision,
                batch_size=settings.embedding_batch_size,
                threads=sizes.torch_threads,
                pool_size=sizes.embedding_pool_size,
            ),
            fallback="EMBEDDING_MODEL_PATH",
        ),
        store=_named(
            _store_variables(settings),
            lambda: _store_config(settings),
            "QDRANT_URL, QDRANT_LOCATION",
        ),
        llm=_llm_config(settings),
        chunking=_named(
            _CHUNKING_VARIABLES,
            lambda: ChunkingConfig(
                max_chars=settings.chunk_size, overlap_ratio=settings.chunk_overlap
            ),
        ),
    )


def build_container(
    settings: Settings,
    factories: AdapterFactories | None = None,
    concurrency: Concurrency | None = None,
) -> Container:
    """Build every adapter and use case once from ``settings``.

    ``factories`` replaces the adapter constructors (tests); ``concurrency`` defaults to
    ``resolve_concurrency(settings)``. Logs the concurrency in use (and any oversubscription
    warning) once. Raises ``ConfigurationError`` naming the variable(s) at fault; adapters
    already built are closed first.
    """
    make = factories if factories is not None else AdapterFactories()
    sizes = resolve_concurrency(settings) if concurrency is None else concurrency
    configs = adapter_configs(settings, sizes)
    _log_concurrency(sizes)
    closers: list[Callable[[], None]] = []
    try:
        store = _named(_store_variables(settings), lambda: make.store(configs.store))
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
        closers.insert(0, embedder.close)
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


def concurrency_for(cpus: int, torch_threads: int) -> Concurrency:
    """The concurrency derived for ``cpus`` available CPUs and ``torch_threads``: the rule.

    - jobs: three quarters of the CPUs rounded up, at most ``MAX_DERIVED_JOBS``;
    - NER pool: CPUs / ``torch_threads`` rounded down, at most ``MAX_DERIVED_NER_POOL``;
    - embedder pool: CPUs / (2 x ``torch_threads``) rounded down, at most
      ``MAX_DERIVED_EMBEDDING_POOL``;
    - OCR workers: the CPUs, at most ``MAX_DERIVED_OCR_WORKERS``;

    each at least 1. With 2 threads this gives the measured sizing (BENCHMARKS 17.4 and 18):
    16 CPUs 12 / 8 / 4 / 4, 8 CPUs 6 / 4 / 2 / 4, 4 CPUs 3 / 2 / 1 / 4. Jobs round up because
    one job too few cost more than one too many where measured (4 CPUs: 2 jobs -22 %, 4 jobs
    -3 %); the pools round down so that a derived pool never exceeds the CPUs (no warning).
    """
    jobs = (3 * cpus + 3) // 4
    return Concurrency(
        cpus=cpus,
        torch_threads=torch_threads,
        max_concurrent_jobs=_at_least_one(min(jobs, MAX_DERIVED_JOBS)),
        ner_pool_size=_at_least_one(min(cpus // torch_threads, MAX_DERIVED_NER_POOL)),
        embedding_pool_size=_at_least_one(
            min(cpus // (2 * torch_threads), MAX_DERIVED_EMBEDDING_POOL)
        ),
        ocr_workers=_at_least_one(min(cpus, MAX_DERIVED_OCR_WORKERS)),
    )


def resolve_concurrency(settings: Settings, cpus: int | None = None) -> Concurrency:
    """The concurrency in use: each of ``settings``' four values as set, or else derived.

    Unset values come from ``concurrency_for`` with ``cpus`` (default ``available_cpus()``)
    and ``TORCH_THREADS`` (unset: ``DEFAULT_TORCH_THREADS``, at most ``cpus``); a set value
    always wins.
    """
    available = available_cpus() if cpus is None else cpus
    threads = _given(settings.torch_threads, min(DEFAULT_TORCH_THREADS, available))
    derived = concurrency_for(available, threads)
    return Concurrency(
        cpus=available,
        torch_threads=threads,
        max_concurrent_jobs=_given(settings.max_concurrent_jobs, derived.max_concurrent_jobs),
        ner_pool_size=_given(settings.ner_pool_size, derived.ner_pool_size),
        embedding_pool_size=_given(settings.embedding_pool_size, derived.embedding_pool_size),
        ocr_workers=_given(settings.ocr_workers, derived.ocr_workers),
    )


def _at_least_one(value: int) -> int:
    return max(1, value)


def _given(value: int | None, derived: int) -> int:
    return derived if value is None else value


def available_cpus(
    cpu_max: Path = CGROUP_CPU_MAX,
    affinity: Callable[[], int | None] | None = None,
    cpu_count: Callable[[], int | None] = os.cpu_count,
    cgroup_v1_cpu: Path = CGROUP_V1_CPU,
) -> int:
    """The whole CPUs this process may use, at least 1.

    The CPUs the process may run on (``os.sched_getaffinity`` where the platform has it,
    otherwise ``cpu_count``), lowered to the cgroup quota (a container's ``--cpus``), rounded
    down: the cgroup v2 ``cpu_max`` file when it exists, otherwise the cgroup v1
    ``cpu.cfs_quota_us`` and ``cpu.cfs_period_us`` in ``cgroup_v1_cpu``. Unreadable or malformed
    files are ignored. ``affinity`` and ``cpu_count`` replace the platform calls (tests).
    """
    count = (affinity or _affinity)()
    cpus = count if count is not None else (cpu_count() or 1)
    quota = _cgroup_quota(cpu_max) if cpu_max.exists() else _cgroup_v1_quota(cgroup_v1_cpu)
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
    return _quota(fields[0], fields[1])


def _cgroup_v1_quota(directory: Path) -> float | None:
    """The CPUs allowed by the cgroup v1 CFS files, ``None`` without a (readable) quota."""
    try:
        quota = (directory / "cpu.cfs_quota_us").read_text(encoding="ascii")
        period = (directory / "cpu.cfs_period_us").read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None
    return _quota(quota.strip(), period.strip())


def _quota(quota_text: str, period_text: str) -> float | None:
    """``quota / period`` in CPUs; ``None`` when either is not a positive integer (v1: ``-1``)."""
    try:
        quota, period = int(quota_text), int(period_text)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return quota / period


_OVERSUBSCRIBED = {
    "NER_POOL_SIZE": (
        "NER_POOL_SIZE x TORCH_THREADS exceeds the CPUs available to the process; "
        "concurrent recognitions will compete for CPUs"
    ),
    "EMBEDDING_POOL_SIZE": (
        "EMBEDDING_POOL_SIZE x TORCH_THREADS exceeds the CPUs available to the process; "
        "concurrent embeddings will compete for CPUs"
    ),
}


def _log_concurrency(concurrency: Concurrency) -> None:
    threads = concurrency.torch_threads
    _logger.info(
        "concurrency in use",
        extra={
            "cpus_available": concurrency.cpus,
            "torch_threads": threads,
            "max_concurrent_jobs": concurrency.max_concurrent_jobs,
            "ner_pool_size": concurrency.ner_pool_size,
            "embedding_pool_size": concurrency.embedding_pool_size,
            "ocr_workers": concurrency.ocr_workers,
        },
    )
    pools = {
        "NER_POOL_SIZE": concurrency.ner_pool_size,
        "EMBEDDING_POOL_SIZE": concurrency.embedding_pool_size,
    }
    for variable, pool_size in pools.items():
        requested = pool_size * threads
        if requested > concurrency.cpus:
            _logger.warning(
                _OVERSUBSCRIBED[variable],
                extra={"cpus_requested": requested, "cpus_available": concurrency.cpus},
            )


def _store_variables(settings: Settings) -> dict[str, str]:
    """``_STORE_VARIABLES``, naming ``QDRANT_API_KEY_FILE`` when the key came from that file."""
    return {**_STORE_VARIABLES, "api_key": settings.key_variable("qdrant_api_key")}


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
    variable = settings.key_variable("llm_api_key")
    api_key = None if key is None else _named({}, lambda: Secret(key.get_secret_value()), variable)
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
            reasoning_effort=settings.llm_reasoning_effort,
            max_tokens_field=settings.llm_max_tokens_field,
            send_temperature=settings.llm_send_temperature,
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
