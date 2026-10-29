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

Device: ``DEVICE`` is resolved here once, by ``resolve_device`` (``auto``: the first GPU when
torch sees one, else the CPU; ``cuda`` without a visible GPU is a ``ConfigurationError``), and
the resolved name ("cpu" or "cuda:0") goes into the three model configurations, so the adapters
never probe the hardware and always agree. The device is part of ``Concurrency`` because the
pool sizes depend on it.

Concurrency: ``MAX_CONCURRENT_JOBS``, ``NER_POOL_SIZE``, ``EMBEDDING_POOL_SIZE``,
``INJECTION_POOL_SIZE`` and ``OCR_WORKERS`` that are not set are derived from the CPUs available
to the process (``available_cpus``: affinity and the cgroup v2 or v1 quota) and the device by
``concurrency_for``; a set value is used as given. ``resolve_concurrency`` is the one place
that does this, and every consumer (the adapter configurations here, the HTTP job limiter in
``app.main``) uses its result. Building the container logs the values in use and the device
(with the GPU's name and memory on the GPU) in one line, and, on the CPU only, one warning per
pool whose size x ``TORCH_THREADS`` (the CPUs its concurrent calls can use) exceeds the
available CPUs; nothing is changed. The pools are checked one by one: a job uses one of them at
a time, and the job limit bounds them together. On the GPU the pools' forward passes do not run
on the CPUs, so that product says nothing.

The injection detector is the Prompt Guard adapter when ``INJECTION_MODEL_PATH`` is set and
``NoDetector`` otherwise; with ``INJECTION_PATTERNS`` (the default) it is wrapped with the
instruction patterns in a ``CompositeDetector`` that keeps the higher score (design D5b).

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

from app.adapters.device import CPU, gpu_description, gpu_visible
from app.adapters.embeddings.config import E5EmbedderConfig
from app.adapters.embeddings.embedder import E5Embedder
from app.adapters.injection.config import InjectionDetectorConfig
from app.adapters.injection.detector import PromptGuardDetector
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
from app.application.no_detector import NoDetector
from app.application.pattern_detector import CompositeDetector, PatternDetector
from app.application.ports import (
    Embedder,
    InjectionDetector,
    LLMClient,
    NameRecognizer,
    OcrEngine,
    VectorStore,
)
from app.application.unconfigured_llm import UnconfiguredLLM
from app.domain.chunking import ChunkingConfig
from app.domain.errors import ConfigurationError
from app.settings import DEFAULT_TORCH_THREADS, DeviceChoice, Settings

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
CPU_INJECTION_POOL = 2
"""``INJECTION_POOL_SIZE`` on the CPU when unset: the detector adapter's default."""
GPU_NER_POOL = 1
"""``NER_POOL_SIZE`` on the GPU when unset. Measured (BENCHMARKS 23, design D3): pools of 1, 2, 4
and 8 came within 3 % of each other, because OCR on the CPUs bounds the jobs and the GPU stays
mostly idle, so the rule of BENCHMARKS 18 (the smallest pool within the best band) gives 1."""
GPU_EMBEDDING_POOL = 1
"""``EMBEDDING_POOL_SIZE`` on the GPU when unset: pools of 1, 2 and 4 within 3 % (BENCHMARKS
23), so the smallest, as ``GPU_NER_POOL``. Queries keep their reserved slot on top of it."""
GPU_INJECTION_POOL = 1
"""``INJECTION_POOL_SIZE`` on the GPU when unset: pools of 1 and 2 within 3 % (BENCHMARKS 23), so
the smallest. Questions keep their reserved slot on top of it."""
GPU_INDEX = 0
"""The GPU the models use when the device is the GPU (multi-GPU is out of scope)."""

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


class ManagedInjectionDetector(InjectionDetector, Protocol):
    """An ``InjectionDetector`` holding a loaded model that must be released."""

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class InferenceDevice:
    """Where the in-process models run: ``name`` ("cpu" or "cuda:<index>") and, on a GPU, its
    name and total memory in MiB (reported in the start-up line)."""

    name: str = CPU
    gpu_name: str | None = None
    gpu_memory_mib: int | None = None

    @property
    def on_gpu(self) -> bool:
        """The models run on a GPU."""
        return self.name != CPU


CPU_DEVICE = InferenceDevice()
"""The CPU, the device of every host without a visible GPU."""


@dataclass(frozen=True, slots=True)
class Concurrency:
    """How much work runs at once, resolved from ``Settings``, the CPUs (``cpus``) and the device.

    ``torch_threads`` torch intra-op threads, ``max_concurrent_jobs`` extraction and ingestion
    jobs, ``ner_pool_size`` concurrent recognitions, ``embedding_pool_size`` concurrent passage
    batches, ``ocr_workers`` Tesseract processes per document, ``injection_pool_size``
    concurrent detector batches and ``device`` where the three models run.
    """

    cpus: int
    max_concurrent_jobs: int
    ner_pool_size: int
    embedding_pool_size: int
    ocr_workers: int
    torch_threads: int = DEFAULT_TORCH_THREADS
    injection_pool_size: int = CPU_INJECTION_POOL
    device: InferenceDevice = CPU_DEVICE


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
    detector: Callable[[InjectionDetectorConfig], ManagedInjectionDetector] = PromptGuardDetector


@dataclass(frozen=True, slots=True)
class AdapterConfigs:
    """Every adapter configuration derived from ``Settings``; ``llm`` is ``None`` without a
    language model and ``injection`` is ``None`` without a detector model."""

    ocr: TesseractOcrConfig
    ner: TransformersNerConfig
    embedder: E5EmbedderConfig
    store: QdrantStoreConfig
    llm: LLMClientConfig | None
    chunking: ChunkingConfig
    injection: InjectionDetectorConfig | None = None


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
    "render_timeout_s": "OCR_RENDER_TIMEOUT_S",
    "max_image_pixels": "OCR_MAX_IMAGE_PIXELS",
    "max_page_chars": "MAX_PAGE_CHARS",
    "max_pages": "MAX_PAGES",
}
_NER_VARIABLES = {
    "model_path": "NER_MODEL_PATH",
    "threads": "TORCH_THREADS",
    "pool_size": "NER_POOL_SIZE",
    "device": "DEVICE",
}
_EMBEDDER_VARIABLES = {
    "model_path": "EMBEDDING_MODEL_PATH",
    "revision": "EMBEDDING_MODEL_REVISION",
    "batch_size": "EMBEDDING_BATCH_SIZE",
    "threads": "TORCH_THREADS",
    "pool_size": "EMBEDDING_POOL_SIZE",
    "device": "DEVICE",
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
    "support_min_share": "RAG_SUPPORT_MIN_SHARE",
    "max_chunks_per_document": "RAG_MAX_CHUNKS_PER_DOCUMENT",
    "injection_threshold": "INJECTION_THRESHOLD",
}
_INGEST_VARIABLES = {
    "batch_size": "EMBEDDING_BATCH_SIZE",
    "injection_threshold": "INJECTION_THRESHOLD",
}
_DETECTOR_VARIABLES = {
    "model_path": "INJECTION_MODEL_PATH",
    "threads": "TORCH_THREADS",
    "pool_size": "INJECTION_POOL_SIZE",
    "batch_size": "EMBEDDING_BATCH_SIZE",
    "device": "DEVICE",
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
                render_timeout_s=settings.ocr_render_timeout_s,
                max_image_pixels=settings.ocr_max_image_pixels,
                max_page_chars=settings.max_page_chars,
            ),
        ),
        ner=_named(
            _NER_VARIABLES,
            lambda: TransformersNerConfig(
                model_path=settings.ner_model_path,
                threads=sizes.torch_threads,
                pool_size=sizes.ner_pool_size,
                device=sizes.device.name,
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
                device=sizes.device.name,
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
        injection=_detector_config(settings, sizes),
    )


def _detector_config(settings: Settings, sizes: Concurrency) -> InjectionDetectorConfig | None:
    if settings.injection_model_path is None:
        return None
    return _named(
        _DETECTOR_VARIABLES,
        lambda: InjectionDetectorConfig(
            model_path=settings.injection_model_path or "",
            threads=sizes.torch_threads,
            pool_size=sizes.injection_pool_size,
            batch_size=settings.embedding_batch_size,
            device=sizes.device.name,
        ),
        fallback="INJECTION_MODEL_PATH",
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
        detector: InjectionDetector = NoDetector()
        if configs.injection is not None:
            injection_config = configs.injection
            loaded = _named(
                _DETECTOR_VARIABLES,
                lambda: make.detector(injection_config),
                fallback="INJECTION_MODEL_PATH",
            )
            closers.insert(0, loaded.close)
            detector = loaded
        else:
            _logger.info("injection detection off (INJECTION_MODEL_PATH is unset)")
        if settings.injection_patterns:
            detector = CompositeDetector([PatternDetector(), detector])
            _logger.info("injection patterns on")
        else:
            _logger.info("injection patterns off (INJECTION_PATTERNS is false)")
        container = Container(
            extract=_named(
                {"threshold": "MATCH_THRESHOLD"},
                lambda: ExtractNames(ocr, recognizer, settings.match_threshold),
            ),
            ingest=_named(
                _INGEST_VARIABLES,
                lambda: IngestDocument(
                    ocr,
                    embedder,
                    store,
                    configs.chunking,
                    settings.embedding_batch_size,
                    detector=detector,
                    injection_threshold=settings.injection_threshold,
                ),
            ),
            ask=_named(_ASK_VARIABLES, lambda: _ask(settings, embedder, store, llm, detector)),
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


def resolve_device(
    choice: DeviceChoice,
    visible: Callable[[], bool] = gpu_visible,
    describe: Callable[[int], tuple[str, int]] = gpu_description,
) -> InferenceDevice:
    """The device of the ``DEVICE`` setting ``choice``: ``cpu``, ``cuda`` (the first GPU) or
    ``auto``.

    ``auto`` is the first GPU when ``visible()`` (torch sees a GPU) and the CPU otherwise. On a
    GPU, ``describe`` gives its name and total memory for the start-up line. Raises
    ``ConfigurationError`` naming ``DEVICE`` for ``cuda`` without a visible GPU. ``visible`` and
    ``describe`` replace the torch calls (tests).
    """
    if choice == "cpu":
        return CPU_DEVICE
    if not visible():
        if choice == "cuda":
            raise ConfigurationError("DEVICE: cuda requested but no GPU is visible")
        return CPU_DEVICE
    name, memory_mib = describe(GPU_INDEX)
    return InferenceDevice(f"cuda:{GPU_INDEX}", gpu_name=name, gpu_memory_mib=memory_mib)


def concurrency_for(
    cpus: int, torch_threads: int, device: InferenceDevice = CPU_DEVICE
) -> Concurrency:
    """The concurrency derived for ``cpus`` available CPUs, ``torch_threads`` and ``device``.

    - jobs: three quarters of the CPUs rounded up, at most ``MAX_DERIVED_JOBS``;
    - NER pool: CPUs / ``torch_threads`` rounded down, at most ``MAX_DERIVED_NER_POOL``;
    - embedder pool: CPUs / (2 x ``torch_threads``) rounded down, at most
      ``MAX_DERIVED_EMBEDDING_POOL``;
    - detector pool: ``CPU_INJECTION_POOL``;
    - OCR workers: the CPUs, at most ``MAX_DERIVED_OCR_WORKERS``;

    each at least 1. On a GPU the three model pools are instead ``GPU_NER_POOL``,
    ``GPU_EMBEDDING_POOL`` and ``GPU_INJECTION_POOL``, measured (BENCHMARKS 23, design D3): the
    forward passes share the one device instead of the CPUs. Jobs
    and OCR workers keep the CPU rule there, because OCR stays on the CPUs and bounds the jobs.
    With 2 threads the CPU rule gives the measured sizing (BENCHMARKS 17.4 and 18):
    16 CPUs 12 / 8 / 4 / 4, 8 CPUs 6 / 4 / 2 / 4, 4 CPUs 3 / 2 / 1 / 4. Jobs round up because
    one job too few cost more than one too many where measured (4 CPUs: 2 jobs -22 %, 4 jobs
    -3 %); the pools round down so that a derived pool never exceeds the CPUs (no warning).
    """
    jobs = (3 * cpus + 3) // 4
    if device.on_gpu:
        ner_pool, embedding_pool = GPU_NER_POOL, GPU_EMBEDDING_POOL
        injection_pool = GPU_INJECTION_POOL
    else:
        ner_pool = min(cpus // torch_threads, MAX_DERIVED_NER_POOL)
        embedding_pool = min(cpus // (2 * torch_threads), MAX_DERIVED_EMBEDDING_POOL)
        injection_pool = CPU_INJECTION_POOL
    return Concurrency(
        cpus=cpus,
        torch_threads=torch_threads,
        max_concurrent_jobs=_at_least_one(min(jobs, MAX_DERIVED_JOBS)),
        ner_pool_size=_at_least_one(ner_pool),
        embedding_pool_size=_at_least_one(embedding_pool),
        ocr_workers=_at_least_one(min(cpus, MAX_DERIVED_OCR_WORKERS)),
        injection_pool_size=_at_least_one(injection_pool),
        device=device,
    )


def resolve_concurrency(
    settings: Settings, cpus: int | None = None, device: InferenceDevice | None = None
) -> Concurrency:
    """The concurrency in use: each of ``settings``' five sizes as set, or else derived.

    Unset values come from ``concurrency_for`` with ``cpus`` (default ``available_cpus()``),
    ``TORCH_THREADS`` (unset: ``DEFAULT_TORCH_THREADS``, at most ``cpus``) and ``device``
    (default ``resolve_device(settings.device)``, the one probe of the hardware); a set value
    always wins. Raises ``ConfigurationError`` when ``DEVICE=cuda`` and no GPU is visible.
    """
    available = available_cpus() if cpus is None else cpus
    resolved = resolve_device(settings.device) if device is None else device
    threads = _given(settings.torch_threads, min(DEFAULT_TORCH_THREADS, available))
    derived = concurrency_for(available, threads, resolved)
    return Concurrency(
        cpus=available,
        torch_threads=threads,
        max_concurrent_jobs=_given(settings.max_concurrent_jobs, derived.max_concurrent_jobs),
        ner_pool_size=_given(settings.ner_pool_size, derived.ner_pool_size),
        embedding_pool_size=_given(settings.embedding_pool_size, derived.embedding_pool_size),
        ocr_workers=_given(settings.ocr_workers, derived.ocr_workers),
        injection_pool_size=_given(settings.injection_pool_size, derived.injection_pool_size),
        device=resolved,
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
    device = concurrency.device
    gpu: dict[str, object] = {}
    if device.on_gpu:
        gpu = {"gpu_name": device.gpu_name, "gpu_memory_mib": device.gpu_memory_mib}
    _logger.info(
        "concurrency in use",
        extra={
            "cpus_available": concurrency.cpus,
            "torch_threads": threads,
            "max_concurrent_jobs": concurrency.max_concurrent_jobs,
            "ner_pool_size": concurrency.ner_pool_size,
            "embedding_pool_size": concurrency.embedding_pool_size,
            "injection_pool_size": concurrency.injection_pool_size,
            "ocr_workers": concurrency.ocr_workers,
            "device": device.name,
            **gpu,
        },
    )
    if device.on_gpu:  # the pools' forward passes do not compete for the CPUs
        return
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


def _ask(
    settings: Settings,
    embedder: Embedder,
    store: VectorStore,
    llm: LLMClient,
    detector: InjectionDetector,
) -> AskQuestion:
    return AskQuestion(
        embedder,
        store,
        llm,
        threshold=settings.rag_threshold,
        top_k=settings.rag_top_k,
        max_chunks=settings.rag_max_chunks,
        max_context_chars=settings.rag_max_context_chars,
        detector=detector,
        injection_threshold=settings.injection_threshold,
        support_min_share=settings.rag_support_min_share,
        max_chunks_per_document=settings.rag_max_chunks_per_document,
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
