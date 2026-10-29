"""The composition root (spec: service-configuration)."""

import dataclasses
import json
import logging
from pathlib import Path

import pytest

import app.bootstrap
from app.adapters.embeddings.config import E5EmbedderConfig
from app.adapters.llm.config import LLMClientConfig
from app.adapters.llm.secret import Secret
from app.adapters.ner.config import TransformersNerConfig
from app.adapters.ocr.engine import TesseractOcrConfig
from app.adapters.vector_store.config import QdrantStoreConfig
from app.application.unconfigured_llm import UnconfiguredLLM
from app.bootstrap import (
    MAX_DERIVED_EMBEDDING_POOL,
    MAX_DERIVED_JOBS,
    MAX_DERIVED_NER_POOL,
    MAX_DERIVED_OCR_WORKERS,
    Concurrency,
    Container,
    adapter_configs,
    available_cpus,
    build_container,
    concurrency_for,
    resolve_concurrency,
)
from app.domain.chunking import ChunkingConfig
from app.domain.errors import ConfigurationError, DependencyUnavailable
from app.domain.types import ChatMessage, OcrPage
from app.observability import JsonFormatter
from tests.bootstrap.support import Recorder, bootstrap_settings, model_directory

API_KEY = "test-key-0123456789"
QDRANT_KEY = "qdrant-key-0123456789"


@pytest.fixture
def models(tmp_path: Path) -> Path:
    model_directory(tmp_path, "ner")
    model_directory(tmp_path, "e5")
    return tmp_path


def _every_setting(models: Path) -> dict[str, object]:
    """A value different from the default for every setting the container uses."""
    return {
        "torch_threads": 3,
        "ner_pool_size": 6,
        "embedding_pool_size": 5,
        "embedding_batch_size": 7,
        "ocr_dpi": 300,
        "ocr_workers": 3,
        "ocr_timeout_s": 45.0,
        "ocr_flatten": False,
        "max_pages": 12,
        "match_threshold": 0.8,
        "chunk_size": 600,
        "chunk_overlap": 0.2,
        "rag_threshold": 0.75,
        "rag_top_k": 9,
        "rag_max_chunks": 4,
        "rag_max_context_chars": 5000,
        "qdrant_url": "http://qdrant.internal:6333",
        "qdrant_location": None,
        "qdrant_api_key": QDRANT_KEY,
        "qdrant_collection_prefix": "docs",
        "qdrant_timeout_s": 4.0,
        "qdrant_attempts": 6,
        "qdrant_deadline_s": 30.0,
        "readiness_timeout_s": 1.5,
        "llm_base_url": "http://ollama:11434/v1",
        "llm_model": "qwen2.5:7b",
        "llm_api_key": API_KEY,
        "llm_connect_timeout_s": 2.0,
        "llm_read_timeout_s": 30.0,
        "llm_total_timeout_s": 40.0,
        "llm_attempts": 2,
        "llm_max_tokens": 256,
        "llm_reasoning_effort": "none",
        "llm_max_tokens_field": "max_completion_tokens",
        "llm_send_temperature": False,
    }


def test_every_setting_reaches_its_adapter_config(models: Path) -> None:
    settings = bootstrap_settings(models, **_every_setting(models))
    recorder = Recorder()

    container = build_container(settings, recorder.factories())

    assert recorder.configs["ocr"] == TesseractOcrConfig(
        dpi=300, flatten=False, timeout_s=45.0, workers=3, max_pages=12
    )
    assert recorder.configs["recognizer"] == TransformersNerConfig(
        model_path=str(models / "ner"), threads=3, pool_size=6
    )
    assert recorder.configs["embedder"] == E5EmbedderConfig(
        model_path=str(models / "e5"), revision="rev-1", batch_size=7, threads=3, pool_size=5
    )
    assert recorder.configs["store"] == QdrantStoreConfig(
        url="http://qdrant.internal:6333",
        collection_prefix="docs",
        timeout_s=4.0,
        attempts=6,
        deadline_s=30.0,
        api_key=QDRANT_KEY,
        ping_timeout_s=1.5,
    )
    assert recorder.configs["llm"] == LLMClientConfig(
        base_url="http://ollama:11434/v1",
        model="qwen2.5:7b",
        api_key=Secret(API_KEY),
        connect_timeout_s=2.0,
        read_timeout_s=30.0,
        total_timeout_s=40.0,
        attempts=2,
        max_tokens=256,
        reasoning_effort="none",
        max_tokens_field="max_completion_tokens",
        send_temperature=False,
    )
    assert container.extract._threshold == 0.8
    assert container.ingest._chunking == ChunkingConfig(max_chars=600, overlap_ratio=0.2)
    assert container.ingest._batch_size == 7
    ask = container.ask
    assert (ask._threshold, ask._top_k, ask._max_chunks, ask._max_context_chars) == (
        0.75,
        9,
        4,
        5000,
    )
    assert container.llm_configured is True


def test_default_model_settings_give_the_default_request_options(models: Path) -> None:
    settings = bootstrap_settings(models, llm_base_url="http://llm/v1", llm_model="m")

    assert adapter_configs(settings).llm == LLMClientConfig("http://llm/v1", "m")


def test_default_settings_give_the_default_adapter_configs(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 16)
    configs = adapter_configs(bootstrap_settings(models))

    assert configs.ocr == TesseractOcrConfig(workers=4)
    assert configs.chunking == ChunkingConfig()
    assert configs.store == QdrantStoreConfig(location=":memory:")
    assert configs.llm is None


@pytest.mark.parametrize("threads", [1, 2, 3])
def test_torch_threads_reaches_both_torch_adapters(models: Path, threads: int) -> None:
    configs = adapter_configs(bootstrap_settings(models, torch_threads=threads))

    assert configs.ner.threads == configs.embedder.threads == threads


def test_default_torch_threads_is_both_adapters_default(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 16)  # pools at their maximum
    configs = adapter_configs(bootstrap_settings(models))

    assert configs.ner == TransformersNerConfig(model_path=str(models / "ner"))
    assert configs.embedder == E5EmbedderConfig(model_path=str(models / "e5"), revision="rev-1")


@pytest.mark.parametrize("adapter", ["recognizer", "embedder"])
def test_a_thread_error_of_either_torch_adapter_names_torch_threads(
    models: Path, adapter: str
) -> None:
    recorder = Recorder()
    recorder.failing[adapter] = ConfigurationError("threads must be a positive integer")

    with pytest.raises(ConfigurationError) as caught:
        build_container(bootstrap_settings(models), recorder.factories())

    assert caught.value.message == "TORCH_THREADS: threads must be a positive integer"


def test_each_adapter_is_built_once_and_closed_by_the_container(models: Path) -> None:
    settings = bootstrap_settings(
        models, llm_base_url="http://llm:8000/v1", llm_model="m", llm_api_key=API_KEY
    )
    recorder = Recorder()

    container = build_container(settings, recorder.factories())
    container.close()

    assert dict(recorder.built) == {"ocr": 1, "recognizer": 1, "embedder": 1, "store": 1, "llm": 1}
    assert recorder.events == ["embedder.close", "recognizer.close", "llm.close", "store.close"]


def test_a_key_from_the_key_file_reaches_the_llm_config(models: Path, tmp_path: Path) -> None:
    """Compose passes the key as a secret file (LLM_API_KEY_FILE), not as a variable."""
    key_file = tmp_path / "llm_api_key"
    key_file.write_text(f"{API_KEY}\n", encoding="utf-8")
    settings = bootstrap_settings(
        models, llm_base_url="http://llm:8000/v1", llm_model="m", llm_api_key_file=str(key_file)
    )

    llm = adapter_configs(settings).llm

    assert llm is not None
    assert llm.api_key == Secret(API_KEY)


def test_a_failing_closer_does_not_stop_the_others(caplog: pytest.LogCaptureFixture) -> None:
    closed: list[str] = []

    def broken() -> None:
        raise OSError("socket already gone")

    container = Container(
        extract=None,  # type: ignore[arg-type]
        ingest=None,  # type: ignore[arg-type]
        ask=None,  # type: ignore[arg-type]
        readiness=(),
        llm_configured=False,
        closers=(broken, lambda: closed.append("store")),
    )

    container.close()

    assert closed == ["store"]
    assert "closing a client failed" in caplog.text


def test_without_a_model_the_service_starts_and_ask_fails_only_at_the_model(
    models: Path,
) -> None:
    recorder = Recorder()

    container = build_container(bootstrap_settings(models), recorder.factories())

    assert "llm" not in recorder.built
    assert container.llm_configured is False
    assert {name: check() for name, check in container.readiness} == {
        "vector_store": "ok",
        "llm": "not_configured",
    }
    assert isinstance(container.ask._llm, UnconfiguredLLM)
    container.close()
    assert recorder.events == ["ping", "embedder.close", "recognizer.close", "store.close"]


def test_unconfigured_model_raises_llm_unavailable() -> None:
    with pytest.raises(DependencyUnavailable) as caught:
        UnconfiguredLLM().complete([ChatMessage("user", "anything")])

    assert caught.value.code == "llm_unavailable"
    assert caught.value.message == "no language model is configured"


def test_store_readiness_check_raises_when_the_store_is_down(models: Path) -> None:
    down = DependencyUnavailable(
        "the vector store could not be reached", "vector_store_unavailable"
    )
    container = build_container(bootstrap_settings(models), Recorder(store_down=down).factories())
    checks = dict(container.readiness)

    with pytest.raises(DependencyUnavailable):
        checks["vector_store"]()


@pytest.mark.parametrize(
    ("overrides", "variable"),
    [
        ({"ner_model_path": "missing-ner"}, "NER_MODEL_PATH"),
        ({"embedding_model_path": "missing-e5"}, "EMBEDDING_MODEL_PATH"),
        ({"embedding_model_revision": " rev "}, "EMBEDDING_MODEL_REVISION"),
        ({"qdrant_location": "/var/qdrant"}, "QDRANT_LOCATION"),
        ({"qdrant_location": None, "qdrant_url": "ftp://qdrant"}, "QDRANT_URL"),
        ({"qdrant_collection_prefix": "no spaces"}, "QDRANT_COLLECTION_PREFIX"),
        ({"llm_base_url": "ftp://llm", "llm_model": "m"}, "LLM_BASE_URL"),
        ({"llm_base_url": "http://u:p@llm/v1", "llm_model": "m"}, "LLM_BASE_URL"),
        (
            {"llm_base_url": "http://llm/v1", "llm_model": "m", "llm_total_timeout_s": 1.0},
            "LLM_TOTAL_TIMEOUT_S",
        ),
        ({"llm_base_url": "http://llm/v1", "llm_model": " "}, "LLM_MODEL"),
    ],
)
def test_invalid_adapter_value_names_the_variable(
    models: Path, overrides: dict[str, object], variable: str
) -> None:
    recorder = Recorder()

    with pytest.raises(ConfigurationError) as caught:
        build_container(bootstrap_settings(models, **overrides), recorder.factories())

    assert caught.value.code == "invalid_configuration"
    assert caught.value.message.startswith(f"{variable}: ")
    assert not recorder.built, "nothing is built before every config is valid"


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_api_key": "has space-0123456789"},
        {"llm_base_url": f"http://llm/{API_KEY}/v1", "llm_api_key": API_KEY},
    ],
)
def test_secret_errors_name_the_variable_but_never_the_key(
    models: Path, overrides: dict[str, object]
) -> None:
    settings = bootstrap_settings(
        models, llm_model="m", **{"llm_base_url": "http://llm/v1", **overrides}
    )

    with pytest.raises(ConfigurationError) as caught:
        build_container(settings, Recorder().factories())

    message = caught.value.message
    assert message.split(":")[0] in {"LLM_API_KEY", "LLM_BASE_URL"}
    assert API_KEY not in message
    assert "has space" not in message
    assert API_KEY not in repr(caught.value)


@pytest.mark.parametrize(
    ("adapter", "variable"),
    [("recognizer", "NER_MODEL_PATH"), ("embedder", "EMBEDDING_MODEL_PATH")],
)
def test_model_load_failure_names_the_variable_and_closes_the_store(
    models: Path, adapter: str, variable: str
) -> None:
    recorder = Recorder()
    recorder.failing[adapter] = ConfigurationError("model could not be loaded from /models/x")

    with pytest.raises(ConfigurationError) as caught:
        build_container(bootstrap_settings(models), recorder.factories())

    assert caught.value.message == f"{variable}: model could not be loaded from /models/x"
    built_first = ["recognizer.close"] if adapter == "embedder" else []
    assert recorder.events == [*built_first, "store.close"]


def test_unexpected_failure_while_building_closes_what_was_built(models: Path) -> None:
    recorder = Recorder()
    recorder.failing["ocr"] = RuntimeError("engine exploded")
    settings = bootstrap_settings(models, llm_base_url="http://llm/v1", llm_model="m")

    with pytest.raises(RuntimeError):
        build_container(settings, recorder.factories())

    assert recorder.events == ["llm.close", "store.close"]


def test_container_wires_the_same_adapters_into_every_use_case(models: Path) -> None:
    container = build_container(bootstrap_settings(models), Recorder().factories())

    assert container.extract._ocr is container.ingest._ocr
    assert container.ingest._embedder is container.ask._embedder
    assert container.ingest._store is container.ask._store
    page: list[OcrPage] = container.extract._ocr.recognize(b"%PDF")
    assert page == []


def _cpu_max(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "cpu.max"
    path.write_text(content, encoding="ascii")
    return path


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("max 100000\n", 8),
        ("200000 100000\n", 2),
        ("250000 100000\n", 2),
        ("50000 100000\n", 1),
        ("1600000 100000\n", 8),
        ("garbage\n", 8),
        ("abc 100000\n", 8),
        ("0 100000\n", 8),
        ("100000 0\n", 8),
        ("", 8),
    ],
)
def test_available_cpus_is_the_affinity_lowered_to_the_cgroup_quota(
    tmp_path: Path, content: str, expected: int
) -> None:
    cpu_max = _cpu_max(tmp_path, content)

    assert available_cpus(cpu_max, affinity=lambda: 8, cpu_count=lambda: 64) == expected


def test_available_cpus_without_a_cgroup_file_is_the_affinity(tmp_path: Path) -> None:
    absent = tmp_path / "absent"
    assert available_cpus(absent, lambda: 6, lambda: 64, cgroup_v1_cpu=absent) == 6


def test_available_cpus_without_affinity_is_the_cpu_count(tmp_path: Path) -> None:
    absent = tmp_path / "absent"
    assert available_cpus(absent, lambda: None, lambda: 12, cgroup_v1_cpu=absent) == 12
    assert available_cpus(absent, lambda: None, lambda: None, cgroup_v1_cpu=absent) == 1
    quota = _cpu_max(tmp_path, "300000 100000")
    assert available_cpus(quota, affinity=lambda: None, cpu_count=lambda: 12) == 3


def test_available_cpus_on_this_platform_is_at_least_one() -> None:
    assert available_cpus() >= 1


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == "app.bootstrap" and record.levelno == logging.WARNING
    ]


def _json(record: logging.LogRecord) -> dict[str, object]:
    line: dict[str, object] = json.loads(JsonFormatter().format(record))
    return line


def test_more_ner_threads_than_cpus_logs_one_warning(
    models: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 4)
    settings = bootstrap_settings(models, ner_pool_size=3, torch_threads=2)

    with caplog.at_level(logging.WARNING, logger="app.bootstrap"):
        build_container(settings, Recorder().factories()).close()

    (record,) = _warnings(caplog)
    assert record.getMessage().startswith("NER_POOL_SIZE x TORCH_THREADS exceeds the CPUs")
    line = JsonFormatter().format(record)
    assert '"cpus_requested": 6' in line
    assert '"cpus_available": 4' in line


def test_more_embedding_threads_than_cpus_logs_one_warning(
    models: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 4)
    settings = bootstrap_settings(models, embedding_pool_size=3, torch_threads=2)

    with caplog.at_level(logging.WARNING, logger="app.bootstrap"):
        build_container(settings, Recorder().factories()).close()

    (record,) = _warnings(caplog)
    assert record.getMessage() == (
        "EMBEDDING_POOL_SIZE x TORCH_THREADS exceeds the CPUs available to the process; "
        "concurrent embeddings will compete for CPUs"
    )
    line = _json(record)
    assert (line["cpus_requested"], line["cpus_available"]) == (6, 4)


def test_both_pools_over_the_cpus_log_one_warning_each(
    models: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 2)
    settings = bootstrap_settings(models, ner_pool_size=2, embedding_pool_size=3, torch_threads=2)

    with caplog.at_level(logging.WARNING, logger="app.bootstrap"):
        build_container(settings, Recorder().factories()).close()

    records = _warnings(caplog)
    assert [record.getMessage().split(" x ")[0] for record in records] == [
        "NER_POOL_SIZE",
        "EMBEDDING_POOL_SIZE",
    ]
    assert [_json(record)["cpus_requested"] for record in records] == [4, 6]


@pytest.mark.parametrize(
    ("ner_pool", "embedding_pool", "threads"), [(2, 2, 2), (4, 4, 1), (1, 1, 4)]
)
def test_pools_within_the_cpus_log_no_warning(
    models: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    ner_pool: int,
    embedding_pool: int,
    threads: int,
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 4)
    settings = bootstrap_settings(
        models, ner_pool_size=ner_pool, embedding_pool_size=embedding_pool, torch_threads=threads
    )

    with caplog.at_level(logging.WARNING, logger="app.bootstrap"):
        build_container(settings, Recorder().factories()).close()

    assert _warnings(caplog) == []


@pytest.mark.parametrize("cpus", [1, 2, 3, 4, 6, 8, 12, 16, 32, 128])
@pytest.mark.parametrize("threads", [1, 2, 4])
def test_derived_values_never_log_a_warning(
    models: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    cpus: int,
    threads: int,
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: cpus)
    # More torch threads than CPUs oversubscribes any pool, so at most one per CPU here.
    settings = bootstrap_settings(models, torch_threads=min(threads, cpus))

    with caplog.at_level(logging.WARNING, logger="app.bootstrap"):
        build_container(settings, Recorder().factories()).close()

    assert _warnings(caplog) == []


# The rule at 2 torch threads: (jobs, NER pool, embedder pool, OCR workers) per CPU count. The
# rows 4, 8 and 16 are the measured sizing (BENCHMARKS 17.4 and 18); the others follow the rule.
RULE_AT_TWO_THREADS = {
    1: (1, 1, 1, 1),
    2: (2, 1, 1, 2),
    3: (3, 1, 1, 3),
    4: (3, 2, 1, 4),
    5: (4, 2, 1, 4),
    6: (5, 3, 1, 4),
    8: (6, 4, 2, 4),
    10: (8, 5, 2, 4),
    12: (9, 6, 3, 4),
    16: (12, 8, 4, 4),
    20: (12, 8, 4, 4),
    24: (12, 8, 4, 4),
    32: (12, 8, 4, 4),
    64: (12, 8, 4, 4),
}


@pytest.mark.parametrize(("cpus", "expected"), sorted(RULE_AT_TWO_THREADS.items()))
def test_the_rule_at_two_threads(cpus: int, expected: tuple[int, int, int, int]) -> None:
    assert concurrency_for(cpus, 2) == Concurrency(cpus, *expected)


@pytest.mark.parametrize("cpus", range(1, 129))
@pytest.mark.parametrize("threads", [1, 2, 3, 4, 8])
def test_the_rule_stays_within_its_bounds(cpus: int, threads: int) -> None:
    derived = concurrency_for(cpus, threads)

    assert min(MAX_DERIVED_JOBS, 3 * cpus / 4) <= derived.max_concurrent_jobs <= MAX_DERIVED_JOBS
    assert 1 <= derived.ner_pool_size <= MAX_DERIVED_NER_POOL
    assert 1 <= derived.embedding_pool_size <= MAX_DERIVED_EMBEDDING_POOL
    assert derived.ocr_workers == min(cpus, MAX_DERIVED_OCR_WORKERS)
    if cpus >= threads:  # a derived pool never asks for more CPUs than there are
        assert derived.ner_pool_size * threads <= cpus
        assert derived.embedding_pool_size * threads <= cpus


@pytest.mark.parametrize("cpus", range(1, 65))
def test_the_rule_never_decreases_with_more_cpus(cpus: int) -> None:
    fewer, more = concurrency_for(cpus, 2), concurrency_for(cpus + 1, 2)

    assert more.max_concurrent_jobs >= fewer.max_concurrent_jobs
    assert more.ner_pool_size >= fewer.ner_pool_size
    assert more.embedding_pool_size >= fewer.embedding_pool_size
    assert more.ocr_workers >= fewer.ocr_workers


def _default(config: type, name: str) -> object:
    (item,) = [item for item in dataclasses.fields(config) if item.name == name]
    return item.default


def test_the_rule_on_16_cpus_gives_the_adapter_defaults_and_the_maximum() -> None:
    derived = concurrency_for(16, 2)

    assert derived.ner_pool_size == _default(TransformersNerConfig, "pool_size")
    assert derived.embedding_pool_size == _default(E5EmbedderConfig, "pool_size")
    assert derived == Concurrency(
        16, MAX_DERIVED_JOBS, MAX_DERIVED_NER_POOL, MAX_DERIVED_EMBEDDING_POOL, 4
    )


def test_unset_values_are_derived_from_the_cpus(models: Path) -> None:
    settings = bootstrap_settings(models)

    assert resolve_concurrency(settings, cpus=8) == Concurrency(8, 6, 4, 2, 4)
    assert resolve_concurrency(settings, cpus=4) == Concurrency(4, 3, 2, 1, 4)


def test_the_derived_pools_follow_torch_threads(models: Path) -> None:
    settings = bootstrap_settings(models, torch_threads=4)

    assert resolve_concurrency(settings, cpus=16) == Concurrency(16, 12, 4, 2, 4, torch_threads=4)


@pytest.mark.parametrize("cpus", [1, 4, 16, 64])
def test_set_values_always_win(models: Path, cpus: int) -> None:
    settings = bootstrap_settings(
        models, max_concurrent_jobs=7, ner_pool_size=5, embedding_pool_size=3, ocr_workers=6
    )

    threads = min(2, cpus)  # TORCH_THREADS unset: 2, at most the CPUs
    assert resolve_concurrency(settings, cpus=cpus) == Concurrency(
        cpus, 7, 5, 3, 6, torch_threads=threads
    )


def test_each_value_is_resolved_on_its_own(models: Path) -> None:
    settings = bootstrap_settings(models, max_concurrent_jobs=20, embedding_pool_size=1)

    assert resolve_concurrency(settings, cpus=8) == Concurrency(8, 20, 4, 1, 4)


def test_resolution_reads_the_available_cpus_by_default(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 4)

    assert resolve_concurrency(bootstrap_settings(models)) == concurrency_for(4, 2)


def test_ocr_workers_and_pools_are_derived_from_the_cgroup_quota(
    models: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    quota = _cpu_max(tmp_path, "200000 100000\n")  # --cpus 2 on a 64-CPU host
    monkeypatch.setattr(
        app.bootstrap,
        "available_cpus",
        lambda: available_cpus(quota, affinity=lambda: 64, cpu_count=lambda: 64),
    )
    recorder = Recorder()

    build_container(bootstrap_settings(models), recorder.factories()).close()

    assert recorder.configs["ocr"] == TesseractOcrConfig(workers=2)
    assert recorder.configs["recognizer"] == TransformersNerConfig(
        model_path=str(models / "ner"), pool_size=1
    )
    assert recorder.configs["embedder"] == E5EmbedderConfig(
        model_path=str(models / "e5"), revision="rev-1", pool_size=1
    )


def test_the_container_uses_a_given_concurrency(models: Path) -> None:
    recorder = Recorder()
    given = Concurrency(
        cpus=3, max_concurrent_jobs=2, ner_pool_size=2, embedding_pool_size=3, ocr_workers=5
    )

    build_container(bootstrap_settings(models), recorder.factories(), given).close()

    assert recorder.configs["ocr"] == TesseractOcrConfig(workers=5)
    assert recorder.configs["recognizer"] == TransformersNerConfig(
        model_path=str(models / "ner"), pool_size=2
    )
    assert recorder.configs["embedder"] == E5EmbedderConfig(
        model_path=str(models / "e5"), revision="rev-1", pool_size=3
    )


@pytest.mark.parametrize(
    ("field", "variable"),
    [
        ("ner_pool_size", "NER_POOL_SIZE"),
        ("embedding_pool_size", "EMBEDDING_POOL_SIZE"),
        ("ocr_workers", "OCR_WORKERS"),
    ],
)
def test_an_invalid_resolved_size_names_its_variable(
    models: Path, field: str, variable: str
) -> None:
    valid = Concurrency(
        cpus=4, max_concurrent_jobs=1, ner_pool_size=1, embedding_pool_size=1, ocr_workers=1
    )

    with pytest.raises(ConfigurationError) as caught:
        adapter_configs(bootstrap_settings(models), dataclasses.replace(valid, **{field: 0}))

    assert caught.value.message.startswith(f"{variable}: ")


def test_building_logs_the_concurrency_in_use_once(
    models: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 8)
    settings = bootstrap_settings(models, max_concurrent_jobs=10)

    with caplog.at_level(logging.INFO, logger="app.bootstrap"):
        build_container(settings, Recorder().factories()).close()

    (record,) = [r for r in caplog.records if r.getMessage() == "concurrency in use"]
    line = _json(record)
    assert (record.name, record.levelno) == ("app.bootstrap", logging.INFO)
    assert {key: line[key] for key in line if key not in {"timestamp", "level", "logger"}} == {
        "message": "concurrency in use",
        "cpus_available": 8,
        "torch_threads": 2,
        "max_concurrent_jobs": 10,
        "ner_pool_size": 4,
        "embedding_pool_size": 2,
        "ocr_workers": 4,
    }
    assert _warnings(caplog) == []


@pytest.mark.parametrize(("cpus", "threads"), [(1, 1), (2, 2), (3, 2), (16, 2)])
def test_unset_torch_threads_is_two_at_most_the_cpus(models: Path, cpus: int, threads: int) -> None:
    resolved = resolve_concurrency(bootstrap_settings(models), cpus=cpus)

    assert resolved.torch_threads == threads
    assert adapter_configs(bootstrap_settings(models), resolved).ner.threads == threads


def test_a_set_torch_threads_wins_over_the_cpus(models: Path) -> None:
    resolved = resolve_concurrency(bootstrap_settings(models, torch_threads=4), cpus=2)

    assert resolved.torch_threads == 4


def test_one_cpu_with_the_defaults_logs_no_warning(
    models: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Formerly 2 torch threads on 1 CPU: both pools warned about oversubscription."""
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 1)
    recorder = Recorder()

    with caplog.at_level(logging.INFO, logger="app.bootstrap"):
        build_container(bootstrap_settings(models), recorder.factories()).close()

    assert _warnings(caplog) == []
    assert recorder.configs["recognizer"] == TransformersNerConfig(
        model_path=str(models / "ner"), threads=1, pool_size=1
    )
    (record,) = [r for r in caplog.records if r.getMessage() == "concurrency in use"]
    assert _json(record)["torch_threads"] == 1


def _cgroup_v1(tmp_path: Path, quota: str, period: str = "100000\n") -> Path:
    directory = tmp_path / "cpu"
    directory.mkdir()
    (directory / "cpu.cfs_quota_us").write_text(quota, encoding="ascii")
    (directory / "cpu.cfs_period_us").write_text(period, encoding="ascii")
    return directory


@pytest.mark.parametrize(
    ("quota", "period", "expected"),
    [
        ("-1\n", "100000\n", 8),  # no quota
        ("200000\n", "100000\n", 2),
        ("150000\n", "100000\n", 1),
        ("50000\n", "100000\n", 1),
        ("1600000\n", "100000\n", 8),
        ("garbage\n", "100000\n", 8),
        ("200000\n", "0\n", 8),
        ("\n", "100000\n", 8),
    ],
)
def test_available_cpus_reads_a_cgroup_v1_quota(
    tmp_path: Path, quota: str, period: str, expected: int
) -> None:
    v1 = _cgroup_v1(tmp_path, quota, period)

    cpus = available_cpus(tmp_path / "absent", lambda: 8, lambda: 64, cgroup_v1_cpu=v1)

    assert cpus == expected


def test_a_cgroup_v2_file_wins_over_v1(tmp_path: Path) -> None:
    v1 = _cgroup_v1(tmp_path, "100000\n")
    v2 = _cpu_max(tmp_path, "max 100000\n")

    assert available_cpus(v2, lambda: 8, lambda: 64, cgroup_v1_cpu=v1) == 8


def test_an_incomplete_cgroup_v1_directory_is_ignored(tmp_path: Path) -> None:
    v1 = tmp_path / "cpu"
    v1.mkdir()
    (v1 / "cpu.cfs_quota_us").write_text("100000\n", encoding="ascii")

    assert available_cpus(tmp_path / "absent", lambda: 8, lambda: 64, cgroup_v1_cpu=v1) == 8


def test_an_invalid_llm_key_from_its_file_names_the_file_variable(
    models: Path, tmp_path: Path
) -> None:
    key_file = tmp_path / "llm_api_key"
    key_file.write_text("key with spaces", encoding="utf-8")
    settings = bootstrap_settings(
        models, llm_base_url="http://llm/v1", llm_model="m", llm_api_key_file=str(key_file)
    )

    with pytest.raises(ConfigurationError) as caught:
        adapter_configs(settings)

    assert caught.value.message.startswith("LLM_API_KEY_FILE: a secret must be")
    assert "spaces" not in caught.value.message


def test_an_invalid_llm_key_from_its_variable_names_the_variable(models: Path) -> None:
    settings = bootstrap_settings(
        models, llm_base_url="http://llm/v1", llm_model="m", llm_api_key="key with spaces"
    )

    with pytest.raises(ConfigurationError) as caught:
        adapter_configs(settings)

    assert caught.value.message.startswith("LLM_API_KEY: a secret must be")


@pytest.mark.parametrize("from_file", [True, False])
def test_a_rejected_qdrant_key_names_where_it_came_from(
    models: Path, tmp_path: Path, from_file: bool
) -> None:
    key_file = tmp_path / "qdrant_api_key"
    key_file.write_text("qdrant-key-0123456789", encoding="utf-8")
    source = (
        {"qdrant_api_key_file": str(key_file)}
        if from_file
        else {"qdrant_api_key": "qdrant-key-0123456789"}
    )
    recorder = Recorder()
    recorder.failing["store"] = ConfigurationError("api_key must be None or a non-empty string")

    with pytest.raises(ConfigurationError) as caught:
        build_container(bootstrap_settings(models, **source), recorder.factories())

    variable = "QDRANT_API_KEY_FILE" if from_file else "QDRANT_API_KEY"
    assert caught.value.message == f"{variable}: api_key must be None or a non-empty string"


def test_the_qdrant_key_file_reaches_the_store(models: Path, tmp_path: Path) -> None:
    key_file = tmp_path / "qdrant_api_key"
    key_file.write_bytes(b"\xef\xbb\xbfqdrant-key-0123456789\n")

    configs = adapter_configs(bootstrap_settings(models, qdrant_api_key_file=str(key_file)))

    assert configs.store.api_key == "qdrant-key-0123456789"
