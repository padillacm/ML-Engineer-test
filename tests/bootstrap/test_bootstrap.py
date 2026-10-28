"""The composition root (spec: service-configuration)."""

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
from app.bootstrap import Container, adapter_configs, available_cpus, build_container
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
        model_path=str(models / "e5"), revision="rev-1", batch_size=7, threads=3
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


def test_default_settings_give_the_default_adapter_configs(models: Path) -> None:
    configs = adapter_configs(bootstrap_settings(models))

    assert configs.ocr == TesseractOcrConfig()
    assert configs.chunking == ChunkingConfig()
    assert configs.store == QdrantStoreConfig(location=":memory:")
    assert configs.llm is None


@pytest.mark.parametrize("threads", [1, 2, 3])
def test_torch_threads_reaches_both_torch_adapters(models: Path, threads: int) -> None:
    configs = adapter_configs(bootstrap_settings(models, torch_threads=threads))

    assert configs.ner.threads == configs.embedder.threads == threads


def test_default_torch_threads_is_both_adapters_default(models: Path) -> None:
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
    assert recorder.events == ["recognizer.close", "llm.close", "store.close"]


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
    assert recorder.events == ["ping", "recognizer.close", "store.close"]


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
    assert available_cpus(tmp_path / "absent", affinity=lambda: 6, cpu_count=lambda: 64) == 6


def test_available_cpus_without_affinity_is_the_cpu_count(tmp_path: Path) -> None:
    absent = tmp_path / "absent"
    assert available_cpus(absent, affinity=lambda: None, cpu_count=lambda: 12) == 12
    assert available_cpus(absent, affinity=lambda: None, cpu_count=lambda: None) == 1
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


@pytest.mark.parametrize(("pool_size", "threads"), [(2, 2), (4, 1), (1, 4)])
def test_ner_threads_within_the_cpus_log_no_warning(
    models: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pool_size: int,
    threads: int,
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 4)
    settings = bootstrap_settings(models, ner_pool_size=pool_size, torch_threads=threads)

    with caplog.at_level(logging.WARNING, logger="app.bootstrap"):
        build_container(settings, Recorder().factories()).close()

    assert _warnings(caplog) == []
