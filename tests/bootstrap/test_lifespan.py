"""Lifespan: settings, one construction, shutdown (spec: service-configuration)."""

import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.bootstrap
import app.main
from app.bootstrap import Concurrency, build_container
from app.domain.errors import ConfigurationError
from app.main import create_app
from app.observability import JsonFormatter
from app.settings import Settings
from tests.api.support import PDF, Fakes, api_settings
from tests.bootstrap.support import Recorder, bootstrap_settings, model_directory

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    yield
    for handler in root.handlers[:]:
        if handler not in saved[0]:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved[1])


@pytest.fixture
def models(tmp_path: Path) -> Path:
    model_directory(tmp_path, "ner")
    model_directory(tmp_path, "e5")
    return tmp_path


def _with_recorder(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> list[Settings]:
    """Make the lifespan build its container from ``recorder``'s fakes; return the settings."""
    seen: list[Settings] = []

    def build(settings: Settings, *, concurrency: Concurrency) -> object:
        seen.append(settings)
        return build_container(settings, recorder.factories(), concurrency)

    monkeypatch.setattr(app.main, "build_container", build)
    return seen


def test_models_are_built_once_at_start_up_and_closed_at_shutdown(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()
    _with_recorder(monkeypatch, recorder)
    settings = bootstrap_settings(models, llm_base_url="http://llm/v1", llm_model="m")

    with TestClient(create_app(settings=settings)) as client:
        assert dict(recorder.built) == {
            "ocr": 1,
            "recognizer": 1,
            "embedder": 1,
            "store": 1,
            "llm": 1,
        }
        for _ in range(2):
            response = client.post(
                "/api/extract", files={"pdf_file": ("a.pdf", PDF, "application/pdf")}
            )
            assert response.status_code == 200
        assert recorder.built["recognizer"] == 1
        assert recorder.events == []

    assert recorder.events == ["embedder.close", "recognizer.close", "llm.close", "store.close"]


def test_an_injected_container_is_used_as_given_and_not_closed() -> None:
    fakes = Fakes()
    settings = api_settings()

    with TestClient(create_app(settings=settings, container=fakes.container(settings))) as client:
        assert client.get("/health").status_code == 200

    assert fakes.closed == []


@pytest.mark.parametrize("enabled", [False, True])
def test_exception_messages_are_logged_only_when_enabled(enabled: bool) -> None:
    fakes = Fakes()
    settings = api_settings(log_exception_messages=enabled)

    with TestClient(create_app(settings=settings, container=fakes.container(settings))):
        (handler,) = [
            h for h in logging.getLogger().handlers if type(h).__name__ == "_ServiceHandler"
        ]
        assert isinstance(handler.formatter, JsonFormatter)
        assert handler.formatter.exception_messages is enabled


def test_the_service_starts_without_a_language_model(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()
    _with_recorder(monkeypatch, recorder)

    with TestClient(create_app(settings=bootstrap_settings(models))) as client:
        ready = client.get("/health/ready")

    assert ready.status_code == 200
    assert ready.json()["checks"] == {"vector_store": "ok", "llm": "not_configured"}
    assert "llm" not in recorder.built


def test_settings_are_read_from_the_environment_at_start_up(
    models: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = Recorder()
    seen = _with_recorder(monkeypatch, recorder)
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NER_MODEL_PATH", str(models / "ner"))
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(models / "e5"))
    monkeypatch.setenv("EMBEDDING_MODEL_REVISION", "rev-env")
    monkeypatch.setenv("QDRANT_LOCATION", ":memory:")
    monkeypatch.setenv("RAG_THRESHOLD", "0.75")
    monkeypatch.setenv("OCR_DPI", "300")
    application = create_app()

    assert seen == []  # creating the application reads nothing
    with TestClient(application):
        (settings,) = seen
        assert settings.embedding_model_revision == "rev-env"
        assert application.state.services.ask._threshold == 0.75
        assert recorder.configs["ocr"].dpi == 300  # type: ignore[attr-defined]


def test_invalid_configuration_stops_start_up_naming_the_variable(models: Path) -> None:
    settings = bootstrap_settings(models, ner_model_path=str(models / "absent"))

    with pytest.raises(ConfigurationError) as caught, TestClient(create_app(settings=settings)):
        pass

    assert caught.value.message.startswith("NER_MODEL_PATH: NER model directory not found")


@pytest.mark.parametrize("variable", ["NER_THREADS", "EMBEDDING_THREADS"])
def test_a_retired_thread_variable_stops_start_up_naming_it(
    models: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variable: str
) -> None:
    recorder = Recorder()
    _with_recorder(monkeypatch, recorder)
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NER_MODEL_PATH", str(models / "ner"))
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(models / "e5"))
    monkeypatch.setenv("EMBEDDING_MODEL_REVISION", "rev-env")
    monkeypatch.setenv("QDRANT_LOCATION", ":memory:")
    monkeypatch.setenv(variable, "1")

    with pytest.raises(ConfigurationError) as caught, TestClient(create_app()):
        pass

    assert f"{variable}: no longer read; set TORCH_THREADS instead" in caught.value.message
    assert not recorder.built


def test_start_up_logs_one_line_without_secrets(
    models: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _with_recorder(monkeypatch, Recorder())
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 1024)  # no CPU warning
    key = "start-key-0123456789"
    settings = bootstrap_settings(
        models, llm_base_url="http://llm/v1", llm_model="m", llm_api_key=key
    )

    with TestClient(create_app(settings=settings)):
        pass

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert [line["message"] for line in lines] == [
        "concurrency in use",
        "service started (language model configured)",
        "service stopped",
    ]
    assert lines[0]["cpus_available"] == 1024
    assert key not in json.dumps(lines)


@pytest.mark.parametrize(("cpus", "jobs"), [(4, 3), (8, 6), (16, 12), (64, 12)])
def test_the_job_limit_is_derived_from_the_cpus_when_unset(
    models: Path, monkeypatch: pytest.MonkeyPatch, cpus: int, jobs: int
) -> None:
    recorder = Recorder()
    _with_recorder(monkeypatch, recorder)
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: cpus)
    application = create_app(settings=bootstrap_settings(models))

    with TestClient(application):
        assert application.state.limiter.free_slots == jobs
        assert recorder.configs["recognizer"].pool_size == min(8, cpus // 2)  # type: ignore[attr-defined]


@pytest.mark.parametrize("cpus", [1, 4, 64])
def test_a_set_job_limit_wins_whatever_the_cpus(
    models: Path, monkeypatch: pytest.MonkeyPatch, cpus: int
) -> None:
    _with_recorder(monkeypatch, Recorder())
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: cpus)
    application = create_app(settings=bootstrap_settings(models, max_concurrent_jobs=5))

    with TestClient(application):
        assert application.state.limiter.free_slots == 5


def test_an_injected_container_still_gets_the_derived_job_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 8)
    fakes = Fakes()
    settings = api_settings(max_concurrent_jobs=None)
    application = create_app(settings=settings, container=fakes.container(settings))

    with TestClient(application):
        assert application.state.limiter.free_slots == 6


def test_importing_the_module_reads_no_environment() -> None:
    script = (
        "import app.main\n"
        "print(type(app.main.app).__name__, hasattr(app.main.app.state, 'settings'))\n"
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.lower() not in Settings.model_fields
    }

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["FastAPI", "False"]
