"""The ``DEVICE`` setting in the composition root: resolution, start-up line and GPU pools
(specs: inference-device, service-configuration). The GPU is simulated by patching
``torch.cuda``, so these run on any host."""

import json
import logging
from pathlib import Path

import pytest

import app.bootstrap
from app.adapters.embeddings.config import E5EmbedderConfig
from app.adapters.injection.config import InjectionDetectorConfig
from app.adapters.ner.config import TransformersNerConfig
from app.bootstrap import (
    CPU_DEVICE,
    GPU_EMBEDDING_POOL,
    GPU_INJECTION_POOL,
    GPU_NER_POOL,
    Concurrency,
    InferenceDevice,
    adapter_configs,
    build_container,
    concurrency_for,
    resolve_concurrency,
    resolve_device,
)
from app.domain.errors import ConfigurationError
from app.observability import JsonFormatter
from tests.bootstrap.support import Recorder, bootstrap_settings, model_directory

GPU_NAME = "NVIDIA GeForce RTX 4070 Ti"
GPU_MIB = 12282
GPU = InferenceDevice("cuda:0", gpu_name=GPU_NAME, gpu_memory_mib=GPU_MIB)


class _Properties:
    name = GPU_NAME
    total_memory = GPU_MIB * 1024 * 1024


@pytest.fixture
def models(tmp_path: Path) -> Path:
    model_directory(tmp_path, "ner")
    model_directory(tmp_path, "e5")
    model_directory(tmp_path, "prompt-guard")
    return tmp_path


@pytest.fixture
def with_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """``torch.cuda`` reports one RTX 4070 Ti."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_properties", lambda index: _Properties())


@pytest.fixture
def without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """``torch.cuda`` reports no GPU, whatever the host has."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)


@pytest.mark.usefixtures("with_gpu")
def test_auto_with_a_gpu_is_the_first_gpu() -> None:
    assert resolve_device("auto") == GPU


@pytest.mark.usefixtures("without_gpu")
def test_auto_without_a_gpu_is_the_cpu() -> None:
    assert resolve_device("auto") == CPU_DEVICE


@pytest.mark.usefixtures("with_gpu")
def test_cpu_is_the_cpu_even_with_a_gpu() -> None:
    assert resolve_device("cpu") == CPU_DEVICE


def test_cpu_does_not_probe_the_hardware() -> None:
    def probe() -> bool:
        raise AssertionError("probed")

    assert resolve_device("cpu", visible=probe) == CPU_DEVICE


@pytest.mark.usefixtures("with_gpu")
def test_cuda_with_a_gpu_is_the_first_gpu() -> None:
    assert resolve_device("cuda") == GPU


@pytest.mark.usefixtures("without_gpu")
def test_cuda_without_a_gpu_is_a_configuration_error_naming_device() -> None:
    with pytest.raises(ConfigurationError) as caught:
        resolve_device("cuda")

    assert caught.value.message == "DEVICE: cuda requested but no GPU is visible"


@pytest.mark.usefixtures("without_gpu")
def test_building_with_cuda_and_no_gpu_fails_before_any_adapter(models: Path) -> None:
    recorder = Recorder()

    with pytest.raises(ConfigurationError) as caught:
        build_container(bootstrap_settings(models, device="cuda"), recorder.factories())

    assert caught.value.message == "DEVICE: cuda requested but no GPU is visible"
    assert not recorder.built


@pytest.mark.usefixtures("with_gpu")
def test_the_resolved_device_reaches_the_three_model_configs(models: Path) -> None:
    settings = bootstrap_settings(
        models, device="auto", injection_model_path=str(models / "prompt-guard")
    )
    recorder = Recorder()

    build_container(settings, recorder.factories(), resolve_concurrency(settings, cpus=8)).close()

    assert recorder.configs["recognizer"] == TransformersNerConfig(
        model_path=str(models / "ner"), pool_size=GPU_NER_POOL, device="cuda:0"
    )
    assert recorder.configs["embedder"] == E5EmbedderConfig(
        model_path=str(models / "e5"),
        revision="rev-1",
        pool_size=GPU_EMBEDDING_POOL,
        device="cuda:0",
    )
    assert recorder.configs["detector"] == InjectionDetectorConfig(
        model_path=str(models / "prompt-guard"), pool_size=GPU_INJECTION_POOL, device="cuda:0"
    )


@pytest.mark.usefixtures("without_gpu")
def test_without_a_gpu_auto_gives_cpu_configs(models: Path) -> None:
    settings = bootstrap_settings(
        models, device="auto", injection_model_path=str(models / "prompt-guard")
    )

    configs = adapter_configs(settings, resolve_concurrency(settings, cpus=8))

    assert configs.injection is not None
    assert (configs.ner.device, configs.embedder.device, configs.injection.device) == (
        "cpu",
        "cpu",
        "cpu",
    )
    assert configs.injection.pool_size == 2


def test_an_invalid_device_in_a_config_names_device(models: Path) -> None:
    sizes = Concurrency(8, 6, 4, 2, 4, device=InferenceDevice("tpu:0"))

    with pytest.raises(ConfigurationError) as caught:
        adapter_configs(bootstrap_settings(models), sizes)

    assert caught.value.message.startswith("DEVICE: device must be")


@pytest.mark.parametrize("cpus", [1, 4, 8, 16, 64])
def test_the_gpu_rule_fixes_the_model_pools_and_keeps_the_cpu_rule_for_jobs_and_ocr(
    cpus: int,
) -> None:
    cpu, gpu = concurrency_for(cpus, 2), concurrency_for(cpus, 2, GPU)

    assert (gpu.ner_pool_size, gpu.embedding_pool_size, gpu.injection_pool_size) == (
        GPU_NER_POOL,
        GPU_EMBEDDING_POOL,
        GPU_INJECTION_POOL,
    )
    assert (gpu.max_concurrent_jobs, gpu.ocr_workers, gpu.torch_threads) == (
        cpu.max_concurrent_jobs,
        cpu.ocr_workers,
        cpu.torch_threads,
    )
    assert gpu.device == GPU
    assert cpu.injection_pool_size == 2


def test_the_gpu_defaults_are_the_measured_values() -> None:
    # BENCHMARKS 23: every pool size tried was within 3 %, so the smallest (rule of section 18).
    assert (GPU_NER_POOL, GPU_EMBEDDING_POOL, GPU_INJECTION_POOL) == (1, 1, 1)


def test_set_pool_sizes_win_on_the_gpu(models: Path) -> None:
    settings = bootstrap_settings(
        models, ner_pool_size=5, embedding_pool_size=3, injection_pool_size=4
    )

    resolved = resolve_concurrency(settings, cpus=8, device=GPU)

    assert (resolved.ner_pool_size, resolved.embedding_pool_size) == (5, 3)
    assert resolved.injection_pool_size == 4


def test_the_cpu_rule_is_unchanged_by_the_device_argument() -> None:
    assert concurrency_for(8, 2, CPU_DEVICE) == concurrency_for(8, 2) == Concurrency(8, 6, 4, 2, 4)


def _concurrency_line(caplog: pytest.LogCaptureFixture) -> dict[str, object]:
    (record,) = [r for r in caplog.records if r.getMessage() == "concurrency in use"]
    line: dict[str, object] = json.loads(JsonFormatter().format(record))
    return line


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "app.bootstrap" and r.levelno >= logging.WARNING]


@pytest.mark.usefixtures("with_gpu")
def test_the_start_up_line_on_the_gpu_names_the_device_the_gpu_and_its_pools(
    models: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 8)

    with caplog.at_level(logging.INFO, logger="app.bootstrap"):
        build_container(bootstrap_settings(models, device="auto"), Recorder().factories()).close()

    line = _concurrency_line(caplog)
    assert (line["device"], line["gpu_name"], line["gpu_memory_mib"]) == (
        "cuda:0",
        GPU_NAME,
        GPU_MIB,
    )
    assert (line["ner_pool_size"], line["embedding_pool_size"]) == (
        GPU_NER_POOL,
        GPU_EMBEDDING_POOL,
    )
    assert (line["injection_pool_size"], line["max_concurrent_jobs"]) == (GPU_INJECTION_POOL, 6)


@pytest.mark.usefixtures("without_gpu")
def test_the_start_up_line_on_the_cpu_says_cpu_without_gpu_fields(
    models: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 8)

    with caplog.at_level(logging.INFO, logger="app.bootstrap"):
        build_container(bootstrap_settings(models, device="auto"), Recorder().factories()).close()

    line = _concurrency_line(caplog)
    assert line["device"] == "cpu"
    assert "gpu_name" not in line
    assert "gpu_memory_mib" not in line


@pytest.mark.usefixtures("with_gpu")
def test_pools_over_the_cpus_warn_only_on_the_cpu(
    models: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(app.bootstrap, "available_cpus", lambda: 2)
    oversized = {"ner_pool_size": 4, "embedding_pool_size": 4, "torch_threads": 2}

    with caplog.at_level(logging.WARNING, logger="app.bootstrap"):
        gpu = bootstrap_settings(models, device="cuda", **oversized)
        build_container(gpu, Recorder().factories()).close()
    on_gpu = _warnings(caplog)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app.bootstrap"):
        cpu = bootstrap_settings(models, device="cpu", **oversized)
        build_container(cpu, Recorder().factories()).close()

    assert on_gpu == []
    assert [r.getMessage().split(" x ")[0] for r in _warnings(caplog)] == [
        "NER_POOL_SIZE",
        "EMBEDDING_POOL_SIZE",
    ]
