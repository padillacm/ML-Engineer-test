"""The two torch builds: extras, image build argument and GPU reservation (spec: deployment).

Wiring checks run everywhere (host and test image): they read ``pyproject.toml``, ``uv.lock``,
the ``Dockerfile`` and the compose files, and evaluate the compose expressions that pick the
variant with the same interpolation helper as ``test_compose_wiring.py``. When the ``docker``
command is on PATH, the merged configuration compose itself produces is checked too. The hygiene
checks look at the torch installed in the running environment of an image (the images set
``TORCH_VARIANT``); on a host without it they are skipped.
"""

import importlib.metadata
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs

from tests.deploy.test_compose_wiring import interpolate

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"
COMPOSE_TEST = ROOT / "docker-compose.test.yml"
COMPOSE_TORCH = ROOT / "docker-compose.torch.yml"

VARIANTS = ("cpu", "cuda")
GPU_PACKAGE = re.compile(r"^(nvidia-|cuda-|triton$)")


def _pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def _requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9._-]+", requirement)
    assert match, requirement
    return match[0].lower()


# --- pyproject.toml and uv.lock ----------------------------------------------------------------


def test_torch_is_only_in_the_two_variant_extras() -> None:
    project = _pyproject()["project"]
    extras = project["optional-dependencies"]

    assert "torch" not in {_requirement_name(r) for r in project["dependencies"]}
    assert extras["cpu"] == extras["cuda"]
    assert [_requirement_name(r) for r in extras["cpu"]] == ["torch"]
    assert "torch" not in {_requirement_name(r) for r in extras["dev"]}


def test_the_variant_extras_conflict() -> None:
    conflicts = _pyproject()["tool"]["uv"]["conflicts"]

    assert conflicts == [[{"extra": "cpu"}, {"extra": "cuda"}]]


def test_each_extra_takes_torch_from_its_own_explicit_index() -> None:
    tool = _pyproject()["tool"]["uv"]
    indexes = {index["name"]: index for index in tool["index"]}
    sources = {source["extra"]: source["index"] for source in tool["sources"]["torch"]}

    assert set(sources) == set(VARIANTS)
    assert indexes[sources["cpu"]]["url"] == "https://download.pytorch.org/whl/cpu"
    assert re.fullmatch(
        r"https://download\.pytorch\.org/whl/cu12\d", indexes[sources["cuda"]]["url"]
    )
    assert sources["cuda"] == "pytorch-" + indexes[sources["cuda"]]["url"].rsplit("/", 1)[1]
    assert all(indexes[name]["explicit"] for name in sources.values())


def test_the_lock_holds_both_builds() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    torch = [package for package in lock["package"] if package["name"] == "torch"]
    cuda_index = next(
        index["url"]
        for index in _pyproject()["tool"]["uv"]["index"]
        if index["name"].startswith("pytorch-cu")
    )

    registries = {package["source"]["registry"] for package in torch}
    assert registries == {"https://download.pytorch.org/whl/cpu", cuda_index}
    assert any(package["version"].endswith("+cpu") for package in torch)


def test_the_cuda_marker_is_registered() -> None:
    markers = _pyproject()["tool"]["pytest"]["ini_options"]["markers"]

    assert any(marker.startswith("cuda: ") for marker in markers)


# --- Dockerfile --------------------------------------------------------------------------------


def _dockerfile() -> str:
    if not DOCKERFILE.is_file():
        # The test image copies the tree without the Dockerfile (.dockerignore): these checks
        # run on the host, where the Dockerfile is.
        pytest.skip("the Dockerfile is not in the test image")
    return DOCKERFILE.read_text(encoding="utf-8")


def _stages() -> dict[str, str]:
    """Each ``FROM ... AS <name>`` stage's text (up to the next FROM)."""
    parts = re.split(r"^(FROM \S+ AS (\S+))$", _dockerfile(), flags=re.MULTILINE)
    return {parts[i + 1]: parts[i] + parts[i + 2] for i in range(1, len(parts) - 1, 3)}


def test_the_build_argument_defaults_to_cpu() -> None:
    preamble = _dockerfile().split("\nFROM ", 1)[0]

    assert re.search(r"^ARG TORCH_VARIANT=cpu$", preamble, re.MULTILINE)


@pytest.mark.parametrize("stage", ["test", "e2e", "runtime"])
def test_every_sync_of_the_python_stages_installs_the_variant(stage: str) -> None:
    text = _stages()[stage]
    syncs = re.findall(r"uv sync[^\n]*", text)

    assert "ARG TORCH_VARIANT" in text
    assert syncs
    assert [sync for sync in syncs if '--extra "${TORCH_VARIANT}"' not in sync] == []


@pytest.mark.parametrize("stage", ["test", "runtime"])
def test_the_variant_is_validated_and_checked_after_the_sync(stage: str) -> None:
    text = _stages()[stage]

    assert re.search(r'case "\$\{TORCH_VARIANT\}" in cpu\|cuda\) ;;', text)
    assert "ENV TORCH_VARIANT=${TORCH_VARIANT}" in text
    assert text.index("ENV TORCH_VARIANT") < text.index("uv sync")
    # The post-sync check refuses a CPU image with NVIDIA packages and a CUDA image without CUDA.
    check = text[text.rindex("uv sync") :]
    assert 'startswith(("nvidia-", "cuda-"))' in check
    assert 'variant == "cuda" and torch.version.cuda is None' in check


def test_no_cuda_base_image() -> None:
    """The CUDA runtime comes with the torch wheel; every stage starts from a non-NVIDIA image."""
    images = re.findall(r"^FROM (\S+)", _dockerfile(), re.MULTILINE)
    python_image = re.search(r"^ARG PYTHON_IMAGE=(\S+)$", _dockerfile(), re.MULTILINE)

    assert python_image
    assert images
    assert [image for image in images if "nvidia" in image or "cuda" in image] == []
    assert python_image[1].startswith("python:3.12-slim")


# --- compose: the variant and the reservation --------------------------------------------------

VARIANT_CASES = [
    ({}, "cpu"),
    ({"COMPOSE_PROFILES": "gpu"}, "cuda"),
    ({"TORCH_VARIANT": "cuda"}, "cuda"),
    ({"TORCH_VARIANT": "cpu", "COMPOSE_PROFILES": "gpu"}, "cpu"),
    ({"TORCH_VARIANT": "", "COMPOSE_PROFILES": ""}, "cpu"),
]


def _torch_service(extending: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    extends = extending["extends"]
    assert extends["file"] == COMPOSE_TORCH.name
    services: dict[str, Any] = _yaml(COMPOSE_TORCH)["services"]
    name = interpolate(extends["service"], env)
    assert name in services, f"{name} is not a service of {COMPOSE_TORCH.name}"
    service: dict[str, Any] = services[name]
    return service


@pytest.mark.parametrize(("env", "variant"), VARIANT_CASES)
def test_the_app_variant_follows_the_gpu_profile(env: dict[str, str], variant: str) -> None:
    service = _torch_service(_yaml(COMPOSE)["services"]["app"], env)

    assert service["build"]["args"]["TORCH_VARIANT"] == variant
    assert ("deploy" in service) == (variant == "cuda")


@pytest.mark.parametrize(
    ("env", "variant"),
    [({}, "cpu"), ({"COMPOSE_PROFILES": "gpu"}, "cpu"), ({"TORCH_VARIANT": "cuda"}, "cuda")],
)
def test_the_test_image_variant_is_cpu_unless_asked(env: dict[str, str], variant: str) -> None:
    service = _torch_service(_yaml(COMPOSE_TEST)["services"]["tests"], env)

    assert service["build"]["args"]["TORCH_VARIANT"] == variant


def test_the_cuda_variant_reserves_the_gpu_like_ollama() -> None:
    services = _yaml(COMPOSE_TORCH)["services"]
    ollama = _yaml(COMPOSE)["services"]["ollama"]

    assert services["torch-cuda"]["deploy"] == {
        "resources": {"reservations": ollama["deploy"]["resources"]["reservations"]}
    }
    assert {name for name, service in services.items() if "deploy" in service} == {"torch-cuda"}


def test_the_extending_services_leave_the_variant_to_the_fragment() -> None:
    """Neither the app nor the tests set TORCH_VARIANT or a reservation themselves."""
    app = _yaml(COMPOSE)["services"]["app"]
    tests = _yaml(COMPOSE_TEST)["services"]["tests"]

    for service in (app, tests):
        assert "TORCH_VARIANT" not in service["build"].get("args", {})
        assert "reservations" not in service.get("deploy", {}).get("resources", {})


def test_the_fragment_is_no_stack_of_its_own() -> None:
    """Only build arguments and the reservation: nothing that could run on its own."""
    for service in _yaml(COMPOSE_TORCH)["services"].values():
        assert set(service) <= {"build", "deploy"}
        assert set(service["build"]) == {"args"}


def _compose_config(env: dict[str, str], tmp_path: Path) -> dict[str, Any]:
    """``docker compose config`` with only ``env`` (an empty env file instead of ``.env``)."""
    empty = tmp_path / "empty.env"
    empty.write_text("", encoding="utf-8")
    clean = {
        key: value
        for key, value in os.environ.items()
        if key not in {"TORCH_VARIANT", "COMPOSE_PROFILES", "COMPOSE_FILE"}
    }
    result = subprocess.run(
        ["docker", "compose", "--env-file", str(empty), "-f", str(COMPOSE), "config"],
        capture_output=True,
        text=True,
        env={**clean, **env},
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    loaded: dict[str, Any] = yaml.safe_load(result.stdout)
    return loaded


@pytest.mark.skipif(shutil.which("docker") is None, reason="needs the docker command")
@pytest.mark.parametrize(("env", "variant"), VARIANT_CASES[:2])
def test_compose_merges_the_variant_into_the_hardened_app(
    env: dict[str, str], variant: str, tmp_path: Path
) -> None:
    app = _compose_config(env, tmp_path)["services"]["app"]

    assert app["build"]["args"]["TORCH_VARIANT"] == variant
    assert app["build"]["args"]["INJECTION_MODEL"] == "1"
    resources = app["deploy"]["resources"]
    assert ("reservations" in resources) == (variant == "cuda")
    assert set(resources["limits"]) == {"pids", "memory"}
    assert app["read_only"] is True
    assert app["cap_drop"] == ["ALL"]
    assert app["security_opt"] == ["no-new-privileges:true"]


# --- hygiene of the installed torch (in an image) ----------------------------------------------

IMAGE_VARIANT = os.environ.get("TORCH_VARIANT", "")
in_an_image = pytest.mark.skipif(
    IMAGE_VARIANT not in VARIANTS, reason="TORCH_VARIANT unset (the images set it)"
)


def _gpu_packages() -> list[str]:
    names = {(d.metadata["Name"] or "").lower() for d in importlib.metadata.distributions()}
    return sorted(name for name in names if GPU_PACKAGE.match(name))


@in_an_image
def test_the_installed_torch_is_the_variant() -> None:
    import torch  # noqa: PLC0415  # only where an image says which build it holds

    if IMAGE_VARIANT == "cpu":
        assert torch.version.cuda is None
        assert torch.__version__.endswith("+cpu")
        assert not torch.cuda.is_available()
        assert _gpu_packages() == []
    else:
        assert torch.version.cuda is not None


@pytest.mark.cuda
def test_torch_sees_the_gpu() -> None:
    import torch  # noqa: PLC0415  # imported by the test, as the skip rule needs the GPU anyway

    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0)
