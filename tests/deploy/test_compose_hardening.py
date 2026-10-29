"""The hardened containers, the detector model's build and the server bounds (spec: deployment).

Runs everywhere (host and test image): it reads ``docker-compose.yml`` (and the other compose
files that build an image), the Dockerfile and the pre-commit configuration. What the options
do to a running stack (``docker inspect``, a write outside tmpfs refused) is checked by
``tests/deploy/test_hardening_smoke.py`` against the stack; these checks keep the wiring from
being edited away.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.yml"
DOCKERFILE = ROOT / "Dockerfile"
PRE_COMMIT = ROOT / ".pre-commit-config.yaml"
BUILDING = {
    "docker-compose.yml": "app",
    "docker-compose.test.yml": "tests",
    "docker-compose.smoke.yml": "smoke",
}
"""Each compose file that builds an image from `base`, and its building service."""

# S108: the container paths under test, not temporary files of this process.
HARDENED = {
    "app": ("8g", ["/tmp:size=1g"]),  # noqa: S108
    "qdrant": ("2g", ["/tmp", "/qdrant/snapshots:size=256m"]),  # noqa: S108
}
PIDS_LIMIT = 512
SERVER_FLAGS = ["--limit-concurrency", "64", "--timeout-keep-alive", "5"]
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PROMPT_GUARD_FILES = [
    "LICENSE",
    "README.md",
    "USE_POLICY.md",
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def _load(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))
    return loaded


def _service(name: str, compose: str = "docker-compose.yml") -> dict[str, Any]:
    service: dict[str, Any] = _load(compose)["services"][name]
    return service


def _dockerfile() -> str:
    if not DOCKERFILE.is_file():
        # The test image copies the tree without the Dockerfile (.dockerignore): these checks
        # run on the host, where the Dockerfile is.
        pytest.skip("the Dockerfile is not in the test image")
    return DOCKERFILE.read_text(encoding="utf-8")


def _stage(name: str) -> str:
    """The text of one Dockerfile stage, from its FROM line to the next one."""
    text = _dockerfile()
    match = re.search(rf"^FROM .* AS {re.escape(name)}\r?$", text, re.MULTILINE)
    assert match, f"no stage {name}"
    rest = text[match.end() :]
    following = re.search(r"^FROM ", rest, re.MULTILINE)
    body = rest if following is None else rest[: following.start()]
    # Without comment lines: the next stage's header comments precede its FROM line.
    return "\n".join(line for line in body.splitlines() if not line.startswith("#"))


# --- Finding: least privilege for the app and Qdrant ----------------------------------------------


@pytest.mark.parametrize("name", list(HARDENED))
def test_the_container_has_a_read_only_root_and_no_capabilities(name: str) -> None:
    service = _service(name)

    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "cap_add" not in service
    assert "privileged" not in service
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert "pid" not in service  # its own PID namespace


@pytest.mark.parametrize("name", list(HARDENED))
def test_the_container_has_process_and_memory_limits(name: str) -> None:
    memory, _ = HARDENED[name]

    assert _service(name)["deploy"]["resources"]["limits"] == {"pids": PIDS_LIMIT, "memory": memory}


@pytest.mark.parametrize("name", list(HARDENED))
def test_the_container_writes_to_tmpfs_and_its_volume_only(name: str) -> None:
    _, tmpfs = HARDENED[name]
    service = _service(name)

    assert service["tmpfs"] == tmpfs
    assert service.get("volumes", []) == (
        ["qdrant_data:/qdrant/storage"] if name == "qdrant" else []
    )


def test_the_app_memory_limit_is_at_least_the_pids_limit_sanity() -> None:
    """Both limits are honoured without swarm only under ``deploy.resources.limits``."""
    assert "pids_limit" not in _service("app")
    assert "mem_limit" not in _service("app")


def test_the_image_points_the_scratch_paths_at_tmp() -> None:
    runtime = _stage("runtime")

    assert re.search(r"^ENV TMPDIR=/tmp \\$", runtime, re.MULTILINE)
    assert "HF_HOME=/tmp/huggingface" in runtime
    assert "TORCH_HOME=/tmp/torch" in runtime


def test_qdrant_keeps_its_init_marker_on_tmpfs() -> None:
    environment = _service("qdrant")["environment"]

    assert environment["QDRANT_INIT_FILE_PATH"] == "/tmp/.qdrant-initialized"  # noqa: S108


# --- Finding: the server bounds its connections ---------------------------------------------------


def _dockerfile_cmd() -> list[str]:
    runtime = _stage("runtime")
    match = re.search(r'^CMD (\["uvicorn".*?\])\s*$', runtime, re.MULTILINE | re.DOTALL)
    assert match
    command: list[str] = json.loads(match[1].replace("\\\n", ""))
    return command


def _compose_uvicorn() -> list[str]:
    script: str = _service("app")["command"][2]
    exec_line = script.split("exec ")[1].replace("\\\n", " ")
    return exec_line.split()


@pytest.mark.parametrize("source", ["dockerfile", "compose"])
def test_the_server_limits_connections_and_closes_idle_ones(source: str) -> None:
    # Read inside the test, not at collection: the Dockerfile is absent in the test image.
    command = _dockerfile_cmd() if source == "dockerfile" else _compose_uvicorn()
    assert command[:5] == ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port"]  # noqa: S104
    assert command[6:8] == ["--http", "h11"]
    assert "--no-server-header" in command
    assert command[-4:] == SERVER_FLAGS


def test_compose_runs_the_image_command() -> None:
    assert _compose_uvicorn() == _dockerfile_cmd()


# --- Finding: the detector model is built in with a secret ----------------------------------------


@pytest.mark.parametrize(("compose", "service"), list(BUILDING.items()))
def test_each_build_passes_the_opt_out_argument_and_the_token_secret(
    compose: str, service: str
) -> None:
    build = _service(service, compose)["build"]

    assert build["args"] == {"INJECTION_MODEL": "${INJECTION_MODEL:-1}"}
    assert build["secrets"] == [{"source": "hf_token${HF_TOKEN:+_from_env}", "target": "hf_token"}]


@pytest.mark.parametrize("compose", ["docker-compose.yml", "docker-compose.test.yml"])
def test_the_token_comes_from_the_variable_or_else_the_placeholder(compose: str) -> None:
    secrets = _load(compose)["secrets"]

    assert secrets["hf_token_from_env"] == {"environment": "HF_TOKEN"}
    assert secrets["hf_token"] == {"file": "./deploy/empty-secret"}


def test_the_token_value_is_never_interpolated_or_passed_to_a_container() -> None:
    for compose in BUILDING:
        text = (ROOT / compose).read_text(encoding="utf-8")
        uses = re.findall(r"\$\{HF_TOKEN([^}]*)\}", text)
        assert set(uses) <= {":+_from_env"}, compose
        assert not re.search(r"^\s+HF_TOKEN:", text, re.MULTILINE), compose


def test_the_model_path_comes_from_the_image_not_from_env() -> None:
    """The image sets it only when built with the model, so the opt-out image has it unset."""
    assert "INJECTION_MODEL_PATH" not in _service("app")["environment"]
    assert "ENV INJECTION_MODEL_PATH=/models/prompt-guard" in _stage("base-1")
    assert "INJECTION_MODEL_PATH" not in _stage("base-0")
    assert "INJECTION_MODEL_PATH" not in _stage("base-common")
    assert re.search(r"^FROM base-\$\{INJECTION_MODEL\} AS base$", _dockerfile(), re.MULTILINE)
    assert re.search(r"^ARG INJECTION_MODEL=1$", _dockerfile(), re.MULTILINE)
    assert "COPY --from=prompt-guard /models/prompt-guard /models/prompt-guard" in _stage("base-1")


def test_the_token_is_a_build_secret_of_the_download_step_only() -> None:
    stage = _stage("prompt-guard")

    assert "RUN --mount=type=secret,id=hf_token,env=HF_TOKEN python - <<'PY'" in stage
    assert "ENV HF_TOKEN" not in stage
    assert "ARG HF_TOKEN" not in _dockerfile()
    assert "--build-arg HF_TOKEN" not in _dockerfile()


def test_every_model_file_is_pinned_by_sha256_at_a_pinned_revision() -> None:
    stage = _stage("prompt-guard")
    args = dict(re.findall(r"^ARG (PROMPT_GUARD_[A-Z0-9_]+)=(\S+)$", stage, re.MULTILINE))

    assert args["PROMPT_GUARD_REPO"] == "meta-llama/Llama-Prompt-Guard-2-86M"
    assert re.fullmatch(r"[0-9a-f]{40}", args["PROMPT_GUARD_REVISION"])
    hashes = {name: value for name, value in args.items() if name.endswith("_SHA256")}
    assert len(hashes) == len(PROMPT_GUARD_FILES)
    assert all(SHA256.match(value) for value in hashes.values()), hashes
    for file in PROMPT_GUARD_FILES:
        assert f'"{file}": "PROMPT_GUARD_' in stage
    for variable in hashes:
        assert f'"{variable}"' in stage  # each pinned hash is checked


def test_the_notice_carries_the_llama_attribution() -> None:
    stage = _stage("prompt-guard")

    assert "Llama 4 Community License" in stage
    assert '"Built with Llama\\n"' in stage
    assert "Llama 4 is licensed under the Llama 4 Community License, Copyright" in stage
    assert "Meta Platforms" in stage
    assert "All Rights Reserved" in stage


# --- Finding: dependencies are audited before a push ----------------------------------------------


def test_the_dependency_audit_runs_on_push() -> None:
    (repo,) = yaml.safe_load(PRE_COMMIT.read_text(encoding="utf-8"))["repos"]
    (audit,) = [hook for hook in repo["hooks"] if hook["id"] == "uv-audit"]

    assert audit["entry"] == "uv audit --locked"
    assert audit["stages"] == ["pre-push"]
    assert audit["always_run"] is True
