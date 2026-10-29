"""The running stack's containers have the hardening options (spec: deployment, hardened stack).

Skipped unless ``SERVICE_URL`` names the service, like the other smoke tests, and unless Docker
is reachable: through the ``docker`` command (a run from the host) or through the Engine API on
the socket ``DOCKER_SOCKET`` names (the ``hardening`` service of ``docker-compose.smoke.yml``
mounts it, read-only, into the test image, which has no docker command). The containers are
found by the labels compose sets (project ``COMPOSE_PROJECT_NAME``, ``docintel`` by default).
Each check reads ``docker inspect`` or runs one shell command inside a container.
"""

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.deployed_service

PROJECT = os.environ.get("COMPOSE_PROJECT_NAME") or "docintel"
GIB = 1024**3
PIDS_LIMIT = 512
# S108: the containers' tmpfs paths under test, not temporary files of this process.
LIMITS = {
    "app": (8 * GIB, {"/tmp": "size=1g"}),  # noqa: S108
    "qdrant": (2 * GIB, {"/tmp": "", "/qdrant/snapshots": "size=256m"}),  # noqa: S108
}
"""Each hardened service, its memory limit in bytes and its tmpfs mounts as Docker records them."""

READ_ONLY_PATHS = ["/app", "/app/app", "/home/app", "/models", "/opt/venv", "/"]
EXEC_TIMEOUT_S = 30.0


class Docker:
    """``docker inspect`` and ``docker exec`` through the command or the Engine API."""

    def __init__(self, command: str | None, socket: str | None) -> None:
        self.command = command
        self.socket = socket
        self.api = (
            None
            if socket is None
            else httpx.Client(
                transport=httpx.HTTPTransport(uds=socket),
                base_url="http://docker",
                timeout=EXEC_TIMEOUT_S,
            )
        )

    def close(self) -> None:
        if self.api is not None:
            self.api.close()

    def container(self, service: str) -> str:
        labels = [f"com.docker.compose.project={PROJECT}", f"com.docker.compose.service={service}"]
        if self.api is None:
            filters = [arg for label in labels for arg in ("--filter", f"label={label}")]
            ids = self._run(["ps", "-q", *filters]).split()
        else:
            response = self.api.get(
                "/containers/json", params={"filters": json.dumps({"label": labels})}
            )
            response.raise_for_status()
            ids = [entry["Id"] for entry in response.json()]
        if len(ids) != 1:
            pytest.skip(f"expected one running container of {PROJECT}/{service}, found {len(ids)}")
        return ids[0]

    def inspect(self, container: str) -> dict[str, Any]:
        if self.api is None:
            (entry,) = json.loads(self._run(["inspect", container]))
        else:
            response = self.api.get(f"/containers/{container}/json")
            response.raise_for_status()
            entry = response.json()
        assert isinstance(entry, dict)
        return entry

    def exec(self, container: str, script: str) -> int:
        """Exit status of ``sh -c script`` inside the container."""
        if self.api is None:
            result = subprocess.run(  # fixed, test-controlled arguments
                [self.command or "docker", "exec", container, "sh", "-c", script],
                capture_output=True,
                timeout=EXEC_TIMEOUT_S,
                check=False,
            )
            return result.returncode
        created = self.api.post(
            f"/containers/{container}/exec",
            json={"Cmd": ["sh", "-c", script], "AttachStdout": True, "AttachStderr": True},
        )
        created.raise_for_status()
        exec_id = created.json()["Id"]
        self.api.post(
            f"/exec/{exec_id}/start", json={"Detach": False, "Tty": False}
        ).raise_for_status()
        state = self.api.get(f"/exec/{exec_id}/json")
        state.raise_for_status()
        code: int = state.json()["ExitCode"]
        return code

    def _run(self, arguments: list[str]) -> str:
        result = subprocess.run(  # fixed, test-controlled arguments
            [self.command or "docker", *arguments],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_S,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout


@pytest.fixture(scope="module")
def docker() -> Iterator[Docker]:
    command = shutil.which("docker")
    socket = os.environ.get("DOCKER_SOCKET") or None
    if command is None and (socket is None or not Path(socket).exists()):
        pytest.skip("needs the docker command or the Engine API socket named by DOCKER_SOCKET")
    engine = Docker(command, None if command is not None else socket)
    yield engine
    engine.close()


@pytest.fixture(scope="module")
def inspected(docker: Docker) -> dict[str, dict[str, Any]]:
    return {service: docker.inspect(docker.container(service)) for service in LIMITS}


@pytest.mark.parametrize("service", list(LIMITS))
def test_the_root_filesystem_is_read_only(
    inspected: dict[str, dict[str, Any]], service: str
) -> None:
    host = inspected[service]["HostConfig"]

    assert host["ReadonlyRootfs"] is True
    assert host["Privileged"] is False


@pytest.mark.parametrize("service", list(LIMITS))
def test_every_capability_is_dropped_and_none_can_be_gained(
    inspected: dict[str, dict[str, Any]], service: str
) -> None:
    host = inspected[service]["HostConfig"]

    assert host["CapDrop"] == ["ALL"]
    assert not host.get("CapAdd")
    assert "no-new-privileges:true" in (host.get("SecurityOpt") or [])


@pytest.mark.parametrize("service", list(LIMITS))
def test_the_process_and_memory_limits_are_set(
    inspected: dict[str, dict[str, Any]], service: str
) -> None:
    memory, _ = LIMITS[service]
    host = inspected[service]["HostConfig"]

    assert host["PidsLimit"] == PIDS_LIMIT
    assert host["Memory"] == memory


@pytest.mark.parametrize("service", list(LIMITS))
def test_the_scratch_space_is_tmpfs(inspected: dict[str, dict[str, Any]], service: str) -> None:
    _, tmpfs = LIMITS[service]

    assert inspected[service]["HostConfig"]["Tmpfs"] == tmpfs


def test_the_app_container_has_no_token_in_its_environment(
    inspected: dict[str, dict[str, Any]],
) -> None:
    variables = [entry.split("=", 1)[0] for entry in inspected["app"]["Config"]["Env"]]

    assert "HF_TOKEN" not in variables


@pytest.mark.parametrize("path", READ_ONLY_PATHS)
def test_a_write_outside_tmpfs_fails_in_the_app(docker: Docker, path: str) -> None:
    container = docker.container("app")

    assert docker.exec(container, f"touch {path}/smoke-write-probe") != 0


def test_a_write_to_tmpfs_works_in_the_app(docker: Docker) -> None:
    container = docker.container("app")

    assert docker.exec(container, "touch /tmp/smoke-write-probe && rm /tmp/smoke-write-probe") == 0


def test_qdrant_writes_to_its_volume_and_tmpfs_only(docker: Docker) -> None:
    container = docker.container("qdrant")

    assert docker.exec(container, "touch /qdrant/smoke-write-probe") != 0
    assert docker.exec(container, "touch /qdrant/config/smoke-write-probe") != 0
    assert docker.exec(container, "touch /qdrant/snapshots/p && rm /qdrant/snapshots/p") == 0
    assert docker.exec(container, "test -f /tmp/.qdrant-initialized") == 0
