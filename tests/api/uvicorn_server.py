"""A real Uvicorn process serving ``tests.api.served_app`` on a free local port.

``served(...)`` starts it, waits until it accepts connections and stops it afterwards. Its
stdout (the service's JSON log lines) is collected as it arrives, so a test can wait for a line
such as the access line of a request whose client has gone.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START_S = 30.0
"""Longest wait for the server to accept connections (imports take a few seconds)."""


class Server:
    """A running server: its port and the JSON log lines it has written so far."""

    def __init__(self, process: subprocess.Popen[str], port: int) -> None:
        self.process = process
        self.port = port
        self._lines: list[dict[str, object]] = []
        self._changed = threading.Condition()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for raw in self.process.stdout:
            if not raw.startswith("{"):
                continue
            try:
                line = json.loads(raw)
            except ValueError:
                continue
            with self._changed:
                self._lines.append(line)
                self._changed.notify_all()

    def lines(self) -> list[dict[str, object]]:
        with self._changed:
            return list(self._lines)

    def wait_for(
        self, matches: Callable[[dict[str, object]], bool], timeout_s: float = 10.0
    ) -> dict[str, object] | None:
        """The first log line ``matches`` accepts, waiting up to ``timeout_s`` for it."""
        deadline = time.monotonic() + timeout_s
        with self._changed:
            while True:
                for line in self._lines:
                    if matches(line):
                        return line
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._changed.wait(remaining)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


@contextmanager
def served(
    settings: Mapping[str, object] | None = None,
    *,
    http: str = "auto",
    env: dict[str, str] | None = None,
    target: str = "tests.api.served_app:create",
) -> Iterator[Server]:
    """Run Uvicorn on ``target`` (an app factory) with ``SERVED_APP_SETTINGS=settings``."""
    port = free_port()
    environment = {
        **os.environ,
        "SERVED_APP_SETTINGS": json.dumps(settings or {}),
        "PYTHONUNBUFFERED": "1",
        **(env or {}),
    }
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        target,
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--http",
        http,
        "--log-level",
        "warning",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    try:
        _wait_until_listening(process, port)
        yield Server(process, port)
    finally:
        process.terminate()
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(10)


def _wait_until_listening(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + START_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise RuntimeError(f"uvicorn exited with {process.returncode}: {output[-2000:]}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("uvicorn did not start listening in time")
