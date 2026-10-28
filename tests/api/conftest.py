"""Fixtures of the HTTP tests: an application over fakes, its client and its log lines."""

import io
import json
import logging
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.observability import JsonFormatter
from tests.api.support import Fakes, api_settings

type ClientFactory = Callable[..., TestClient]


class LogLines:
    """Every log line written while the fixture is active, as JSON objects."""

    def __init__(self) -> None:
        self.stream = io.StringIO()

    def clear(self) -> None:
        self.stream.seek(0)
        self.stream.truncate()

    @property
    def text(self) -> str:
        return self.stream.getvalue()

    def lines(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in self.text.splitlines() if line.startswith("{")]

    def with_message(self, message: str) -> list[dict[str, object]]:
        return [line for line in self.lines() if line["message"] == message]


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    """The lifespan configures logging process-wide; put the root logger back afterwards."""
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    yield
    for handler in root.handlers[:]:
        if handler not in saved[0]:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved[1])


@pytest.fixture
def logs() -> Iterator[LogLines]:
    """Formats every record reaching the root logger with the service's JSON formatter.

    The record is formatted in the thread that logs it, so the request id is the one the
    service sees.
    """
    captured = LogLines()
    handler = logging.StreamHandler(captured.stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    yield captured
    root.removeHandler(handler)


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()


@pytest.fixture
def make_client(fakes: Fakes) -> Iterator[ClientFactory]:
    """Start an application over ``fakes`` with settings from ``api_settings(**overrides)``."""
    clients: list[TestClient] = []

    def make(**overrides: object) -> TestClient:
        active = api_settings(**overrides)
        client = TestClient(create_app(settings=active, container=fakes.container(active)))
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: ClientFactory) -> TestClient:
    return make_client()
