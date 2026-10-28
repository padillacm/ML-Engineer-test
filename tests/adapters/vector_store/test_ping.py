"""``QdrantVectorStore.ping``: the one-request readiness probe, with its own short timeout."""

import time
from typing import Any

import pytest
from qdrant_client import QdrantClient

from app.adapters.vector_store import QdrantStoreConfig, QdrantVectorStore, qdrant_store
from app.domain.errors import DependencyTimeout, DependencyUnavailable
from tests.adapters.vector_store.flaky_client import FlakyClient, connect_error, read_timeout
from tests.adapters.vector_store.servers import silent_server

MEMORY = QdrantStoreConfig(location=":memory:")


def _flaky(*failures: BaseException) -> tuple[QdrantVectorStore, FlakyClient]:
    flaky = FlakyClient(QdrantClient(location=":memory:"), {"get_collections": list(failures)})
    store = QdrantVectorStore(MEMORY, client=flaky.as_client(), sleep=lambda _: None)
    return store, flaky


def test_ping_answers_before_any_collection_is_bound() -> None:
    store, flaky = _flaky()

    store.ping()

    assert flaky.calls == {"get_collections": 1}
    assert store.collection is None


@pytest.mark.parametrize(
    ("failure", "error_type", "code"),
    [
        (connect_error(), DependencyUnavailable, "vector_store_unavailable"),
        (read_timeout(), DependencyTimeout, "vector_store_timeout"),
    ],
)
def test_ping_fails_after_one_request_without_retrying(
    failure: BaseException, error_type: type[Exception], code: str
) -> None:
    store, flaky = _flaky(failure, failure, failure)

    with pytest.raises(error_type) as caught:
        store.ping()

    assert getattr(caught.value, "code", None) == code
    assert flaky.calls == {"get_collections": 1}


def test_programming_errors_propagate() -> None:
    store, _ = _flaky(KeyError("bug"))

    with pytest.raises(KeyError):
        store.ping()


@pytest.mark.filterwarnings("ignore:Failed to obtain server version:UserWarning")
def test_ping_of_a_hung_server_ends_within_its_own_timeout() -> None:
    with silent_server() as url:
        store = QdrantVectorStore(
            QdrantStoreConfig(url=url, timeout_s=10, attempts=3, ping_timeout_s=1)
        )
        started = time.monotonic()

        with pytest.raises(DependencyTimeout) as raised:
            store.ping()

        elapsed = time.monotonic() - started
        store.close()

    assert raised.value.code == "vector_store_timeout"
    assert 0.9 <= elapsed < 2.0  # ping_timeout_s, not timeout_s (10 s)


class _RecordingClient:
    """Stands in for ``QdrantClient``: records how each instance was built and closed."""

    built: list["_RecordingClient"] = []  # noqa: RUF012 - shared on purpose, reset per test

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.calls: list[str] = []
        _RecordingClient.built.append(self)

    def get_collections(self) -> None:
        self.calls.append("get_collections")

    def close(self) -> None:
        self.closed = True


def test_server_store_pings_through_a_probe_client_with_the_ping_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingClient.built = []
    monkeypatch.setattr(qdrant_store, "QdrantClient", _RecordingClient)
    config = QdrantStoreConfig(url="http://qdrant:6333", timeout_s=10, ping_timeout_s=1.5)

    store = QdrantVectorStore(config)
    store.ping()
    store.close()

    main, probe = _RecordingClient.built
    assert main.kwargs["timeout"] == 10
    assert probe.kwargs["timeout"] == 2  # qdrant-client takes whole seconds; rounded up
    assert probe.kwargs["check_compatibility"] is False  # the main client checks the version
    assert (main.calls, probe.calls) == ([], ["get_collections"])
    assert main.closed
    assert probe.closed


def test_memory_store_pings_through_its_only_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingClient.built = []
    monkeypatch.setattr(qdrant_store, "QdrantClient", _RecordingClient)

    store = QdrantVectorStore(MEMORY)
    store.ping()
    store.close()

    (only,) = _RecordingClient.built
    assert only.calls == ["get_collections"]
    assert only.closed
