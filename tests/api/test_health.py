"""Liveness and readiness (spec: http-api, service-configuration)."""

import pytest
from fastapi.testclient import TestClient

from app.domain.errors import DependencyUnavailable
from tests.api.conftest import ClientFactory, LogLines
from tests.api.support import Fakes


def test_liveness_answers_ok_without_calling_any_dependency(
    make_client: ClientFactory, fakes: Fakes
) -> None:
    probed: list[str] = []

    def probe() -> str:
        probed.append("store")
        return "ok"

    fakes.readiness = (("vector_store", probe),)
    client = make_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert probed == []
    assert fakes.embedder.query_calls == []
    assert fakes.embedder.passage_calls == []


def test_readiness_reports_every_check(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"vector_store": "ok", "llm": "configured"},
    }


def _store_down() -> str:
    raise DependencyUnavailable("the vector store could not be reached", "vector_store_unavailable")


def _broken() -> str:
    raise RuntimeError("unexpected probe failure")


@pytest.mark.parametrize("probe", [_store_down, _broken])
def test_store_down_makes_readiness_503_naming_the_check_while_liveness_is_200(
    make_client: ClientFactory, fakes: Fakes, logs: LogLines, probe: object
) -> None:
    fakes.readiness = (("vector_store", probe), ("llm", lambda: "configured"))  # type: ignore[assignment]
    client = make_client()

    ready = client.get("/health/ready")
    alive = client.get("/health")

    assert ready.status_code == 503
    assert ready.json() == {
        "status": "not_ready",
        "checks": {"vector_store": "unavailable", "llm": "configured"},
    }
    assert alive.status_code == 200
    (failed,) = logs.with_message("readiness check failed")
    assert failed["check"] == "vector_store"
    assert failed["ready"] is False


def test_readiness_without_a_language_model_is_still_ready(
    make_client: ClientFactory, fakes: Fakes
) -> None:
    fakes.readiness = (("vector_store", lambda: "ok"), ("llm", lambda: "not_configured"))
    client = make_client()

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["llm"] == "not_configured"


def test_openapi_documents_the_endpoints_and_the_ingest_addition(
    make_client: ClientFactory,
) -> None:
    client = make_client(expose_docs=True)  # off by default (spec: http-api)

    schema = client.get("/openapi.json").json()

    assert set(schema["paths"]) == {
        "/health",
        "/health/ready",
        "/api/extract",
        "/api/ingest",
        "/api/ask",
    }
    ingest = schema["components"]["schemas"]["IngestResponseOut"]
    assert set(ingest["properties"]) == {"status", "chunks_stored", "document_id"}
    assert client.get("/docs").status_code == 200
