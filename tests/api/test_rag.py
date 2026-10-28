"""``POST /api/ingest`` and ``POST /api/ask`` (spec: http-api, grounded-answering)."""

import hashlib
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.application.ingest_document import document_id_for
from app.application.unconfigured_llm import UnconfiguredLLM
from app.domain.errors import DependencyTimeout, DependencyUnavailable
from app.domain.prompting import REFUSAL
from tests.api.conftest import ClientFactory
from tests.api.support import PDF, Fakes
from tests.fakes.llm import FakeLLMClient

QUESTION = "Who signed the contract before the notary in Madrid?"


def _ingest(client: TestClient, pdf: bytes = PDF) -> Any:
    return client.post("/api/ingest", files={"pdf_file": ("doc.pdf", pdf, "application/pdf")})


def test_ingesting_twice_reports_indexed_then_already_indexed(client: TestClient) -> None:
    first = _ingest(client)
    second = _ingest(client)

    assert first.status_code == second.status_code == 200
    expected_id = document_id_for(hashlib.sha256(PDF).hexdigest())
    assert first.json() == {"status": "indexed", "chunks_stored": 1, "document_id": expected_id}
    assert second.json() == {
        "status": "already_indexed",
        "chunks_stored": 1,
        "document_id": expected_id,
    }


def test_ingest_validates_the_upload(client: TestClient, fakes: Fakes) -> None:
    response = _ingest(client, b"<html>not a pdf</html>")

    assert response.status_code == 415
    assert response.json()["code"] == "not_a_pdf"
    with pytest.raises(RuntimeError):
        fakes.store.document_chunk_count("any")  # never bound: nothing reached the store


def test_store_down_during_ingest_is_503_with_retry_after(client: TestClient, fakes: Fakes) -> None:
    fakes.store.fail_next(
        1,
        DependencyUnavailable("the vector store could not be reached", "vector_store_unavailable"),
    )

    response = _ingest(client)

    assert response.status_code == 503
    assert response.json()["code"] == "vector_store_unavailable"
    assert response.headers["Retry-After"] == "5"


def test_question_is_answered_with_the_use_case_sources(client: TestClient) -> None:
    _ingest(client)

    response = client.post("/api/ask", json={"question": QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The notary was there."
    assert body["sources"] == [
        "The contract was signed by María González on 3 March 2024 in Madrid before the notary."
    ]


def test_unrelated_question_is_refused_without_calling_the_model(
    client: TestClient, fakes: Fakes
) -> None:
    _ingest(client)

    response = client.post("/api/ask", json={"question": "zebra quantum xylophone"})

    assert response.status_code == 200
    assert response.json() == {"answer": REFUSAL, "sources": []}
    assert isinstance(fakes.llm, FakeLLMClient)
    assert fakes.llm.calls == []


def test_whitespace_question_is_422_and_nothing_is_embedded_or_searched(
    client: TestClient, fakes: Fakes
) -> None:
    response = client.post("/api/ask", json={"question": "\t  "})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_question"
    assert fakes.embedder.query_calls == []
    with pytest.raises(RuntimeError):
        fakes.store.document_chunk_count("any")  # never bound: ensure_collection was not called


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "x" * 2001}, {}, {"q": "x"}])
def test_question_outside_the_contract_is_422_invalid_request(
    client: TestClient, body: dict[str, str]
) -> None:
    response = client.post("/api/ask", json=body)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_malformed_json_is_422_invalid_request(client: TestClient) -> None:
    response = client.post(
        "/api/ask", content=b'{"question": ', headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_model_timeout_is_504_with_retry_after(make_client: ClientFactory, fakes: Fakes) -> None:
    fakes.llm = FakeLLMClient([DependencyTimeout("the model did not answer", "llm_timeout")])
    client = make_client()
    _ingest(client)

    response = client.post("/api/ask", json={"question": QUESTION})

    assert response.status_code == 504
    assert response.json()["code"] == "llm_timeout"
    assert response.headers["Retry-After"] == "5"


def test_without_a_model_ask_is_503_while_extract_and_ingest_work(
    make_client: ClientFactory, fakes: Fakes
) -> None:
    fakes.llm = UnconfiguredLLM()
    client = make_client()

    ingested = _ingest(client)
    extracted = client.post("/api/extract", files={"pdf_file": ("doc.pdf", PDF, "application/pdf")})
    asked = client.post("/api/ask", json={"question": QUESTION})
    refused = client.post("/api/ask", json={"question": "zebra quantum xylophone"})

    assert ingested.status_code == 200
    assert extracted.status_code == 200
    assert asked.status_code == 503
    assert asked.json()["code"] == "llm_unavailable"
    assert asked.json()["detail"] == "no language model is configured"
    assert refused.status_code == 200
    assert refused.json()["answer"] == REFUSAL
