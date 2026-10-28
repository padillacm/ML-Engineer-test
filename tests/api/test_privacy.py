"""No personal data, question, answer or file name in any log line (spec: observability)."""

from fastapi.testclient import TestClient

from tests.api.conftest import ClientFactory, LogLines
from tests.api.support import PDF, Fakes, names_json
from tests.fakes.llm import FakeLLMClient

QUESTION = "Who signed the contract before the notary in Madrid?"
ANSWER = "Signed by the notary Ximena Quintanilla."
FORBIDDEN = (
    "María",
    "Maria",
    "González",
    "Gonzalez",
    "Gonz",
    "Who signed",
    "Ximena",
    "Quintanilla",
    "secret-report",
)


def _assert_clean(logs: LogLines) -> None:
    text = logs.text
    assert text, "nothing was logged"
    for word in FORBIDDEN:
        assert word not in text, f"{word!r} reached the logs"


def _service_messages(logs: LogLines) -> list[str]:
    """The sorted messages of the service's own request lines (not start-up or the client's)."""
    return sorted(
        str(line["message"])
        for line in logs.lines()
        if str(line["logger"]).startswith("app") and "request_id" in line
    )


def test_extraction_logs_no_names(client: TestClient, logs: LogLines) -> None:
    response = client.post(
        "/api/extract",
        files={"pdf_file": ("secret-report María González.pdf", PDF, "application/pdf")},
        data={"names": names_json(("María", "González"), ("Maria", "Gonzalez"))},
    )

    assert response.status_code == 200
    assert response.json()["extracted_names"][0]["name"] == "María González"
    assert _service_messages(logs) == ["extraction completed", "request"]
    (outcome,) = logs.with_message("extraction completed")
    assert outcome["logger"] == "app.application.extract_names"
    assert (outcome["names_found"], outcome["requested_names"], outcome["matches"]) == (1, 2, 1)
    assert (outcome["pages"], outcome["size_bytes"]) == (1, len(PDF))
    assert isinstance(outcome["ocr_ms"], float)
    assert isinstance(outcome["ner_ms"], float)
    _assert_clean(logs)


def test_rejected_names_are_not_logged(client: TestClient, logs: LogLines) -> None:
    response = client.post(
        "/api/extract",
        files={"pdf_file": ("a.pdf", PDF, "application/pdf")},
        data={"names": '[{"first_name": "María"}]'},
    )

    assert response.status_code == 422
    _assert_clean(logs)


def test_ingestion_and_questions_log_no_text(
    make_client: ClientFactory, fakes: Fakes, logs: LogLines
) -> None:
    fakes.llm = FakeLLMClient(default=ANSWER)
    client = make_client()

    ingested = client.post(
        "/api/ingest", files={"pdf_file": ("secret-report.pdf", PDF, "application/pdf")}
    )
    asked = client.post("/api/ask", json={"question": QUESTION})
    too_long = client.post("/api/ask", json={"question": QUESTION * 100})

    assert ingested.status_code == 200
    assert asked.json()["answer"] == ANSWER
    assert too_long.status_code == 422
    assert _service_messages(logs) == [
        "ingestion completed",
        "question completed",
        "request",
        "request",
        "request",
        "request failed",
    ]
    (ingestion,) = logs.with_message("ingestion completed")
    assert ingestion["chunks_stored"] == 1
    assert ingestion["already_indexed"] is False
    assert ingestion["document_id"] == ingested.json()["document_id"]
    (answered,) = logs.with_message("question completed")
    assert (answered["refused"], answered["refusal_reason"], answered["sources"]) == (
        False,
        None,
        1,
    )
    assert (answered["hits"], answered["hits_kept"]) == (1, 1)
    assert isinstance(answered["top_score"], float)
    assert isinstance(answered["llm_ms"], float)
    _assert_clean(logs)


def test_failing_readiness_check_logs_no_exception_text(
    make_client: ClientFactory, fakes: Fakes, logs: LogLines
) -> None:
    credential_url = "http://admin:" + "-".join(("s3cr3t", "key")) + "@qdrant:6333"

    def unreachable() -> str:
        raise ValueError(f"connection to {credential_url} refused")

    fakes.readiness = (("vector_store", unreachable), ("llm", lambda: "configured"))
    client = make_client()

    response = client.get("/health/ready")

    assert response.status_code == 503
    (failed,) = logs.with_message("readiness check failed")
    assert "in unreachable" in str(failed["exception"])
    assert "s3cr3t" not in logs.text
    assert "admin:" not in logs.text


def test_rejected_question_is_not_echoed_in_the_response(client: TestClient) -> None:
    response = client.post("/api/ask", json={"question": QUESTION * 100})

    assert response.status_code == 422
    assert "Who signed" not in response.text
