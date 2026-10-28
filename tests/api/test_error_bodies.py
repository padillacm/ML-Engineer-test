"""422 bodies say what failed and where, never the rejected value (spec: http-api).

FastAPI's validation errors carry the rejected ``input`` (and ``ctx``); echoed, a 422 would
return a question, a document sent as a plain form field or a whole request body.
"""

import json

import pytest
from fastapi.testclient import TestClient

from tests.api.support import PDF

SECRET = "-".join(("Ximena", "Quintanilla"))
DETAIL_KEYS = {"type", "loc", "msg"}


def _details(response_json: object) -> list[dict[str, object]]:
    assert isinstance(response_json, dict)
    assert response_json["code"] == "invalid_request"
    details = response_json["detail"]
    assert isinstance(details, list)
    assert details
    for entry in details:
        assert set(entry) == DETAIL_KEYS
    return details


def test_too_long_question_is_not_echoed(client: TestClient) -> None:
    question = f"Did {SECRET} sign? " + "a" * 2000

    response = client.post("/api/ask", json={"question": question}, headers={"X-Request-ID": "q1"})

    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "type": "string_too_long",
                "loc": ["body", "question"],
                "msg": "String should have at most 2000 characters",
            }
        ],
        "code": "invalid_request",
        "request_id": "q1",
    }
    assert SECRET not in response.text


def test_long_question_gets_a_small_422(client: TestClient) -> None:
    question = SECRET + "a" * 60_000  # under the 64 KiB JSON limit, over 2,000 characters

    response = client.post("/api/ask", json={"question": question})

    assert response.status_code == 422
    assert len(response.content) < 1024


def test_document_sent_as_a_plain_form_field_is_not_echoed(client: TestClient) -> None:
    boundary = "plainfield"
    document = PDF + f"page one: signed by {SECRET}".encode()
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="pdf_file"\r\n\r\n'.encode()
        + document
        + f"\r\n--{boundary}--\r\n".encode()
    )

    response = client.post(
        "/api/extract",
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    assert response.status_code == 422
    (entry,) = _details(response.json())
    assert entry["loc"] == ["body", "pdf_file"]
    assert SECRET not in response.text
    assert "%PDF" not in response.text


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        (json.dumps({"question": SECRET}).encode(), "text/plain"),
        (json.dumps([SECRET]).encode(), "application/json"),
        (b'{"question": "' + SECRET.encode(), "application/json"),
        (json.dumps({"question": {"text": SECRET}}).encode(), "application/json"),
    ],
    ids=["text-plain", "json-list", "invalid-json", "object-question"],
)
def test_malformed_question_bodies_are_not_echoed(
    client: TestClient, content: bytes, content_type: str
) -> None:
    response = client.post("/api/ask", content=content, headers={"Content-Type": content_type})

    assert response.status_code == 422
    _details(response.json())
    assert SECRET not in response.text


def test_missing_file_names_the_field(client: TestClient) -> None:
    response = client.post("/api/extract", data={"names": json.dumps([SECRET])})

    assert response.status_code == 422
    (entry,) = _details(response.json())
    assert entry == {"type": "missing", "loc": ["body", "pdf_file"], "msg": "Field required"}
    assert SECRET not in response.text
