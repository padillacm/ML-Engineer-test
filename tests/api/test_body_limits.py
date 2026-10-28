"""Body limits chosen by route, and declared lengths of any size (spec: http-api).

Uploads may carry ``MAX_UPLOAD_MB`` of PDF plus the form allowance (a ``names`` field as large
as the multipart parser accepts, and the envelope); every other route accepts at most
``JSON_BODY_LIMIT_BYTES`` (a question is 2,000 characters).
"""

import asyncio
import inspect
import json

import pytest
from fastapi.testclient import TestClient
from starlette.formparsers import MultiPartParser
from starlette.requests import Request
from starlette.types import Message, Receive, Scope, Send

from app.api.middleware import BodyLimitMiddleware, route_path
from app.api.uploads import MAX_NAME_CHARS
from app.main import (
    FORM_ALLOWANCE_BYTES,
    FORM_ENVELOPE_BYTES,
    JSON_BODY_LIMIT_BYTES,
    MULTIPART_FIELD_LIMIT_BYTES,
)
from tests.api.conftest import ClientFactory, LogLines
from tests.api.support import MIB, PDF, Fakes, chunked, multipart

TOO_LARGE = {"detail": "the request body is larger than this endpoint accepts", "code": "too_large"}


def _oversized_question() -> bytes:
    return json.dumps({"question": "q" * JSON_BODY_LIMIT_BYTES}).encode()


def test_declared_question_body_over_the_json_limit_is_413_before_reading(
    client: TestClient, fakes: Fakes, logs: LogLines
) -> None:
    response = client.post(
        "/api/ask",
        content=_oversized_question(),
        headers={"Content-Type": "application/json", "X-Request-ID": "ask-big"},
    )

    assert response.status_code == 413
    assert response.json() == {**TOO_LARGE, "request_id": "ask-big"}
    assert fakes.embedder.query_calls == []
    (access,) = logs.with_message("request")
    assert access["route"] == "unmatched"  # refused before routing


def test_streamed_question_body_over_the_json_limit_is_413(
    client: TestClient, fakes: Fakes
) -> None:
    response = client.post(
        "/api/ask",
        content=chunked(_oversized_question(), size=8 * 1024)(),
        headers={"Content-Type": "application/json", "X-Request-ID": "ask-chunked"},
    )

    assert response.status_code == 413
    assert response.json() == {**TOO_LARGE, "request_id": "ask-chunked"}
    assert fakes.embedder.query_calls == []


def test_longest_escaped_question_fits_the_json_limit(client: TestClient) -> None:
    # 2,000 astral characters, each written as a 12-byte surrogate-pair escape.
    body = json.dumps({"question": "\U0001f600" * 2000}, ensure_ascii=True).encode()
    assert len(body) < JSON_BODY_LIMIT_BYTES

    response = client.post("/api/ask", content=body, headers={"Content-Type": "application/json"})

    assert response.status_code == 200


@pytest.mark.parametrize("path", ["/health", "/nope"])
def test_routes_without_uploads_have_the_json_limit(client: TestClient, path: str) -> None:
    response = client.post(path, content=b"x" * (JSON_BODY_LIMIT_BYTES + 1))

    assert response.status_code == 413
    assert response.json()["code"] == "too_large"


def test_upload_route_with_trailing_slash_keeps_the_upload_limit(client: TestClient) -> None:
    body, content_type = multipart(PDF + b"0" * (512 * 1024))

    response = client.post(
        "/api/ingest/", content=body, headers={"Content-Type": content_type}, follow_redirects=False
    )

    assert response.status_code != 413


def test_pdf_near_the_limit_with_max_names_names_is_accepted(make_client: ClientFactory) -> None:
    client = make_client(max_names=1000)
    names = json.dumps([{"first_name": "María José", "last_name": "González Rodríguez"}] * 1000)
    pdf = PDF + b"0" * (MIB - 4 * 1024 - len(PDF))  # 1,020 KiB, under the 1 MiB limit
    body, content_type = multipart(pdf, {"names": names})
    assert len(body) > MIB + 64 * 1024  # more than the former fixed 64 KiB envelope

    response = client.post("/api/extract", content=body, headers={"Content-Type": content_type})

    assert response.status_code == 200


def test_field_limit_is_the_multipart_parsers() -> None:
    # FastAPI parses forms with ``request.form()``'s defaults.
    default = inspect.signature(Request.form).parameters["max_part_size"].default

    assert default == MULTIPART_FIELD_LIMIT_BYTES
    assert MultiPartParser.max_part_size == MULTIPART_FIELD_LIMIT_BYTES
    assert FORM_ALLOWANCE_BYTES == MULTIPART_FIELD_LIMIT_BYTES + FORM_ENVELOPE_BYTES


def _longest_names(budget: int) -> str:
    """As many names of ``MAX_NAME_CHARS`` escaped characters (6 bytes each) as fit ``budget``
    bytes of JSON."""
    entry = {"first_name": "é" * MAX_NAME_CHARS, "last_name": "ñ" * MAX_NAME_CHARS}
    one = len(json.dumps([entry]))
    return json.dumps([entry] * ((budget - 2) // (one - 1)))


def test_pdf_at_the_limit_with_a_names_field_near_the_parser_limit_is_not_cut(
    make_client: ClientFactory,
) -> None:
    client = make_client(max_names=1000)
    names = _longest_names(MULTIPART_FIELD_LIMIT_BYTES - 1024)
    assert MULTIPART_FIELD_LIMIT_BYTES - 4096 < len(names) < MULTIPART_FIELD_LIMIT_BYTES
    pdf = PDF + b"0" * (MIB - len(PDF))  # exactly MAX_UPLOAD_MB
    body, content_type = multipart(pdf, {"names": names, "language": "es"})
    assert len(body) > MIB + MULTIPART_FIELD_LIMIT_BYTES - 4096

    response = client.post("/api/extract", content=body, headers={"Content-Type": content_type})

    assert response.status_code == 200


def test_names_field_over_the_parser_limit_is_refused_by_the_parser(client: TestClient) -> None:
    body, content_type = multipart(PDF, {"names": "[" + " " * MULTIPART_FIELD_LIMIT_BYTES + "]"})

    response = client.post("/api/extract", content=body, headers={"Content-Type": content_type})

    assert response.status_code == 400  # the parser's refusal, not the body limit's 413
    assert response.json()["code"] == "http_error"


def test_pdf_over_the_exact_limit_is_still_413_inside_the_envelope(client: TestClient) -> None:
    body, content_type = multipart(PDF + b"0" * (MIB + 1 - len(PDF)))

    response = client.post("/api/ingest", content=body, headers={"Content-Type": content_type})

    assert response.status_code == 413
    assert response.json()["code"] == "too_large"


def test_huge_declared_length_is_413_not_500(client: TestClient, logs: LogLines) -> None:
    response = client.post(
        "/api/extract",
        content=b"x",
        headers={"Content-Length": "9" * 5000, "Content-Type": "multipart/form-data; boundary=x"},
    )

    assert response.status_code == 413
    assert response.json()["code"] == "too_large"
    assert not logs.with_message("unexpected error")


# --- declared lengths at the ASGI level ----------------------------------------------------


async def _read_everything(scope: Scope, receive: Receive, send: Send) -> None:
    while (await receive()).get("more_body", False):
        pass
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"read"})


def _status(content_length: bytes, limit: int = 100) -> tuple[int, int]:
    """The status and the number of ``receive`` calls for one 10-byte request."""
    sent: list[Message] = []
    received = 0

    async def receive() -> Message:
        nonlocal received
        received += 1
        return {"type": "http.request", "body": b"x" * 10, "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/x",
        "headers": [(b"content-length", content_length)],
    }
    asyncio.run(BodyLimitMiddleware(_read_everything, limit=lambda _: limit)(scope, receive, send))
    return int(sent[0]["status"]), received


@pytest.mark.parametrize(
    "value",
    [b"9" * 5000, b"1" + b"0" * 18, b"9" * 19, b" " + b"9" * 4301 + b" "],
    ids=["5000-digits", "10**18", "19-nines", "4301-digits-padded"],
)
def test_lengths_over_18_digits_are_over_any_limit(value: bytes) -> None:
    # 413 before the application reads anything (the 10-byte body is then drained).
    assert _status(value, limit=10**18 - 1) == (413, 1)


@pytest.mark.parametrize(
    "value",
    [b"0" * 5000 + b"10", b"10", b"abc", b"\xb2", b"-5", b"1e9", b""],
    ids=["leading-zeros", "plain", "letters", "superscript-two", "negative", "exponent", "empty"],
)
def test_other_lengths_are_left_to_the_body_count(value: bytes) -> None:
    assert _status(value) == (200, 1)


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ({"path": "/api/ask"}, "/api/ask"),
        ({"path": "/api/extract/"}, "/api/extract"),
        ({"path": "/svc/api/ingest", "root_path": "/svc"}, "/api/ingest"),
        ({"path": "/"}, "/"),
        ({"path": "//"}, "/"),
    ],
)
def test_route_path(scope: Scope, expected: str) -> None:
    assert route_path(scope) == expected
