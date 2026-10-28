"""The single error mapping and the uniform error body (spec: http-api)."""

from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import errors
from app.domain.cancellation import NEVER_CANCELLED, Cancellation
from app.domain.errors import (
    KNOWN_CODES,
    Category,
    ConfigurationError,
    DependencyTimeout,
    DependencyUnavailable,
    IncompatibleEmbedding,
    InvalidInput,
    ProcessingFailed,
    ServiceError,
)
from app.domain.types import OcrPage
from tests.api.conftest import ClientFactory, LogLines
from tests.api.support import PDF, Fakes

EXPECTED_STATUS = {
    "not_a_pdf": 415,
    "pdf_unreadable": 422,
    "empty_file": 400,
    "too_large": 413,
    "too_many_pages": 413,
    "invalid_names": 422,
    "invalid_question": 422,
    "unsupported_language": 422,
    "ocr_failed": 500,
    "ner_failed": 500,
    "embedding_failed": 500,
    "llm_unavailable": 503,
    "llm_timeout": 504,
    "vector_store_unavailable": 503,
    "vector_store_timeout": 504,
    "service_busy": 503,
    "incompatible_embedding": 500,
    "geometry_error": 500,
    "request_cancelled": 499,
    "invalid_configuration": 500,
}

_ERROR_TYPES: dict[Category, Callable[..., ServiceError]] = {
    Category.INVALID_INPUT: InvalidInput,
    Category.PROCESSING_FAILED: ProcessingFailed,
    Category.DEPENDENCY_UNAVAILABLE: DependencyUnavailable,
    Category.DEPENDENCY_TIMEOUT: DependencyTimeout,
    Category.INCOMPATIBLE_EMBEDDING: IncompatibleEmbedding,
}


def test_every_known_code_has_exactly_one_status() -> None:
    assert set(errors.STATUS_BY_CODE) == KNOWN_CODES
    assert set(errors.CATEGORY_BY_CODE) == KNOWN_CODES


def test_statuses_follow_the_spec() -> None:
    assert dict(errors.STATUS_BY_CODE) == EXPECTED_STATUS


@pytest.mark.parametrize("code", sorted(KNOWN_CODES))
def test_each_code_raised_with_its_category_gets_its_status(code: str) -> None:
    error = _ERROR_TYPES[errors.CATEGORY_BY_CODE[code]]("message", code=code)

    assert errors.status_for(error) == EXPECTED_STATUS[code]


def test_retry_after_is_five_seconds_or_the_busy_wait_rounded_up() -> None:
    assert errors.retry_after_s("vector_store_unavailable", 0.2) == 5
    assert errors.retry_after_s("llm_timeout", 30) == 5
    assert errors.retry_after_s("service_busy", 2.1) == 3
    assert errors.retry_after_s("service_busy", 0.2) == 1


class _Failing:
    """OCR engine that raises a scripted exception."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def recognize(
        self, pdf: bytes, language: str | None = None, *, cancel: Cancellation = NEVER_CANCELLED
    ) -> list[OcrPage]:
        raise self.error


@pytest.mark.parametrize(
    ("error", "status", "retry_after"),
    [
        (ProcessingFailed("page 2 could not be read", "ocr_failed"), 500, None),
        (DependencyUnavailable("unreachable", "vector_store_unavailable"), 503, "5"),
        (DependencyTimeout("too slow", "vector_store_timeout"), 504, "5"),
    ],
)
def test_core_errors_have_the_uniform_body(
    make_client: ClientFactory,
    fakes: Fakes,
    logs: LogLines,
    error: ServiceError,
    status: int,
    retry_after: str | None,
) -> None:
    fakes.ocr = _Failing(error)
    client = make_client()

    response = client.post(
        "/api/extract",
        files={"pdf_file": ("a.pdf", PDF, "application/pdf")},
        headers={"X-Request-ID": "req-1"},
    )

    assert response.status_code == status
    assert response.json() == {"detail": error.message, "code": error.code, "request_id": "req-1"}
    assert response.headers.get("Retry-After") == retry_after
    (failed,) = logs.with_message("request failed")
    assert failed == {
        **failed,
        "code": error.code,
        "category": str(error.category),
        "status": status,
        "request_id": "req-1",
    }


def test_configuration_error_during_a_request_is_500_without_its_details(
    make_client: ClientFactory, fakes: Fakes
) -> None:
    fakes.ocr = _Failing(ConfigurationError("model directory /secret/path is missing"))
    client = make_client()

    response = client.post("/api/extract", files={"pdf_file": ("a.pdf", PDF, "application/pdf")})

    assert response.status_code == 500
    assert response.json()["code"] == "invalid_configuration"
    assert "/secret/path" not in response.text


def test_unexpected_exception_is_500_internal_error_without_its_text(
    make_client: ClientFactory, fakes: Fakes, logs: LogLines
) -> None:
    fakes.ocr = _Failing(RuntimeError("boom-internal-detail"))
    client = make_client()

    response = client.post(
        "/api/extract",
        files={"pdf_file": ("a.pdf", PDF, "application/pdf")},
        headers={"X-Request-ID": "req-500"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "internal server error",
        "code": "internal_error",
        "request_id": "req-500",
    }
    assert "boom-internal-detail" not in response.text
    assert response.headers["X-Request-ID"] == "req-500"
    (logged,) = logs.with_message("unexpected error")
    assert logged["code"] == "internal_error"
    assert "Traceback" in str(logged["exception"])
    (access,) = logs.with_message("request")
    assert access["status"] == 500


def test_unknown_route_is_404_with_the_uniform_body(client: TestClient, logs: LogLines) -> None:
    response = client.get("/nope", headers={"X-Request-ID": "r-404"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found", "code": "not_found", "request_id": "r-404"}
    (access,) = logs.with_message("request")
    assert access["route"] == "unmatched"


def test_wrong_method_is_405_and_keeps_the_allow_header(client: TestClient) -> None:
    response = client.get("/api/extract")

    assert response.status_code == 405
    assert response.json()["code"] == "method_not_allowed"
    assert response.headers["Allow"] == "POST"


def test_no_exception_handler_is_registered_for_unexpected_errors(client: TestClient) -> None:
    # Starlette would run it in ServerErrorMiddleware, outside RequestContextMiddleware, which
    # already turns every unexpected exception into the 500 (so it would never be reached).
    assert isinstance(client.app, FastAPI)
    assert Exception not in client.app.exception_handlers


def test_unexpected_exception_text_is_in_neither_body_nor_logs(
    make_client: ClientFactory, fakes: Fakes, logs: LogLines
) -> None:
    document_text = "-".join(("contract", "signed", "by", "Ximena", "Quintanilla"))
    fakes.ocr = _Failing(RuntimeError(f"tokenizer choked on: {document_text}"))
    client = make_client()  # raises in the test if any exception reached the server

    response = client.post("/api/extract", files={"pdf_file": ("a.pdf", PDF, "application/pdf")})

    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert "Quintanilla" not in response.text
    assert "Quintanilla" not in logs.text
    (logged,) = logs.with_message("unexpected error")
    exception = str(logged["exception"])
    assert "in recognize" in exception  # the frame that raised is still there
    assert exception.endswith("RuntimeError: <message redacted>")
