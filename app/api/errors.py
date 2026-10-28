"""One mapping from failures to HTTP responses, and the exception handlers that apply it.

Every error body is ``{"detail": <message>, "code": <code>, "request_id": <id>}``. Core errors
take their status from ``STATUS_BY_CODE``; request-validation errors are 422 ``invalid_request``
with a ``detail`` list of ``{"type", "loc", "msg"}`` entries (FastAPI's list without ``input``
and ``ctx``, which echo the request: a question, a document sent as a plain field); framework
HTTP errors keep their status. 503 and 504 carry ``Retry-After``.

Anything else is a 500 ``internal_error`` with a generic message, rendered by
``RequestContextMiddleware`` (outermost), which catches every exception the application raises
before the response starts; no ``Exception`` handler is registered, because Starlette would run
it in ``ServerErrorMiddleware``, outside that middleware, where it is never reached.

Each failure is logged once with its code, category and status, never with its message or the
request's content; an unexpected exception is logged with its stack trace, whose exception
messages the service's formatters redact (``app.observability``) unless
``LOG_EXCEPTION_MESSAGES`` is set for local debugging.
"""

import logging
import math
from collections.abc import Mapping
from types import MappingProxyType

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.domain.errors import Category, ServiceError
from app.observability import request_id_var

STATUS_BY_CATEGORY: Mapping[Category, int] = MappingProxyType(
    {
        Category.INVALID_INPUT: 422,
        Category.PROCESSING_FAILED: 500,
        Category.DEPENDENCY_UNAVAILABLE: 503,
        Category.DEPENDENCY_TIMEOUT: 504,
        Category.INCOMPATIBLE_EMBEDDING: 500,
    }
)
"""Default status of each category."""

CATEGORY_BY_CODE: Mapping[str, Category] = MappingProxyType(
    {
        "not_a_pdf": Category.INVALID_INPUT,
        "pdf_unreadable": Category.INVALID_INPUT,
        "empty_file": Category.INVALID_INPUT,
        "too_large": Category.INVALID_INPUT,
        "too_many_pages": Category.INVALID_INPUT,
        "invalid_names": Category.INVALID_INPUT,
        "invalid_question": Category.INVALID_INPUT,
        "unsupported_language": Category.INVALID_INPUT,
        "invalid_configuration": Category.INVALID_INPUT,
        "ocr_failed": Category.PROCESSING_FAILED,
        "ner_failed": Category.PROCESSING_FAILED,
        "embedding_failed": Category.PROCESSING_FAILED,
        "geometry_error": Category.PROCESSING_FAILED,
        "request_cancelled": Category.PROCESSING_FAILED,
        "llm_unavailable": Category.DEPENDENCY_UNAVAILABLE,
        "vector_store_unavailable": Category.DEPENDENCY_UNAVAILABLE,
        "service_busy": Category.DEPENDENCY_UNAVAILABLE,
        "llm_timeout": Category.DEPENDENCY_TIMEOUT,
        "vector_store_timeout": Category.DEPENDENCY_TIMEOUT,
        "incompatible_embedding": Category.INCOMPATIBLE_EMBEDDING,
    }
)
"""The category each core code is raised with."""

_STATUS_OVERRIDES: Mapping[str, int] = MappingProxyType(
    {
        "empty_file": 400,
        "too_large": 413,
        "too_many_pages": 413,
        "not_a_pdf": 415,
        # The client went away; the status only reaches the access log (nginx's convention).
        "request_cancelled": 499,
        # Raised while serving a request it is a server bug, not the client's input.
        "invalid_configuration": 500,
    }
)

STATUS_BY_CODE: Mapping[str, int] = MappingProxyType(
    {
        code: _STATUS_OVERRIDES.get(code, STATUS_BY_CATEGORY[category])
        for code, category in CATEGORY_BY_CODE.items()
    }
)
"""The single code -> status table (spec http-api); a test keeps it equal to ``KNOWN_CODES``."""

DEPENDENCY_RETRY_AFTER_S = 5
"""``Retry-After`` of a dependency failure (503/504) other than ``service_busy``."""

INVALID_REQUEST = "invalid_request"
INTERNAL_ERROR = "internal_error"
INTERNAL_MESSAGE = "internal server error"
_CONFIGURATION_MESSAGE = "the service is misconfigured"
_INTERNAL_CATEGORY = "internal"
_RETRY_STATUSES = frozenset({503, 504})
_FIRST_SERVER_ERROR = 500
_HTTP_CODES: Mapping[int, str] = MappingProxyType(
    {404: "not_found", 405: "method_not_allowed", 413: "too_large"}
)

_logger = logging.getLogger(__name__)


def status_for(error: ServiceError) -> int:
    """The HTTP status of a core error: its code's, else its category's default."""
    return STATUS_BY_CODE.get(error.code, STATUS_BY_CATEGORY[error.category])


def retry_after_s(code: str, busy_wait_s: float) -> int:
    """Seconds a client should wait before retrying a 503/504 with ``code``."""
    if code == "service_busy":
        return max(1, math.ceil(busy_wait_s))
    return DEPENDENCY_RETRY_AFTER_S


def error_body(detail: object, code: str) -> dict[str, object]:
    """The uniform error body, with the current request id."""
    return {"detail": detail, "code": code, "request_id": request_id_var.get()}


def error_response(
    status: int,
    detail: object,
    code: str,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """A JSON error response with the uniform body."""
    return JSONResponse(error_body(detail, code), status_code=status, headers=headers)


def internal_error_response() -> JSONResponse:
    """The 500 ``internal_error`` response; it says nothing about the failure."""
    return error_response(500, INTERNAL_MESSAGE, INTERNAL_ERROR)


def log_failure(code: str, category: str, status: int) -> None:
    """Log a failed request once: code, category and status, no message."""
    level = logging.ERROR if status >= _FIRST_SERVER_ERROR else logging.WARNING
    _logger.log(
        level, "request failed", extra={"code": code, "category": category, "status": status}
    )


def install_error_handlers(app: FastAPI) -> None:
    """Register the handlers of core, validation and framework errors (unexpected ones are
    handled by ``RequestContextMiddleware``)."""
    app.add_exception_handler(ServiceError, _service_error)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _http_error)  # type: ignore[arg-type]


async def _service_error(request: Request, error: ServiceError) -> JSONResponse:
    status = status_for(error)
    log_failure(error.code, str(error.category), status)
    detail = _CONFIGURATION_MESSAGE if error.code == "invalid_configuration" else error.message
    headers = None
    if status in _RETRY_STATUSES:
        busy_wait_s = float(getattr(request.app.state.settings, "busy_wait_s", 1.0))
        headers = {"Retry-After": str(retry_after_s(error.code, busy_wait_s))}
    return error_response(status, detail, error.code, headers)


async def _validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    del request
    log_failure(INVALID_REQUEST, str(Category.INVALID_INPUT), 422)
    return error_response(422, validation_details(error), INVALID_REQUEST)


def validation_details(error: RequestValidationError) -> list[dict[str, object]]:
    """Each validation error as ``{"type", "loc", "msg"}``: what failed and where, never the
    rejected value (pydantic's ``input`` and ``ctx`` are dropped; its messages name the rule,
    such as "String should have at most 2000 characters", not the value)."""
    return [
        {
            "type": str(entry.get("type", "")),
            "loc": [part if isinstance(part, int) else str(part) for part in entry.get("loc", ())],
            "msg": str(entry.get("msg", "")),
        }
        for entry in error.errors()
    ]


async def _http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
    del request
    code = _HTTP_CODES.get(error.status_code, "http_error")
    client_error = error.status_code < _FIRST_SERVER_ERROR
    category = str(Category.INVALID_INPUT) if client_error else _INTERNAL_CATEGORY
    log_failure(code, category, error.status_code)
    return error_response(error.status_code, error.detail, code, error.headers)


def log_unexpected(error: BaseException) -> None:
    """Log an unexpected exception once, with its stack trace (messages redacted by the
    service's formatters)."""
    _logger.error(
        "unexpected error",
        exc_info=(type(error), error, error.__traceback__),
        extra={"code": INTERNAL_ERROR, "category": _INTERNAL_CATEGORY, "status": 500},
    )
