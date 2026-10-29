"""The API documentation is off unless enabled, and the API routes carry protective headers
(spec: http-api, "API documentation is served only when enabled", "API responses carry
protective headers")."""

from collections.abc import Mapping

import httpx2
from fastapi.testclient import TestClient

from app.domain.cancellation import NEVER_CANCELLED, Cancellation
from app.domain.types import OcrPage
from tests.api.conftest import ClientFactory
from tests.api.support import PDF, Fakes

PROTECTIVE: Mapping[str, str] = {
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
}
"""Spec http-api, written out so a change to the headers has to change this test too."""

DOC_PATHS = ("/docs", "/redoc", "/openapi.json")


class _Failing:
    """An OCR engine that raises an unexpected error, for a 500 rendered by the middleware."""

    def recognize(
        self, pdf: bytes, language: str | None = None, *, cancel: Cancellation = NEVER_CANCELLED
    ) -> list[OcrPage]:
        raise RuntimeError("boom")


def _extract(client: TestClient, pdf: bytes = PDF) -> httpx2.Response:
    return client.post("/api/extract", files={"pdf_file": ("a.pdf", pdf, "application/pdf")})


def _protective(response: httpx2.Response) -> dict[str, str | None]:
    return {name: response.headers.get(name) for name in PROTECTIVE}


# --- documentation --------------------------------------------------------------------------


def test_documentation_is_404_not_found_by_default(client: TestClient) -> None:
    for path in DOC_PATHS:
        response = client.get(path)
        assert response.status_code == 404, path
        assert response.json()["code"] == "not_found", path
    # The API and the audit viewer answer as usual.
    assert client.get("/health").status_code == 200
    assert _extract(client).status_code == 200
    assert client.get("/audit").status_code == 200


def test_documentation_answers_when_enabled(make_client: ClientFactory) -> None:
    client = make_client(expose_docs=True)

    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    schema = client.get("/openapi.json").json()
    assert {"/api/extract", "/api/ingest", "/api/ask"} <= set(schema["paths"])
    assert client.get("/audit").status_code == 200  # the viewer is unaffected


def test_documentation_is_included_once_however_often_the_application_starts(
    make_client: ClientFactory,
) -> None:
    client = make_client(expose_docs=True)
    with TestClient(client.app):  # the lifespan runs a second time on the same application
        pass

    docs = [route for route in client.app.routes if getattr(route, "path", None) == "/docs"]  # type: ignore[attr-defined]
    assert len(docs) == 1
    assert client.get("/docs").status_code == 200


# --- protective headers ---------------------------------------------------------------------


def test_api_and_health_responses_carry_the_protective_headers(client: TestClient) -> None:
    responses = {
        "liveness": client.get("/health"),
        "readiness": client.get("/health/ready"),
        "extraction": _extract(client),
        "extraction refused": _extract(client, b"not a pdf"),
        "invalid question": client.post("/api/ask", json={}),
        "unknown api path": client.get("/api/nothing"),
    }

    assert [response.status_code for response in responses.values()] == [
        200,
        200,
        200,
        415,
        422,
        404,
    ]
    for name, response in responses.items():
        assert _protective(response) == dict(PROTECTIVE), name


def test_the_500_rendered_by_the_middleware_carries_the_protective_headers(
    make_client: ClientFactory, fakes: Fakes
) -> None:
    fakes.ocr = _Failing()
    client = make_client()

    response = _extract(client)

    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert _protective(response) == dict(PROTECTIVE)


def test_a_rate_limited_response_carries_the_protective_headers(
    make_client: ClientFactory,
) -> None:
    client = make_client(rate_limit_uploads_per_minute=1, rate_limit_uploads_burst=1)
    assert _extract(client).status_code == 200

    response = _extract(client)

    assert response.status_code == 429
    assert _protective(response) == dict(PROTECTIVE)


def test_other_paths_keep_their_own_headers(client: TestClient) -> None:
    unknown = client.get("/no/such/route")
    page = client.get("/audit")
    stylesheet = client.get("/audit/static/viewer.css")

    assert unknown.status_code == 404
    assert _protective(unknown) == dict.fromkeys(PROTECTIVE)
    assert page.status_code == 200
    assert "content-security-policy" in page.headers  # the viewer's own headers
    assert stylesheet.status_code == 200
    assert stylesheet.headers["cache-control"] == "no-cache"  # the viewer's caching, not ours
