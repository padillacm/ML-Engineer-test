"""Smoke tests of a running stack over HTTP (spec: deployment, smoke-testing a running stack).

Skipped unless ``SERVICE_URL`` names the service (``docker-compose.smoke.yml`` sets it for the
``smoke`` service). Everything goes through ``httpx`` against the live service; the sample PDFs
are read from ``sample_pdfs/`` and their page sizes from pypdfium2 (a runtime dependency).
The stack's vector store is persistent, so the first ingestion of a sample may already find it
indexed by an earlier run; the second one must always report ``already_indexed``. The
answerable questions run in one of two forms, chosen by readiness: with a model they must be
answered from the samples, without one they must fail with 503 ``llm_unavailable``.
"""

import io
import json
import os
import struct
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pypdfium2 as pdfium
import pytest

from app.domain.prompting import REFUSAL

pytestmark = pytest.mark.deployed_service

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "sample_pdfs"
MEMO = "company_memo.pdf"
MINUTES = "meeting_minutes.pdf"
REPORT = "research_report.pdf"
# A person each sample names on its first page (as in tests/integration/test_http_e2e.py).
PRESENT = {
    MEMO: ("Robert", "Chen"),
    MINUTES: ("Elizabeth", "Park"),
    REPORT: ("Priya", "Sharma"),
}
ABSENT = ("Zebulon", "Quackenbush")
MIN_MATCH_SCORE = 0.9

# Question -> a fact the answer must contain (case-insensitive), and the sample holding it.
ANSWERABLE = {
    "Who was promoted to Vice President of Engineering?": ("Robert Chen", MEMO),
    "How many peer-reviewed publications did the research department produce in 2024?": (
        "23",
        REPORT,
    ),
}
OFF_TOPIC_QUESTION = "What is the boiling point of mercury on Venus?"

# Generous: OCR of a sample takes seconds, a local model's first answer includes loading it.
TIMEOUT_S = 300.0


@pytest.fixture(scope="module")
def client() -> Iterator[httpx.Client]:
    with httpx.Client(base_url=os.environ["SERVICE_URL"], timeout=TIMEOUT_S) as http:
        yield http


@pytest.fixture(scope="module")
def readiness(client: httpx.Client) -> dict[str, Any]:
    response = client.get("/health/ready")
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


@pytest.fixture(scope="module")
def ingested(client: httpx.Client) -> dict[str, dict[str, Any]]:
    """The memo and the report, ingested once for the module (or already indexed)."""
    bodies = {}
    for name in (MEMO, REPORT):
        response = _post_pdf(client, "/api/ingest", name)
        assert response.status_code == 200, response.text
        bodies[name] = response.json()
    return bodies


def _post_pdf(
    client: httpx.Client,
    path: str,
    name: str,
    data: bytes | None = None,
    form: dict[str, str] | None = None,
    request_id: str | None = None,
) -> httpx.Response:
    content = (SAMPLES_DIR / name).read_bytes() if data is None else data
    headers = {} if request_id is None else {"X-Request-ID": request_id}
    return client.post(
        path,
        files={"pdf_file": (name, content, "application/pdf")},
        data=form or {},
        headers=headers,
    )


def _ask(client: httpx.Client, question: str) -> httpx.Response:
    return client.post("/api/ask", json={"question": question})


def _page_sizes(name: str) -> list[tuple[float, float]]:
    document = pdfium.PdfDocument(SAMPLES_DIR / name)
    try:
        sizes = [page.get_size() for page in document]
    finally:
        document.close()
    return [(float(width), float(height)) for width, height in sizes]


def _names_form(*pairs: tuple[str, str]) -> dict[str, str]:
    return {"names": json.dumps([{"first_name": f, "last_name": last} for f, last in pairs])}


def _png() -> bytes:
    """A valid 1x1 PNG, built with the standard library."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(b"\x00\xff\xff\xff")
    out = io.BytesIO()
    out.write(b"\x89PNG\r\n\x1a\n")
    out.write(chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b""))
    return out.getvalue()


def _assert_error(response: httpx.Response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["code"] == code, body
    assert body["detail"]
    assert body["request_id"] == response.headers["X-Request-ID"]


# --- Health ----------------------------------------------------------------------------------


def test_liveness_echoes_the_request_id(client: httpx.Client) -> None:
    response = client.get("/health", headers={"X-Request-ID": "smoke-live"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"] == "smoke-live"
    assert "server" not in response.headers


def test_a_request_without_an_id_gets_a_generated_one(client: httpx.Client) -> None:
    request_id = client.get("/health").headers["X-Request-ID"]

    assert len(request_id) == 32
    int(request_id, 16)


def test_readiness_reports_the_store_ready_and_the_model_state(
    readiness: dict[str, Any],
) -> None:
    print(f"\n[readiness] {readiness}")
    assert readiness["status"] == "ready"
    assert readiness["checks"]["vector_store"] == "ok"
    assert readiness["checks"]["llm"] in {"configured", "not_configured"}


# --- Extraction ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", [MEMO, MINUTES, REPORT])
def test_extract_finds_the_named_person_with_boxes_on_its_pages(
    client: httpx.Client, name: str
) -> None:
    present = PRESENT[name]
    full = f"{present[0]} {present[1]}"

    response = _post_pdf(client, "/api/extract", name, form=_names_form(present, ABSENT))

    assert response.status_code == 200, response.text
    body = response.json()
    extracted = body["extracted_names"]
    assert extracted
    sizes = _page_sizes(name)
    for entry in extracted:
        box = entry["bounding_box"]
        assert 0 <= box["page_number"] < len(sizes), entry
        width, height = sizes[box["page_number"]]
        assert box["width"] > 0, entry
        assert box["height"] > 0, entry
        assert box["x"] >= 0, entry
        assert box["y"] >= 0, entry
        assert box["x"] + box["width"] <= width + 1e-6, entry
        assert box["y"] + box["height"] <= height + 1e-6, entry
    matches = body["fuzzy_matches"]
    present_matches = [m for m in matches if m["matched_name"] == full]
    assert present_matches, matches
    assert all(m["score"] >= MIN_MATCH_SCORE for m in present_matches), matches
    assert not [m for m in matches if m["matched_name"] == " ".join(ABSENT)], matches


# --- Ingestion -------------------------------------------------------------------------------


def test_ingesting_a_sample_again_reports_already_indexed(
    client: httpx.Client, ingested: dict[str, dict[str, Any]]
) -> None:
    first = ingested[MEMO]
    assert first["status"] in {"indexed", "already_indexed"}, first
    assert first["chunks_stored"] > 0, first

    again = _post_pdf(client, "/api/ingest", MEMO, request_id="smoke-ingest-again")

    assert again.status_code == 200, again.text
    assert again.headers["X-Request-ID"] == "smoke-ingest-again"
    assert again.json() == {
        "status": "already_indexed",
        "chunks_stored": first["chunks_stored"],
        "document_id": first["document_id"],
    }


# --- Questions -------------------------------------------------------------------------------


@pytest.mark.parametrize("question", list(ANSWERABLE))
def test_an_answerable_question_is_answered_by_the_model(
    client: httpx.Client,
    readiness: dict[str, Any],
    ingested: dict[str, dict[str, Any]],
    question: str,
) -> None:
    if readiness["checks"]["llm"] != "configured":
        pytest.skip("no language model configured (readiness)")
    fact, _ = ANSWERABLE[question]

    response = _ask(client, question)

    assert response.status_code == 200, response.text
    body = response.json()
    print(f"\n[answer] {question} -> {body}")
    assert fact.casefold() in body["answer"].casefold(), body
    assert body["answer"] != REFUSAL
    assert body["sources"], body


@pytest.mark.parametrize("question", list(ANSWERABLE))
def test_an_answerable_question_without_a_model_is_llm_unavailable(
    client: httpx.Client,
    readiness: dict[str, Any],
    ingested: dict[str, dict[str, Any]],
    question: str,
) -> None:
    if readiness["checks"]["llm"] != "not_configured":
        pytest.skip("a language model is configured (readiness)")

    response = _ask(client, question)

    _assert_error(response, 503, "llm_unavailable")


def test_an_off_topic_question_is_refused(
    client: httpx.Client, ingested: dict[str, dict[str, Any]]
) -> None:
    response = _ask(client, OFF_TOPIC_QUESTION)

    assert response.status_code == 200, response.text
    assert response.json() == {"answer": REFUSAL, "sources": []}


# --- Validation ------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/api/extract", "/api/ingest"])
def test_a_png_renamed_pdf_is_refused_as_not_a_pdf(client: httpx.Client, path: str) -> None:
    response = _post_pdf(client, path, "scan.pdf", data=_png())

    _assert_error(response, 415, "not_a_pdf")
