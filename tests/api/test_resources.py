"""Bounded heavy work and released upload buffers (spec: http-api)."""

import io
import threading
from typing import Any

import anyio
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from app.api.dependencies import JobLimiter
from app.api.uploads import parse_language, parse_names, read_pdf
from app.domain.cancellation import Cancellation
from app.domain.errors import ConfigurationError, InvalidInput
from app.domain.types import NamePair
from tests.api.conftest import ClientFactory
from tests.api.support import PDF, PNG, BlockingOcr, Fakes, RecordedSpool

WAIT_S = 10.0


def test_saturated_service_answers_busy_and_the_running_job_completes(
    make_client: ClientFactory, fakes: Fakes
) -> None:
    blocking = BlockingOcr()
    fakes.ocr = blocking
    client = make_client(max_concurrent_jobs=1, busy_wait_s=0.2)
    first: dict[str, Any] = {}

    def run_first() -> None:
        first["response"] = client.post(
            "/api/extract", files={"pdf_file": ("a.pdf", PDF, "application/pdf")}
        )

    worker = threading.Thread(target=run_first)
    worker.start()
    try:
        assert blocking.entered.wait(WAIT_S)
        second = client.post("/api/ingest", files={"pdf_file": ("b.pdf", PDF, "application/pdf")})
        health = client.get("/health")
        question = client.post("/api/ask", json={"question": "zebra quantum"})
    finally:
        blocking.release.set()
        worker.join(WAIT_S)

    assert second.status_code == 503
    assert second.json()["code"] == "service_busy"
    assert second.headers["Retry-After"] == "1"
    assert health.status_code == 200
    assert question.status_code == 200  # questions do not wait for a job slot
    assert first["response"].status_code == 200
    assert first["response"].json()["extracted_names"][0]["name"] == "María González"


def test_limiter_frees_its_slot_after_a_failure() -> None:
    limiter = JobLimiter(1, 0.0, max_questions=1)

    def fail(cancel: Cancellation) -> None:
        raise RuntimeError("job failed")

    async def scenario() -> str:
        with pytest.raises(RuntimeError):
            await limiter.run_job(fail)
        assert limiter.free_slots == 1
        return await limiter.run_job(lambda _: "done")

    assert anyio.run(scenario) == "done"
    assert limiter.free_slots == 1


@pytest.mark.parametrize(
    ("jobs", "wait", "questions"),
    [
        (0, 1.0, 1),
        (True, 1.0, 1),
        (1, -1.0, 1),
        (1, float("nan"), 1),
        (1, float("inf"), 1),
        (1, 1.0, 0),
        (1, 1.0, True),
    ],
)
def test_limiter_rejects_invalid_values(jobs: int, wait: float, questions: int) -> None:
    with pytest.raises(ConfigurationError):
        JobLimiter(jobs, wait, max_questions=questions)


# --- uploads -------------------------------------------------------------------------------


def _upload(data: bytes) -> UploadFile:
    return UploadFile(io.BytesIO(data), filename="any.pdf")


@pytest.mark.parametrize(
    ("data", "code"),
    [(PDF, None), (PNG, "not_a_pdf"), (b"", "empty_file")],
    ids=["pdf", "png", "empty"],
)
def test_read_pdf_closes_the_upload_on_success_and_failure(data: bytes, code: str | None) -> None:
    upload = _upload(data)

    if code is None:
        assert read_pdf(upload, 1000) == PDF
    else:
        with pytest.raises(InvalidInput) as caught:
            read_pdf(upload, 1000)
        assert caught.value.code == code
    assert upload.file.closed


def test_read_pdf_reads_one_byte_past_the_limit_only() -> None:
    reads: list[int] = []

    class Spy(io.BytesIO):
        def read(self, size: int | None = -1, /) -> bytes:
            reads.append(-1 if size is None else size)
            return super().read(size)

    upload = UploadFile(Spy(PDF + b"0" * 5000), filename="any.pdf")

    with pytest.raises(InvalidInput) as caught:
        read_pdf(upload, 100)

    assert caught.value.code == "too_large"
    assert reads == [101]
    assert upload.file.closed


@pytest.mark.parametrize(
    ("data", "status"),
    [(PDF, 200), (PNG, 415), (b"", 400), (PDF + b"0" * (2 << 20), 413)],
    ids=["pdf", "png", "empty", "oversized"],
)
def test_uploaded_file_is_closed_when_the_request_ends(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, data: bytes, status: int
) -> None:
    RecordedSpool.instances = []
    monkeypatch.setattr("starlette.formparsers.SpooledTemporaryFile", RecordedSpool)

    response = client.post("/api/ingest", files={"pdf_file": ("a.pdf", data, "application/pdf")})

    assert response.status_code == status
    assert all(spool.closed for spool in RecordedSpool.instances)
    if status != 413:  # an oversized body is refused before the form is parsed
        assert len(RecordedSpool.instances) == 1


def test_names_are_parsed_into_core_pairs() -> None:
    raw = '[{"first_name": "Ana", "last_name": "Ruiz", "extra": 1}]'

    assert parse_names(raw, 5) == [NamePair("Ana", "Ruiz")]
    assert parse_names("[]", 5) == []


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, None), ("", None), ("  ", None), ("FR", "fr")]
)
def test_language_hint_is_optional_and_normalised(raw: str | None, expected: str | None) -> None:
    assert parse_language(raw) == expected
