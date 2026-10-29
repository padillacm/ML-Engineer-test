"""Many requests, half of them failing, release their uploads (spec: http-api).

Scenario "Many requests": 200 requests, half failing validation, leave the open file
descriptors and temporary files where they were. Every spooled upload Starlette creates is
tracked and must be closed when its request ends; uploads above 1 MB roll over to a real
temporary file, which must be gone from the temporary directory afterwards (on POSIX the file
is unlinked at once, so the descriptor count of the process is checked there too).
"""

import gc
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient
from starlette import formparsers

from tests.api.conftest import ClientFactory
from tests.api.support import MIB, PDF, multipart

REQUESTS = 200
SPOOL_LIMIT = 1024 * 1024  # Starlette keeps uploads up to 1 MB in memory
BIG_PDF = PDF + b"0" * (SPOOL_LIMIT + 256 * 1024)  # rolls over to disk, under the 2 MiB limit


class TrackedSpool(tempfile.SpooledTemporaryFile[bytes]):
    """``SpooledTemporaryFile`` that remembers every instance and whether it went to disk."""

    created: ClassVar[list["TrackedSpool"]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        TrackedSpool.created.append(self)

    @property
    def rolled_to_disk(self) -> bool:
        return bool(getattr(self, "_rolled", False))


def _open_fds() -> int | None:
    fd_dir = Path("/proc/self/fd")
    return len(list(fd_dir.iterdir())) if fd_dir.is_dir() else None


def _form(pdf: bytes, fields: dict[str, str] | None = None) -> dict[str, Any]:
    body, content_type = multipart(pdf, fields)
    return {"content": body, "headers": {"Content-Type": content_type}}


SMALL_UPLOAD = {"files": {"pdf_file": ("a.pdf", PDF, "application/pdf")}}
VALID: dict[int, tuple[str, dict[str, Any]]] = {
    0: ("/api/ingest", _form(BIG_PDF)),
    2: ("/api/extract", SMALL_UPLOAD),
    4: ("/api/ingest", SMALL_UPLOAD),
    6: ("/api/extract", _form(BIG_PDF, {"names": "[]"})),
    8: ("/api/ask", {"json": {"question": "who signed the contract?"}}),
}
INVALID: dict[int, tuple[str, dict[str, Any]]] = {
    1: ("/api/extract", _form(b"not a pdf" * 200_000)),  # spooled to disk, then 415
    3: ("/api/extract", {**SMALL_UPLOAD, "data": {"names": "not json"}}),  # 422
    5: ("/api/extract", _form(BIG_PDF, {"language": "xx"})),  # spooled, then 422
    7: ("/api/ask", {"json": {"question": "a" * 2001}}),  # 422
    9: ("/api/ingest", _form(PDF + b"0" * (2 * MIB))),  # spooled, over the PDF limit: 413
}
"""Requests by ``index % 10``: even ones are valid, odd ones fail validation."""


def _send(client: TestClient, index: int) -> int:
    path, request = {**VALID, **INVALID}[index % 10]
    return client.post(path, **request).status_code


def test_uploads_are_released_after_many_requests(
    make_client: ClientFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", TrackedSpool)
    TrackedSpool.created = []
    # Hundreds of uploads from one address: the upload rate limit is off for this test.
    client = make_client(max_upload_mb=2, rate_limit_uploads_per_minute=0)
    _send(client, 2)  # warm up: lazy imports and pools open their descriptors once
    gc.collect()
    fds_before = _open_fds()
    TrackedSpool.created = []

    statuses = [_send(client, index) for index in range(REQUESTS)]

    gc.collect()
    valid = statuses[0::2]
    invalid = statuses[1::2]
    assert all(status == 200 for status in valid), sorted(set(valid))
    assert all(400 <= status < 500 for status in invalid), sorted(set(invalid))
    spools = TrackedSpool.created
    assert len(spools) >= 120  # one per upload request
    assert sum(spool.rolled_to_disk for spool in spools) >= 60
    assert all(spool.closed for spool in spools)
    assert list(tmp_path.iterdir()) == []
    fds_after = _open_fds()
    if fds_before is not None and fds_after is not None:
        assert fds_after <= fds_before + 2, (fds_before, fds_after)


@pytest.mark.skipif(sys.platform == "win32", reason="/proc/self/fd exists on Linux only")
def test_descriptor_count_is_observable() -> None:
    before = _open_fds()
    assert before is not None
    with open(os.devnull, "rb"):
        assert _open_fds() == before + 1


def test_tracked_spool_records_rollover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    TrackedSpool.created = []
    with TrackedSpool(max_size=10) as spool:
        spool.write(json.dumps("x" * 20).encode())
        assert spool.rolled_to_disk
    assert TrackedSpool.created == [spool]
    assert spool.closed
