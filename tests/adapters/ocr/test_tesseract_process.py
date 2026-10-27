"""Real processes behind the Tesseract command: timeouts kill the whole process tree (POSIX).

The command is a small shell script instead of Tesseract, so these tests are about process
handling only. They run in the container (``tesseract`` marker) and need ``/proc``.
"""

import dataclasses
import sys
import threading
import time
from pathlib import Path

import pytest

from app.adapters.ocr import TesseractOcrConfig, TesseractOcrEngine
from app.adapters.ocr.tesseract import TesseractCli
from app.domain.errors import ProcessingFailed
from tests.adapters.ocr.conftest import RenderSpy
from tests.adapters.ocr.pdfs import PageSpec, build_pdf

pytestmark = [
    pytest.mark.tesseract,
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups and /proc"),
]

# An unusual duration identifies the processes started by these tests.
SLEEP = "3017"
PNG = b"\x89PNG not really an image"


def _script(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _sleepers() -> list[int]:
    """PIDs of live (non-zombie) ``sleep SLEEP`` processes."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if arguments[:2] == [b"sleep", SLEEP.encode()]:
            found.append(int(entry.name))
    return found


def _wait_until_no_sleepers(deadline_s: float = 3.0) -> list[int]:
    end = time.monotonic() + deadline_s
    while (left := _sleepers()) and time.monotonic() < end:
        time.sleep(0.05)
    return left


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(f"exec sleep {SLEEP}", id="exec"),
        pytest.param(f"sleep {SLEEP} & sleep {SLEEP}; wait", id="wrapper-with-children"),
    ],
)
def test_timeout_kills_the_command_and_every_child(tmp_path: Path, body: str) -> None:
    command = _script(tmp_path, "fake-tesseract", body)
    assert _sleepers() == []
    cli = TesseractCli(command=command, dpi=72, psm=3, oem=1, timeout_s=1.0)

    started = time.monotonic()
    with pytest.raises(ProcessingFailed) as caught:
        cli(PNG, "eng")
    elapsed = time.monotonic() - started

    assert caught.value.code == "ocr_failed"
    assert "did not finish" in caught.value.message
    assert elapsed < 5.0
    assert _wait_until_no_sleepers() == []


def test_engine_timeout_stops_the_document_and_leaves_nothing_running(
    tmp_path: Path, rendered_pages: RenderSpy
) -> None:
    command = _script(tmp_path, "fake-tesseract", f"sleep {SLEEP} & sleep {SLEEP}; wait")
    config = dataclasses.replace(
        TesseractOcrConfig(dpi=72, workers=2), tesseract_cmd=command, timeout_s=1.0
    )
    pdf = build_pdf([PageSpec() for _ in range(12)])

    started = time.monotonic()
    with pytest.raises(ProcessingFailed) as caught:
        TesseractOcrEngine(config).recognize(pdf, "en")
    elapsed = time.monotonic() - started

    assert caught.value.code == "ocr_failed"
    assert elapsed < 5.0
    assert len(rendered_pages.indexes) <= 4  # the in-flight window, not the 12 pages
    assert _wait_until_no_sleepers() == []
    assert not [thread for thread in threading.enumerate() if thread.name.startswith("tesseract")]


def test_command_killed_by_a_signal_fails(tmp_path: Path) -> None:
    command = _script(tmp_path, "fake-tesseract", "kill -9 $$")

    with pytest.raises(ProcessingFailed) as caught:
        TesseractCli(command=command, dpi=72, psm=3, oem=1, timeout_s=5.0)(PNG, "eng")

    assert "code -9" in caught.value.message
