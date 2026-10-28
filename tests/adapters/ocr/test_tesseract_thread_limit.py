"""The Tesseract child process is single-threaded even when the parent sets no limit.

The image no longer sets ``OMP_THREAD_LIMIT`` globally (it would also cap torch), so the runner
must set it for each Tesseract process. The command is a shell script that prints the variable.
"""

import sys
from pathlib import Path

import pytest

from app.adapters.ocr.tesseract import TesseractCli

pytestmark = [
    pytest.mark.tesseract,
    pytest.mark.skipif(sys.platform == "win32", reason="runs a POSIX shell script"),
]


@pytest.mark.parametrize("parent", [None, "8"])
def test_child_gets_one_omp_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent: str | None
) -> None:
    if parent is None:
        monkeypatch.delenv("OMP_THREAD_LIMIT", raising=False)
    else:
        monkeypatch.setenv("OMP_THREAD_LIMIT", parent)
    script = tmp_path / "print-limit"
    script.write_text('#!/bin/sh\nprintf "%s" "${OMP_THREAD_LIMIT-unset}"\n', encoding="utf-8")
    script.chmod(0o755)
    cli = TesseractCli(command=str(script), dpi=72, psm=3, oem=1, timeout_s=10.0)

    assert cli(b"png", "eng") == "1"
