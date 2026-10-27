"""Installed Tesseract, unreadable inputs and the no-temporary-files guarantee (real Tesseract)."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from app.adapters.ocr import TesseractOcrEngine
from app.domain.errors import InvalidInput
from tests.adapters.ocr.pdfs import encrypted_pdf

pytestmark = pytest.mark.tesseract

SAMPLE = Path(__file__).resolve().parents[2] / "sample_pdfs" / "company_memo.pdf"
REQUIRED_MODELS = {"eng", "spa", "por", "fra", "deu"}


def _tesseract(*arguments: str) -> str:
    completed = subprocess.run(
        ["tesseract", *arguments],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return completed.stdout + completed.stderr


def test_tesseract_version_and_models_are_the_expected_ones() -> None:
    version = _tesseract("--version")
    languages = _tesseract("--list-langs")
    print(f"tesseract --version:\n{version}\ntesseract --list-langs:\n{languages}")

    assert version.startswith("tesseract 5."), version
    assert set(languages.split()) >= REQUIRED_MODELS, languages


@pytest.mark.parametrize(
    "pdf",
    [
        pytest.param(encrypted_pdf(), id="encrypted"),
        pytest.param(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\ngarbage\n", id="corrupt"),
        pytest.param(b"\x00\x01not a pdf at all", id="not-a-pdf"),
    ],
)
def test_unreadable_pdfs_are_rejected_as_pdf_unreadable(pdf: bytes) -> None:
    with pytest.raises(InvalidInput) as raised:
        TesseractOcrEngine().recognize(pdf)

    assert raised.value.code == "pdf_unreadable"


def test_recognising_creates_no_file_in_the_temp_or_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = SAMPLE.read_bytes()
    temp_dir = tmp_path / "temp"
    work_dir = tmp_path / "work"
    temp_dir.mkdir()
    work_dir.mkdir()
    for variable in ("TMPDIR", "TMP", "TEMP"):
        monkeypatch.setenv(variable, str(temp_dir))
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    monkeypatch.chdir(work_dir)

    pages = TesseractOcrEngine().recognize(pdf)

    assert pages[0].words
    assert list(temp_dir.iterdir()) == []
    assert list(work_dir.iterdir()) == []
