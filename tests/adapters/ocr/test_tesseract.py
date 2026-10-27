import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from app.adapters.ocr.tesseract import TesseractCli, parse_tsv
from app.domain.errors import ProcessingFailed
from app.domain.geometry import PageRaster
from app.domain.types import Box, PageWord
from tests.adapters.ocr.fake_runner import TSV_HEADER

FIXTURES = Path(__file__).parent / "fixtures"
# 1000 x 500 pixels showing a 500 x 250 pt page: 0.5 pt per pixel on both axes.
RASTER = PageRaster(1000, 500, 500.0, 250.0)
PNG = b"\x89PNG fake image bytes"
FAKE_PID = 4_242_424


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _cli(command: str = "tesseract") -> TesseractCli:
    return TesseractCli(command=command, dpi=200, psm=3, oem=1, timeout_s=12.5)


def _row(left: str = "10", conf: str = "90", text: str = "word") -> str:
    return f"{TSV_HEADER}\n5\t1\t1\t1\t1\t1\t{left}\t10\t10\t10\t{conf}\t{text}\n"


class TestParseTsv:
    def test_header_only_gives_no_words(self) -> None:
        assert parse_tsv(_fixture("header_only.tsv.txt"), RASTER, 0) == ()

    def test_word_rows_become_page_words_in_points(self) -> None:
        words = parse_tsv(_fixture("page_words.tsv.txt"), RASTER, 3)

        assert words == (
            PageWord("Ramírez", Box(3, 50.0, 25.0, 100.0, 20.0)),
            PageWord("Müller", Box(3, 160.0, 25.0, 50.0, 20.0)),
            PageWord("São", Box(3, 300.0, 25.0, 75.0, 20.0)),
            PageWord("edge", Box(3, 450.0, 230.0, 50.0, 20.0)),
            PageWord("beyond", Box(3, 475.0, 240.0, 25.0, 10.0)),
        )

    def test_blank_negative_confidence_and_empty_boxes_are_dropped(self) -> None:
        texts = [word.text for word in parse_tsv(_fixture("page_words.tsv.txt"), RASTER, 0)]

        assert "ghost" not in texts  # conf -1
        assert "outside" not in texts  # starts at the right edge of the image
        assert "thin" not in texts  # zero width
        assert all(text.strip() for text in texts)

    def test_crlf_line_endings_are_accepted(self) -> None:
        crlf = _fixture("page_words.tsv.txt").replace("\n", "\r\n")

        assert parse_tsv(crlf, RASTER, 0) == parse_tsv(_fixture("page_words.tsv.txt"), RASTER, 0)

    def test_huge_but_finite_coordinates_are_clipped_to_the_page(self) -> None:
        assert parse_tsv(_row(left="-500"), RASTER, 0) == ()  # ends left of the page

        text = f"{TSV_HEADER}\n5\t1\t1\t1\t1\t1\t0\t0\t99999999999\t99999999999\t90\tbig\n"
        (big,) = parse_tsv(text, RASTER, 0)
        assert big.box == Box(0, 0.0, 0.0, 500.0, 250.0)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "Error: could not read image\n",
            TSV_HEADER + "\n5\t1\t1\t1\t1\t1\t10\t10\n",
            _row(left="ten"),
            _row(conf="high"),
            _row(left="9" * 400),
            _row(left="9" * 5000),
            _row(conf="nan"),
            _row(conf="inf"),
            _row(conf="-inf"),
        ],
        ids=[
            "empty",
            "no-header",
            "short-row",
            "bad-int",
            "bad-conf",
            "overflowing-int",
            "int-over-digit-limit",
            "nan-conf",
            "inf-conf",
            "minus-inf-conf",
        ],
    )
    def test_unparsable_output_fails(self, text: str) -> None:
        with pytest.raises(ProcessingFailed) as caught:
            parse_tsv(text, RASTER, 0)

        assert caught.value.code == "ocr_failed"
        assert "word" not in caught.value.message


class FakeStream:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakePopen:
    """Stands in for ``subprocess.Popen``; ``answers`` are consumed by successive communicate."""

    instances: ClassVar[list["FakePopen"]] = []

    def __init__(
        self,
        args: list[str],
        *,
        answers: list[tuple[bytes, bytes] | BaseException],
        returncode: int = 0,
        **kwargs: Any,
    ) -> None:
        self.args = args
        self.kwargs = kwargs
        self.answers = answers
        self.returncode: int | None = None
        self.final_returncode = returncode
        self.pid = FAKE_PID
        self.inputs: list[bytes | None] = []
        self.timeouts: list[float | None] = []
        self.killed = False
        self.waited = False
        self.stdin = FakeStream()
        self.stdout = FakeStream()
        self.stderr = FakeStream()
        FakePopen.instances.append(self)

    def __enter__(self) -> "FakePopen":
        return self

    def __exit__(self, *exc: object) -> None:
        self.waited = True

    def communicate(
        self, input: bytes | None = None, timeout: float | None = None
    ) -> tuple[bytes, bytes]:
        self.inputs.append(input)
        self.timeouts.append(timeout)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        self.returncode = self.final_returncode
        return answer

    def kill(self) -> None:
        self.killed = True

    def wait(self) -> int:
        self.waited = True
        return -9


@pytest.fixture
def popen(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch ``subprocess.Popen``; set ``answers`` / ``returncode`` in the returned dict."""
    setup: dict[str, Any] = {"answers": [(b"", b"")], "returncode": 0, "killpg": []}
    FakePopen.instances = []

    def make(args: list[str], **kwargs: Any) -> FakePopen:
        return FakePopen(
            args, answers=list(setup["answers"]), returncode=setup["returncode"], **kwargs
        )

    def killpg(pid: int, sig: int) -> None:
        setup["killpg"].append((pid, sig))

    monkeypatch.setattr(subprocess, "Popen", make)
    monkeypatch.setattr(os, "killpg", killpg, raising=False)
    return setup


def _timeout() -> subprocess.TimeoutExpired:
    return subprocess.TimeoutExpired(["tesseract"], 12.5, output=b"partial Ramirez")


class TestTesseractCli:
    def test_runs_tesseract_through_stdin_and_stdout_single_threaded(
        self, popen: dict[str, Any]
    ) -> None:
        popen["answers"] = [("Ramírez".encode(), b"")]

        assert _cli()(PNG, "eng+spa") == "Ramírez"

        (process,) = FakePopen.instances
        assert process.args == [
            "tesseract",
            "stdin",
            "stdout",
            "--oem",
            "1",
            "--psm",
            "3",
            "--dpi",
            "200",
            "-l",
            "eng+spa",
            "tsv",
        ]
        assert process.inputs == [PNG]
        assert process.timeouts == [12.5]
        assert process.kwargs["stdin"] == subprocess.PIPE
        assert process.kwargs["stdout"] == subprocess.PIPE
        assert process.kwargs["stderr"] == subprocess.PIPE
        assert process.kwargs["start_new_session"] is (sys.platform != "win32")
        assert "shell" not in process.kwargs
        assert process.kwargs["env"]["OMP_THREAD_LIMIT"] == "1"
        assert process.waited

    def test_missing_executable_fails(self) -> None:
        with pytest.raises(ProcessingFailed) as caught:
            _cli("no-such-tesseract-executable-3f9a")(PNG, "eng")

        assert caught.value.code == "ocr_failed"
        assert isinstance(caught.value.__cause__, FileNotFoundError)

    def test_timeout_kills_the_process_group_and_fails(self, popen: dict[str, Any]) -> None:
        popen["answers"] = [_timeout(), (b"", b"")]

        with pytest.raises(ProcessingFailed) as caught:
            _cli()(PNG, "eng")

        assert caught.value.code == "ocr_failed"
        assert "12.5" in caught.value.message
        assert "Ramirez" not in str(caught.value)
        assert isinstance(caught.value.__cause__, subprocess.TimeoutExpired)
        (process,) = FakePopen.instances
        assert process.killed
        assert len(process.inputs) == 2  # reaped after the kill
        sigkill = getattr(signal, "SIGKILL", None)
        expected = [] if sys.platform == "win32" else [(FAKE_PID, sigkill)]
        assert popen["killpg"] == expected

    def test_descendant_holding_the_pipes_does_not_block_the_failure(
        self, popen: dict[str, Any]
    ) -> None:
        popen["answers"] = [_timeout(), _timeout()]

        with pytest.raises(ProcessingFailed):
            _cli()(PNG, "eng")

        (process,) = FakePopen.instances
        assert process.stdout.closed
        assert process.stderr.closed
        assert process.waited

    def test_interrupted_wait_kills_the_process_and_propagates(self, popen: dict[str, Any]) -> None:
        popen["answers"] = [KeyboardInterrupt(), (b"", b"")]

        with pytest.raises(KeyboardInterrupt):
            _cli()(PNG, "eng")

        assert FakePopen.instances[0].killed

    def test_non_zero_exit_fails_without_repeating_stderr(self, popen: dict[str, Any]) -> None:
        popen["answers"] = [(b"", b"Ramirez secret")]
        popen["returncode"] = 1

        with pytest.raises(ProcessingFailed) as caught:
            _cli()(PNG, "eng")

        assert caught.value.code == "ocr_failed"
        assert "code 1" in caught.value.message
        assert "Ramirez" not in str(caught.value)

    def test_output_that_is_not_utf8_fails(self, popen: dict[str, Any]) -> None:
        popen["answers"] = [(b"\xff\xfe\xfa", b"")]

        with pytest.raises(ProcessingFailed) as caught:
            _cli()(PNG, "eng")

        assert caught.value.code == "ocr_failed"
        assert isinstance(caught.value.__cause__, UnicodeDecodeError)
