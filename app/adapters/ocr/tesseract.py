"""The Tesseract command-line runner and the parser of its word-level TSV output.

The image goes to the process through stdin and the TSV comes back through stdout, so no file
is ever written. Every failure becomes ``ProcessingFailed`` (code ``ocr_failed``) with a message
that names the problem but never repeats the image or the recognised text.

Cancellation: with a cancellable token, a small watcher thread polls the ``Cancellation``
every ``CANCEL_POLL_S`` while Tesseract runs (``communicate`` itself keeps waiting in one call:
retrying it after a timeout would stop feeding the rest of the image to stdin). Once the token
is cancelled the watcher sends ``SIGTERM`` to the process (on POSIX its whole process group)
and, if it has not exited within ``CANCEL_GRACE_S``, ``SIGKILL``. The runner's own
``communicate`` then returns and reaps the process, and ``ProcessingFailed`` (code
``request_cancelled``) is raised, so neither a running process nor a zombie is left.
"""

import contextlib
import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from app.domain.cancellation import CANCELLED_MESSAGE, NEVER_CANCELLED, Cancellation
from app.domain.errors import GeometryError, ProcessingFailed
from app.domain.geometry import PageRaster, clamp
from app.domain.types import Box, PageWord, PixelBox

_HEADER = (
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
)
_WORD_LEVEL = 5
_MAX_CONFIDENCE = 100
_POSIX = sys.platform != "win32"
_REAP_TIMEOUT_S = 5.0

CANCEL_POLL_S = 0.05
"""How often a running Tesseract process checks the cancellation token."""

CANCEL_GRACE_S = 0.5
"""How long a cancelled Tesseract process may take to exit after ``SIGTERM`` before it is killed."""


class TesseractRunner(Protocol):
    """Recognises one PNG image with the given ``+``-joined Tesseract models; returns TSV.

    Implementations raise ``ProcessingFailed`` (code ``ocr_failed``, or ``request_cancelled``
    once ``cancel`` is cancelled, stopping the recognition in progress) and nothing else. They
    are called from worker threads, so they must not share mutable state between calls.
    """

    def __call__(
        self, png: bytes, languages: str, *, cancel: Cancellation = NEVER_CANCELLED
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class TesseractCli:
    """Runs the ``tesseract`` executable once per image, single-threaded, with a timeout."""

    command: str
    dpi: int
    psm: int
    oem: int
    timeout_s: float

    def __call__(
        self, png: bytes, languages: str, *, cancel: Cancellation = NEVER_CANCELLED
    ) -> str:
        """Return the TSV produced by Tesseract for ``png``.

        On POSIX the process starts in a new session; on timeout its whole process group is
        killed, so wrapper scripts and their children do not survive. On Windows only the
        process itself is killed. Once ``cancel`` is cancelled the process is stopped (see the
        module notes) and ``request_cancelled`` is raised.
        """
        cancel.raise_if_cancelled()
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed argument list, no shell
                self._arguments(languages),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "OMP_THREAD_LIMIT": "1"},
                start_new_session=_POSIX,
            )
        except OSError as error:
            raise ProcessingFailed("tesseract could not be started", "ocr_failed") from error
        watch = None if cancel is NEVER_CANCELLED else _CancelWatch(process, cancel)
        with process:
            try:
                stdout, _ = process.communicate(png, timeout=self.timeout_s)
            except subprocess.TimeoutExpired as error:
                stopped = _finish(watch)  # before reaping: no signal after the reap
                _terminate(process)
                if stopped:
                    raise ProcessingFailed(CANCELLED_MESSAGE, "request_cancelled") from None
                raise ProcessingFailed(
                    f"tesseract did not finish within {self.timeout_s} s", "ocr_failed"
                ) from error
            except BaseException:
                _finish(watch)
                _terminate(process)
                raise
            stopped = _finish(watch)
        if stopped:
            raise ProcessingFailed(CANCELLED_MESSAGE, "request_cancelled")
        if process.returncode != 0:
            raise ProcessingFailed(f"tesseract exited with code {process.returncode}", "ocr_failed")
        try:
            return stdout.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProcessingFailed("tesseract output is not valid UTF-8", "ocr_failed") from error

    def _arguments(self, languages: str) -> list[str]:
        return [
            self.command,
            "stdin",
            "stdout",
            "--oem",
            str(self.oem),
            "--psm",
            str(self.psm),
            "--dpi",
            str(self.dpi),
            "-l",
            languages,
            "tsv",
        ]


class _CancelWatch:
    """Stops ``process`` from a watcher thread once ``cancel`` is cancelled (see module notes).

    It only sends signals; the runner's ``communicate`` reaps the process. ``finish`` ends the
    watch, waits for the thread and tells whether the process was stopped.
    """

    def __init__(self, process: subprocess.Popen[bytes], cancel: Cancellation) -> None:
        self._process = process
        self._cancel = cancel
        self._done = threading.Event()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="ocr-cancel-watch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._done.wait(CANCEL_POLL_S):
            if self._cancel.cancelled:
                self._stopped = True
                _signal_group(self._process, _TERMINATE)
                if not self._done.wait(CANCEL_GRACE_S):
                    _signal_group(self._process, _KILL)
                return

    def finish(self) -> bool:
        """End the watch (idempotent); ``True`` when the process was stopped."""
        self._done.set()
        self._thread.join()
        return self._stopped


def _finish(watch: _CancelWatch | None) -> bool:
    return watch is not None and watch.finish()


_TERMINATE = getattr(signal, "SIGTERM", 15)
_KILL = getattr(signal, "SIGKILL", 9)


def _signal_group(process: subprocess.Popen[bytes], sig: int) -> None:
    """Send ``sig`` to the process group of ``process`` (POSIX) or end the process (Windows).

    The watcher signals only while the runner is still inside ``communicate``, which reaps the
    process only after its pipes close, so the group id still belongs to this process tree
    (short of a few microseconds between the reap and ``finish``).
    """
    if sys.platform != "win32":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, sig)
    else:
        with contextlib.suppress(OSError):
            process.kill()


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """Kill ``process`` (and on POSIX its whole process group), then reap it."""
    if sys.platform != "win32":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    try:
        process.communicate(timeout=_REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # A descendant that left the process group still holds the pipes open: stop reading.
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        process.wait()


def parse_tsv(tsv: str, raster: PageRaster, page: int) -> tuple[PageWord, ...]:
    """Return the words of a Tesseract TSV in reading order, with boxes in page points.

    Only word rows with non-blank text and a confidence of at least 0 are kept (a non-numeric,
    non-finite or overflowing field, or a confidence above 100, fails with ``ocr_failed``); each
    kept word carries its confidence. Boxes are
    converted with ``raster`` and clipped to the page, and words left without area are dropped.
    Each kept word gets the 0-based index of its ``(block_num, par_num, line_num)`` triple in
    order of first appearance among the kept words, so the lines of a page are numbered
    consecutively from 0 in reading order; its paragraph index is that of its
    ``(block_num, par_num)`` pair, numbered the same way.
    """
    lines = tsv.splitlines()
    if not lines or tuple(lines[0].split("\t")) != _HEADER:
        raise ProcessingFailed("tesseract output has no TSV header", "ocr_failed")
    line_indices: dict[tuple[int, int, int], int] = {}
    paragraph_indices: dict[tuple[int, int], int] = {}
    words: list[PageWord] = []
    for number, line in enumerate(lines[1:], start=2):
        row = _parse_row(line, number, raster, page)
        if row is not None:
            text, box, key, confidence = row
            line_index = line_indices.setdefault(key, len(line_indices))
            paragraph_index = paragraph_indices.setdefault(key[:2], len(paragraph_indices))
            words.append(PageWord(text, box, line_index, paragraph_index, confidence))
    return tuple(words)


def _parse_row(
    line: str, number: int, raster: PageRaster, page: int
) -> tuple[str, Box, tuple[int, int, int], float] | None:
    """Return the text, page-space box, line key and confidence of a kept word row, or ``None``."""
    fields = line.split("\t", len(_HEADER) - 1)
    if len(fields) != len(_HEADER):
        raise ProcessingFailed(f"tesseract TSV line {number} is malformed", "ocr_failed")
    invalid = ProcessingFailed(f"tesseract TSV line {number} has an invalid field", "ocr_failed")
    try:
        level = int(fields[0])
        block, paragraph, text_line = (int(value) for value in fields[2:5])
        left, top, width, height = (int(value) for value in fields[6:10])
        confidence = float(fields[10])
    except (ValueError, OverflowError) as error:
        raise invalid from error
    if not isfinite(confidence) or confidence > _MAX_CONFIDENCE:
        raise invalid
    text = fields[11]
    if level != _WORD_LEVEL or not text.strip() or confidence < 0:
        return None
    try:
        box = clamp(
            raster.to_points(page, PixelBox(left, top, width, height)),
            raster.page_width_pt,
            raster.page_height_pt,
        )
    except GeometryError:
        return None
    except OverflowError as error:
        raise invalid from error
    if box.width <= 0 or box.height <= 0:
        return None
    return text, box, (block, paragraph, text_line), confidence
