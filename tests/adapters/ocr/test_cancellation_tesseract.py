"""Cancelling a job while real Tesseract processes read its pages (container, POSIX ``/proc``).

A watcher thread cancels the token as soon as a given page has been read, as the HTTP edge
does when the client disconnects. The Tesseract process of the page in progress is stopped at
once (well within one page time), no later page is started, and no Tesseract process of this
test process is left, running or unreaped.
"""

import dataclasses
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from app.adapters.ocr import TesseractOcrConfig, TesseractOcrEngine
from app.adapters.ocr import tesseract as tesseract_module
from app.adapters.ocr.tesseract import TesseractCli
from app.domain.cancellation import NEVER_CANCELLED, Cancellation
from app.domain.errors import ProcessingFailed
from tests.adapters.ocr.pdfs import A4, ENGLISH_LINES, PageSpec, build_pdf, image_only_pdf

pytestmark = [
    pytest.mark.tesseract,
    pytest.mark.skipif(sys.platform == "win32", reason="reads child processes from /proc"),
]

CONFIG = TesseractOcrConfig(workers=1)  # the benchmarked 200 DPI
LATENCY_LIMIT_S = 1.0
SLEEP = "3019"  # identifies the processes of the script test


def _children(name: str) -> list[int]:
    """PIDs of the processes called ``name`` whose parent is this process (zombies included)."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8")
        except OSError:
            continue
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        if fields.get("Name", "").strip() == name and fields.get("PPid", "").strip() == str(
            os.getpid()
        ):
            found.append(int(entry.name))
    return found


def _sleepers() -> list[int]:
    """PIDs of live or unreaped ``sleep SLEEP`` processes, whoever their parent is."""
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


class _Recording:
    """Runs real Tesseract; records per call (in call order) how it ended and how long it took."""

    def __init__(self, config: TesseractOcrConfig, cancel_after_call: int) -> None:
        self._cli = TesseractCli(
            command=config.tesseract_cmd,
            dpi=config.dpi,
            psm=config.psm,
            oem=config.oem,
            timeout_s=config.timeout_s,
        )
        self._cancel_after_call = cancel_after_call
        self._lock = threading.Lock()
        self.started: list[int] = []
        self.finished: list[int] = []
        self.stopped: list[int] = []
        self.seconds: list[float] = []
        self.call_read = threading.Event()
        self.call_started = threading.Event()

    def __call__(
        self, png: bytes, languages: str, *, cancel: Cancellation = NEVER_CANCELLED
    ) -> str:
        with self._lock:
            call = len(self.started)
            self.started.append(call)
        if call == self._cancel_after_call:
            self.call_started.set()
        begin = time.monotonic()
        try:
            result = self._cli(png, languages, cancel=cancel)
        except ProcessingFailed as error:
            if error.code == "request_cancelled":
                with self._lock:
                    self.stopped.append(call)
            raise
        with self._lock:
            self.finished.append(call)
            self.seconds.append(time.monotonic() - begin)
        if call == self._cancel_after_call:
            self.call_read.set()
        return result


class _Sampler(threading.Thread):
    """Counts this process's Tesseract children every few milliseconds until stopped."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.most = 0

    def run(self) -> None:
        while not self.stop.wait(0.01):
            self.most = max(self.most, len(_children("tesseract")))


def _run(
    pages: int,
    cancel_after_call: int,
    config: TesseractOcrConfig = CONFIG,
    *,
    while_running_s: float | None = None,
) -> tuple[ProcessingFailed, _Recording, _Sampler, float]:
    """Recognise ``pages`` A4 pages, cancelling once call ``cancel_after_call`` has answered,
    or, with ``while_running_s``, that long after the call started (while it runs).

    Returns the error, the recording, the sampler and the seconds from cancel to raise.
    """
    pdf = image_only_pdf([PageSpec(size=A4, lines=ENGLISH_LINES * 8)] * pages)  # ~0.9 s each
    runner = _Recording(config, cancel_after_call)
    engine = TesseractOcrEngine(config, runner=runner)
    cancel = Cancellation()
    cancelled_at: list[float] = []

    def watch() -> None:
        event = runner.call_read if while_running_s is None else runner.call_started
        if event.wait(timeout=120):
            time.sleep(while_running_s or 0.0)
            cancelled_at.append(time.monotonic())
            cancel.cancel()

    watcher = threading.Thread(target=watch, daemon=True)
    sampler = _Sampler()
    assert _children("tesseract") == []
    watcher.start()
    sampler.start()
    try:
        with pytest.raises(ProcessingFailed) as caught:
            engine.recognize(pdf, "en", cancel=cancel)
        raised_at = time.monotonic()
    finally:
        sampler.stop.set()
        sampler.join()
        watcher.join(timeout=5)
    assert caught.value.code == "request_cancelled"
    return caught.value, runner, sampler, raised_at - cancelled_at[0]


def _nothing_left() -> None:
    assert _children("tesseract") == []
    assert not [t for t in threading.enumerate() if t.name.startswith("tesseract")]


def test_cancelling_after_page_one_stops_the_page_in_progress_at_once() -> None:
    _, runner, sampler, latency = _run(pages=6, cancel_after_call=0)

    assert runner.finished == [0]
    assert runner.started[:1] == [0]
    assert len(runner.started) <= 2  # page 1 and, at most, page 2 (stopped while running)
    assert runner.stopped == runner.started[1:]  # nothing after page 1 ran to the end
    assert latency < LATENCY_LIMIT_S
    assert latency < runner.seconds[0]  # well under one page time
    assert sampler.most >= 1  # the probe does see Tesseract while it runs
    _nothing_left()


def test_cancelling_after_the_last_page_was_handed_over_stops_it_too() -> None:
    _, runner, _, latency = _run(pages=3, cancel_after_call=2, while_running_s=0.2)

    assert runner.finished == [0, 1]
    assert runner.stopped == [2]  # the last page was running and was stopped
    assert latency < LATENCY_LIMIT_S
    assert latency < min(runner.seconds)
    _nothing_left()


def test_cancelling_with_parallel_workers_stops_every_running_page() -> None:
    _, runner, _, latency = _run(
        pages=6, cancel_after_call=0, config=dataclasses.replace(CONFIG, workers=2)
    )

    assert len(runner.started) < 6
    assert sorted(runner.finished + runner.stopped) == sorted(runner.started)
    assert latency < LATENCY_LIMIT_S
    _nothing_left()


def test_a_command_ignoring_sigterm_is_killed_after_the_grace_period(tmp_path: Path) -> None:
    script = tmp_path / "stubborn-tesseract"
    script.write_text(f"#!/bin/sh\ntrap '' TERM\nsleep {SLEEP} & wait\n", encoding="utf-8")
    script.chmod(0o755)
    cli = TesseractCli(command=str(script), dpi=72, psm=3, oem=1, timeout_s=30.0)
    cancel = Cancellation()
    errors: list[ProcessingFailed] = []

    def run() -> None:
        try:
            cli(build_pdf([PageSpec()]), "eng", cancel=cancel)
        except ProcessingFailed as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    deadline = time.monotonic() + 5
    while not _children("stubborn-tesser") and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _children("stubborn-tesser")  # /proc truncates names to 15 characters
    time.sleep(0.2)  # let the script start its sleep and ignore SIGTERM
    started = time.monotonic()
    cancel.cancel()
    worker.join(timeout=10)
    latency = time.monotonic() - started

    assert [error.code for error in errors] == ["request_cancelled"]
    assert tesseract_module.CANCEL_GRACE_S <= latency < tesseract_module.CANCEL_GRACE_S + 1.0
    assert _children("stubborn-tesser") == []
    end = time.monotonic() + 3
    while _sleepers() and time.monotonic() < end:  # the killed sleep is reaped by init
        time.sleep(0.05)
    assert _sleepers() == []
