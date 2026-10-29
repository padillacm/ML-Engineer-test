"""Tesseract implementation of the ``OcrEngine`` port.

Pages are rendered one after another on the calling thread and their PNG images are handed to
a small thread pool whose threads only wait on single-threaded Tesseract processes. At most
``2 x workers`` rendered pages exist at once, and rendering stops as soon as one page fails.
Results are reassembled in page order.

Cancellation: the ``Cancellation`` token is checked before every page is rendered (on the
calling thread), by a worker before it hands a rendered page to Tesseract, and by the runner
while Tesseract runs. Once it is cancelled no further page is rendered or started, the
Tesseract processes already running are terminated (killed after a short grace period, and
reaped; see ``app.adapters.ocr.tesseract``), every thread of the pool is joined and the
rendered images are released before ``request_cancelled`` is raised, so nothing keeps running
after ``recognize`` returns. A page being rendered when the token is cancelled still finishes
rendering (PDFium cannot be interrupted).

Thread safety: ``recognize`` may be called from several threads at once. PDFium is not
thread-safe, so every PDFium call (open, render, close) is serialised by the module-level lock
of ``app.adapters.ocr.rendering``; the Tesseract processes of all calls still run concurrently.

Coordinates: word boxes are in points in the displayed frame of the page (after ``/Rotate``,
relative to the CropBox, origin at the top-left). Drawing or redacting on a rotated page needs
the page's derotation matrix (PyMuPDF ``page.derotation_matrix``) to map boxes back.
"""

import os
import threading
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import partial
from itertools import chain
from math import isfinite

from app.adapters.ocr.preprocess import flatten_illumination
from app.adapters.ocr.rendering import ImageTransform, RenderedPage, open_pdf, render_pages
from app.adapters.ocr.tesseract import TesseractCli, TesseractRunner, parse_tsv
from app.domain.cancellation import NEVER_CANCELLED, Cancellation
from app.domain.errors import ConfigurationError, InvalidInput, ProcessingFailed
from app.domain.geometry import PageText
from app.domain.language import SUPPORTED_LANGUAGES, detect_language, validate_language
from app.domain.types import OcrPage

TESSERACT_MODELS: dict[str, str] = {
    "en": "eng",
    "es": "spa",
    "pt": "por",
    "fr": "fra",
    "de": "deu",
}

_MAX_PSM = 13
_MAX_OEM = 3
_DEFAULT_MAX_WORKERS = 4
_PAGES_IN_FLIGHT_PER_WORKER = 2


@dataclass(frozen=True, slots=True)
class TesseractOcrConfig:
    """Settings of ``TesseractOcrEngine``; the defaults are the benchmarked configuration.

    ``languages`` are engine-neutral codes, read together when the document language is
    unknown. Detection reads pages with all of them until ``detect_language`` decides, giving
    up after ``detect_max_pages`` pages that contain alphabetic words. ``workers=None`` uses
    ``min(4, cpu_count)`` concurrent Tesseract processes (the machine's CPUs: a container's CPU
    quota is not seen; the service passes the value it derives from the CPUs available to the
    process, quota included). ``max_pages`` is the largest page
    count accepted; a longer document is refused after opening, before any page is loaded or
    rendered. Invalid values raise ``ConfigurationError``.
    """

    dpi: int = 200
    flatten: bool = True
    psm: int = 3
    oem: int = 1
    languages: tuple[str, ...] = SUPPORTED_LANGUAGES
    detect_min_words: int = 20
    detect_max_pages: int = 3
    timeout_s: float = 60.0
    workers: int | None = None
    max_page_pixels: int = 40_000_000
    max_pages: int = 50
    tesseract_cmd: str = "tesseract"

    def __post_init__(self) -> None:
        _require(_is_positive_int(self.dpi), "dpi must be a positive integer")
        _require(isinstance(self.flatten, bool), "flatten must be a boolean")
        _require(_in_range(self.psm, _MAX_PSM), f"psm must be an integer in 0..{_MAX_PSM}")
        _require(_in_range(self.oem, _MAX_OEM), f"oem must be an integer in 0..{_MAX_OEM}")
        _require(_valid_languages(self.languages), _LANGUAGES_MESSAGE)
        _require(_is_positive_int(self.detect_min_words), "detect_min_words must be >= 1")
        _require(_is_positive_int(self.detect_max_pages), "detect_max_pages must be >= 1")
        _require(_is_positive_number(self.timeout_s), "timeout_s must be a finite number > 0")
        _require(
            self.workers is None or _is_positive_int(self.workers),
            "workers must be None or a positive integer",
        )
        _require(_is_positive_int(self.max_page_pixels), "max_page_pixels must be >= 1")
        _require(_is_positive_int(self.max_pages), "max_pages must be a positive integer")
        _require(
            isinstance(self.tesseract_cmd, str) and bool(self.tesseract_cmd.strip()),
            "tesseract_cmd must be a non-empty string",
        )


_LANGUAGES_MESSAGE = (
    f"languages must be a non-empty tuple of distinct codes from: {', '.join(SUPPORTED_LANGUAGES)}"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _in_range(value: object, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= maximum


def _is_positive_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(value)
        and value > 0
    )


def _valid_languages(languages: object) -> bool:
    return (
        isinstance(languages, tuple)
        and bool(languages)
        and len(set(languages)) == len(languages)
        and all(code in SUPPORTED_LANGUAGES for code in languages)
    )


class TesseractOcrEngine:
    """Recognises scanned PDFs with the Tesseract CLI; satisfies the ``OcrEngine`` port.

    ``runner`` replaces the Tesseract process, for tests; by default it is a ``TesseractCli``
    built from ``config``. Safe to call from several threads at once (see the module notes).
    Errors: ``InvalidInput`` (``pdf_unreadable``, ``too_large``, ``too_many_pages``,
    ``unsupported_language``) and ``ProcessingFailed`` (``ocr_failed``, or
    ``request_cancelled`` once the ``cancel`` token of ``recognize`` is cancelled).
    """

    def __init__(
        self, config: TesseractOcrConfig | None = None, runner: TesseractRunner | None = None
    ) -> None:
        self._config = config if config is not None else TesseractOcrConfig()
        self._runner: TesseractRunner = runner if runner is not None else self._default_runner()
        self._workers = self._config.workers or min(_DEFAULT_MAX_WORKERS, os.cpu_count() or 1)
        self._combined = "+".join(TESSERACT_MODELS[code] for code in self._config.languages)

    @property
    def config(self) -> TesseractOcrConfig:
        return self._config

    def _default_runner(self) -> TesseractRunner:
        config = self._config
        return TesseractCli(
            command=config.tesseract_cmd,
            dpi=config.dpi,
            psm=config.psm,
            oem=config.oem,
            timeout_s=config.timeout_s,
        )

    def recognize(
        self,
        pdf: bytes,
        language: str | None = None,
        *,
        cancel: Cancellation = NEVER_CANCELLED,
    ) -> list[OcrPage]:
        """Return one ``OcrPage`` per page of ``pdf``, in page order.

        With a ``language`` hint every page is read with that model only. Without one, pages
        are read in order with all configured models, accumulating their text, until the
        language is detected or ``detect_max_pages`` pages with alphabetic words have been
        read; the whole document (those pages again included) is then read with the detected
        model. When detection is inconclusive the combined results already obtained are kept
        and only the remaining pages are read, with the combined model.

        Every page reports the language it was read with: the hint, the detected language, or
        ``None`` when detection was inconclusive (all pages then report ``None``).

        A document with more than ``max_pages`` pages raises ``InvalidInput`` (code
        ``too_many_pages``) before any page is rendered or read.

        ``cancel`` is checked before each page is rendered, before each is recognised and while
        Tesseract runs; once cancelled, running Tesseract processes are stopped, no further
        page starts and ``ProcessingFailed`` (code ``request_cancelled``) is raised.
        """
        code = None if language is None else self._supported(language)
        with open_pdf(pdf, max_pages=self._config.max_pages) as document:
            pages = _until_cancelled(
                render_pages(
                    document, self._config.dpi, self._config.max_page_pixels, self._transform()
                ),
                cancel,
            )
            if code is not None:
                return self._recognize_all(pages, code, cancel)
            return self._recognize_detecting(pages, cancel)

    def _supported(self, language: str) -> str:
        code = validate_language(language)
        if code not in self._config.languages:
            raise InvalidInput(
                f"language must be one of: {', '.join(self._config.languages)}",
                "unsupported_language",
            )
        return code

    def _model(self, language: str | None) -> str:
        return self._combined if language is None else TESSERACT_MODELS[language]

    def _transform(self) -> ImageTransform | None:
        if not self._config.flatten:
            return None
        return partial(flatten_illumination, dpi=self._config.dpi)

    def _recognize_detecting(
        self, pages: Iterator[RenderedPage], cancel: Cancellation
    ) -> list[OcrPage]:
        seen: list[RenderedPage] = []
        combined: list[OcrPage] = []
        texts: list[str] = []
        for rendered in pages:
            seen.append(rendered)
            result = self._recognize_page(rendered, None, cancel)
            combined.append(result)
            if not _has_alphabetic_word(result):
                continue
            texts.append(PageText.from_words(result.words).text)
            detected = detect_language(" ".join(texts), min_words=self._config.detect_min_words)
            if detected is not None and detected in self._config.languages:
                return self._recognize_all(chain(seen, pages), detected, cancel)
            if len(texts) >= self._config.detect_max_pages:
                break
        seen.clear()
        return combined + self._recognize_all(pages, None, cancel)

    def _recognize_all(
        self, pages: Iterable[RenderedPage], language: str | None, cancel: Cancellation
    ) -> list[OcrPage]:
        """OCR ``pages`` concurrently, holding at most ``2 x workers`` rendered pages at once.

        ``language`` is a supported code, or ``None`` for the combined model.

        The next page is rendered only when a slot is free. After the first failure, or once
        ``cancel`` is cancelled, no further page is rendered or started, running processes
        finish (or time out) and the failure of the earliest failed page (else
        ``request_cancelled``) is raised.
        """
        window = _Window(_PAGES_IN_FLIGHT_PER_WORKER * self._workers)
        futures: list[Future[OcrPage]] = []
        iterator = iter(pages)
        pool = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="tesseract")
        try:
            while window.acquire():
                rendered = next(iterator, None)
                if rendered is None:
                    # Every page is submitted: wait for all of them, or for the first failure.
                    wait(futures, return_when=FIRST_EXCEPTION)
                    break
                future = pool.submit(
                    self._recognize_unless_stopped, window, rendered, language, cancel
                )
                del rendered
                future.add_done_callback(window.release)
                futures.append(future)
        finally:
            # Pages still queued here are abandoned: a page failed, rendering raised or the
            # request was cancelled. Running pages finish; every worker thread is joined.
            pool.shutdown(wait=True, cancel_futures=True)
        return _gather(futures, cancel)

    def _recognize_unless_stopped(
        self,
        window: "_Window",
        rendered: RenderedPage,
        language: str | None,
        cancel: Cancellation,
    ) -> OcrPage:
        if window.failed or cancel.cancelled:
            raise _Abandoned
        return self._recognize_page(rendered, language, cancel)

    def _recognize_page(
        self, rendered: RenderedPage, language: str | None, cancel: Cancellation
    ) -> OcrPage:
        model = self._model(language)
        try:
            tsv = self._runner(rendered.png, model, cancel=cancel)
            words = parse_tsv(tsv, rendered.raster, rendered.index)
        except ProcessingFailed as error:
            if error.code == "request_cancelled":
                raise
            raise ProcessingFailed(
                f"OCR failed on page {rendered.index}: {error.message}", "ocr_failed"
            ) from error
        raster = rendered.raster
        return OcrPage(rendered.index, raster.page_width_pt, raster.page_height_pt, words, language)


class _Abandoned(Exception):
    """A queued page that was not started: another page had failed or the job was cancelled."""


class _Window:
    """Free slots for rendered pages; ``acquire`` waits for one and fails once a page failed."""

    def __init__(self, size: int) -> None:
        self._condition = threading.Condition()
        self._free = size
        self._failed = False

    def acquire(self) -> bool:
        with self._condition:
            self._condition.wait_for(lambda: self._free > 0 or self._failed)
            if self._failed:
                return False
            self._free -= 1
            return True

    @property
    def failed(self) -> bool:
        with self._condition:
            return self._failed

    def release(self, future: Future[OcrPage]) -> None:
        failed = not future.cancelled() and future.exception() is not None
        with self._condition:
            self._free += 1
            self._failed = self._failed or failed
            self._condition.notify_all()


def _has_alphabetic_word(page: OcrPage) -> bool:
    return any(char.isalpha() for word in page.words for char in word.text)


def _until_cancelled(pages: Iterator[RenderedPage], cancel: Cancellation) -> Iterator[RenderedPage]:
    """Yield ``pages``, checking ``cancel`` before each page is rendered (one per ``next``)."""
    while True:
        cancel.raise_if_cancelled()
        rendered = next(pages, None)
        if rendered is None:
            return
        yield rendered
        del rendered


def _gather(futures: list[Future[OcrPage]], cancel: Cancellation) -> list[OcrPage]:
    """Return the results in page order, or raise the failure of the earliest failed page.

    Without a page failure, a cancelled token raises ``request_cancelled`` even when every page
    was read: a cancelled job never returns results.
    """
    for future in futures:
        if not future.cancelled() and not isinstance(future.exception(), _Abandoned | None):
            future.result()
    cancel.raise_if_cancelled()
    return [future.result() for future in futures]
