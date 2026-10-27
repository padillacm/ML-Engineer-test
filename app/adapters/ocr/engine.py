"""Tesseract implementation of the ``OcrEngine`` port.

Pages are rendered one after another on the calling thread and their PNG images are handed to
a small thread pool whose threads only wait on single-threaded Tesseract processes. At most
``2 x workers`` rendered pages exist at once, and rendering stops as soon as one page fails.
Results are reassembled in page order.

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
    ``min(4, cpu_count)`` concurrent Tesseract processes. Invalid values raise
    ``ConfigurationError``.
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
    Errors: ``InvalidInput`` (``pdf_unreadable``, ``too_large``,
    ``unsupported_language``) and ``ProcessingFailed`` (``ocr_failed``).
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

    def recognize(self, pdf: bytes, language: str | None = None) -> list[OcrPage]:
        """Return one ``OcrPage`` per page of ``pdf``, in page order.

        With a ``language`` hint every page is read with that model only. Without one, pages
        are read in order with all configured models, accumulating their text, until the
        language is detected or ``detect_max_pages`` pages with alphabetic words have been
        read; the whole document (those pages again included) is then read with the detected
        model. When detection is inconclusive the combined results already obtained are kept
        and only the remaining pages are read, with the combined model.
        """
        model = None if language is None else self._model_for(language)
        with open_pdf(pdf) as document:
            pages = render_pages(
                document, self._config.dpi, self._config.max_page_pixels, self._transform()
            )
            if model is not None:
                return self._recognize_all(pages, model)
            return self._recognize_detecting(pages)

    def _model_for(self, language: str) -> str:
        code = validate_language(language)
        if code not in self._config.languages:
            raise InvalidInput(
                f"language must be one of: {', '.join(self._config.languages)}",
                "unsupported_language",
            )
        return TESSERACT_MODELS[code]

    def _transform(self) -> ImageTransform | None:
        if not self._config.flatten:
            return None
        return partial(flatten_illumination, dpi=self._config.dpi)

    def _recognize_detecting(self, pages: Iterator[RenderedPage]) -> list[OcrPage]:
        seen: list[RenderedPage] = []
        combined: list[OcrPage] = []
        texts: list[str] = []
        for rendered in pages:
            seen.append(rendered)
            result = self._recognize_page(rendered, self._combined)
            combined.append(result)
            if not _has_alphabetic_word(result):
                continue
            texts.append(PageText.from_words(result.words).text)
            detected = detect_language(" ".join(texts), min_words=self._config.detect_min_words)
            if detected is not None and detected in self._config.languages:
                return self._recognize_all(chain(seen, pages), TESSERACT_MODELS[detected])
            if len(texts) >= self._config.detect_max_pages:
                break
        seen.clear()
        return combined + self._recognize_all(pages, self._combined)

    def _recognize_all(self, pages: Iterable[RenderedPage], model: str) -> list[OcrPage]:
        """OCR ``pages`` concurrently, holding at most ``2 x workers`` rendered pages at once.

        The next page is rendered only when a slot is free. After the first failure no further
        page is rendered or started, running processes finish (or time out) and the failure of
        the earliest failed page is raised.
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
                future = pool.submit(self._recognize_unless_stopped, window, rendered, model)
                del rendered
                future.add_done_callback(window.release)
                futures.append(future)
        finally:
            # Pages still queued here are abandoned: a page failed or rendering raised.
            pool.shutdown(wait=True, cancel_futures=True)
        return _gather(futures)

    def _recognize_unless_stopped(
        self, window: "_Window", rendered: RenderedPage, model: str
    ) -> OcrPage:
        if window.failed:
            raise _Abandoned
        return self._recognize_page(rendered, model)

    def _recognize_page(self, rendered: RenderedPage, model: str) -> OcrPage:
        try:
            words = parse_tsv(self._runner(rendered.png, model), rendered.raster, rendered.index)
        except ProcessingFailed as error:
            raise ProcessingFailed(
                f"OCR failed on page {rendered.index}: {error.message}", "ocr_failed"
            ) from error
        raster = rendered.raster
        return OcrPage(rendered.index, raster.page_width_pt, raster.page_height_pt, words)


class _Abandoned(Exception):
    """A queued page that was not started because another page had already failed."""


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


def _gather(futures: list[Future[OcrPage]]) -> list[OcrPage]:
    """Return the results in page order, or raise the failure of the earliest failed page."""
    for future in futures:
        if not future.cancelled() and not isinstance(future.exception(), _Abandoned | None):
            future.result()
    return [future.result() for future in futures]
