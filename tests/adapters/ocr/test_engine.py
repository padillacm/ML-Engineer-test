import dataclasses
import math
import os
import random
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium
import pytest

from app.adapters.ocr import TesseractOcrConfig, TesseractOcrEngine
from app.adapters.ocr.preprocess import flatten_illumination
from app.adapters.ocr.rendering import open_pdf, render_pages
from app.adapters.ocr.tesseract import TesseractCli
from app.application.ports import OcrEngine
from app.domain.cancellation import NEVER_CANCELLED, Cancellation
from app.domain.errors import ConfigurationError, InvalidInput, ProcessingFailed
from app.domain.types import Box, OcrPage
from tests.adapters.ocr.conftest import RenderSpy
from tests.adapters.ocr.fake_runner import FakeTesseract, decode
from tests.adapters.ocr.pdfs import (
    A4,
    PageSpec,
    build_pdf,
    declare_page_count,
    encrypted_pdf,
    image_only_pdf,
)

COMBINED = "eng+spa+por+fra+deu"
SPANISH = (
    "el contrato de los servicios tiene una vigencia del año y las partes que firman el "
    "presente documento se obligan a su cumplimiento hasta la fecha de pago de las facturas"
)
PORTUGUESE = (
    "o contrato foi assinado pelas partes e os bens serao entregues na sede da empresa com "
    "uma garantia que nao pode ser alterada sem o acordo das partes"
)
NO_STOP_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor"
)
# 72 DPI keeps the images tiny: one pixel per point.
FAST = TesseractOcrConfig(dpi=72)


def _pages(count: int, blank: frozenset[int] = frozenset()) -> bytes:
    """``count`` small pages of distinct widths (200, 210, ... pt) so each image is identifiable."""
    return build_pdf(
        [
            PageSpec(size=(200.0 + 10 * index, 300.0), lines=() if index in blank else ("Text",))
            for index in range(count)
        ]
    )


def _plain(pdf: bytes) -> np.ndarray:
    """The first page of ``pdf`` rendered at 72 DPI without any transform."""
    with open_pdf(pdf) as document:
        first = next(render_pages(document, 72, 40_000_000))
    return decode(first.png)


def _engine(runner: FakeTesseract, **options: Any) -> TesseractOcrEngine:
    return TesseractOcrEngine(dataclasses.replace(FAST, **options), runner=runner)


class TestConfig:
    def test_defaults_are_the_benchmarked_configuration(self) -> None:
        config = TesseractOcrEngine().config

        assert config.dpi == 200
        assert config.flatten is True
        assert (config.psm, config.oem) == (3, 1)
        assert config.languages == ("en", "es", "pt", "fr", "de")
        assert config.detect_min_words == 20
        assert config.detect_max_pages == 3
        assert config.timeout_s == 60.0
        assert config.workers is None
        assert config.max_page_pixels == 40_000_000
        assert config.max_pages == 50
        assert config.tesseract_cmd == "tesseract"

    def test_default_runner_is_built_from_the_config(self) -> None:
        engine = TesseractOcrEngine(TesseractOcrConfig(dpi=300, psm=6, timeout_s=5.0))

        assert engine._runner == TesseractCli("tesseract", dpi=300, psm=6, oem=1, timeout_s=5.0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dpi", 0),
            ("dpi", -72),
            ("dpi", 1.5),
            ("dpi", True),
            ("flatten", "yes"),
            ("psm", 14),
            ("psm", -1),
            ("oem", 4),
            ("languages", ()),
            ("languages", ("it",)),
            ("languages", ("en", "en")),
            ("languages", ["en"]),
            ("detect_min_words", 0),
            ("detect_max_pages", 0),
            ("detect_max_pages", True),
            ("timeout_s", 0),
            ("timeout_s", -1.0),
            ("timeout_s", math.inf),
            ("timeout_s", math.nan),
            ("timeout_s", False),
            ("workers", 0),
            ("max_page_pixels", 0),
            ("max_pages", 0),
            ("max_pages", -1),
            ("max_pages", True),
            ("max_pages", 1.5),
            ("max_pages", None),
            ("tesseract_cmd", ""),
            ("tesseract_cmd", "  "),
        ],
    )
    def test_invalid_values_are_rejected(self, field: str, value: object) -> None:
        with pytest.raises(ConfigurationError) as caught:
            TesseractOcrConfig(**{field: value})  # type: ignore[arg-type]

        assert caught.value.code == "invalid_configuration"
        assert field in caught.value.message

    def test_valid_non_default_values_are_accepted(self) -> None:
        config = TesseractOcrConfig(
            dpi=300,
            flatten=False,
            psm=6,
            oem=0,
            languages=("es",),
            timeout_s=1,
            workers=2,
            max_pages=1,
        )

        assert config.languages == ("es",)
        assert config.max_pages == 1


class TestPages:
    def test_engine_satisfies_the_port(self) -> None:
        assert isinstance(TesseractOcrEngine(runner=FakeTesseract()), OcrEngine)

    def test_every_page_is_returned_in_order_including_the_first(self) -> None:
        runner = FakeTesseract()

        pages = _engine(runner).recognize(_pages(3), "en")

        assert [page.page for page in pages] == [0, 1, 2]
        assert [(page.width, page.height) for page in pages] == [
            (200.0, 300.0),
            (210.0, 300.0),
            (220.0, 300.0),
        ]
        assert [page.words[0].text for page in pages] == ["eng:200", "eng:210", "eng:220"]

    def test_blank_page_has_its_size_and_no_words(self) -> None:
        pages = _engine(FakeTesseract()).recognize(_pages(3, blank=frozenset({1})), "en")

        assert pages[1].words == ()
        assert (pages[1].width, pages[1].height) == (210.0, 300.0)
        assert pages[0].words
        assert pages[2].words

    def test_word_boxes_are_converted_to_points_of_their_page(self) -> None:
        engine = _engine(FakeTesseract(), dpi=144)

        pages = engine.recognize(build_pdf([PageSpec(size=A4, rotation=90)]), "en")

        assert (pages[0].width, pages[0].height) == (842.0, 595.0)
        assert pages[0].words[0].box == Box(0, 0.0, 0.0, 4.0, 4.0)

    def test_scanned_image_only_pdf_is_recognised(self) -> None:
        pdf = image_only_pdf([PageSpec(), PageSpec(lines=())])

        pages = _engine(FakeTesseract()).recognize(pdf, "en")

        assert [len(page.words) > 0 for page in pages] == [True, False]

    def test_one_and_four_workers_give_identical_results(self) -> None:
        pdf = _pages(10, blank=frozenset({4}))
        rng = random.Random(3)
        delays = {200 + 10 * index: rng.uniform(0.0, 0.03) for index in range(10)}

        def runner() -> FakeTesseract:
            return FakeTesseract(lambda languages: "some words", delay=delays.__getitem__)

        sequential = _engine(runner(), workers=1).recognize(pdf, "en")
        concurrent = _engine(runner(), workers=4).recognize(pdf, "en")

        assert concurrent == sequential
        assert [page.page for page in concurrent] == list(range(10))

    def test_flattening_is_applied_by_default(self) -> None:
        runner = FakeTesseract()
        pdf = build_pdf([PageSpec()])

        _engine(runner).recognize(pdf, "en")

        assert np.array_equal(decode(runner.pngs[0]), flatten_illumination(_plain(pdf), dpi=72))

    def test_without_flattening_the_rendered_page_is_passed_through(self) -> None:
        runner = FakeTesseract()
        pdf = build_pdf([PageSpec()])

        _engine(runner, flatten=False).recognize(pdf, "en")

        assert np.array_equal(decode(runner.pngs[0]), _plain(pdf))


class TestFailures:
    def test_failure_on_page_two_closes_the_document_and_stops_the_document(
        self, opened_documents: list[pdfium.PdfDocument], rendered_pages: RenderSpy
    ) -> None:
        runner = FakeTesseract(fail_width=220, delay=lambda width: 0.0 if width == 220 else 0.02)
        pdf = _pages(12)

        with pytest.raises(ProcessingFailed) as caught:
            _engine(runner, workers=4).recognize(pdf, "en")

        assert caught.value.code == "ocr_failed"
        assert "page 2" in caught.value.message
        assert isinstance(caught.value.__cause__, ProcessingFailed)
        assert [document.raw is None for document in opened_documents] == [True]
        # Pages beyond the in-flight window (2 x 4 workers) were never rendered or read.
        assert len(rendered_pages.indexes) <= 8
        assert set(runner.widths) <= {200 + 10 * index for index in range(8)}
        assert not [t for t in threading.enumerate() if t.name.startswith("tesseract")]

    def test_failure_on_the_first_page_stops_rendering_at_once(
        self, rendered_pages: RenderSpy
    ) -> None:
        # OCR is much slower than rendering, so without the window every page would be rendered.
        runner = FakeTesseract(fail_width=200, delay=lambda width: 0.1 if width == 200 else 0.2)

        started = time.monotonic()
        with pytest.raises(ProcessingFailed) as caught:
            _engine(runner, workers=2, flatten=False).recognize(_pages(20), "en")
        elapsed = time.monotonic() - started

        assert "page 0" in caught.value.message
        assert len(rendered_pages.indexes) <= 4
        assert len(runner.calls) <= 2
        assert elapsed < 1.0

    def test_rendered_pages_waiting_for_ocr_are_bounded(self, rendered_pages: RenderSpy) -> None:
        runner = FakeTesseract(delay=lambda width: 0.03)
        held: list[int] = []

        def tracking(png: bytes, languages: str, *, cancel: Cancellation = NEVER_CANCELLED) -> str:
            with lock:
                # Pages rendered and not yet finished by OCR, this one included.
                held.append(len(rendered_pages.indexes) - len(done))
            try:
                return runner(png, languages)
            finally:
                with lock:
                    done.append(1)

        lock = threading.Lock()
        done: list[int] = []
        config = dataclasses.replace(FAST, workers=2, flatten=False)
        engine = TesseractOcrEngine(config, runner=tracking)

        pages = engine.recognize(_pages(20), "en")

        assert [page.page for page in pages] == list(range(20))
        assert len(rendered_pages.indexes) == 20
        assert max(held) <= 4  # the 2 x 2 window
        assert runner.max_active <= 2

    def test_concurrent_calls_from_several_threads_get_their_own_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inside = 0
        overlaps: list[int] = []
        guard = threading.Lock()
        real_render = pdfium.PdfPage.render

        def render(page: pdfium.PdfPage, *args: Any, **kwargs: Any) -> Any:
            nonlocal inside
            with guard:
                inside += 1
                overlaps.append(inside)
            try:
                time.sleep(0.002)
                return real_render(page, *args, **kwargs)
            finally:
                with guard:
                    inside -= 1

        monkeypatch.setattr(pdfium.PdfPage, "render", render)
        counts = (3, 4, 5, 6)
        documents = {
            count: build_pdf(
                [PageSpec(size=(100.0 + 30 * count + 3 * page, 150.0)) for page in range(count)]
            )
            for count in counts
        }
        engine = _engine(FakeTesseract(delay=lambda width: 0.005), workers=2)
        results: dict[int, list[OcrPage]] = {}
        errors: list[BaseException] = []
        barrier = threading.Barrier(len(counts))

        def work(count: int) -> None:
            try:
                barrier.wait()
                results[count] = engine.recognize(documents[count], "en")
            except BaseException as error:  # noqa: BLE001 - reported by the assertion below
                errors.append(error)

        threads = [threading.Thread(target=work, args=(count,)) for count in counts]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert sorted(results) == list(counts)
        for count, pages in results.items():
            widths = [100 + 30 * count + 3 * page for page in range(count)]
            assert [page.words[0].text for page in pages] == [f"eng:{w}" for w in widths]
        assert max(overlaps) == 1  # PDFium was never entered by two threads at once

    def test_failure_during_detection_is_reported(self) -> None:
        runner = FakeTesseract(fail_width=200)

        with pytest.raises(ProcessingFailed) as caught:
            _engine(runner).recognize(_pages(2))

        assert "page 0" in caught.value.message
        assert runner.calls == [COMBINED]

    def test_encrypted_pdf_is_unreadable(self) -> None:
        runner = FakeTesseract()

        with pytest.raises(InvalidInput) as caught:
            _engine(runner).recognize(encrypted_pdf(), "en")

        assert caught.value.code == "pdf_unreadable"
        assert runner.calls == []

    def test_corrupt_pdf_is_unreadable(self) -> None:
        with pytest.raises(InvalidInput) as caught:
            _engine(FakeTesseract()).recognize(b"%PDF-1.7\n\x00\x01 broken", None)

        assert caught.value.code == "pdf_unreadable"

    def test_page_over_the_pixel_budget_is_too_large_before_any_ocr(self) -> None:
        runner = FakeTesseract()

        with pytest.raises(InvalidInput) as caught:
            _engine(runner, max_page_pixels=200 * 300).recognize(_pages(2), "en")

        assert caught.value.code == "too_large"
        assert runner.calls == []

    def test_no_files_are_created(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        temp_dir = tmp_path / "temp"
        work_dir = tmp_path / "work"
        temp_dir.mkdir()
        work_dir.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
        for variable in ("TMP", "TEMP", "TMPDIR"):
            monkeypatch.setenv(variable, str(temp_dir))
        monkeypatch.chdir(work_dir)

        _engine(FakeTesseract(lambda languages: SPANISH), workers=2).recognize(_pages(3))

        assert tempfile.gettempdir() == str(temp_dir)
        assert os.listdir(temp_dir) == []
        assert os.listdir(work_dir) == []


class TestPageLimit:
    def test_one_page_over_the_limit_is_refused_before_any_rendering(
        self, opened_documents: list[pdfium.PdfDocument], rendered_pages: RenderSpy
    ) -> None:
        runner = FakeTesseract()

        with pytest.raises(InvalidInput) as caught:
            _engine(runner, max_pages=3).recognize(_pages(4), "en")

        assert caught.value.code == "too_many_pages"
        assert caught.value.message == "the PDF has more than 3 pages; at most 3 are accepted"
        assert rendered_pages.indexes == []
        assert runner.calls == []
        assert [document.raw is None for document in opened_documents] == [True]

    def test_limit_applies_while_detecting_the_language(self, rendered_pages: RenderSpy) -> None:
        runner = FakeTesseract(lambda languages: SPANISH)

        with pytest.raises(InvalidInput) as caught:
            _engine(runner, max_pages=3).recognize(_pages(4))

        assert caught.value.code == "too_many_pages"
        assert rendered_pages.indexes == []
        assert runner.calls == []

    def test_exactly_the_limit_is_recognised(
        self, opened_documents: list[pdfium.PdfDocument], rendered_pages: RenderSpy
    ) -> None:
        runner = FakeTesseract()

        pages = _engine(runner, max_pages=3).recognize(_pages(3), "en")

        assert [page.page for page in pages] == [0, 1, 2]
        assert rendered_pages.indexes == [0, 1, 2]
        assert runner.calls == ["eng"] * 3
        assert [document.raw is None for document in opened_documents] == [True]

    def test_default_limit_refuses_51_pages(self, rendered_pages: RenderSpy) -> None:
        runner = FakeTesseract()
        pdf = build_pdf([PageSpec(lines=()) for _ in range(51)])

        with pytest.raises(InvalidInput) as caught:
            TesseractOcrEngine(runner=runner).recognize(pdf, "en")

        assert caught.value.code == "too_many_pages"
        assert caught.value.message == ("the PDF has more than 50 pages; at most 50 are accepted")
        assert rendered_pages.indexes == []
        assert runner.calls == []


class TestLanguage:
    def test_hint_reads_every_page_with_that_model_only(self) -> None:
        runner = FakeTesseract(lambda languages: SPANISH)

        pages = _engine(runner).recognize(_pages(3), " ES ")

        assert runner.calls == ["spa", "spa", "spa"]
        assert [page.language for page in pages] == ["es", "es", "es"]

    @pytest.mark.parametrize("hint", ["it", "spa", ""])
    def test_unsupported_hint_is_rejected_before_opening_the_pdf(self, hint: str) -> None:
        runner = FakeTesseract()

        with pytest.raises(InvalidInput) as caught:
            _engine(runner).recognize(b"not even a pdf", hint)

        assert caught.value.code == "unsupported_language"
        assert runner.calls == []

    def test_hint_outside_the_configured_languages_is_rejected(self) -> None:
        with pytest.raises(InvalidInput) as caught:
            _engine(FakeTesseract(), languages=("en", "es")).recognize(_pages(1), "de")

        assert caught.value.code == "unsupported_language"
        assert "en, es" in caught.value.message

    def test_detected_language_is_used_for_every_page(self) -> None:
        runner = FakeTesseract(lambda languages: SPANISH)

        pages = _engine(runner).recognize(_pages(3, blank=frozenset({0})))

        assert runner.calls[:2] == [COMBINED, COMBINED]
        assert runner.calls[2:] == ["spa", "spa", "spa"]
        assert pages[0].words == ()
        assert [page.words[0].text for page in pages[1:]] == ["spa:210", "spa:220"]
        assert [page.language for page in pages] == ["es", "es", "es"]

    def test_inconclusive_detection_reads_every_page_once_with_all_languages(self) -> None:
        runner = FakeTesseract(lambda languages: NO_STOP_WORDS)

        pages = _engine(runner).recognize(_pages(3))

        assert runner.calls == [COMBINED] * 3
        assert [page.words[0].text for page in pages] == [
            f"{COMBINED}:200",
            f"{COMBINED}:210",
            f"{COMBINED}:220",
        ]
        assert [page.language for page in pages] == [None, None, None]

    def test_too_little_text_everywhere_reads_once_with_all_languages(self) -> None:
        runner = FakeTesseract(lambda languages: "el")

        pages = _engine(runner).recognize(_pages(3))

        assert runner.calls == [COMBINED] * 3
        assert [page.page for page in pages] == [0, 1, 2]
        assert [page.language for page in pages] == [None, None, None]

    def test_first_page_of_names_does_not_decide_and_later_pages_do(self) -> None:
        names = " ".join(f"Name{index} Surname{index} 2024" for index in range(15))
        runner = FakeTesseract(page_text={200: names, 210: SPANISH, 220: SPANISH})

        pages = _engine(runner).recognize(_pages(3))

        assert runner.calls == [COMBINED, COMBINED, "spa", "spa", "spa"]
        assert [page.words[0].text for page in pages] == ["spa:200", "spa:210", "spa:220"]
        assert [page.language for page in pages] == ["es", "es", "es"]

    def test_inconclusive_pages_are_never_read_twice(self) -> None:
        runner = FakeTesseract(lambda languages: NO_STOP_WORDS)

        pages = _engine(runner, workers=2).recognize(_pages(6))

        assert runner.calls == [COMBINED] * 6
        assert sorted(runner.widths) == [200 + 10 * index for index in range(6)]
        assert [page.page for page in pages] == list(range(6))
        assert {page.language for page in pages} == {None}

    def test_detection_gives_up_after_the_configured_pages_with_text(self) -> None:
        runner = FakeTesseract(lambda languages: NO_STOP_WORDS, page_text={230: SPANISH})

        pages = _engine(runner, detect_max_pages=2).recognize(_pages(4, blank=frozenset({0})))

        # Page 0 is blank and does not count; pages 1 and 2 are inconclusive: stop there,
        # so the Spanish page 3 is read with the combined model like the rest.
        assert runner.calls == [COMBINED] * 4
        assert [page.language for page in pages] == [None] * 4

    def test_detection_threshold_is_configurable(self) -> None:
        runner = FakeTesseract(lambda languages: "el contrato y los pagos de las facturas")

        _engine(runner, detect_min_words=5).recognize(_pages(2))

        assert runner.calls == [COMBINED, "spa", "spa"]

    def test_combined_model_follows_the_configured_languages(self) -> None:
        runner = FakeTesseract(lambda languages: PORTUGUESE)

        _engine(runner, languages=("en", "es")).recognize(_pages(2))

        assert runner.calls == ["eng+spa", "eng+spa"]


class TestDeclaredPageCount:
    def test_one_page_declaring_a_million_pages_is_recognised(
        self, opened_documents: list[pdfium.PdfDocument], rendered_pages: RenderSpy
    ) -> None:
        runner = FakeTesseract()
        pdf = declare_page_count(_pages(1), 1_000_000)

        pages = TesseractOcrEngine(FAST, runner=runner).recognize(pdf, "en")

        assert [page.page for page in pages] == [0]
        assert pages[0].words[0].text == "eng:200"
        assert rendered_pages.indexes == [0]
        assert runner.calls == ["eng"]
        assert [document.raw is None for document in opened_documents] == [True]

    def test_overstated_document_is_recognised_while_detecting_the_language(
        self, rendered_pages: RenderSpy
    ) -> None:
        runner = FakeTesseract(lambda languages: SPANISH)
        pdf = declare_page_count(_pages(2), 1_000_000)

        pages = _engine(runner).recognize(pdf)

        assert [page.page for page in pages] == [0, 1]
        assert [page.language for page in pages] == ["es", "es"]
        assert sorted(set(rendered_pages.indexes)) == [0, 1]

    def test_a_real_5000_page_document_is_refused_fast_without_rendering(
        self, five_thousand_pages: bytes, rendered_pages: RenderSpy
    ) -> None:
        runner = FakeTesseract()

        started = time.monotonic()
        with pytest.raises(InvalidInput) as caught:
            TesseractOcrEngine(runner=runner).recognize(five_thousand_pages, "en")
        elapsed = time.monotonic() - started

        assert caught.value.code == "too_many_pages"
        assert rendered_pages.indexes == []
        assert runner.calls == []
        assert elapsed < 1.0


class _CancellingRunner:
    """Wraps a ``FakeTesseract`` and cancels ``cancel`` around call number ``on_call``.

    By default the token is cancelled when that call starts (the fake then stops at once, as
    ``TesseractCli`` does); with ``after_answer`` it is cancelled once that call has answered.
    ``cancelled_at`` is the ``time.monotonic()`` of the cancellation.
    """

    def __init__(
        self,
        runner: FakeTesseract,
        cancel: Cancellation,
        on_call: int,
        *,
        after_answer: bool = False,
    ) -> None:
        self._runner = runner
        self._cancel = cancel
        self._on_call = on_call
        self._after_answer = after_answer
        self._lock = threading.Lock()
        self._started = 0
        self.cancelled_at: float | None = None

    def __call__(
        self, png: bytes, languages: str, *, cancel: Cancellation = NEVER_CANCELLED
    ) -> str:
        with self._lock:
            self._started += 1
            number = self._started
        if number == self._on_call and not self._after_answer:
            self._cancel_now()
        result = self._runner(png, languages, cancel=cancel)
        if number == self._on_call and self._after_answer:
            self._cancel_now()
        return result

    def _cancel_now(self) -> None:
        self.cancelled_at = time.monotonic()
        self._cancel.cancel()


def _no_tesseract_threads() -> bool:
    return not [thread for thread in threading.enumerate() if thread.name.startswith("tesseract")]


def _recognize(
    runner: _CancellingRunner,
    cancel: Cancellation,
    pdf: bytes,
    language: str | None,
    **options: Any,
) -> float:
    """Recognise ``pdf`` expecting ``request_cancelled``; return the cancel-to-raise seconds."""
    engine = TesseractOcrEngine(dataclasses.replace(FAST, **options), runner=runner)
    with pytest.raises(ProcessingFailed) as caught:
        engine.recognize(pdf, language, cancel=cancel)
    raised_at = time.monotonic()
    assert caught.value.code == "request_cancelled"
    assert runner.cancelled_at is not None
    return raised_at - runner.cancelled_at


class TestCancellation:
    def test_a_cancelled_token_stops_before_anything_is_rendered(
        self, opened_documents: list[pdfium.PdfDocument], rendered_pages: RenderSpy
    ) -> None:
        runner = FakeTesseract()
        cancel = Cancellation()
        cancel.cancel()

        for language in ("en", None):
            with pytest.raises(ProcessingFailed) as caught:
                _engine(runner).recognize(_pages(3), language, cancel=cancel)
            assert caught.value.code == "request_cancelled"

        assert rendered_pages.indexes == []
        assert runner.calls == []
        assert [document.raw is None for document in opened_documents] == [True, True]

    def test_cancelled_during_a_page_stops_it_at_once_and_starts_no_later_page(
        self, opened_documents: list[pdfium.PdfDocument], rendered_pages: RenderSpy
    ) -> None:
        cancel = Cancellation()
        fake = FakeTesseract(delay=lambda width: 0.05 if width == 200 else 2.0)
        runner = _CancellingRunner(fake, cancel, on_call=2)  # page 1

        latency = _recognize(runner, cancel, _pages(6), "en", workers=1)

        assert fake.widths == [200, 210]  # page 2 and later never reached Tesseract
        assert fake.finished == [200]
        assert fake.stopped == [210]  # the page in progress was stopped, not read to the end
        assert latency < 0.5  # far below the 2 s the page would have taken
        assert len(rendered_pages.indexes) <= 3  # pages 0 and 1, at most one queued behind
        assert [document.raw is None for document in opened_documents] == [True]
        assert _no_tesseract_threads()

    def test_parallel_workers_are_all_stopped_and_leave_nothing_running(
        self, rendered_pages: RenderSpy
    ) -> None:
        cancel = Cancellation()
        fake = FakeTesseract(delay=lambda width: 2.0)
        runner = _CancellingRunner(fake, cancel, on_call=4)  # the four workers are busy

        latency = _recognize(runner, cancel, _pages(20), "en", workers=4)

        assert len(fake.widths) == 4
        assert fake.finished == []
        assert sorted(fake.stopped) == sorted(fake.widths)
        assert fake.active == 0
        assert latency < 0.5
        assert len(rendered_pages.indexes) <= 8  # the 2 x 4 window
        assert _no_tesseract_threads()

    def test_cancelled_after_the_last_page_was_handed_over_stops_it_too(self) -> None:
        cancel = Cancellation()
        fake = FakeTesseract(delay=lambda width: 0.05 if width < 220 else 2.0)
        runner = _CancellingRunner(fake, cancel, on_call=3)  # page 2, the last one

        latency = _recognize(runner, cancel, _pages(3), "en", workers=1)

        assert fake.finished == [200, 210]
        assert fake.stopped == [220]
        assert latency < 0.5
        assert _no_tesseract_threads()

    def test_cancelled_once_every_page_is_read_returns_no_result(self) -> None:
        cancel = Cancellation()
        fake = FakeTesseract()
        runner = _CancellingRunner(fake, cancel, on_call=3, after_answer=True)

        _recognize(runner, cancel, _pages(3), "en", workers=1)

        assert sorted(fake.finished) == [200, 210, 220]

    def test_cancelled_while_detecting_reads_no_further_page(self) -> None:
        cancel = Cancellation()
        fake = FakeTesseract(lambda languages: NO_STOP_WORDS, delay=lambda width: 2.0)
        runner = _CancellingRunner(fake, cancel, on_call=1)

        latency = _recognize(runner, cancel, _pages(4), None)

        assert fake.calls == [COMBINED]
        assert fake.stopped == [200]
        assert latency < 0.5

    def test_cancelled_after_detection_stops_the_second_reading(self) -> None:
        cancel = Cancellation()
        fake = FakeTesseract(lambda languages: SPANISH, delay=lambda width: 0.02)
        runner = _CancellingRunner(fake, cancel, on_call=2)  # the first Spanish page

        _recognize(runner, cancel, _pages(4), None, workers=1)

        assert fake.calls == [COMBINED, "spa"]
        assert fake.stopped == [200]
        assert _no_tesseract_threads()

    def test_a_page_failure_is_reported_even_when_the_job_is_cancelled(self) -> None:
        cancel = Cancellation()
        fake = FakeTesseract(fail_width=200)
        runner = _CancellingRunner(fake, cancel, on_call=1, after_answer=True)
        engine = TesseractOcrEngine(dataclasses.replace(FAST, workers=1), runner=runner)

        with pytest.raises(ProcessingFailed) as caught:
            engine.recognize(_pages(3), "en", cancel=cancel)

        assert caught.value.code in {"ocr_failed", "request_cancelled"}
        assert fake.widths == [200]
        assert _no_tesseract_threads()

    def test_a_live_token_reads_every_page(self) -> None:
        pages = _engine(FakeTesseract(), workers=2).recognize(
            _pages(5), "en", cancel=Cancellation()
        )

        assert [page.page for page in pages] == list(range(5))
