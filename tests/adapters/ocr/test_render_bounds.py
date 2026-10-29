"""The render bounds: the time budget (progressive rendering) and the image pre-check."""

import threading
import time
import zlib
from pathlib import Path

import numpy as np
import pymupdf
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest

from app.adapters.ocr import TesseractOcrConfig, TesseractOcrEngine
from app.adapters.ocr.rendering import (
    DEFAULT_RENDER_LIMITS,
    MAX_FORM_DEPTH,
    PDFIUM_LOCK,
    RenderedPage,
    RenderLimits,
    open_pdf,
    render_pages,
)
from app.domain.errors import InvalidInput
from tests.adapters.ocr.fake_runner import FakeTesseract, decode
from tests.adapters.ocr.pdfs import (
    A4,
    ENGLISH_LINES,
    PAGE_DICT,
    PageSpec,
    build_pdf,
    cropped_pdf,
    image_only_pdf,
    inherited_boxes_pdf,
    many_strokes_pdf,
    page_tree_pdf,
)
from tests.adversarial import hostile_pdfs as hostile

SAMPLES_DIR = Path(__file__).resolve().parents[3] / "sample_pdfs"
SAMPLES = sorted(SAMPLES_DIR.glob("*.pdf"))
# 100,000 strokes: about four seconds of drawing at 200 DPI on this class of machine.
HOSTILE = many_strokes_pdf(100_000)
TINY_BUDGET_S = 0.05
# The budget plus "one progress step": PDFium polls every hundred objects (milliseconds here);
# the slack covers a slow CI host, not a design margin.
STEP_SLACK_S = 0.5


def _render(
    pdf: bytes, dpi: int, limits: RenderLimits = DEFAULT_RENDER_LIMITS
) -> list[RenderedPage]:
    with open_pdf(pdf) as document:
        return list(render_pages(document, dpi, 40_000_000, None, limits))


def _unbounded_render(pdf: bytes, index: int, dpi: int) -> np.ndarray:
    """The page through pypdfium2's own ``PdfPage.render``: the previous code of the adapter."""
    with PDFIUM_LOCK:
        document = pdfium.PdfDocument(pdf)
        try:
            page = document[index]
            try:
                bitmap = page.render(scale=dpi / 72, grayscale=True, may_draw_forms=False)
                try:
                    return np.array(bitmap.to_numpy(), copy=True)
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            document.close()


class _SteppingClock:
    """A monotonic clock that advances ``step`` seconds on every reading."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step
        self.readings = 0

    def __call__(self) -> float:
        self.readings += 1
        value = self.now
        self.now += self.step
        return value


def _lock_is_free() -> bool:
    """Whether another thread can take the PDFium lock now (an RLock lies to its owner)."""
    taken: list[bool] = []

    def probe() -> None:
        acquired = PDFIUM_LOCK.acquire(timeout=1.0)
        taken.append(acquired)
        if acquired:
            PDFIUM_LOCK.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return taken == [True]


class TestTimeBudget:
    def test_hostile_page_stops_within_the_budget_and_the_renderer_stays_usable(self) -> None:
        limits = RenderLimits(timeout_s=TINY_BUDGET_S)

        started = time.perf_counter()
        with pytest.raises(InvalidInput) as caught:
            _render(HOSTILE, 200, limits)
        elapsed = time.perf_counter() - started
        print(f"hostile page stopped after {elapsed * 1000:.1f} ms (budget {TINY_BUDGET_S} s)")

        assert caught.value.code == "pdf_unreadable"
        assert caught.value.message == "page 0 exceeded the render budget of 0.05 s"
        assert elapsed < TINY_BUDGET_S + STEP_SLACK_S
        assert _lock_is_free()
        # Another document renders afterwards, with the same limits.
        (page,) = _render(build_pdf([PageSpec(size=A4, lines=ENGLISH_LINES)]), 72, limits)
        assert decode(page.png).min() < 128

    def test_the_failing_page_is_named(self) -> None:
        # A blank page, then the hostile content (object 5, after the two page kids).
        content = zlib.compress(b"0 0 m 200 200 l S\n" * 100_000, 1)
        stream = b"<< /Length %d /Filter /FlateDecode >>\nstream\n%s\nendstream" % (
            len(content),
            content,
        )
        pdf = page_tree_pdf(
            [PAGE_DICT, PAGE_DICT.replace(b" >>", b" /Contents 5 0 R >>")], extra=[stream]
        )

        with pytest.raises(InvalidInput) as caught:
            _render(pdf, 200, RenderLimits(timeout_s=TINY_BUDGET_S))

        assert caught.value.message == "page 1 exceeded the render budget of 0.05 s"

    def test_the_budget_is_measured_on_the_injected_clock(self) -> None:
        # Every reading advances the clock by the whole budget: the first pause poll already
        # finds it spent, whatever the page holds.
        clock = _SteppingClock(step=10.0)
        pdf = image_only_pdf([PageSpec(size=A4, lines=ENGLISH_LINES)], dpi=72)

        with pytest.raises(InvalidInput) as caught:
            _render(pdf, 72, RenderLimits(timeout_s=10.0, clock=clock))

        assert caught.value.code == "pdf_unreadable"
        assert caught.value.message == "page 0 exceeded the render budget of 10 s"
        assert clock.readings >= 3  # the start, at least one poll, the check after the pause

    def test_a_page_within_the_budget_renders_with_a_ticking_clock(self) -> None:
        clock = _SteppingClock(step=0.001)
        pdf = image_only_pdf([PageSpec(size=A4, lines=ENGLISH_LINES)], dpi=72)

        (page,) = _render(pdf, 72, RenderLimits(timeout_s=10.0, clock=clock))

        assert np.array_equal(decode(page.png), _unbounded_render(pdf, 0, 72))
        assert clock.readings >= 2  # the start, then at least one pause poll while drawing

    def test_a_render_pdfium_reports_as_failed_is_unreadable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            pdfium_c, "FPDF_RenderPageBitmap_Start", lambda *args: pdfium_c.FPDF_RENDER_FAILED
        )

        with pytest.raises(InvalidInput) as caught:
            _render(build_pdf([PageSpec()]), 72)

        assert caught.value.code == "pdf_unreadable"
        assert caught.value.message == "page 0 of the PDF cannot be rendered"
        assert isinstance(caught.value.__cause__, pdfium.PdfiumError)

    def test_the_engine_passes_its_render_timeout(self) -> None:
        engine = TesseractOcrEngine(
            TesseractOcrConfig(dpi=200, render_timeout_s=TINY_BUDGET_S), runner=FakeTesseract()
        )

        with pytest.raises(InvalidInput) as caught:
            engine.recognize(HOSTILE, "en")

        assert caught.value.code == "pdf_unreadable"
        assert "page 0 exceeded the render budget" in caught.value.message


@pytest.mark.parametrize(
    ("pdf", "dpi"),
    [
        *(pytest.param(path.read_bytes(), 200, id=path.name) for path in SAMPLES),
        *(
            pytest.param(
                build_pdf([PageSpec(size=A4, rotation=rotation)]), 72, id=f"rotate-{rotation}"
            )
            for rotation in (0, 90, 180, 270)
        ),
        *(
            pytest.param(cropped_pdf(rotation), 72, id=f"cropbox-rotate-{rotation}")
            for rotation in (0, 90, 180, 270)
        ),
        *(
            pytest.param(inherited_boxes_pdf(rotation), 72, id=f"inherited-boxes-rotate-{rotation}")
            for rotation in (0, 90, 180, 270, -90, 450)
        ),
        pytest.param(image_only_pdf([PageSpec(size=A4, lines=ENGLISH_LINES)]), 150, id="scan"),
    ],
)
def test_bounded_render_equals_the_unbounded_render_pixel_for_pixel(pdf: bytes, dpi: int) -> None:
    assert SAMPLES, "the sample scans must be present"

    rendered = _render(pdf, dpi)

    assert rendered
    for page in rendered:
        expected = _unbounded_render(pdf, page.index, dpi)
        assert decode(page.png).shape == expected.shape
        assert np.array_equal(decode(page.png), expected)


def _two_images_pdf() -> bytes:
    """One A4 page holding two copies of the same 595 x 842 scan image."""
    with pymupdf.open(stream=image_only_pdf([PageSpec(size=A4)], dpi=72), filetype="pdf") as src:
        pixmap = src[0].get_pixmap(dpi=72, colorspace=pymupdf.csGRAY, alpha=False)
        with pymupdf.open() as out:
            page = out.new_page(width=A4[0], height=A4[1])
            page.insert_image(pymupdf.Rect(0, 0, 300, 400), pixmap=pixmap)
            page.insert_image(pymupdf.Rect(300, 400, 595, 842), pixmap=pixmap)
            data: bytes = out.tobytes()
            return data


def _image_in_form_xobject_pdf() -> bytes:
    """One page that shows the scan page through a form XObject (``show_pdf_page``)."""
    with (
        pymupdf.open(stream=image_only_pdf([PageSpec(size=A4)], dpi=72), filetype="pdf") as src,
        pymupdf.open() as out,
    ):
        page = out.new_page(width=A4[0], height=A4[1])
        page.show_pdf_page(page.rect, src, 0)
        data: bytes = out.tobytes()
        return data


def _declared_pixels(pdf: bytes) -> int:
    """The declared size of the images of page 0, through PyMuPDF (the reference)."""
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        return sum(int(info[2]) * int(info[3]) for info in document[0].get_images(full=True))


class TestImagePreCheck:
    def test_a_page_declaring_more_pixels_than_the_limit_is_refused_without_rendering(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pdf = image_only_pdf([PageSpec(size=A4, lines=ENGLISH_LINES)], dpi=72)
        declared = _declared_pixels(pdf)
        assert declared == 595 * 842

        def no_rendering(*args: object) -> None:
            raise AssertionError("rendered despite the image pre-check")

        monkeypatch.setattr(pdfium_c, "FPDF_RenderPageBitmap_Start", no_rendering)
        with pytest.raises(InvalidInput) as caught:
            _render(pdf, 72, RenderLimits(max_image_pixels=declared - 1))

        assert caught.value.code == "too_large"
        assert caught.value.message == f"page 0 holds more than {declared - 1} decoded image pixels"

    def test_a_page_declaring_exactly_the_limit_renders(self) -> None:
        pdf = image_only_pdf([PageSpec(size=A4, lines=ENGLISH_LINES)], dpi=72)

        (page,) = _render(pdf, 72, RenderLimits(max_image_pixels=_declared_pixels(pdf)))

        assert decode(page.png).min() < 128

    def test_the_images_of_a_page_are_summed(self) -> None:
        pdf = _two_images_pdf()
        one_image = 595 * 842

        (page,) = _render(pdf, 72, RenderLimits(max_image_pixels=2 * one_image))
        assert decode(page.png).shape == (842, 595)

        with pytest.raises(InvalidInput) as caught:
            _render(pdf, 72, RenderLimits(max_image_pixels=2 * one_image - 1))
        assert caught.value.code == "too_large"

    def test_images_inside_a_form_xobject_are_counted(self) -> None:
        pdf = _image_in_form_xobject_pdf()

        with pytest.raises(InvalidInput) as caught:
            _render(pdf, 72, RenderLimits(max_image_pixels=595 * 842 - 1))

        assert caught.value.code == "too_large"
        (page,) = _render(pdf, 72, RenderLimits(max_image_pixels=595 * 842))
        assert page.raster.width_px == 595

    def test_the_pre_check_is_a_lookup_not_a_decode(self) -> None:
        # A 20,000 x 20,000 image (400 M declared pixels) in a tiny file: refused in
        # milliseconds, far below what decoding it would take.
        pdf = _huge_declared_image_pdf()

        started = time.perf_counter()
        with pytest.raises(InvalidInput) as caught:
            _render(pdf, 72)
        elapsed = time.perf_counter() - started

        assert caught.value.code == "too_large"
        assert caught.value.message == "page 0 holds more than 80000000 decoded image pixels"
        assert elapsed < 0.5

    def test_a_form_nested_at_the_poll_depth_is_walked_and_rendered(self) -> None:
        pdf = hostile.nested_forms_pdf(depth=MAX_FORM_DEPTH, strokes=3)

        (page,) = _render(pdf, 72)

        assert decode(page.png).min() < 128  # the strokes inside the deepest form are drawn

    def test_a_form_nested_beyond_the_poll_depth_is_refused_before_drawing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pdfium_c, "FPDF_RenderPageBitmap_Start", self._no_drawing)
        pdf = hostile.nested_forms_pdf(depth=MAX_FORM_DEPTH + 1, strokes=3)

        with pytest.raises(InvalidInput) as caught:
            _render(pdf, 72)

        assert caught.value.code == "pdf_unreadable"
        assert caught.value.message == "page 0 nests form XObjects deeper than 8 levels"

    @staticmethod
    def _no_drawing(*args: object) -> None:
        raise AssertionError("rendered despite the pre-check")

    def test_the_engine_passes_its_image_limit(self) -> None:
        pdf = image_only_pdf([PageSpec(size=A4, lines=ENGLISH_LINES)], dpi=72)
        engine = TesseractOcrEngine(
            TesseractOcrConfig(dpi=72, max_image_pixels=1_000), runner=FakeTesseract()
        )

        with pytest.raises(InvalidInput) as caught:
            engine.recognize(pdf, "en")

        assert caught.value.code == "too_large"
        assert caught.value.message == "page 0 holds more than 1000 decoded image pixels"


def _huge_declared_image_pdf() -> bytes:
    """A page drawing one image XObject whose dictionary declares 20,000 x 20,000 pixels."""
    image = (
        b"<< /Type /XObject /Subtype /Image /Width 20000 /Height 20000 /ColorSpace /DeviceGray"
        b" /BitsPerComponent 8 /Length 1 >>\nstream\n\x00\nendstream"
    )
    content = b"q 200 0 0 200 0 0 cm /Im0 Do Q"
    stream = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content)
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200]"
        b" /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
    )
    return page_tree_pdf([page], extra=[image, stream])


def test_the_default_limits_are_the_engine_defaults() -> None:
    config = TesseractOcrConfig()

    assert DEFAULT_RENDER_LIMITS.timeout_s == config.render_timeout_s
    assert DEFAULT_RENDER_LIMITS.max_image_pixels == config.max_image_pixels
    assert DEFAULT_RENDER_LIMITS.clock is time.monotonic


def _image_in_annotation_pdf(side: int = 20000) -> bytes:
    """A page with no content whose stamp annotation's appearance stream draws one image
    XObject declaring ``side`` x ``side`` pixels (a few hundred kilobytes of flate data)."""
    raw = zlib.compress(b"\x80" * (side * side), 9)
    image = (
        b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray"
        b" /BitsPerComponent 8 /Filter /FlateDecode /Length %d >>\nstream\n"
        % (side, side, len(raw))
        + raw
        + b"\nendstream"
    )
    drawing = b"q 200 0 0 200 0 0 cm /Im0 Do Q\n"
    appearance = (
        b"<< /Type /XObject /Subtype /Form /BBox [0 0 200 200]"
        b" /Resources << /XObject << /Im0 4 0 R >> >> /Length %d >>\nstream\n"
        % len(drawing)
        + drawing
        + b"\nendstream"
    )
    annotation = b"<< /Type /Annot /Subtype /Stamp /Rect [0 0 200 200] /F 4 /AP << /N 5 0 R >> >>"
    page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Annots [6 0 R] >>"
    return page_tree_pdf([page], extra=[image, appearance, annotation])


class TestAnnotationImages:
    """Review finding: ``FPDF_ANNOT`` draws annotation appearance streams, whose images the
    pre-check did not count, so a stamp annotation carried an image of any size past
    ``max_image_pixels``."""

    def test_an_image_in_an_annotation_appearance_is_counted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pdfium_c, "FPDF_RenderPageBitmap_Start", TestImagePreCheck._no_drawing)

        with pytest.raises(InvalidInput) as caught:
            _render(_image_in_annotation_pdf(), 72, RenderLimits(max_image_pixels=1_000_000))

        assert caught.value.code == "too_large"
        assert caught.value.message == "page 0 holds more than 1000000 decoded image pixels"

    def test_an_annotation_image_within_the_limit_is_drawn(self) -> None:
        (page,) = _render(_image_in_annotation_pdf(side=100), 72)

        assert decode(page.png).min() < 255  # the stamp's gray image is on the page
