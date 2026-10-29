"""Boxes in the displayed frame of rotated, cropped and mixed-size pages (spec: audit-viewer).

The PDF (``tests/e2e/frames.py``) has upright displayed text on pages with ``/Rotate`` 0, 90,
180 and 270, a crop box offset from its media box (alone and with ``/Rotate 90``), A4,
Letter and A5-landscape pages in one file, a ``/UserUnit 2`` page and pages scanned as JPEG
2000 and CCITT G4. It goes through the real service, and the viewer draws the response over
pdf.js's rendering: every expected name must be found with a box on its glyphs, every box
element must sit at the response's points x scale within 1 px at two zoom levels, every
rendered canvas must cover its page element within 1 px, and every box must pass the ink check
on the rendered canvas, also at a device pixel ratio of 2. The ``/UserUnit`` page is drawn in
the service's frame (user units) with a page audit warning; a page whose ``/Rotate`` is a
string gets one too.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("playwright.sync_api", reason="Playwright is installed only in the e2e image")

from tests.e2e import frames, ink
from tests.e2e.viewer import PIXEL_TOLERANCE, Session, position_errors, response_json

pytestmark = pytest.mark.e2e

LISTED = frames.LISTED_NAMES
NAMES_TEXT = frames.NAMES_TEXT
MIN_IOU = 0.5
USER_UNIT_PAGE = next(i for i, spec in enumerate(frames.PAGES) if spec.user_unit != 1)
USER_UNIT_WARNING = "page uses /UserUnit 2: the service's boxes are in user units, not points"
CODEC_PAGES = {spec.codec: i for i, spec in enumerate(frames.PAGES) if spec.codec != "jpeg"}
FALLBACK_DECODERS = {
    "jpx": "/audit/vendor/pdfjs/openjpeg_nowasm_fallback.js",
    "ccitt": "/audit/vendor/pdfjs/jbig2_nowasm_fallback.js",
}


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    overlap = width * height
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def _box(entry: dict[str, Any]) -> tuple[float, float, float, float]:
    box = entry["bounding_box"]
    return (box["x"], box["y"], box["width"], box["height"])


@pytest.fixture(scope="module")
def audited(
    open_viewer: Callable[..., Session], frame_document: frames.FrameDocument, artifacts: Path
) -> tuple[Session, dict[str, Any]]:
    session = open_viewer()
    response = session.extract(frame_document.pdf, NAMES_TEXT, "en", filename="frames.pdf")
    assert response.status == 200, response.text()
    body = response_json(response)
    (artifacts / "frames_response.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    return session, body


def ink_failures(
    session: Session, body: dict[str, Any], label: str, artifacts: Path | None = None
) -> list[str]:
    """Boxes that fail the ink check on the canvas pdf.js rendered, page by page, and pages
    whose rendered canvas does not cover the page element within ``PIXEL_TOLERANCE``.

    With ``artifacts``, every box's measure is written to ``ink_<label>.json`` there.
    """
    failures = []
    measures = []
    pages = sorted({entry["bounding_box"]["page_number"] for entry in body["extracted_names"]})
    for index in pages:
        canvas = session.render(index)
        if canvas.misfit() > PIXEL_TOLERANCE:
            failures.append(f"{label} page {index}: canvas offsets {canvas.offsets} px")
        for number, entry in enumerate(body["extracted_names"]):
            box = entry["bounding_box"]
            if box["page_number"] != index:
                continue
            measured = ink.measure(canvas.gray, canvas.to_pixels(box))
            measures.append(
                {"box": number, "name": entry["name"], "page": index, "inside": measured.inside}
                | {"ring": measured.ring, "ratio": min(measured.ratio, 1e9)}
            )
            if not measured.passes():
                failures.append(
                    f"{label} box-{number} {entry['name']!r} page {index}: inside "
                    f"{measured.inside:.3f}, ring {measured.ring:.3f}, ratio {measured.ratio:.2f}"
                )
    if artifacts is not None:
        slug = re.sub(r"[^A-Za-z0-9.-]+", "_", label)
        path = artifacts / f"ink_{slug}.json"
        path.write_text(json.dumps(measures, indent=1), encoding="utf-8")
    return failures


def test_every_expected_name_is_found_on_its_glyphs(
    audited: tuple[Session, dict[str, Any]], frame_document: frames.FrameDocument
) -> None:
    _, body = audited
    found = [
        (entry["name"], entry["bounding_box"]["page_number"]) for entry in body["extracted_names"]
    ]

    assert sorted(found) == sorted((name.name, name.page) for name in frame_document.names)
    by_name = {entry["name"]: entry for entry in body["extracted_names"]}
    for expected in frame_document.names:
        drawn = (expected.x, expected.y, expected.width, expected.height)
        assert _iou(_box(by_name[expected.name]), drawn) >= MIN_IOU, expected
    assert not set(frame_document.hidden_names) & set(by_name)


def test_pages_have_the_displayed_sizes(
    audited: tuple[Session, dict[str, Any]], frame_document: frames.FrameDocument
) -> None:
    session, _ = audited
    pages = session.page.locator(".page")

    sizes = [
        (
            float(pages.nth(i).get_attribute("data-width-pt") or 0),
            float(pages.nth(i).get_attribute("data-height-pt") or 0),
        )
        for i in range(pages.count())
    ]

    assert len(sizes) == len(frame_document.displayed_sizes)
    for actual, expected in zip(sizes, frame_document.displayed_sizes, strict=True):
        assert actual == pytest.approx(expected, abs=0.5)
    session.check()


def test_matches_are_marked(audited: tuple[Session, dict[str, Any]]) -> None:
    session, body = audited
    matched = {match["extracted_name"] for match in body["fuzzy_matches"]}

    assert matched == set(LISTED)
    for number, entry in enumerate(body["extracted_names"]):
        box = session.page.locator(f"#box-{number}")
        assert box.get_attribute("data-matched") == str(entry["name"] in matched).lower()


@pytest.mark.parametrize("zoom", ["fit", "2"])
def test_boxes_sit_at_points_times_scale(
    audited: tuple[Session, dict[str, Any]], zoom: str
) -> None:
    session, body = audited
    session.zoom(zoom)

    rects = session.box_rects()

    assert position_errors(rects, body) == []
    scales = {rect["scale"] for rect in rects}
    if zoom == "2":
        assert scales == {2.0}
    else:
        widest = max(
            float(session.page.locator(".page").nth(i).get_attribute("data-width-pt") or 0)
            for i in range(session.page_count())
        )
        available = session.page.evaluate(
            "() => { const p = document.getElementById('pages'); const s = getComputedStyle(p);"
            " return p.clientWidth - parseFloat(s.paddingLeft) - parseFloat(s.paddingRight); }"
        )
        assert len(scales) == 1
        assert scales.pop() * widest == pytest.approx(available, abs=1)
    session.check()


def test_every_box_passes_the_ink_check(
    audited: tuple[Session, dict[str, Any]], artifacts: Path
) -> None:
    session, body = audited
    session.zoom("2")

    assert ink_failures(session, body, "zoom 2", artifacts) == []
    session.check()


def test_no_box_is_reported_outside_its_page(audited: tuple[Session, dict[str, Any]]) -> None:
    session, _ = audited

    assert session.page.locator("li[data-warning]").count() == 0
    assert session.page.locator(".box[data-warning]").count() == 0


def test_only_the_user_unit_page_has_a_page_warning(
    audited: tuple[Session, dict[str, Any]], frame_document: frames.FrameDocument
) -> None:
    """pdf.js applies /UserUnit 2, the service does not: the page is drawn in the service's
    frame (user units, the size PDFium reports) and flagged; numeric /Rotate pages are not."""
    session, body = audited
    page = session.page

    where = f"Page {USER_UNIT_PAGE + 1} (page_number {USER_UNIT_PAGE})"
    assert session.page_warnings() == [
        {"page": USER_UNIT_PAGE, "kind": "user-unit", "text": f"{where}: {USER_UNIT_WARNING}"}
    ]
    assert page.locator(".page-frame[data-page-warning]").count() == 1
    frame = page.locator(".page-frame").nth(USER_UNIT_PAGE)
    assert frame.get_attribute("data-page-warning") == "user-unit"
    assert USER_UNIT_WARNING in (frame.locator("figcaption").text_content() or "")
    view = page.locator(f'.page[data-page-index="{USER_UNIT_PAGE}"]')
    assert view.get_attribute("data-user-unit") == "2"
    width, height = frame_document.displayed_sizes[USER_UNIT_PAGE]
    assert float(view.get_attribute("data-width-pt") or 0) == pytest.approx(width, abs=0.5)
    assert float(view.get_attribute("data-height-pt") or 0) == pytest.approx(height, abs=0.5)
    assert "1 page with an audit warning" in (page.text_content("#status") or "")
    number = next(
        i
        for i, entry in enumerate(body["extracted_names"])
        if entry["bounding_box"]["page_number"] == USER_UNIT_PAGE
    )
    label = page.get_attribute(f"#box-{number}", "aria-label") or ""
    assert label.endswith(f"page audit warning: {USER_UNIT_WARNING}")


def test_the_jpx_and_ccitt_pages_use_the_fallback_decoders(
    audited: tuple[Session, dict[str, Any]],
) -> None:
    session, _ = audited
    session.zoom("1")

    for codec, index in CODEC_PAGES.items():
        canvas = session.render(index)
        assert (canvas.gray < 128).mean() > 0.002, codec  # the scanned text is drawn
        assert FALLBACK_DECODERS[codec] in {
            url.removeprefix(session.origin) for _, url in session.requests
        }, codec
    session.check()


def test_a_rotate_that_is_not_a_number_is_flagged(open_viewer: Callable[..., Session]) -> None:
    """``/Rotate (90)``: PDFium reads 0, pdf.js turns the page; neither exposes the raw value,
    but pdf.js passes the string through, which the viewer reports."""
    document = frames.build((frames.STRING_ROTATE_PAGE,))
    session = open_viewer()
    canned = json.dumps(frames.canned_response(document))
    session.page.route(
        "**/api/extract",
        lambda route: route.fulfill(status=200, body=canned, content_type="application/json"),
    )

    session.extract(document.pdf, filename="string_rotate.pdf")

    (warning,) = session.page_warnings()
    assert (warning["page"], warning["kind"]) == (0, "rotate")
    assert 'page /Rotate is "90", not a number multiple of 90' in warning["text"]
    session.check()


def test_device_pixel_ratio_2(
    open_viewer: Callable[..., Session], frame_document: frames.FrameDocument, artifacts: Path
) -> None:
    session = open_viewer(dpr=2)
    response = session.extract(frame_document.pdf, NAMES_TEXT, "en", filename="frames.pdf")
    assert response.status == 200
    body = response_json(response)

    for zoom in ("fit", "1.5"):
        session.zoom(zoom)
        assert position_errors(session.box_rects(), body) == [], zoom
    canvas = session.render(1)  # /Rotate 90
    assert canvas.dpr == 2
    assert canvas.gray.shape[1] == pytest.approx(2 * canvas.css_width, abs=1)
    assert ink_failures(session, body, "zoom 1.5, dpr 2", artifacts) == []
    session.check()


@pytest.mark.parametrize(
    "sample", ["company_memo.pdf", "meeting_minutes.pdf", "research_report.pdf"]
)
def test_sample_pdfs(
    open_viewer: Callable[..., Session], samples: dict[str, bytes], sample: str, artifacts: Path
) -> None:
    session = open_viewer()
    response = session.extract(samples[sample], "", None, filename=sample)
    assert response.status == 200
    body = response_json(response)
    assert body["extracted_names"]

    for zoom in ("fit", "2"):
        session.zoom(zoom)
        assert position_errors(session.box_rects(), body) == [], zoom
    assert ink_failures(session, body, sample, artifacts) == []
    session.check()
