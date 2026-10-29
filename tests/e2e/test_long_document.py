"""A long document in the audit viewer (spec: audit-viewer, long documents stay responsive).

The frame PDF's ten pages six times over (60 pages, 120 names) with a canned response at the
names' drawn glyph boxes (route interception: the service accepts at most ``MAX_PAGES`` 50 and
would take a minute). The list and every box exist as soon as the response is drawn, before
any page is rendered; scrolling through every page renders each one; afterwards only a few
canvases hold pixels; and repeated extractions of the same document do not grow the page's DOM
or its event listeners (CDP ``Performance.getMetrics`` after a forced garbage collection).
"""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("playwright.sync_api", reason="Playwright is installed only in the e2e image")

from playwright.sync_api import Route

from tests.e2e import frames
from tests.e2e.viewer import Session, position_errors

pytestmark = pytest.mark.e2e

COPIES = 6
MAX_LIVE_CANVASES = 8
"""Pages rendered near the viewport plus those not yet released (BENCHMARKS 20.4: 4)."""
RUNS = 4
NODE_GROWTH = 50
LISTENER_GROWTH = 5
SCROLL_TIMEOUT_MS = 180_000

RECORD_RENDERED = """
() => {
  window.__renderedPages = new Set(
    [...document.querySelectorAll('.page[data-rendered="true"]')].map((p) => p.dataset.pageIndex)
  );
  new MutationObserver((mutations) => {
    for (const { target } of mutations) {
      if (target.dataset.rendered === "true") window.__renderedPages.add(target.dataset.pageIndex);
    }
  }).observe(document.getElementById("pages"), {
    attributes: true,
    attributeFilter: ["data-rendered"],
    subtree: true,
  });
}
"""


@pytest.fixture(scope="module")
def long_document(frame_document: frames.FrameDocument) -> frames.FrameDocument:
    return frames.repeat(frame_document, COPIES)


def _serve_canned(session: Session, body: dict[str, Any]) -> None:
    text = json.dumps(body)

    def handle(route: Route) -> None:
        route.fulfill(status=200, body=text, content_type="application/json")

    session.page.route("**/api/extract", handle)


def _scroll_all(session: Session) -> None:
    session.page.evaluate(
        "() => window.__auditViewer.scrollAllPages()"
    )  # each page in turn, awaiting its rendering
    session.page.wait_for_function(
        f"() => [...document.querySelectorAll('canvas.page-canvas')]"
        f".filter((c) => c.width > 0).length <= {MAX_LIVE_CANVASES}",
        timeout=SCROLL_TIMEOUT_MS,
    )


def test_sixty_pages_list_at_once_render_every_page_and_keep_few_canvases(
    open_viewer: Callable[..., Session], long_document: frames.FrameDocument, artifacts: Path
) -> None:
    session = open_viewer()
    body = frames.canned_response(long_document)
    _serve_canned(session, body)
    pages = len(long_document.pages)

    session.extract(long_document.pdf, frames.NAMES_TEXT, filename="long.pdf")

    page = session.page
    assert page.get_attribute("body", "data-state") == "done"
    assert page.locator(".page").count() == pages
    assert page.locator(".box").count() == len(body["extracted_names"])
    assert page.locator("#name-list li[data-box-id]").count() == len(body["extracted_names"])
    assert page.locator("#name-list li.list-page").count() == pages
    assert page.locator('.page[data-rendered="true"]').count() <= MAX_LIVE_CANVASES
    assert position_errors(session.box_rects(), body) == []

    page.evaluate(RECORD_RENDERED)
    page.set_default_timeout(SCROLL_TIMEOUT_MS)
    started = time.perf_counter()
    _scroll_all(session)
    scroll_s = time.perf_counter() - started

    rendered = page.evaluate("() => [...window.__renderedPages].map(Number).sort((a, b) => a - b)")
    assert rendered == list(range(pages))
    assert session.live_canvases() <= MAX_LIVE_CANVASES
    assert page.locator('.page[data-rendered="true"]').count() <= MAX_LIVE_CANVASES
    assert f"on {pages} pages" in (page.text_content("#status") or "")
    record = {"pages": pages, "scroll_s": scroll_s, "live_canvases": session.live_canvases()}
    (artifacts / "long_document.json").write_text(json.dumps(record), encoding="utf-8")
    session.check()


def test_repeated_extractions_do_not_grow_the_page(
    open_viewer: Callable[..., Session], long_document: frames.FrameDocument, artifacts: Path
) -> None:
    session = open_viewer()
    _serve_canned(session, frames.canned_response(long_document))
    session.page.set_default_timeout(SCROLL_TIMEOUT_MS)
    cdp = session.context.new_cdp_session(session.page)
    cdp.send("Performance.enable")
    samples: list[dict[str, float]] = []

    for _ in range(RUNS):
        session.extract(long_document.pdf, filename="long.pdf")
        _scroll_all(session)
        cdp.send("HeapProfiler.collectGarbage")
        metrics = cdp.send("Performance.getMetrics")["metrics"]
        samples.append({metric["name"]: metric["value"] for metric in metrics})

    keep = ("Nodes", "JSEventListeners", "JSHeapUsedSize", "Documents", "Frames")
    rows = [{name: sample.get(name) for name in keep} for sample in samples]
    (artifacts / "long_document_runs.json").write_text(json.dumps(rows), encoding="utf-8")
    # The first run loads pdf.js and warms its caches; the later ones must not grow.
    first, *later = samples[1:]
    for sample in later:
        assert sample["Nodes"] <= first["Nodes"] + NODE_GROWTH, samples
        assert sample["JSEventListeners"] <= first["JSEventListeners"] + LISTENER_GROWTH, samples
        assert sample["JSHeapUsedSize"] <= first["JSHeapUsedSize"] * 1.5 + 5e6, samples
    session.check()
