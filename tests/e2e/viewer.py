"""Driving the audit viewer in headless Chromium (Playwright, sync API) for tests/e2e.

Imported only by the e2e test modules after ``pytest.importorskip("playwright")`` and by the
benchmark scripts, so the host suite never imports Playwright. A ``Session`` is one browser
context with one page of the viewer; it records every request, console error, page error,
dialog and CSP violation (a ``securitypolicyviolation`` listener added before any page script
runs), and ``Session.check`` fails on anything the page must not do (spec: audit-viewer, the
page is safe with untrusted documents).
"""

import base64
import io
import json
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from playwright.sync_api import (
    Browser,
    BrowserContext,
    ConsoleMessage,
    Dialog,
    Page,
    Response,
    Route,
)

from tests.e2e.ink import Gray

VIEWPORT = {"width": 1280, "height": 900}
EXTRACT_TIMEOUT_MS = 300_000
RENDER_TIMEOUT_MS = 60_000
PIXEL_TOLERANCE = 1.0
"""Spec: every box element at the response's points x scale within 1 CSS pixel."""

CSP_LISTENER = """
window.__cspViolations = [];
document.addEventListener("securitypolicyviolation", (event) => {
  window.__cspViolations.push({
    directive: event.violatedDirective,
    blocked: event.blockedURI,
    source: event.sourceFile,
    line: event.lineNumber,
  });
});
"""

# Every request the page may make, relative to the app's origin (spec: no other requests).
ALLOWED_REQUESTS = (
    ("GET", re.compile(r"/audit")),
    ("GET", re.compile(r"/audit/static/[\w.-]+")),
    ("GET", re.compile(r"/audit/vendor/pdfjs/[\w.-]+")),
    ("POST", re.compile(r"/api/extract")),
)
# Chromium logs this console error for every HTTP error status (or network failure) a fetch
# receives.
HTTP_ERROR_CONSOLE = re.compile(r"^Failed to load resource: ")

BOX_RECTS = """
() => [...document.querySelectorAll(".box")].map((box) => {
  const page = box.closest(".page");
  const overlay = page.querySelector(".page-overlay");
  const p = page.getBoundingClientRect();
  const o = overlay.getBoundingClientRect();
  const r = box.getBoundingClientRect();
  return {
    id: box.id,
    page: Number(page.dataset.pageIndex),
    scale: Number(page.dataset.scale),
    left: r.left - p.left,
    top: r.top - p.top,
    width: r.width,
    height: r.height,
    overlayLeft: o.left - p.left,
    overlayTop: o.top - p.top,
    hidden: box.hidden,
  };
})
"""

PAGE_CANVAS = """
(index) => {
  const page = document.querySelector(`.page[data-page-index="${index}"]`);
  const canvas = page.querySelector("canvas.page-canvas");
  const rect = page.getBoundingClientRect();
  const drawn = canvas.getBoundingClientRect();
  return {
    png: canvas.toDataURL("image/png"),
    width: canvas.width,
    height: canvas.height,
    cssWidth: rect.width,
    cssHeight: rect.height,
    offsets: [
      drawn.left - rect.left,
      drawn.top - rect.top,
      drawn.width - rect.width,
      drawn.height - rect.height,
    ],
    scale: Number(page.dataset.scale),
    dpr: window.devicePixelRatio,
  };
}
"""

PAGE_WARNINGS = """
() => [...document.querySelectorAll("#page-warnings li")].map((li) => ({
  page: Number(li.dataset.page),
  kind: li.dataset.kind,
  text: li.textContent,
}))
"""

# Records every change of the text of each live region (role status or alert, aria-live), by
# region id, with consecutive repeats dropped: what a screen reader would be asked to read.
WATCH_ANNOUNCEMENTS = """
() => {
  window.__announced = [];
  for (const region of document.querySelectorAll('[aria-live], [role="status"], [role="alert"]')) {
    const record = () => {
      const text = region.hidden ? "" : region.textContent;
      const last = window.__announced.findLast((entry) => entry.id === region.id);
      if (last ? last.text !== text : text !== "") window.__announced.push({ id: region.id, text });
    };
    new MutationObserver(record).observe(region, {
      childList: true,
      characterData: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["hidden"],
    });
  }
}
"""

LIVE_CANVASES = """
() => [...document.querySelectorAll("canvas.page-canvas")].filter((c) => c.width > 0).length
"""

PAGE_RENDERED = """
(index) => {
  const page = document.querySelector(`.page[data-page-index="${index}"]`);
  return page !== null && page.dataset.rendered === "true";
}
"""


def origin_of(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


@dataclass
class Canvas:
    """A page's rendered canvas as grey pixels, with its CSS size and viewport scale."""

    gray: Gray
    css_width: float
    css_height: float
    scale: float
    dpr: float
    offsets: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """The canvas element's rectangle minus its page element's (left, top, width, height)."""

    def misfit(self) -> float:
        """How far, in CSS pixels, the drawn canvas is from covering its page exactly."""
        return max(abs(value) for value in self.offsets)

    def to_pixels(self, box: dict[str, Any]) -> tuple[float, float, float, float]:
        """A response box (points) in pixels of this canvas."""
        kx = self.gray.shape[1] / self.css_width * self.scale
        ky = self.gray.shape[0] / self.css_height * self.scale
        return (box["x"] * kx, box["y"] * ky, box["width"] * kx, box["height"] * ky)


@dataclass
class Session:
    """One browser context with the viewer open, and everything it was seen doing."""

    context: BrowserContext
    url: str
    page: Page = field(init=False)
    requests: list[tuple[str, str]] = field(default_factory=list)
    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    dialogs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.context.add_init_script(CSP_LISTENER)
        self.context.on(
            "request", lambda request: self.requests.append((request.method, request.url))
        )
        self.page = self.context.new_page()
        self.page.on("console", self._console)
        self.page.on("pageerror", lambda error: self.page_errors.append(str(error)))
        self.page.on("dialog", self._dialog)
        self.page.goto(self.url, wait_until="domcontentloaded")
        self.page.evaluate("() => window.__auditViewer.ready")
        state = self.page.get_attribute("body", "data-state")
        assert state == "idle", f"the viewer did not start (state {state}): pdf.js missing?"

    @property
    def origin(self) -> str:
        return origin_of(self.url)

    def _console(self, message: ConsoleMessage) -> None:
        if message.type == "error":
            self.console_errors.append(message.text)

    def _dialog(self, dialog: Dialog) -> None:
        self.dialogs.append(f"{dialog.type}: {dialog.message}")
        dialog.dismiss()

    # ---------- extraction ----------

    def choose(
        self,
        data: bytes,
        names: str = "",
        language: str | None = None,
        filename: str = "document.pdf",
        mime: str = "application/pdf",
    ) -> None:
        self.page.set_input_files(
            "#pdf-file", files=[{"name": filename, "mimeType": mime, "buffer": data}]
        )
        self.page.fill("#names", names)
        self.page.select_option("#language", language or "auto")

    def extract(
        self, data: bytes, names: str = "", language: str | None = None, **kwargs: str
    ) -> Response:
        """Choose ``data`` and extract; return the ``/api/extract`` response once drawn."""
        self.choose(data, names, language, **kwargs)
        with self.page.expect_response(_is_extract, timeout=EXTRACT_TIMEOUT_MS) as info:
            self.page.click("#extract")
        self.page.wait_for_selector(
            "body[data-state=done], body[data-state=error]", timeout=EXTRACT_TIMEOUT_MS
        )
        return info.value

    # ---------- view ----------

    def zoom(self, value: str) -> None:
        self.page.select_option("#zoom", value)

    def filter(self, value: str) -> None:
        self.page.select_option("#filter", value)

    def box_rects(self) -> list[dict[str, Any]]:
        rects: list[dict[str, Any]] = self.page.evaluate(BOX_RECTS)
        return rects

    def page_count(self) -> int:
        count: int = self.page.locator(".page").count()
        return count

    def render(self, index: int) -> Canvas:
        """Scroll page ``index`` into view, wait for its rendering and read its canvas."""
        self.page.evaluate(
            '(i) => document.querySelector(`.page[data-page-index="${i}"]`)'
            ".scrollIntoView({ block: 'start' })",
            index,
        )
        self.page.wait_for_function(PAGE_RENDERED, arg=index, timeout=RENDER_TIMEOUT_MS)
        data = self.page.evaluate(PAGE_CANVAS, index)
        png = base64.b64decode(data["png"].split(",", 1)[1])
        image = Image.open(io.BytesIO(png)).convert("L")
        gray = np.asarray(image, dtype=np.uint8)
        assert gray.shape == (data["height"], data["width"])
        offsets = tuple(float(value) for value in data["offsets"])
        return Canvas(
            gray,
            data["cssWidth"],
            data["cssHeight"],
            data["scale"],
            data["dpr"],
            (offsets[0], offsets[1], offsets[2], offsets[3]),
        )

    def page_warnings(self) -> list[dict[str, Any]]:
        """The page audit warnings the page lists: ``{page, kind, text}``."""
        warnings: list[dict[str, Any]] = self.page.evaluate(PAGE_WARNINGS)
        return warnings

    def live_canvases(self) -> int:
        """How many page canvases hold pixels (a released page's canvas is 0 x 0)."""
        count: int = self.page.evaluate(LIVE_CANVASES)
        return count

    def watch_announcements(self) -> None:
        self.page.evaluate(WATCH_ANNOUNCEMENTS)

    def announcements(self, region: str) -> list[str]:
        """The texts live region ``region`` (an element id) took since ``watch_announcements``."""
        entries: list[dict[str, str]] = self.page.evaluate("() => window.__announced")
        return [entry["text"] for entry in entries if entry["id"] == region]

    def hold_extract(self) -> "HeldRoute":
        return HeldRoute(self.page)

    def screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(path), full_page=False)

    # ---------- safety ----------

    def csp_violations(self) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = self.page.evaluate("() => window.__cspViolations ?? []")
        return violations

    def foreign_requests(self) -> list[tuple[str, str]]:
        """Requests other than the page, its own files and ``POST /api/extract``."""
        foreign = []
        for method, url in self.requests:
            if url.startswith(("blob:", "data:")):
                continue  # in-memory URLs: the downloaded JSON, pdf.js's own data
            parts = urllib.parse.urlsplit(url)
            same_origin = f"{parts.scheme}://{parts.netloc}" == self.origin
            path = parts.path
            allowed = any(m == method and rule.fullmatch(path) for m, rule in ALLOWED_REQUESTS)
            if not (same_origin and allowed and not parts.query):
                foreign.append((method, url))
        return foreign

    def check(self, http_errors: bool = False) -> None:
        """Fail on a CSP violation, a foreign request, a dialog, a page error or console error.

        ``http_errors``: the test expects an HTTP error status from the API, which Chromium
        reports as a console error; only that message is then allowed.
        """
        console = [
            text
            for text in self.console_errors
            if not (http_errors and HTTP_ERROR_CONSOLE.match(text))
        ]
        assert self.csp_violations() == []
        assert self.foreign_requests() == []
        assert self.dialogs == []
        assert self.page_errors == []
        assert console == []

    def close(self) -> None:
        self.context.close()


class HeldRoute:
    """Holds every ``/api/extract`` request unanswered until the test fulfils it (a slow API)."""

    def __init__(self, page: Page) -> None:
        self.page = page
        self.routes: list[Route] = []
        page.route("**/api/extract", self._hold)

    def _hold(self, route: Route) -> None:
        self.routes.append(route)

    def wait(self, timeout_ms: int = 10_000) -> Route:
        """The first held request, once it has arrived."""
        waited = 0
        while not self.routes:
            assert waited < timeout_ms, "no /api/extract request was made"
            self.page.wait_for_timeout(50)
            waited += 50
        return self.routes[0]

    def fulfil(self, status: int, body: str, request_id: str | None = None) -> None:
        headers = {"X-Request-ID": request_id} if request_id else {}
        route = self.routes.pop(0)
        route.fulfill(status=status, body=body, content_type="application/json", headers=headers)

    def release(self) -> None:
        """Stop holding: later requests reach the service."""
        self.page.unroute("**/api/extract")


def _is_extract(response: Response) -> bool:
    method: str = response.request.method
    return method == "POST" and urllib.parse.urlsplit(response.url).path == "/api/extract"


def open_session(
    browser: Browser, url: str, dpr: float = 1, viewport: dict[str, int] | None = None
) -> Session:
    context = browser.new_context(
        viewport=viewport or VIEWPORT, device_scale_factor=dpr, accept_downloads=True
    )
    return Session(context, url)


def response_json(response: Response) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(response.body())
    return body


def position_errors(rects: list[dict[str, Any]], body: dict[str, Any]) -> list[str]:
    """Boxes whose element is not at the response's box x scale within ``PIXEL_TOLERANCE``.

    The expected rectangle is computed from the RESPONSE (``body``), not from the element's
    data attributes; ``rects`` come from ``Session.box_rects`` (relative to the page element).
    """
    extracted = body["extracted_names"]
    errors = []
    by_id = {rect["id"]: rect for rect in rects}
    for index, entry in enumerate(extracted):
        rect = by_id.get(f"box-{index}")
        box = entry["bounding_box"]
        if rect is None:
            errors.append(f"box-{index} ({entry['name']}): no element")
            continue
        scale = rect["scale"]
        expected = (box["x"] * scale, box["y"] * scale, box["width"] * scale, box["height"] * scale)
        actual = (rect["left"], rect["top"], rect["width"], rect["height"])
        off = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
        overlay_off = max(abs(rect["overlayLeft"]), abs(rect["overlayTop"]))
        if (
            rect["page"] != box["page_number"]
            or off > PIXEL_TOLERANCE
            or overlay_off > PIXEL_TOLERANCE
        ):
            errors.append(
                f"box-{index} ({entry['name']}): page {rect['page']} vs {box['page_number']}, "
                f"rect {actual} vs {expected} (scale {scale}), overlay offset {overlay_off}"
            )
    if len(rects) != len(extracted):
        errors.append(f"{len(rects)} box elements for {len(extracted)} extracted names")
    return errors
