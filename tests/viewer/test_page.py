"""The page has no inline script or style, which its Content-Security-Policy would block
(design D5, D7). Skipped while ``app/web/index.html`` does not exist yet."""

from html.parser import HTMLParser
from pathlib import PurePosixPath

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.viewer import (
    DEFAULT_PDFJS_DIR,
    PAGE,
    PLAIN_NAME,
    STATIC_TYPES,
    WEB_DIR,
    viewer_router,
)


class InlineFinder(HTMLParser):
    """Collects what ``script-src 'self'`` and ``style-src 'self'`` without ``'unsafe-inline'``
    refuse: a script element with a body or without ``src``, a ``style`` element, a ``style``
    attribute, an event-handler attribute (``onclick``) and a ``javascript:`` URL."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found: list[str] = []
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        names = {name.lower(): value or "" for name, value in attrs}
        line = self.getpos()[0]
        if tag == "script":
            self._in_script = True
            if not names.get("src"):
                self.found.append(f"line {line}: <script> without src")
        if tag == "style":
            self.found.append(f"line {line}: <style> element")
        if "style" in names:
            self.found.append(f"line {line}: style attribute on <{tag}>")
        self.found.extend(
            f"line {line}: {name} attribute on <{tag}>" for name in names if name.startswith("on")
        )
        self.found.extend(
            f"line {line}: javascript: URL in {name}"
            for name, value in names.items()
            if value.strip().lower().startswith("javascript:")
        )

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script and data.strip():
            self.found.append(f"line {self.getpos()[0]}: inline script body")


def inline_code(html: str) -> list[str]:
    finder = InlineFinder()
    finder.feed(html)
    finder.close()
    return finder.found


def test_the_page_has_no_inline_script_or_style() -> None:
    page = WEB_DIR / PAGE
    if not page.is_file():
        pytest.skip("app/web/index.html does not exist yet")

    assert inline_code(page.read_text(encoding="utf-8")) == []


def test_the_finder_reports_every_kind_of_inline_code() -> None:
    html = (
        "<html><head><style>p{}</style>\n"
        '<script>alert(1)</script><script type="module" src="/audit/static/a.mjs"></script>\n'
        '</head><body style="margin:0"><button onclick="go()">x</button>\n'
        '<a href=" JavaScript:go()">y</a><script src="/audit/static/b.js">\n</script>'
        '<script src=""></script></body></html>'
    )

    assert inline_code(html) == [
        "line 1: <style> element",
        "line 2: <script> without src",
        "line 2: inline script body",
        "line 3: style attribute on <body>",
        "line 3: onclick attribute on <button>",
        "line 4: javascript: URL in href",
        "line 5: <script> without src",
    ]


def test_a_clean_page_passes() -> None:
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Audit</title>'
        '<link rel="stylesheet" href="/audit/static/viewer.css"></head><body>'
        '<main id="app"></main><script type="module" src="/audit/static/viewer.mjs"></script>'
        "</body></html>"
    )

    assert inline_code(html) == []


def test_every_file_of_the_page_directory_is_served() -> None:
    """A file the page needs but the routes refuse (subdirectory, other extension) would only
    show as a broken page in the browser."""
    if not WEB_DIR.is_dir():
        pytest.skip("app/web does not exist yet")
    names = sorted(entry.name for entry in WEB_DIR.iterdir() if entry.name != PAGE)

    app = FastAPI()
    app.include_router(viewer_router(DEFAULT_PDFJS_DIR, WEB_DIR))
    with TestClient(app) as client:
        statuses = {name: client.get(f"/audit/static/{name}").status_code for name in names}

    assert names
    assert [entry.name for entry in WEB_DIR.iterdir() if entry.is_dir()] == []
    assert statuses == dict.fromkeys(names, 200)
    assert all(
        PLAIN_NAME.fullmatch(name) and PurePosixPath(name).suffix in STATIC_TYPES for name in names
    )
