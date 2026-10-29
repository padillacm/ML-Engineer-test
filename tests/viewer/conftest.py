"""Fixtures of the viewer tests: temporary page and pdf.js directories and an application over
fakes that serves them, so the tests depend neither on ``app/web`` nor on a fetched pdf.js."""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import viewer
from app.main import create_app
from tests.api.support import Fakes, api_settings

type ViewerClient = Callable[..., TestClient]

PAGE_HTML = (
    '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><title>Audit</title>\n'
    '<link rel="stylesheet" href="/audit/static/viewer.css"></head>\n'
    '<body><script type="module" src="/audit/static/viewer.mjs"></script></body></html>\n'
)
STATIC_FILES = {
    "viewer.mjs": "export const ready = true;\n",
    "names.js": "export function parse() { return []; }\n",
    "viewer.css": "body { margin: 0; }\n",
    "icon.svg": '<svg xmlns="http://www.w3.org/2000/svg"/>\n',
    # Present but never served: the extension is not allowed.
    "notes.txt": "not served\n",
    "other.html": "<p>not served</p>\n",
    "viewer.mjs.map": "{}\n",
}
PDFJS = {
    "pdf.min.mjs": "export const version = 'test';\n",
    "pdf.worker.min.mjs": "export {};\n",
    "jbig2_nowasm_fallback.js": "export default async function JBig2() {}\n",
    "openjpeg_nowasm_fallback.js": "export default async function OpenJPEG() {}\n",
    "LICENSE": "Apache License 2.0 (test copy)\n",
    "LICENSE_JBIG2": "PDFium licence (test copy)\n",
    "LICENSE_OPENJPEG": "OpenJPEG licence (test copy)\n",
    "LICENSE_PDFJS_JBIG2": "pdf.js JBIG2 licence (test copy)\n",
    "LICENSE_PDFJS_OPENJPEG": "pdf.js OpenJPEG licence (test copy)\n",
}
OUTSIDE = "outside.js"
"""A file next to the page directory, which a traversal would reach."""


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    """The lifespan configures logging process-wide; put the root logger back afterwards."""
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    yield
    for handler in root.handlers[:]:
        if handler not in saved[0]:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved[1])


@pytest.fixture
def web_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A page directory in place of ``app/web``, with a file outside it and a subdirectory."""
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_bytes(PAGE_HTML.encode())
    for name, text in STATIC_FILES.items():
        (web / name).write_bytes(text.encode())
    (web / "modules").mkdir()
    (web / "modules" / "inner.js").write_bytes(b"export {};\n")
    (tmp_path / OUTSIDE).write_bytes(b"export const secret = 1;\n")
    monkeypatch.setattr(viewer, "WEB_DIR", web)
    return web


@pytest.fixture
def pdfjs_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "pdfjs"
    directory.mkdir()
    for name, text in PDFJS.items():
        (directory / name).write_bytes(text.encode())
    return directory


@pytest.fixture
def make_viewer(web_dir: Path, pdfjs_dir: Path) -> Iterator[ViewerClient]:
    """Start an application over fakes serving ``web_dir`` and ``pdfjs_dir``; ``overrides``
    are settings (``pdfjs_dir`` defaults to the fixture's)."""
    clients: list[TestClient] = []

    def make(**overrides: object) -> TestClient:
        active = api_settings(**{"pdfjs_dir": str(pdfjs_dir), **overrides})
        client = TestClient(create_app(settings=active, container=Fakes().container(active)))
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_viewer: ViewerClient) -> TestClient:
    return make_viewer()
