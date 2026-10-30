"""Serve the documentation site (docsify, ``docs/index.html``) on this machine (standard library).

    uv run python scripts/serve_docs.py              # then open http://127.0.0.1:3000/docs/
    uv run python scripts/serve_docs.py --port 8080

Only the documentation is served: the files of ``docs/`` under ``/docs/``, and the two
deliverables at the repository root, ``/DESIGN.md`` and ``/DECISIONS.md``, which the site renders
as its Design and Decisions pages. Every other path is a 404, so nothing else of the repository
(``.env``, the code, ``.git``) can be read through the server, which ``python -m http.server`` run
from the root would allow. A requested path never becomes a file path by joining: each segment
must be a plain name (no separator, no leading dot), the file must be a regular file with an
allowed extension, and it must lie inside ``docs/``. The server binds to ``127.0.0.1`` by default.

docsify itself and its plugins are loaded by the browser from a CDN, pinned by version and by
Subresource Integrity hash in ``docs/index.html``.
"""

import argparse
import contextlib
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlsplit

ROOT: Final = Path(__file__).resolve().parents[1]
DOCS_PREFIX: Final = "/docs/"
ROOT_FILES: Final = frozenset({"DESIGN.md", "DECISIONS.md"})
"""Files of the repository root the site renders (as ``/DESIGN.md``, ``/DECISIONS.md``)."""

CONTENT_TYPES: Final = {
    ".html": "text/html; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}
"""Extension -> content type of the files served; no other file is served."""

PLAIN_NAME: Final = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
"""A path segment served: no separator, no leading dot (`_sidebar.md` is docsify's)."""
DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 3000


@dataclass(frozen=True, slots=True)
class Redirect:
    location: str


@dataclass(frozen=True, slots=True)
class File:
    path: Path
    content_type: str


def resolve(url_path: str, root: Path = ROOT) -> Redirect | File | None:
    """What ``url_path`` (the path of a request, query and fragment ignored) serves under
    ``root``: a redirect to the site, a file, or ``None`` (404)."""
    path = unquote(urlsplit(url_path).path)
    if path in {"", "/", "/docs"}:
        return Redirect(DOCS_PREFIX)
    if path.lstrip("/") in ROOT_FILES and path.count("/") == 1:
        return _file(root / path.lstrip("/"), root)
    if not path.startswith(DOCS_PREFIX):
        return None
    relative = path[len(DOCS_PREFIX) :] or "index.html"
    segments = relative.split("/")
    if not all(PLAIN_NAME.fullmatch(segment) for segment in segments):
        return None
    docs = root / "docs"
    candidate = docs.joinpath(*segments)
    if not candidate.resolve().is_relative_to(docs.resolve()):
        return None
    return _file(candidate, root)


def _file(path: Path, root: Path) -> File | None:
    content_type = CONTENT_TYPES.get(path.suffix.lower())
    if content_type is None or path.is_symlink() or not path.is_file():
        return None
    if not path.resolve().is_relative_to(root.resolve()):
        return None
    return File(path, content_type)


class DocsHandler(BaseHTTPRequestHandler):
    """GET and HEAD of the documentation; everything else is refused."""

    root: Path = ROOT
    server_version = "docs"
    sys_version = ""

    def do_GET(self) -> None:
        self._answer(body=True)

    def do_HEAD(self) -> None:
        self._answer(body=False)

    def _answer(self, *, body: bool) -> None:
        target = resolve(self.path, self.root)
        if isinstance(target, Redirect):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", target.location)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = target.path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", target.content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if body:
            self.wfile.write(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the documentation site.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="interface to bind (local only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to listen on")
    args = parser.parse_args(argv)
    with ThreadingHTTPServer((args.host, args.port), DocsHandler) as server:
        host, port = server.server_address[:2]
        print(f"documentation at http://{host!s}:{port}/docs/ (Ctrl+C to stop)", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
