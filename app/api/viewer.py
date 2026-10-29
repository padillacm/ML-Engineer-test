"""The audit viewer's page and its files (specs: audit-viewer, http-api; design D5).

``viewer_router(pdfjs_dir)`` serves, from fixed directories, each route for GET and HEAD:

- ``/audit`` (and ``/audit/``): the page, ``app/web/index.html``;
- ``/audit/static/{name}``: a file of ``app/web/`` with an extension of ``STATIC_TYPES``
  (scripts, the stylesheet, images); the page itself only at ``/audit``;
- ``/audit/vendor/pdfjs/{name}``: one of the pinned pdf.js files (``PDFJS_FILES``: the
  library, its worker, the two image decoders it imports with WebAssembly off, and the
  licences) in ``pdfjs_dir``, or 503 with a message saying how to get them when it is missing;
- any other path under ``/audit/``: 404 with the viewer's headers, so the router's
  trailing-slash redirect (which builds its ``Location`` from the request's ``Host`` header)
  never answers for a viewer path.

A requested name never becomes a path. It must be a plain file name (``PLAIN_NAME``: no
separator, no dot segment, no NUL), and it is looked up among the regular files actually in
the directory (symbolic links are not followed); only a listed entry is opened. Anything else
answers 404 with the uniform error body. ``{name:path}`` lets the routes see every path under
their prefix, so a name with a separator is refused here, with the viewer's headers, instead of
falling through to the unknown-route 404.

Every response of these routes, errors included, carries ``SECURITY_HEADERS`` and a
``Cache-Control``: ``no-store`` on the page and on errors, ``no-cache`` (revalidated) on the
page's own files, which change with the code, and one day on the pdf.js files, which change only
with the pin (their URL has no version, so a longer cache could outlive a bump). A file
response carries an ``ETag``; a request whose ``If-None-Match`` names it gets 304 with the same
headers and no body. The API routes never get these headers. ``app.main`` includes the router
only when ``AUDIT_VIEWER_ENABLED`` is on; off, every ``/audit`` path is an unknown route (404).
"""

import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

PAGE_PATH = "/audit"
STATIC_PREFIX = "/audit/static/"
PDFJS_PREFIX = "/audit/vendor/pdfjs/"

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
"""The page and its files (``app/web``), part of the service code."""
DEFAULT_PDFJS_DIR = Path(__file__).resolve().parents[2] / "vendor" / "pdfjs"
"""``vendor/pdfjs`` at the project root, where ``scripts/fetch_pdfjs.py`` writes the files;
``/app/vendor/pdfjs`` in the images (which also set ``PDFJS_DIR`` to it)."""

PAGE = "index.html"
JAVASCRIPT = "text/javascript; charset=utf-8"
HTML = "text/html; charset=utf-8"

STATIC_TYPES: Mapping[str, str] = MappingProxyType(
    {
        ".js": JAVASCRIPT,
        ".mjs": JAVASCRIPT,
        ".css": "text/css; charset=utf-8",
        ".svg": "image/svg+xml",
    }
)
"""Extension -> content type of the files ``/audit/static/`` serves; no other file is served."""

TEXT = "text/plain; charset=utf-8"

PDFJS_FILES: Mapping[str, str] = MappingProxyType(
    {
        "pdf.min.mjs": JAVASCRIPT,
        "pdf.worker.min.mjs": JAVASCRIPT,
        # pdf.js 6 imports these (a module each) for CCITT/JBIG2 and JPEG 2000 images when the
        # page turns WebAssembly off; no .wasm file is served.
        "jbig2_nowasm_fallback.js": JAVASCRIPT,
        "openjpeg_nowasm_fallback.js": JAVASCRIPT,
        "LICENSE": TEXT,
        "LICENSE_JBIG2": TEXT,
        "LICENSE_OPENJPEG": TEXT,
        "LICENSE_PDFJS_JBIG2": TEXT,
        "LICENSE_PDFJS_OPENJPEG": TEXT,
    }
)
"""The pdf.js files kept from the pinned release (``deploy/pdfjs.lock``) -> content type."""

PLAIN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
"""A file name the routes look up: no separator, backslash, NUL or leading dot (``..``)."""

CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'none'",
        "script-src 'self'",
        "worker-src 'self'",
        "style-src 'self'",
        "img-src 'self'",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    )
)

SECURITY_HEADERS: Mapping[str, str] = MappingProxyType(
    {
        "Content-Security-Policy": CONTENT_SECURITY_POLICY,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
    }
)
"""On every response of the viewer's routes."""

METHODS = ["GET", "HEAD"]
"""Every viewer route answers both (a HEAD gets the GET's headers without the body)."""

NO_STORE = "no-store"
REVALIDATE = "no-cache"
PDFJS_CACHE = "public, max-age=86400"

NOT_FOUND = "Not Found"
PDFJS_MISSING = (
    "the audit viewer's pdf.js files are not installed: run "
    "`uv run python scripts/fetch_pdfjs.py` (the Docker images include them)"
)


def pdfjs_directory(configured: str | None) -> Path:
    """The directory of the pdf.js files: ``PDFJS_DIR`` when set, else ``DEFAULT_PDFJS_DIR``."""
    return Path(configured) if configured is not None else DEFAULT_PDFJS_DIR


def viewer_router(pdfjs_dir: Path, web_dir: Path | None = None) -> APIRouter:
    """The viewer's routes over ``web_dir`` (default ``WEB_DIR``) and ``pdfjs_dir``."""
    web = web_dir if web_dir is not None else WEB_DIR
    router = APIRouter(include_in_schema=False)

    @router.api_route(STATIC_PREFIX + "{name:path}", methods=METHODS)
    async def static_file(request: Request, name: str) -> Response:
        media_type = STATIC_TYPES.get(PurePosixPath(name).suffix)
        if media_type is None or name == PAGE:
            raise _error(404, NOT_FOUND)
        return _serve(request, web, name, media_type, REVALIDATE)

    @router.api_route(PDFJS_PREFIX + "{name:path}", methods=METHODS)
    async def pdfjs_file(request: Request, name: str) -> Response:
        media_type = PDFJS_FILES.get(name)
        if media_type is None:
            raise _error(404, NOT_FOUND)
        return _serve(
            request, pdfjs_dir, name, media_type, PDFJS_CACHE, missing=(503, PDFJS_MISSING)
        )

    @router.api_route(PAGE_PATH, methods=METHODS)
    @router.api_route(PAGE_PATH + "/", methods=METHODS)
    async def page(request: Request) -> Response:
        return _serve(request, web, PAGE, HTML, NO_STORE)

    @router.api_route(PAGE_PATH + "/{rest:path}", methods=METHODS)
    async def unknown(rest: str) -> Response:
        # Every other path under /audit: a 404 with the viewer's headers here, so that the
        # router's trailing-slash redirect (built from the request's Host header) never answers
        # for a viewer path such as /audit/static.
        del rest
        raise _error(404, NOT_FOUND)

    return router


def _serve(
    request: Request,
    directory: Path,
    name: str,
    media_type: str,
    cache: str,
    missing: tuple[int, str] = (404, NOT_FOUND),
) -> Response:
    """The file ``name`` of ``directory`` with the viewer's headers; ``missing`` when absent;
    304 (same headers, no body) when ``If-None-Match`` names its current entity tag."""
    found = _listed_file(directory, name)
    if found is None:
        raise _error(*missing)
    path, stat = found
    response = FileResponse(
        path,
        media_type=media_type,
        headers={**SECURITY_HEADERS, "Cache-Control": cache},
        stat_result=stat,
    )
    etag = response.headers["etag"]
    if _matches(request.headers.get("if-none-match"), etag):
        return Response(
            status_code=304,
            headers={**SECURITY_HEADERS, "Cache-Control": cache, "ETag": etag},
        )
    return response


def _matches(if_none_match: str | None, etag: str) -> bool:
    """Whether an ``If-None-Match`` value names ``etag`` (weak comparison, RFC 9110 13.1.2)."""
    if if_none_match is None:
        return False
    tags = {tag.strip().removeprefix("W/") for tag in if_none_match.split(",")}
    return "*" in tags or etag.removeprefix("W/") in tags


def _listed_file(directory: Path, name: str) -> tuple[str, os.stat_result] | None:
    """The path and status of the regular file ``name`` listed in ``directory`` (no symbolic
    link), else ``None``; ``name`` must be a ``PLAIN_NAME`` and is never joined to a path."""
    if PLAIN_NAME.fullmatch(name) is None:
        return None
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name == name and entry.is_file(follow_symlinks=False):
                    return entry.path, entry.stat(follow_symlinks=False)
    except OSError:
        return None
    return None


def _error(status: int, message: str) -> StarletteHTTPException:
    """An HTTP error the service's handler renders with the uniform body and these headers."""
    return StarletteHTTPException(
        status, message, headers={**SECURITY_HEADERS, "Cache-Control": NO_STORE}
    )
