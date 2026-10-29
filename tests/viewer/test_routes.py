"""The audit viewer's page, files and pdf.js routes (specs: audit-viewer, http-api; design D5)."""

import asyncio
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import viewer
from app.api.errors import install_error_handlers
from app.main import create_app, include_viewer
from tests.api.support import PDF, Fakes, api_settings, names_json
from tests.viewer.conftest import OUTSIDE, PAGE_HTML, PDFJS, STATIC_FILES, ViewerClient

CSP = (
    "default-src 'none'; script-src 'self'; worker-src 'self'; style-src 'self'; "
    "img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)
"""Design D5, written out so a change to the policy has to change this test too."""
SECURITY = {
    "content-security-policy": CSP,
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
}
JS = "text/javascript; charset=utf-8"


class Response(Protocol):
    """What the tests read of a test client's response."""

    @property
    def status_code(self) -> int: ...
    @property
    def headers(self) -> Mapping[str, str]: ...
    def json(self) -> Any: ...


def _security_headers(response: Response) -> dict[str, str | None]:
    return {name: response.headers.get(name) for name in SECURITY}


def _assert_not_found(response: Response) -> None:
    assert response.status_code == 404
    body = response.json()
    assert (body["code"], body["detail"]) == ("not_found", "Not Found")


# --- Page and files ------------------------------------------------------------------------------


def test_the_page_is_served_with_the_security_headers_and_no_store(client: TestClient) -> None:
    response = client.get("/audit")

    assert response.status_code == 200
    assert response.text == PAGE_HTML
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["cache-control"] == "no-store"
    assert _security_headers(response) == SECURITY


@pytest.mark.parametrize(
    ("name", "content_type"),
    [
        ("viewer.mjs", JS),
        ("names.js", JS),
        ("viewer.css", "text/css; charset=utf-8"),
        ("icon.svg", "image/svg+xml"),
    ],
)
def test_the_page_files_are_served_with_their_type(
    client: TestClient, name: str, content_type: str
) -> None:
    response = client.get(f"/audit/static/{name}")

    assert response.status_code == 200
    assert response.text == STATIC_FILES[name]
    assert response.headers["content-type"] == content_type
    assert response.headers["cache-control"] == "no-cache"
    assert _security_headers(response) == SECURITY


@pytest.mark.parametrize(
    ("name", "content_type"),
    [
        ("pdf.min.mjs", JS),
        ("pdf.worker.min.mjs", JS),
        ("jbig2_nowasm_fallback.js", JS),
        ("openjpeg_nowasm_fallback.js", JS),
        ("LICENSE", "text/plain; charset=utf-8"),
        ("LICENSE_JBIG2", "text/plain; charset=utf-8"),
        ("LICENSE_OPENJPEG", "text/plain; charset=utf-8"),
        ("LICENSE_PDFJS_JBIG2", "text/plain; charset=utf-8"),
        ("LICENSE_PDFJS_OPENJPEG", "text/plain; charset=utf-8"),
    ],
)
def test_the_pdfjs_files_are_served_with_their_type_and_a_cache(
    client: TestClient, name: str, content_type: str
) -> None:
    response = client.get(f"/audit/vendor/pdfjs/{name}")

    assert response.status_code == 200
    assert response.text == PDFJS[name]
    assert response.headers["content-type"] == content_type
    assert response.headers["cache-control"] == "public, max-age=86400"
    assert _security_headers(response) == SECURITY


def test_the_viewer_goes_through_the_service_middleware(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Request id and access line as for the API; the route template, never the raw path."""
    with caplog.at_level(logging.INFO, logger="app.access"):
        response = client.get("/audit/static/viewer.mjs", headers={"X-Request-ID": "audit-1"})

    assert response.headers["x-request-id"] == "audit-1"
    (access,) = [record for record in caplog.records if record.name == "app.access"]
    assert access.__dict__["route"] == "/audit/static/{name:path}"
    assert access.__dict__["status"] == 200


def test_the_api_gets_no_viewer_headers(client: TestClient) -> None:
    responses = [
        client.get("/health"),
        client.post(
            "/api/extract",
            files={"pdf_file": ("scan.pdf", PDF, "application/pdf")},
            data={"names": names_json(("Maria", "Gonzalez"))},
        ),
        client.get("/no/such/route"),
    ]

    assert [response.status_code for response in responses] == [200, 200, 404]
    for response in responses:
        assert set(SECURITY).isdisjoint(response.headers)


def test_the_viewer_is_not_in_the_api_schema(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert [path for path in paths if path.startswith("/audit")] == []


@pytest.mark.parametrize(
    "path", ["/audit", "/audit/", "/audit/static/viewer.mjs", "/audit/vendor/pdfjs/LICENSE"]
)
def test_head_answers_with_the_headers_of_get_and_no_body(client: TestClient, path: str) -> None:
    got, head = client.get(path), client.head(path)

    assert head.status_code == 200
    assert head.content == b""
    for name in ("content-type", "content-length", "cache-control", "etag", *SECURITY):
        assert head.headers[name] == got.headers[name], name


@pytest.mark.parametrize("path", ["/audit/static/missing.js", "/audit/vendor/pdfjs/x.mjs"])
def test_head_of_an_unknown_file_is_404_with_the_viewer_headers(
    client: TestClient, path: str
) -> None:
    response = client.head(path)

    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"
    assert _security_headers(response) == SECURITY


def test_the_page_is_also_served_at_audit_slash_without_a_redirect(client: TestClient) -> None:
    response = client.get("/audit/", headers={"host": "attacker.example"}, follow_redirects=False)

    assert response.status_code == 200
    assert response.text == PAGE_HTML
    assert "location" not in response.headers
    assert response.headers["cache-control"] == "no-store"
    assert _security_headers(response) == SECURITY


@pytest.mark.parametrize(
    "path",
    [
        "/audit/static",
        "/audit/vendor/pdfjs",
        "/audit/vendor",
        "/audit/vendor/",
        "/audit/other",
        "/audit/static.mjs",
        "/audit/index.html",
    ],
)
def test_other_viewer_paths_are_404_never_a_redirect_to_the_host_header(
    client: TestClient, path: str
) -> None:
    response = client.get(path, headers={"host": "attacker.example"}, follow_redirects=False)

    _assert_not_found(response)
    assert "location" not in response.headers
    assert response.headers["cache-control"] == "no-store"
    assert _security_headers(response) == SECURITY


# --- Conditional requests ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "cache"),
    [
        ("/audit/static/viewer.mjs", "no-cache"),
        ("/audit/static/viewer.css", "no-cache"),
        ("/audit/vendor/pdfjs/pdf.min.mjs", "public, max-age=86400"),
    ],
)
@pytest.mark.parametrize("form", ["{etag}", "W/{etag}", '"other", {etag}', "*"])
def test_a_matching_if_none_match_is_304_with_the_headers(
    client: TestClient, path: str, cache: str, form: str
) -> None:
    etag = client.get(path).headers["etag"]

    response = client.get(path, headers={"If-None-Match": form.format(etag=etag)})

    assert response.status_code == 304
    assert response.content == b""
    assert response.headers["etag"] == etag
    assert response.headers["cache-control"] == cache
    assert _security_headers(response) == SECURITY
    assert client.head(path, headers={"If-None-Match": etag}).status_code == 304


def test_another_entity_tag_gets_the_file(client: TestClient) -> None:
    response = client.get("/audit/static/viewer.mjs", headers={"If-None-Match": '"stale"'})

    assert response.status_code == 200
    assert response.text == STATIC_FILES["viewer.mjs"]


def test_a_changed_file_gets_a_new_entity_tag(client: TestClient, web_dir: Path) -> None:
    before = client.get("/audit/static/viewer.mjs").headers["etag"]
    (web_dir / "viewer.mjs").write_bytes(b"export const ready = false; // changed")

    response = client.get("/audit/static/viewer.mjs", headers={"If-None-Match": before})

    assert response.status_code == 200
    assert response.headers["etag"] != before
    assert response.text.endswith("// changed")


# --- Unknown names -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/audit/static/missing.js",
        "/audit/static/notes.txt",  # present, extension not allowed
        "/audit/static/other.html",  # HTML is only the page itself
        "/audit/static/index.html",  # ... at /audit
        "/audit/static/viewer.mjs.map",
        "/audit/static/VIEWER.MJS",  # exact names only, also on a case-insensitive disk
        "/audit/static/modules",  # a directory
        "/audit/static/modules/inner.js",  # no subdirectories
        "/audit/static/",
        "/audit/vendor/pdfjs/pdf.mjs",  # not one of the kept files
        "/audit/vendor/pdfjs/openjpeg.wasm",
        "/audit/vendor/pdfjs/pdf.min.mjs.map",
        "/audit/vendor/pdfjs/",
    ],
)
def test_an_unknown_file_is_404_with_the_viewer_headers(client: TestClient, path: str) -> None:
    response = client.get(path)

    _assert_not_found(response)
    assert response.headers["cache-control"] == "no-store"
    assert _security_headers(response) == SECURITY


@pytest.mark.parametrize(
    "path",
    [
        "/audit/static/..%2F..%2Fsettings.py",
        "/audit/static/..%2Foutside.js",
        "/audit/static/%2e%2e%2foutside.js",
        "/audit/static/%2E%2E/outside.js",
        "/audit/static/..%5Coutside.js",
        "/audit/static/%2Fetc%2Fpasswd",
        "/audit/static/viewer.mjs%00.js",
        "/audit/vendor/pdfjs/..%2Fweb%2Fviewer.mjs",
        "/audit/vendor/pdfjs/%2e%2e%2fpdfjs%2fLICENSE",
    ],
)
def test_encoded_traversal_attempts_are_404(client: TestClient, path: str) -> None:
    _assert_not_found(client.get(path))


def _raw_get(app: FastAPI, path: str) -> int:
    """The status of a GET whose ASGI path is ``path`` exactly (as a server could deliver it),
    bypassing the HTTP client, which would normalise dot segments."""
    statuses: list[int] = []
    received: list[str] = []
    done = asyncio.Event()

    async def receive() -> dict[str, object]:
        # The request once; then the client is gone, after the response (file responses
        # listen for the disconnect while they send).
        if not received:
            received.append("request")
            return {"type": "http.request", "body": b"", "more_body": False}
        await done.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        if message["type"] == "http.response.start":
            statuses.append(int(message["status"]))  # type: ignore[call-overload]
        elif message["type"] == "http.response.body" and not message.get("more_body", False):
            done.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))  # type: ignore[arg-type]
    (status,) = statuses
    return status


@pytest.fixture
def bare_viewer(web_dir: Path, pdfjs_dir: Path) -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(viewer.viewer_router(pdfjs_dir, web_dir))
    return app


@pytest.mark.parametrize(
    "path",
    [
        f"/audit/static/../{OUTSIDE}",
        "/audit/static/../web/viewer.mjs",
        "/audit/static/./viewer.mjs",
        "/audit/static/..",
        f"/audit/static/..\\{OUTSIDE}",
        "/audit/static/modules\\inner.js",
        "/audit/static//etc/passwd",
        "/audit/static/C:/Windows/win.ini",
        "/audit/static/C:\\Windows\\system.js",
        "/audit/static/viewer.mjs\x00.js",
        "/audit/static/viewer.mjs\n.js",
        "/audit/static/.viewer.mjs",
        "/audit/vendor/pdfjs/../pdfjs/LICENSE",
        "/audit/vendor/pdfjs/./pdf.min.mjs",
        "/audit/vendor/pdfjs//pdf.min.mjs",
    ],
)
def test_raw_traversal_attempts_are_404(bare_viewer: FastAPI, path: str) -> None:
    assert _raw_get(bare_viewer, path) == 404


def test_the_raw_requests_reach_files_that_exist(bare_viewer: FastAPI) -> None:
    """The raw helper itself serves the files, so its 404s above are the guard's."""
    assert _raw_get(bare_viewer, "/audit/static/viewer.mjs") == 200
    assert _raw_get(bare_viewer, "/audit/vendor/pdfjs/LICENSE") == 200


@pytest.mark.skipif(os.name == "nt", reason="symbolic links need privileges on Windows")
def test_a_symbolic_link_in_the_page_directory_is_not_followed(
    client: TestClient, web_dir: Path
) -> None:
    (web_dir / "linked.js").symlink_to(web_dir.parent / OUTSIDE)

    _assert_not_found(client.get("/audit/static/linked.js"))


def test_a_missing_page_directory_is_404(make_viewer: ViewerClient, web_dir: Path) -> None:
    for entry in web_dir.rglob("*"):
        if entry.is_file():
            entry.unlink()
    (web_dir / "modules").rmdir()
    web_dir.rmdir()
    client = make_viewer()

    _assert_not_found(client.get("/audit"))
    _assert_not_found(client.get("/audit/static/viewer.mjs"))


# --- pdf.js missing ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PDFJS))
def test_a_missing_pdfjs_file_is_503_with_how_to_get_it(
    make_viewer: ViewerClient, tmp_path: Path, name: str
) -> None:
    client = make_viewer(pdfjs_dir=str(tmp_path / "not-fetched"))

    response = client.get(f"/audit/vendor/pdfjs/{name}")

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "http_error"
    assert "scripts/fetch_pdfjs.py" in body["detail"]
    assert str(tmp_path) not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert _security_headers(response) == SECURITY
    # The page and an unknown name are unaffected.
    assert client.get("/audit").status_code == 200
    _assert_not_found(client.get("/audit/vendor/pdfjs/other.mjs"))


def test_the_pdfjs_directory_defaults_to_vendor_at_the_project_root() -> None:
    root = Path(viewer.__file__).resolve().parents[2]

    assert viewer.pdfjs_directory(None) == root / "vendor" / "pdfjs" == viewer.DEFAULT_PDFJS_DIR
    assert viewer.pdfjs_directory("/srv/pdfjs") == Path("/srv/pdfjs")
    assert root / "app" / "web" == viewer.WEB_DIR


def test_the_default_pdfjs_directory_is_used_when_unset(
    make_viewer: ViewerClient, pdfjs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(viewer, "DEFAULT_PDFJS_DIR", pdfjs_dir)
    client = make_viewer(pdfjs_dir=None)

    assert client.get("/audit/vendor/pdfjs/LICENSE").text == PDFJS["LICENSE"]


# --- Turned off ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/audit",
        "/audit/",
        "/audit/static/viewer.mjs",
        "/audit/vendor/pdfjs/pdf.min.mjs",
        "/audit/vendor/pdfjs/missing",
    ],
)
def test_turned_off_every_viewer_path_is_an_unknown_route(
    make_viewer: ViewerClient, path: str
) -> None:
    client = make_viewer(audit_viewer_enabled=False)

    response = client.get(path)

    _assert_not_found(response)
    assert set(SECURITY).isdisjoint(response.headers)


def test_turned_off_the_routes_do_not_exist_and_the_api_works(make_viewer: ViewerClient) -> None:
    enabled, disabled = make_viewer(), make_viewer(audit_viewer_enabled=False)

    response = disabled.post(
        "/api/extract",
        files={"pdf_file": ("scan.pdf", PDF, "application/pdf")},
        data={"names": names_json(("Maria", "Gonzalez"))},
    )

    # FastAPI lists an included router as one entry: the viewer's is the only difference.
    assert len(disabled.app.routes) == len(enabled.app.routes) - 1  # type: ignore[attr-defined]
    assert not getattr(disabled.app.state, "viewer_included", False)  # type: ignore[attr-defined]
    assert response.status_code == 200
    assert response.json()["fuzzy_matches"][0]["matched_name"] == "Maria Gonzalez"


def test_the_routes_exist_only_once_the_settings_are_loaded(pdfjs_dir: Path) -> None:
    active = api_settings(pdfjs_dir=str(pdfjs_dir))
    app = create_app(settings=active, container=Fakes().container(active))
    before = len(app.routes)

    with TestClient(app) as client:
        assert len(app.routes) == before + 1
        assert client.get("/audit/vendor/pdfjs/LICENSE").status_code == 200
    # A second start of the same application does not include them again.
    with TestClient(app) as client:
        assert len(app.routes) == before + 1
        assert client.get("/audit/vendor/pdfjs/LICENSE").status_code == 200
    include_viewer(app, active)
    assert len(app.routes) == before + 1
