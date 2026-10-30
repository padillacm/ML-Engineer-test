"""``scripts/serve_docs.py``: the documentation and nothing else of the repository is served."""

import http.client
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts.serve_docs import DocsHandler, File, Redirect, main, resolve


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A repository-shaped folder: the docs, the two deliverables and files that must stay out."""
    docs = tmp_path / "docs"
    (docs / "img").mkdir(parents=True)
    (docs / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (docs / "README.md").write_text("# Home", encoding="utf-8")
    (docs / "_sidebar.md").write_text("- [Home](README.md)", encoding="utf-8")
    (docs / "img" / "flow.svg").write_text("<svg/>", encoding="utf-8")
    (docs / "capacity_estimate.py").write_text("print()", encoding="utf-8")
    (docs / ".hidden.md").write_text("secret", encoding="utf-8")
    (tmp_path / "DESIGN.md").write_text("# Design", encoding="utf-8")
    (tmp_path / "DECISIONS.md").write_text("# Decisions", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Repo", encoding="utf-8")
    (tmp_path / ".env").write_text("LLM_API_KEY=sk-not-for-you", encoding="utf-8")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("app = None", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("path", ["", "/", "/docs", "/?x=1"])
def test_the_root_redirects_to_the_site(root: Path, path: str) -> None:
    assert resolve(path, root) == Redirect("/docs/")


@pytest.mark.parametrize(
    ("path", "relative", "content_type"),
    [
        ("/docs/", "docs/index.html", "text/html; charset=utf-8"),
        ("/docs/README.md", "docs/README.md", "text/markdown; charset=utf-8"),
        ("/docs/_sidebar.md", "docs/_sidebar.md", "text/markdown; charset=utf-8"),
        ("/docs/img/flow.svg", "docs/img/flow.svg", "image/svg+xml"),
        ("/docs/README.md?id=intro", "docs/README.md", "text/markdown; charset=utf-8"),
        ("/DESIGN.md", "DESIGN.md", "text/markdown; charset=utf-8"),
        ("/DECISIONS.md", "DECISIONS.md", "text/markdown; charset=utf-8"),
    ],
)
def test_the_documentation_is_served(
    root: Path, path: str, relative: str, content_type: str
) -> None:
    assert resolve(path, root) == File(root / relative, content_type)


@pytest.mark.parametrize(
    "path",
    [
        "/.env",
        "/README.md",
        "/app/main.py",
        "/docs/../.env",
        "/docs/%2e%2e/.env",
        "/docs/..%2f.env",
        "/docs/.hidden.md",
        "/docs/capacity_estimate.py",
        "/docs/missing.md",
        "/docs/img",
        "/docs/img/",
        "/docs//README.md",
        "/docs/README.md%00.html",
        "/docs\\README.md",
        "/x/DESIGN.md",
        "/DESIGN.md/",
    ],
)
def test_anything_else_is_not_found(root: Path, path: str) -> None:
    assert resolve(path, root) is None


def test_a_symbolic_link_out_of_the_docs_is_not_followed(root: Path) -> None:
    link = root / "docs" / "leak.md"
    try:
        link.symlink_to(root / ".env")
    except OSError:
        pytest.skip("symbolic links are not available here")
    assert resolve("/docs/leak.md", root) is None


@pytest.fixture
def server(root: Path) -> Iterator[tuple[str, int]]:
    handler = type("Handler", (DocsHandler,), {"root": root, "log_message": lambda *a: None})
    from http.server import ThreadingHTTPServer  # noqa: PLC0415 - only this fixture serves

    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        host, port = httpd.server_address[:2]
        yield str(host), int(port)
        httpd.shutdown()
        thread.join()


def _get(server: tuple[str, int], path: str, method: str = "GET") -> http.client.HTTPResponse:
    connection = http.client.HTTPConnection(*server, timeout=5)
    connection.request(method, path)
    return connection.getresponse()


def test_a_page_is_served_with_its_type_and_no_sniffing(server: tuple[str, int]) -> None:
    response = _get(server, "/docs/README.md")

    assert response.status == 200
    assert response.read() == b"# Home"
    assert response.getheader("Content-Type") == "text/markdown; charset=utf-8"
    assert response.getheader("X-Content-Type-Options") == "nosniff"


def test_head_has_the_headers_and_no_body(server: tuple[str, int]) -> None:
    response = _get(server, "/DESIGN.md", "HEAD")

    assert response.status == 200
    assert response.getheader("Content-Length") == str(len("# Design"))
    assert response.read() == b""


def test_the_server_redirects_and_refuses(server: tuple[str, int]) -> None:
    redirect = _get(server, "/")
    assert redirect.status == 302
    assert redirect.getheader("Location") == "/docs/"

    refused = _get(server, "/.env")
    assert refused.status == 404
    assert b"sk-not-for-you" not in refused.read()


def test_other_methods_are_refused(server: tuple[str, int]) -> None:
    assert _get(server, "/docs/README.md", "POST").status == 501


def test_main_rejects_a_bad_port() -> None:
    with pytest.raises(SystemExit):
        main(["--port", "not-a-number"])
