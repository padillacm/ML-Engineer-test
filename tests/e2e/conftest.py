"""Fixtures of the browser end-to-end tests of the audit viewer (marker ``e2e``).

The tests run only in the ``e2e`` image against a running stack (docker-compose.e2e.yml sets
``VIEWER_URL``); on the host Playwright is not installed and every module is skipped by its own
``pytest.importorskip``. ``VIEWER_URL`` is the page's address (``http://docintel-app:8000/audit``).
Nothing here imports Playwright at module level, so collecting this directory on the host
never needs it.

Artefacts (screenshots, the frame PDF, JSON results) go to ``E2E_ARTIFACTS`` (default
``/e2e-artifacts``, the compose file's mounted output directory).
"""

import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.e2e import frames

if TYPE_CHECKING:
    from tests.e2e.viewer import Session

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "sample_pdfs"
DEFAULT_ARTIFACTS = "/e2e-artifacts"


def viewer_url_or_skip() -> str:
    url = os.environ.get("VIEWER_URL", "").strip()
    if not url:
        pytest.skip(
            "VIEWER_URL unset; run the browser suite against the stack: docker compose "
            "-f docker-compose.yml -f docker-compose.e2e.yml run --rm e2e"
        )
    return url


@pytest.fixture(scope="session")
def viewer_url() -> str:
    return viewer_url_or_skip()


@pytest.fixture(scope="session")
def artifacts() -> Path:
    path = Path(os.environ.get("E2E_ARTIFACTS", DEFAULT_ARTIFACTS))
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def browser(viewer_url: str) -> Iterator[Any]:
    from playwright.sync_api import sync_playwright  # noqa: PLC0415 - e2e image only

    with sync_playwright() as playwright:
        chromium = playwright.chromium.launch(headless=True, args=["--disable-dev-shm-usage"])
        yield chromium
        chromium.close()


def _slug(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text).strip("_")


@pytest.fixture(scope="module")
def open_viewer(
    browser: Any, viewer_url: str, artifacts: Path, request: pytest.FixtureRequest
) -> Iterator[Callable[..., "Session"]]:
    """Open a new viewer session (a fresh browser context); screenshots kept on teardown."""
    from tests.e2e.viewer import open_session  # noqa: PLC0415 - imports Playwright

    sessions: list[Session] = []

    def factory(dpr: float = 1, viewport: dict[str, int] | None = None) -> "Session":
        session = open_session(browser, viewer_url, dpr, viewport)
        sessions.append(session)
        return session

    yield factory
    module = _slug(request.node.name.removesuffix(".py"))
    for number, session in enumerate(sessions):
        session.screenshot(artifacts / f"{module}_{number}.png")
        session.close()


@pytest.fixture(scope="session")
def frame_document(artifacts: Path) -> frames.FrameDocument:
    document = frames.build()
    (artifacts / "frames.pdf").write_bytes(document.pdf)
    return document


@pytest.fixture(scope="session")
def samples() -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(SAMPLES_DIR.glob("*.pdf"))}
