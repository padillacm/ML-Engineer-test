"""The e2e stage's browser: the Chrome Headless Shell that Playwright pins, fetched by the
Dockerfile's ``chromium`` stage with a sha256 (not by ``playwright install``, which checks none).

On the host the Dockerfile is checked (its build context leaves it out of the images); in the
e2e image (marker ``e2e``) the browser directory must be the one this Playwright version looks
for, so a Playwright bump without a new pin fails here instead of at launch.
"""

import json
import os
import re
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile"
ARG = re.compile(r"^ARG (CHROMIUM_HEADLESS_SHELL_\w+)=(\S+)$", re.MULTILINE)


def _stage(text: str, name: str) -> str:
    for block in re.split(r"^FROM ", text, flags=re.MULTILINE)[1:]:
        header, _, body = block.partition("\n")
        if header.split(" AS ")[-1].strip() == name:
            return body
    raise AssertionError(f"no stage {name}")


def test_the_chromium_stage_pins_the_archive_and_the_e2e_stage_uses_it() -> None:
    if not DOCKERFILE.is_file():
        pytest.skip("the Dockerfile is not in the image (.dockerignore); checked on the host")
    text = DOCKERFILE.read_text(encoding="utf-8")
    chromium, e2e = _stage(text, "chromium"), _stage(text, "e2e")
    args = dict(ARG.findall(chromium))

    assert re.fullmatch(r"[0-9a-f]{64}", args["CHROMIUM_HEADLESS_SHELL_SHA256"])
    assert re.fullmatch(r"\d+\.\d+\.\d+\.\d+", args["CHROMIUM_HEADLESS_SHELL_VERSION"])
    revision = args["CHROMIUM_HEADLESS_SHELL_REVISION"]
    assert "hashlib.sha256(data).hexdigest()" in chromium
    assert "COPY --from=chromium /ms-playwright /ms-playwright" in e2e
    assert "playwright install-deps chromium-headless-shell" in e2e
    assert re.search(r"playwright install(?!-deps)", e2e) is None
    assert f"/ms-playwright/chromium_headless_shell-{revision}/DEPENDENCIES_VALIDATED" in e2e


@pytest.mark.e2e
def test_the_installed_browser_is_the_one_this_playwright_pins() -> None:
    playwright = pytest.importorskip("playwright", reason="Playwright is only in the e2e image")
    browsers = Path(playwright.__file__).parent / "driver" / "package" / "browsers.json"
    pinned = {entry["name"]: entry for entry in json.loads(browsers.read_text())["browsers"]}
    revision = pinned["chromium-headless-shell"]["revision"]
    root = Path(os.environ["PLAYWRIGHT_BROWSERS_PATH"]) / f"chromium_headless_shell-{revision}"

    assert os.access(root / "chrome-headless-shell-linux64" / "chrome-headless-shell", os.X_OK)
    assert (root / "INSTALLATION_COMPLETE").is_file()
    assert (root / "DEPENDENCIES_VALIDATED").is_file()
