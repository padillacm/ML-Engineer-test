"""The pinned pdf.js release: one pin (``deploy/pdfjs.lock``) for the Dockerfile, the host script,
the image hygiene check and the service, and the installed files match it (design D2; spec:
deployment, the images carry the pinned viewer library).

The installed-file checks run where the files are: in the test image always (it sets
``PDFJS_DIR``, so a missing directory fails), on the host after ``scripts/fetch_pdfjs.py``.
"""

import hashlib
import os
import re
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs

from app.api.viewer import DEFAULT_PDFJS_DIR, PDFJS_FILES
from scripts.fetch_pdfjs import DEFAULT_LOCK, FILES, Pin, read_lock

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
SMOKE = ROOT / "docker-compose.smoke.yml"
IMAGE_PDFJS_DIR = "/app/vendor/pdfjs"

CVE_2024_4367_FIXED = (4, 2, 67)
CVE_2026_16633_FIXED = (6, 2, 108)


@pytest.fixture(scope="module")
def pin() -> Pin:
    return read_lock(DEFAULT_LOCK)


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _dockerfile() -> str:
    """The Dockerfile's text; skipped in the test image, whose build context leaves it out."""
    if not DOCKERFILE.is_file():
        pytest.skip("the Dockerfile is not in the image (.dockerignore); checked on the host")
    return DOCKERFILE.read_text(encoding="utf-8")


def _stages() -> dict[str, str]:
    """Each Dockerfile stage name -> its instructions."""
    text = _dockerfile()
    stages = {}
    for block in re.split(r"^FROM ", text, flags=re.MULTILINE)[1:]:
        header, _, body = block.partition("\n")
        stages[header.split(" AS ")[-1].strip()] = body
    return stages


def test_the_lock_is_at_the_project_root() -> None:
    assert DEFAULT_LOCK == ROOT / "deploy" / "pdfjs.lock"


def test_the_pinned_version_is_past_the_script_execution_fixes(pin: Pin) -> None:
    assert _version(pin.version) >= CVE_2026_16633_FIXED > CVE_2024_4367_FIXED
    assert pin.tarball_url.endswith(f"/pdfjs-dist-{pin.version}.tgz")


def test_the_service_serves_exactly_the_files_kept_from_the_tarball() -> None:
    assert set(PDFJS_FILES) == set(FILES)
    assert {member for member, _ in FILES.values()} == {
        "package/build/pdf.min.mjs",
        "package/build/pdf.worker.min.mjs",
        "package/wasm/jbig2_nowasm_fallback.js",
        "package/wasm/openjpeg_nowasm_fallback.js",
        "package/LICENSE",
        "package/wasm/LICENSE_JBIG2",
        "package/wasm/LICENSE_OPENJPEG",
        "package/wasm/LICENSE_PDFJS_JBIG2",
        "package/wasm/LICENSE_PDFJS_OPENJPEG",
    }
    # Each file is served under its own name, so names must not collide when flattened.
    assert all(member.rsplit("/", 1)[1] == name for name, (member, _) in FILES.items())


def test_no_webassembly_is_kept() -> None:
    """The page turns WebAssembly off (its CSP has no 'wasm-unsafe-eval')."""
    assert [name for name in FILES if name.endswith(".wasm")] == []


def test_the_pdfjs_stage_runs_the_script_with_the_lock() -> None:
    stage = _stages()["pdfjs"]

    assert "COPY deploy/pdfjs.lock scripts/fetch_pdfjs.py" in stage
    assert "fetch_pdfjs.py --lock" in stage
    assert "chmod -R a=rX" in stage
    assert re.search(r"^FROM \$\{PYTHON_IMAGE\} AS pdfjs$", _dockerfile(), re.M)
    assert "PYTHON_IMAGE=python:3.12-slim-trixie@sha256:" in _dockerfile()


@pytest.mark.parametrize("stage", ["test", "runtime"])
def test_the_test_and_runtime_images_copy_the_files_read_only(stage: str) -> None:
    body = _stages()[stage]

    assert f"COPY --from=pdfjs /pdfjs {IMAGE_PDFJS_DIR}" in body
    assert f"PDFJS_DIR={IMAGE_PDFJS_DIR}" in body
    # No --chown: the files stay root's, with the stage's read-only modes.
    assert "COPY --chown=app:app --from=pdfjs" not in body


def test_the_image_path_is_the_default_inside_the_images() -> None:
    """``/app/app/api/viewer.py`` in the images: the default and ``PDFJS_DIR`` agree."""
    assert DEFAULT_PDFJS_DIR.relative_to(ROOT) == Path("vendor", "pdfjs")


def test_the_hygiene_check_reads_the_same_lock() -> None:
    hygiene = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))["services"]["hygiene"]

    assert "./deploy/pdfjs.lock:/run/pdfjs.lock:ro" in hygiene["volumes"]
    script = hygiene["command"][2]
    for name, (_, entry) in FILES.items():
        assert f'"$${entry}" {name} ' in script, name
    assert f'[ "$$(ls -A /app/vendor/pdfjs | wc -l)" = {len(FILES)} ]' in script


def _installed_dir() -> Path:
    configured = os.environ.get("PDFJS_DIR", "").strip()
    if configured:
        return Path(configured)
    if not DEFAULT_PDFJS_DIR.is_dir():
        pytest.skip("pdf.js is not fetched on this host (uv run python scripts/fetch_pdfjs.py)")
    return DEFAULT_PDFJS_DIR


def test_the_installed_files_have_the_pinned_hashes(pin: Pin) -> None:
    directory = _installed_dir()

    actual = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in PDFJS_FILES
    }

    assert actual == dict(pin.file_sha256)


@pytest.mark.skipif(not os.environ.get("PDFJS_DIR"), reason="only in the images (PDFJS_DIR set)")
def test_in_the_image_the_files_are_root_owned_and_read_only() -> None:
    directory = Path(os.environ["PDFJS_DIR"])

    assert str(directory) == IMAGE_PDFJS_DIR
    assert sorted(path.name for path in directory.iterdir()) == sorted(PDFJS_FILES)
    for path in (directory, *directory.iterdir()):
        assert path.stat().st_uid == 0, path
        assert not os.access(path, os.W_OK), path
