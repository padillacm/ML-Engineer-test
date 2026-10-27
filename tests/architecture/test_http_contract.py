"""The fixed HTTP contract (app/models/schemas.py) must not drift."""

import hashlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "app" / "models" / "schemas.py"
SCHEMAS_SHA256 = "5b235f7e3060867c53b075c839a8c64aaf14ce6532a620ea17daa7d736789e0f"


def _digest(content: bytes) -> str:
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def test_schemas_match_recorded_hash() -> None:
    """The real guard: ``SCHEMAS_SHA256`` pins the HTTP contract file.

    If the contract is ever changed on purpose, update the constant deliberately in the same
    change; never update it just to make this test pass.
    """
    assert _digest(SCHEMAS.read_bytes()) == SCHEMAS_SHA256


def test_schemas_match_committed_version() -> None:
    """Cross-check against git HEAD; skipped when git or the file at HEAD is unavailable.

    The recorded hash above stays the authority: after a deliberate contract change this test
    fails until the change is committed.
    """
    try:
        result = subprocess.run(
            ["git", "show", "HEAD:app/models/schemas.py"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
    except OSError:
        pytest.skip("git is not available")
    if result.returncode != 0:
        pytest.skip("git history or the file at HEAD is not available")
    assert _digest(result.stdout) == SCHEMAS_SHA256
