"""Importing the core must not load any heavy or vendor module."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEAVY = (
    "spacy",
    "chromadb",
    "sentence_transformers",
    "pytesseract",
    "torch",
    "transformers",
    "fitz",
    "fastapi",
    "pydantic",
    "httpx",
    "qdrant_client",
)

SCRIPT = f"""
import importlib
import json
import pkgutil
import sys

for name in ("app.domain", "app.application"):
    package = importlib.import_module(name)
    for info in pkgutil.walk_packages(package.__path__, name + "."):
        importlib.import_module(info.name)
print(json.dumps(sorted(m for m in {HEAVY!r} if m in sys.modules)))
"""


def test_core_import_loads_no_heavy_module() -> None:
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    offenders = json.loads(result.stdout.strip().splitlines()[-1])
    assert not offenders, f"importing app.domain / app.application loaded: {offenders}"
