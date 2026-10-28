"""Suite-wide hooks: tests that need a local tool or model run only where it is available.

- ``tesseract``: needs the Tesseract executable on PATH.
- ``ner_model``: needs the local NER model directory named by ``NER_MODEL_PATH`` (baked into
  the container image at ``/models``; absent on the host).
"""

import os
import shutil
from pathlib import Path

import pytest

_CONTAINER_HINT = (
    "run the container suite: "
    "docker build --target test -t docintel-test . && docker run --rm docintel-test"
)

_NO_TESSERACT = pytest.mark.skip(reason=f"tesseract executable not on PATH; {_CONTAINER_HINT}")
_NO_NER_MODEL = pytest.mark.skip(
    reason=f"NER model not available (NER_MODEL_PATH unset or not a directory); {_CONTAINER_HINT}"
)


def _ner_model_available() -> bool:
    path = os.environ.get("NER_MODEL_PATH")
    return path is not None and path != "" and Path(path).is_dir()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``tesseract``/``ner_model``-marked tests when the tool or model is missing."""
    missing = []
    if shutil.which("tesseract") is None:
        missing.append(("tesseract", _NO_TESSERACT))
    if not _ner_model_available():
        missing.append(("ner_model", _NO_NER_MODEL))
    for item in items:
        for keyword, skip in missing:
            if keyword in item.keywords:
                item.add_marker(skip)
