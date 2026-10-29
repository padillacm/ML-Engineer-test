"""Suite-wide hooks: tests that need a local tool, model or server run only where it is available.

- ``tesseract``: needs the Tesseract executable on PATH.
- ``ner_model``: needs the local NER model directory named by ``NER_MODEL_PATH`` (baked into
  the container image at ``/models``; absent on the host).
- ``embedding_model``: needs the local embedding model directory named by
  ``EMBEDDING_MODEL_PATH`` (baked into the container image; absent on the host).
- ``qdrant_server``: needs a Qdrant server at ``QDRANT_URL`` (set by the test compose file).
- ``llm_server``: needs an OpenAI-compatible LLM server at ``LLM_BASE_URL``.
- ``deployed_service``: needs a running stack at ``SERVICE_URL`` (set by
  docker-compose.smoke.yml for the ``smoke`` service).
"""

import os
import shutil
from pathlib import Path

import pytest

_CONTAINER_HINT = (
    "run the container suite: "
    "docker build --target test -t docintel-test . && docker run --rm docintel-test"
)
_COMPOSE_HINT = "run the compose suite: docker compose -f docker-compose.test.yml run --rm tests"

_NO_TESSERACT = pytest.mark.skip(reason=f"tesseract executable not on PATH; {_CONTAINER_HINT}")
_NO_NER_MODEL = pytest.mark.skip(
    reason=f"NER model not available (NER_MODEL_PATH unset or not a directory); {_CONTAINER_HINT}"
)
_NO_EMBEDDING_MODEL = pytest.mark.skip(
    reason=(
        "embedding model not available (EMBEDDING_MODEL_PATH unset or not a directory); "
        f"{_CONTAINER_HINT}"
    )
)
_NO_QDRANT = pytest.mark.skip(reason=f"QDRANT_URL unset; {_COMPOSE_HINT}")
_NO_LLM = pytest.mark.skip(reason="LLM_BASE_URL unset (needs an OpenAI-compatible server)")
_NO_SERVICE = pytest.mark.skip(
    reason=(
        "SERVICE_URL unset; run the smoke suite against the stack: docker compose "
        "-f docker-compose.yml -f docker-compose.smoke.yml run --rm smoke"
    )
)


def _directory_from_env(name: str) -> bool:
    path = os.environ.get(name)
    return path is not None and path != "" and Path(path).is_dir()


def _env_set(name: str) -> bool:
    return os.environ.get(name, "").strip() != ""


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip marked tests whose tool, model or server is missing."""
    missing = []
    if shutil.which("tesseract") is None:
        missing.append(("tesseract", _NO_TESSERACT))
    if not _directory_from_env("NER_MODEL_PATH"):
        missing.append(("ner_model", _NO_NER_MODEL))
    if not _directory_from_env("EMBEDDING_MODEL_PATH"):
        missing.append(("embedding_model", _NO_EMBEDDING_MODEL))
    if not _env_set("QDRANT_URL"):
        missing.append(("qdrant_server", _NO_QDRANT))
    if not _env_set("LLM_BASE_URL"):
        missing.append(("llm_server", _NO_LLM))
    if not _env_set("SERVICE_URL"):
        missing.append(("deployed_service", _NO_SERVICE))
    for item in items:
        for keyword, skip in missing:
            if item.get_closest_marker(keyword) is not None:
                item.add_marker(skip)
