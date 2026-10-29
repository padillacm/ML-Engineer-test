"""Suite-wide hooks: tests that need a local tool, model or server run only where it is available.

- ``tesseract``: needs the Tesseract executable on PATH.
- ``ner_model``: needs the local NER model directory named by ``NER_MODEL_PATH`` (baked into
  the container image at ``/models``; absent on the host).
- ``embedding_model``: needs the local embedding model directory named by
  ``EMBEDDING_MODEL_PATH`` (baked into the container image; absent on the host).
- ``injection_model``: needs the local Prompt Guard model directory named by
  ``INJECTION_MODEL_PATH`` (baked into the container image; absent on the host unless the
  model was downloaded there).
- ``qdrant_server``: needs a Qdrant server at ``QDRANT_URL`` (set by the test compose file).
- ``llm_server``: needs an OpenAI-compatible LLM server at ``LLM_BASE_URL``.
- ``deployed_service``: needs a running stack at ``SERVICE_URL`` (set by
  docker-compose.smoke.yml for the ``smoke`` service).
- ``cuda``: needs a CUDA GPU visible to torch (``torch.cuda.is_available()``): the CUDA build
  of torch and a GPU, as in the CUDA image or the host's ``.venv-cuda``; the CPU build always
  skips them.
"""

import os
import shutil
from collections.abc import Callable
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
_NO_INJECTION_MODEL = pytest.mark.skip(
    reason=(
        "Prompt Guard model not available (INJECTION_MODEL_PATH unset or not a directory); "
        f"{_CONTAINER_HINT}"
    )
)
_NO_QDRANT = pytest.mark.skip(reason=f"QDRANT_URL unset; {_COMPOSE_HINT}")
_NO_LLM = pytest.mark.skip(reason="LLM_BASE_URL unset (needs an OpenAI-compatible server)")
_NO_CUDA = pytest.mark.skip(
    reason=(
        "no CUDA GPU visible to torch; run with the CUDA build and a GPU: "
        "UV_PROJECT_ENVIRONMENT=.venv-cuda uv run pytest -m cuda"
    )
)
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


def _cuda_available() -> bool:
    import torch  # noqa: PLC0415 - heavy; imported when the markers are checked, not at import

    return bool(torch.cuda.is_available())


_REQUIREMENTS: tuple[tuple[str, Callable[[], bool], pytest.MarkDecorator], ...] = (
    ("tesseract", lambda: shutil.which("tesseract") is not None, _NO_TESSERACT),
    ("ner_model", lambda: _directory_from_env("NER_MODEL_PATH"), _NO_NER_MODEL),
    ("embedding_model", lambda: _directory_from_env("EMBEDDING_MODEL_PATH"), _NO_EMBEDDING_MODEL),
    ("injection_model", lambda: _directory_from_env("INJECTION_MODEL_PATH"), _NO_INJECTION_MODEL),
    ("qdrant_server", lambda: _env_set("QDRANT_URL"), _NO_QDRANT),
    ("llm_server", lambda: _env_set("LLM_BASE_URL"), _NO_LLM),
    ("deployed_service", lambda: _env_set("SERVICE_URL"), _NO_SERVICE),
    ("cuda", _cuda_available, _NO_CUDA),
)
"""Each marker, whether what it needs is available, and the skip applied when it is not."""


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip marked tests whose tool, model or server is missing."""
    missing = [(keyword, skip) for keyword, available, skip in _REQUIREMENTS if not available()]
    for item in items:
        for keyword, skip in missing:
            if item.get_closest_marker(keyword) is not None:
                item.add_marker(skip)
