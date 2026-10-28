"""Isolate every settings test from the real environment and from any ``.env`` file."""

from pathlib import Path

import pytest

from app.settings import Settings

REQUIRED_ENV = {
    "NER_MODEL_PATH": "/models/ner",
    "EMBEDDING_MODEL_PATH": "/models/e5",
    "EMBEDDING_MODEL_REVISION": "rev-1",
    "QDRANT_URL": "http://qdrant:6333",
}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No settings variable is set and the working directory has no ``.env``."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def required_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set only the required variables."""
    for name, value in REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)
    return dict(REQUIRED_ENV)
