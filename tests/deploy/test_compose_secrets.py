"""The language model and Qdrant keys reach the app only as secret files (spec: deployment).

Runs everywhere (host and test image): it only reads ``docker-compose.yml`` and the placeholder
secret. What compose actually does with them (``docker inspect``, ``docker compose config``, the
keys arriving at the model server and at Qdrant) was verified against a running stack
(``docs/BENCHMARKS.md`` sections 19.7 and 19.8); these checks keep the wiring from being edited
away.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.yml"
PLACEHOLDER = ROOT / "deploy" / "empty-secret"
KEYS = [("LLM_API_KEY", "llm_api_key"), ("QDRANT_API_KEY", "qdrant_api_key")]


def _compose() -> str:
    return COMPOSE.read_text(encoding="utf-8")


def test_the_placeholder_secret_is_empty() -> None:
    """Mounted when a key is unset; any content would be sent as a key."""
    assert PLACEHOLDER.read_bytes() == b""


@pytest.mark.parametrize(("variable", "secret"), KEYS)
def test_the_app_reads_the_key_from_the_secret_file_not_from_a_variable(
    variable: str, secret: str
) -> None:
    text = _compose()

    assert not re.search(rf"^      {variable}:", text, re.MULTILINE)  # not in its environment
    assert re.search(rf"^      {variable}_FILE: /run/secrets/{secret}\r?$", text, re.MULTILINE)
    assert re.search(rf"^        target: {secret}\r?$", text, re.MULTILINE)


@pytest.mark.parametrize(("variable", "secret"), KEYS)
def test_the_secret_comes_from_the_variable_or_else_the_placeholder(
    variable: str, secret: str
) -> None:
    text = _compose()

    assert f"- source: {secret}${{{variable}:+_from_env}}" in text
    assert re.search(
        rf"^  {secret}_from_env:\r?\n    environment: {variable}\r?$", text, re.MULTILINE
    )
    assert re.search(rf"^  {secret}:\r?\n    file: \./deploy/empty-secret\r?$", text, re.MULTILINE)


@pytest.mark.parametrize("variable", [variable for variable, _ in KEYS])
def test_the_key_value_is_never_interpolated(variable: str) -> None:
    """``${KEY}`` or ``${KEY:-...}`` would put the key into the resolved file."""
    uses = re.findall(rf"\$\{{{variable}([^}}]*)\}}", _compose())

    assert uses
    assert set(uses) == {":+_from_env"}


def test_qdrant_requires_the_key_only_when_one_is_set() -> None:
    text = _compose()

    assert 'key="$$(cat /run/secrets/qdrant_api_key)"' in text
    assert 'if [ -n "$$key" ]; then export QDRANT__SERVICE__API_KEY="$$key"; fi' in text
    assert "QDRANT__SERVICE__API_KEY:" not in text  # never a container variable
