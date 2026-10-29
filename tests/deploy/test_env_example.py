"""``.env.example`` lists every setting with its current default and no secret (spec: deployment).

Runs everywhere (host and test image): it only reads ``.env.example``, the compose files and the
``Settings`` class. Lines are ``NAME=value``, commented out (``# NAME=value``) so that a
copied file changes nothing; a value shown for a setting with a default must be that default,
so a default changed in ``app/settings.py`` cannot leave a stale example behind.
"""

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter
from pydantic_core import PydanticUndefined

from app.settings import RETIRED_VARIABLES, Settings

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / ".env.example"
COMPOSE_FILES = sorted(ROOT.glob("docker-compose*.yml"))

SECRETS = ("LLM_API_KEY", "QDRANT_API_KEY")
ASSIGNMENT = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=(.*)$")
# ${NAME}, ${NAME:-default}, ${NAME-default}, ${NAME:?error} ... in a compose file; `$${NAME}` is
# a shell variable of a service's command, escaped from compose.
COMPOSE_VARIABLE = re.compile(r"(?<!\$)\$\{([A-Z][A-Z0-9_]*)")


def _assignments() -> list[tuple[str, str]]:
    lines = EXAMPLE.read_text(encoding="utf-8").splitlines()
    return [
        (match[1], match[2].strip()) for line in lines if (match := ASSIGNMENT.match(line.strip()))
    ]


def _listed() -> dict[str, list[str]]:
    listed: dict[str, list[str]] = {}
    for name, value in _assignments():
        listed.setdefault(name, []).append(value)
    return listed


def _settings_fields() -> list[str]:
    return [name for name in Settings.model_fields if name not in RETIRED_VARIABLES]


def test_every_setting_is_listed() -> None:
    listed = _listed()

    missing = [name.upper() for name in _settings_fields() if name.upper() not in listed]

    assert missing == []


def test_retired_variables_are_not_listed() -> None:
    listed = _listed()

    assert [name for name in RETIRED_VARIABLES.values() if name in listed] == []


def test_the_compose_files_are_found() -> None:
    assert [path.name for path in COMPOSE_FILES] == [
        "docker-compose.e2e.yml",
        "docker-compose.smoke.yml",
        "docker-compose.test.yml",
        "docker-compose.torch.yml",
        "docker-compose.yml",
    ]


@pytest.mark.parametrize("compose", COMPOSE_FILES, ids=lambda path: path.name)
def test_every_compose_variable_is_listed(compose: Path) -> None:
    used = set(COMPOSE_VARIABLE.findall(compose.read_text(encoding="utf-8")))
    assert used, f"the pattern found no ${{...}} variable in {compose.name}"

    assert sorted(used - set(_listed())) == []


def test_shell_variables_of_commands_are_not_taken_for_compose_variables() -> None:
    assert COMPOSE_VARIABLE.findall("$${LLM_BASE_URL:-} ${APP_PORT:-8000} $$HOME") == ["APP_PORT"]


def test_the_example_says_how_to_quote_a_value_with_a_dollar() -> None:
    text = EXAMPLE.read_text(encoding="utf-8")

    assert "single quotes" in text
    assert "LLM_API_KEY='abc$def'" in text


@pytest.mark.parametrize("name", SECRETS)
def test_secrets_have_no_value(name: str) -> None:
    values = _listed().get(name)

    assert values, f"{name} is not listed"
    assert values == [""] * len(values)


def _with_default() -> list[str]:
    """Fields with a default value (not required, not unset by default)."""
    return [
        name
        for name in _settings_fields()
        if Settings.model_fields[name].default not in (PydanticUndefined, None)
    ]


@pytest.mark.parametrize("field", _with_default())
def test_a_shown_value_is_the_settings_default(field: str) -> None:
    info = Settings.model_fields[field]
    default: Any = info.default
    values = [value for value in _listed().get(field.upper(), []) if value]
    assert values, f"{field.upper()}: show its default"
    assert info.annotation is not None
    adapter: TypeAdapter[Any] = TypeAdapter(info.annotation)

    assert [adapter.validate_python(value) for value in values] == [default] * len(values)


def test_every_line_is_a_comment_or_blank() -> None:
    """Nothing is active, so copying the example to ``.env`` changes no setting."""
    lines = EXAMPLE.read_text(encoding="utf-8").splitlines()

    assert [line for line in lines if line.strip() and not line.startswith("#")] == []
