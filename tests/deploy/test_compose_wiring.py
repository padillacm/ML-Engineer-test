"""How docker-compose.yml wires the service, its settings and the local model (spec: deployment).

Runs everywhere (host and test image): it parses the compose file and evaluates its ``${...}``
expressions with the rules compose documents (``:-`` default when unset or empty, ``:+``
replacement when set and non-empty). The shell snippets of the start commands run with ``sh``
when the platform has it. What compose does with all this (containers started or refused, the
model reached, nothing else in the environment) was checked against a running stack
(``docs/BENCHMARKS.md`` section 19.8); these checks keep the wiring from being edited away.
"""

import ipaddress
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs

from app.settings import RETIRED_VARIABLES, Settings

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.yml"

# Settings compose does not pass from .env: the image sets the model paths and revision, the
# store target and the key files are fixed below, and the keys arrive only as secret files.
NOT_PASSED = {
    "ner_model_path",
    "embedding_model_path",
    "embedding_model_revision",
    "qdrant_location",
    "llm_api_key",
    "qdrant_api_key",
}
FIXED = {
    "QDRANT_URL": "http://qdrant:6333",
    "QDRANT_API_KEY_FILE": "/run/secrets/qdrant_api_key",
    "LLM_API_KEY_FILE": "/run/secrets/llm_api_key",
}
DERIVED = {"LLM_BASE_URL", "LLM_MODEL"}
NOT_SETTINGS = {"OLLAMA_URL"}  # read by the start command only
DEFAULT_PREFIX = "10.213.47"
OLLAMA_URL = f"http://{DEFAULT_PREFIX}.250:11434"


def _compose() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    return loaded


def _service(name: str) -> dict[str, Any]:
    service: dict[str, Any] = _compose()["services"][name]
    return service


def _environment(name: str) -> dict[str, str]:
    return {key: str(value) for key, value in _service(name)["environment"].items()}


def interpolate(text: str, env: Mapping[str, str]) -> str:
    """Compose's ``${NAME}``, ``${NAME:-default}`` and ``${NAME:+replacement}``, nested."""
    out: list[str] = []
    index = 0
    while index < len(text):
        if text.startswith("${", index):
            end = _closing_brace(text, index + 2)
            out.append(_expand(text[index + 2 : end], env))
            index = end + 1
        else:
            out.append(text[index])
            index += 1
    return "".join(out)


def _closing_brace(text: str, start: int) -> int:
    depth = 1
    for position in range(start, len(text)):
        if text.startswith("${", position):
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return position
    raise AssertionError(f"unbalanced expression in {text!r}")


def _expand(body: str, env: Mapping[str, str]) -> str:
    match = re.fullmatch(r"([A-Z][A-Z0-9_]*)(?:(:-|:\+)(.*))?", body, re.DOTALL)
    assert match, f"unsupported expression ${{{body}}}"
    name, operator, rest = match[1], match[2], match[3]
    value = env.get(name, "")
    if operator == ":-":
        return value or interpolate(rest, env)
    if operator == ":+":
        return interpolate(rest, env) if value else ""
    return value


def test_the_interpolation_helper_follows_compose() -> None:
    assert interpolate("${A:-x}", {}) == "x"
    assert interpolate("${A:-x}", {"A": ""}) == "x"
    assert interpolate("${A:-x}", {"A": "a"}) == "a"
    assert interpolate("${A:+y}", {}) == ""
    assert interpolate("${A:+y}", {"A": "a"}) == "y"
    assert interpolate("p${A:-${B:+q${C:-r}}}s", {"B": "1"}) == "pqrs"
    assert interpolate("${A}", {"A": "a"}) == "a"


# --- Finding: only the service's own settings reach it (no env_file) --------------------------


def _passed_settings() -> set[str]:
    return {
        name.upper()
        for name in Settings.model_fields
        if name not in RETIRED_VARIABLES and name not in NOT_PASSED
    }


def test_no_service_reads_the_whole_env_file() -> None:
    """``env_file: .env`` copied every variable of .env, tokens included, into the container."""
    services = _compose()["services"]

    assert [name for name, service in services.items() if "env_file" in service] == []


def test_the_app_gets_exactly_the_settings_it_reads() -> None:
    environment = _environment("app")

    assert set(environment) - NOT_SETTINGS == _passed_settings()


def test_each_setting_is_passed_as_given_or_empty() -> None:
    """An empty value counts as unset in ``Settings`` (``env_ignore_empty``): the default."""
    environment = _environment("app")

    for name, value in environment.items():
        if name in FIXED:
            assert value == FIXED[name]
        elif name not in DERIVED | NOT_SETTINGS:
            assert value == f"${{{name}:-}}", name


def test_a_setting_unset_in_env_is_empty_for_the_app() -> None:
    environment = _environment("app")

    assert interpolate(environment["LOG_LEVEL"], {}) == ""
    assert interpolate(environment["LOG_LEVEL"], {"LOG_LEVEL": "DEBUG"}) == "DEBUG"


# --- Finding: explicit Ollama wiring at a fixed internal address --------------------------------


def _base_url(env: Mapping[str, str]) -> str:
    return interpolate(_environment("app")["LLM_BASE_URL"], env)


def _model(env: Mapping[str, str]) -> str:
    return interpolate(_environment("app")["LLM_MODEL"], env)


@pytest.mark.parametrize(
    ("env", "base_url", "model"),
    [
        ({}, "", ""),
        ({"COMPOSE_PROFILES": "gpu"}, f"{OLLAMA_URL}/v1", "gemma4:e4b-it-qat"),
        (
            {"COMPOSE_PROFILES": "gpu", "LLM_MODEL": "gemma4:12b-it-qat"},
            f"{OLLAMA_URL}/v1",
            "gemma4:12b-it-qat",
        ),
        (
            {"COMPOSE_PROFILES": "gpu", "BACKEND_NETWORK_PREFIX": "10.9.8"},
            "http://10.9.8.250:11434/v1",
            "gemma4:e4b-it-qat",
        ),
        (
            {"LLM_BASE_URL": "https://api.openai.com/v1", "LLM_MODEL": "m"},
            "https://api.openai.com/v1",
            "m",
        ),
        # A hosted URL without LLM_MODEL: no Gemma default, so the settings refuse to start.
        ({"LLM_BASE_URL": "https://api.openai.com/v1"}, "https://api.openai.com/v1", ""),
    ],
)
def test_the_model_server_follows_compose_profiles_or_llm_base_url(
    env: dict[str, str], base_url: str, model: str
) -> None:
    assert (_base_url(env), _model(env)) == (base_url, model)


def test_the_app_never_looks_up_a_host_name_for_ollama() -> None:
    """A name lookup is forwarded to the host's resolvers while Ollama is down (outside answers)."""
    text = COMPOSE.read_text(encoding="utf-8")

    assert "getent" not in text
    assert "http://ollama:" not in text
    assert ipaddress.ip_address(_base_url({"COMPOSE_PROFILES": "gpu"}).split("/")[2].split(":")[0])


def test_ollama_has_the_fixed_address_inside_the_internal_subnet() -> None:
    ollama = _service("ollama")
    backend = _compose()["networks"]["backend"]
    (pool,) = backend["ipam"]["config"]

    address = ipaddress.ip_address(interpolate(ollama["networks"]["backend"]["ipv4_address"], {}))
    subnet = ipaddress.ip_network(interpolate(pool["subnet"], {}))
    dynamic = ipaddress.ip_network(interpolate(pool["ip_range"], {}))

    assert backend["internal"] is True
    assert list(ollama["networks"]) == ["backend"]  # no route out while serving
    assert f"http://{address}:11434" == interpolate(_environment("app")["OLLAMA_URL"], {})
    assert address in subnet
    assert address not in dynamic  # never handed to another container
    assert dynamic.subnet_of(subnet)  # type: ignore[arg-type]


def test_the_model_is_downloaded_by_a_separate_server_on_the_egress_network_only() -> None:
    pull = _service("ollama-pull")

    assert pull["networks"] == ["egress"]
    assert pull["volumes"] == _service("ollama")["volumes"]
    assert "deploy" not in pull  # no GPU needed to download
    assert _service("ollama")["depends_on"] == {
        "ollama-pull": {"condition": "service_completed_successfully"}
    }


def test_a_warm_volume_is_not_downloaded_again() -> None:
    script = _service("ollama-pull")["command"][0]

    assert 'ollama show "$$LLM_MODEL" > /dev/null 2>&1 || ollama pull "$$LLM_MODEL"' in script


# --- Finding: a failed download stops the app from starting --------------------------------------


def test_the_app_starts_only_after_the_model_check_succeeded() -> None:
    depends_on = _service("app")["depends_on"]

    assert depends_on["llm-warmup"] == {"condition": "service_completed_successfully"}
    # No dependency on a profile service: with `required: false` compose ignores its failure.
    assert set(depends_on) == {"qdrant", "llm-warmup"}
    assert all(entry.get("required", True) for entry in depends_on.values())


def test_the_model_check_sees_the_app_s_model_server() -> None:
    warmup = _service("llm-warmup")
    app, check = _environment("app"), _environment("llm-warmup")

    assert "profiles" not in warmup  # runs with and without the profile
    assert warmup["command"] == ["python", "-m", "app.llm_warmup"]
    assert warmup["image"] == _service("app")["image"]
    assert warmup["pull_policy"] == "never"
    assert {name: check[name] for name in ("LLM_BASE_URL", "LLM_MODEL", "OLLAMA_URL")} == {
        name: app[name] for name in ("LLM_BASE_URL", "LLM_MODEL", "OLLAMA_URL")
    }
    assert warmup["depends_on"] == {"ollama": {"condition": "service_healthy", "required": False}}


# --- The shell snippets ---------------------------------------------------------------------------


def _sh() -> str:
    found = shutil.which("sh")
    if found is None or os.name == "nt":  # Git's sh on Windows mangles some quoting
        pytest.skip("needs a POSIX sh")
    return found


def _run(script: str, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # the compose file's own script, test-controlled env
        [_sh(), "-c", script],
        env={"PATH": os.environ.get("PATH", ""), **env},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _unescaped(script: str) -> str:
    return script.replace("$$", "$")


def _app_start(env: Mapping[str, str]) -> str:
    """The reasoning effort the app's start command hands to the service."""
    script = _unescaped(_service("app")["command"][2])
    probe = script.split("exec uvicorn")[0] + 'printf "%s" "${LLM_REASONING_EFFORT-<unset>}"'
    result = _run(probe, env)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize(
    ("env", "effort"),
    [
        ({"LLM_BASE_URL": f"{OLLAMA_URL}/v1", "OLLAMA_URL": OLLAMA_URL}, "none"),
        ({"LLM_BASE_URL": f"{OLLAMA_URL}/v1/", "OLLAMA_URL": OLLAMA_URL}, "none"),
        (
            {
                "LLM_BASE_URL": f"{OLLAMA_URL}/v1",
                "OLLAMA_URL": OLLAMA_URL,
                "LLM_REASONING_EFFORT": "low",
            },
            "low",
        ),
        (
            {
                "LLM_BASE_URL": f"{OLLAMA_URL}/v1",
                "OLLAMA_URL": OLLAMA_URL,
                "LLM_REASONING_EFFORT": "",
            },
            "none",
        ),
        ({"LLM_BASE_URL": "https://api.openai.com/v1", "OLLAMA_URL": OLLAMA_URL}, "<unset>"),
        ({"LLM_BASE_URL": "", "OLLAMA_URL": OLLAMA_URL}, "<unset>"),
    ],
)
def test_thinking_is_off_whenever_the_app_uses_the_stack_ollama(
    env: dict[str, str], effort: str
) -> None:
    assert _app_start(env) == effort


def _pull_guard(env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    script = _unescaped(_service("ollama-pull")["command"][0])
    guard = script.split("ollama serve")[0] + "echo guard-passed"
    return _run(guard, {"OLLAMA_URL": OLLAMA_URL, **env})


def test_the_download_starts_when_the_app_uses_the_stack_ollama() -> None:
    result = _pull_guard({"APP_LLM_BASE_URL": f"{OLLAMA_URL}/v1"})

    assert (result.returncode, result.stdout.strip()) == (0, "guard-passed")


def test_the_profile_alone_fails_naming_compose_profiles() -> None:
    """``--profile gpu`` without COMPOSE_PROFILES: the app would run without the model."""
    result = _pull_guard({"APP_LLM_BASE_URL": ""})

    assert result.returncode == 1
    assert "start with COMPOSE_PROFILES=gpu" in result.stderr


def test_a_hosted_model_with_the_profile_fails() -> None:
    result = _pull_guard({"APP_LLM_BASE_URL": "https://api.openai.com/v1"})

    assert result.returncode == 1
    assert "LLM_BASE_URL is not the stack's Ollama" in result.stderr
