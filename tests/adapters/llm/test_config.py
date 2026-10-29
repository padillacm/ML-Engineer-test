"""Validation of ``LLMClientConfig`` and masking of ``Secret``."""

import copy
import pickle
import traceback

import pytest

from app.adapters.llm import LLMClientConfig, Secret
from app.domain.errors import ConfigurationError
from tests.adapters.llm.helpers import KEY


def test_defaults() -> None:
    config = LLMClientConfig("http://ollama:11434/v1", "qwen2.5")

    assert config.api_key is None
    assert (config.connect_timeout_s, config.read_timeout_s) == (5.0, 60.0)
    assert config.attempts == 3
    assert (config.backoff_base_s, config.backoff_cap_s, config.retry_after_cap_s) == (
        0.5,
        8.0,
        30.0,
    )
    assert config.total_timeout_s == 90.0
    assert (config.max_tokens, config.temperature) == (512, 0.0)
    assert config.reasoning_effort is None
    assert config.max_tokens_field == "max_tokens"
    assert config.send_temperature is True


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("http://ollama:11434/v1", "http://ollama:11434/v1"),
        ("http://ollama:11434/v1/", "http://ollama:11434/v1"),
        ("  https://api.example.com/v1// ", "https://api.example.com/v1"),
        ("https://api.example.com", "https://api.example.com"),
    ],
)
def test_base_url_is_normalised(given: str, expected: str) -> None:
    config = LLMClientConfig(given, "m")

    assert config.base_url == expected
    assert config.completions_url == f"{expected}/chat/completions"


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "   ",
        "ollama:11434/v1",
        "ftp://host/v1",
        "http:///v1",
        "http://user:pass@host/v1",
        "http://host/v1?key=x",
        "http://host/v1#frag",
        "http://host:notaport/v1",
    ],
)
def test_invalid_base_url_is_rejected(base_url: str) -> None:
    with pytest.raises(ConfigurationError):
        LLMClientConfig(base_url, "m")


def test_base_url_containing_the_key_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as info:
        LLMClientConfig(f"https://host/{KEY}/v1", "m", api_key=Secret(KEY))

    assert KEY not in str(info.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": ""},
        {"model": "  "},
        {"api_key": KEY},
        {"connect_timeout_s": 0},
        {"read_timeout_s": -1.0},
        {"read_timeout_s": float("inf")},
        {"attempts": 0},
        {"attempts": 1.5},
        {"attempts": True},
        {"backoff_base_s": -0.1},
        {"backoff_base_s": 2.0, "backoff_cap_s": 1.0},
        {"retry_after_cap_s": -1},
        {"total_timeout_s": 0},
        {"total_timeout_s": -5.0},
        {"total_timeout_s": float("inf")},
        {"total_timeout_s": float("nan")},
        {"total_timeout_s": True},
        {"total_timeout_s": 4.0, "connect_timeout_s": 5.0},
        {"max_tokens": 0},
        {"temperature": -0.1},
        {"temperature": 2.5},
        {"temperature": float("nan")},
        {"reasoning_effort": ""},
        {"reasoning_effort": "None"},
        {"reasoning_effort": "off"},
        {"reasoning_effort": " low"},
        {"reasoning_effort": 0},
        {"reasoning_effort": ["low"]},
        {"max_tokens_field": "max_output_tokens"},
        {"max_tokens_field": "MAX_TOKENS"},
        {"max_tokens_field": None},
        {"max_tokens_field": ["max_tokens"]},
        {"send_temperature": "false"},
        {"send_temperature": 0},
        {"send_temperature": None},
    ],
)
def test_invalid_values_are_rejected(overrides: dict[str, object]) -> None:
    settings: dict[str, object] = {"base_url": "http://h/v1", "model": "m", **overrides}

    with pytest.raises(ConfigurationError):
        LLMClientConfig(**settings)  # type: ignore[arg-type]


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh"])
def test_every_reasoning_effort_is_accepted(effort: str) -> None:
    assert LLMClientConfig("http://h/v1", "m", reasoning_effort=effort).reasoning_effort == effort


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("send_temperature", [True, False])
def test_limit_field_and_temperature_switch_are_accepted(
    field: str, send_temperature: bool
) -> None:
    config = LLMClientConfig(
        "http://h/v1", "m", max_tokens_field=field, send_temperature=send_temperature
    )

    assert (config.max_tokens_field, config.send_temperature) == (field, send_temperature)


def test_option_errors_are_fixed_text() -> None:
    with pytest.raises(ConfigurationError) as info:
        LLMClientConfig("http://h/v1", "m", reasoning_effort="sk-secret-value")

    assert info.value.message == (
        "reasoning_effort must be None or one of none, minimal, low, medium, high, xhigh"
    )
    with pytest.raises(ConfigurationError) as other:
        LLMClientConfig("http://h/v1", "m", max_tokens_field="sk-secret-value")

    assert "sk-secret-value" not in str(other.value)


def test_key_is_masked_in_config_text() -> None:
    config = LLMClientConfig("https://api.example.com/v1", "m", api_key=Secret(KEY))

    assert KEY not in repr(config)
    assert KEY not in str(config)
    assert KEY not in f"{config}"
    assert "***" in repr(config)


def test_secret_masks_and_reveals() -> None:
    secret = Secret(KEY)

    assert secret.reveal() == KEY
    assert KEY not in repr(secret)
    assert KEY not in str(secret)
    assert secret == Secret(KEY)
    assert secret != Secret("other")
    with pytest.raises(TypeError):
        hash(secret)


def test_total_timeout_may_equal_the_connect_timeout() -> None:
    config = LLMClientConfig("http://h/v1", "m", connect_timeout_s=5.0, total_timeout_s=5.0)

    assert config.total_timeout_s == 5.0


@pytest.mark.parametrize("value", ["", "   ", None, 123])
def test_blank_secret_is_rejected(value: object) -> None:
    with pytest.raises(ConfigurationError):
        Secret(value)  # type: ignore[arg-type]


UNUSABLE_KEYS = [
    "sk key",
    " sk-key",
    "sk-key ",
    "sk-key\n",
    "sk-key\r\nX-Injected: 1",
    "sk-\tkey",
    "sk-\x00key",
    "sk-\x1bkey",
    "sk-\x7fkey",
    "sk-" + chr(0xA0) + "key",  # no-break space
    "sk-" + chr(0x2028) + "key",  # line separator
    "sk-cl" + chr(0xE9),  # e acute
    "sk-" + "".join(map(chr, (0x43A, 0x43B, 0x44E, 0x447))),  # Cyrillic
    "sk-" + chr(0x1F511),  # emoji
]


@pytest.mark.parametrize("key", UNUSABLE_KEYS)
def test_unusable_key_is_rejected_without_showing_it(key: str) -> None:
    with pytest.raises(ConfigurationError) as info:
        Secret(key)

    text = repr(info.value) + "".join(traceback.format_exception(info.value))
    assert key not in text
    assert key.strip() not in text
    assert info.value.__cause__ is None
    assert info.value.__context__ is None


@pytest.mark.parametrize("key", [KEY, "!", "~" * 200, "Ab0-._~+/=!#$%&'*"])
def test_printable_ascii_keys_are_accepted(key: str) -> None:
    assert Secret(key).reveal() == key


def test_secret_cannot_be_pickled() -> None:
    secret = Secret(KEY)

    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        with pytest.raises(TypeError, match=r"^Secret values cannot be pickled$") as info:
            pickle.dumps(secret, protocol=protocol)
        assert KEY not in str(info.value)


def test_secret_copies_are_the_same_object() -> None:
    secret = Secret(KEY)

    assert copy.copy(secret) is secret
    assert copy.deepcopy(secret) is secret


def test_config_with_a_key_cannot_be_pickled() -> None:
    config = LLMClientConfig("https://api.example.com/v1", "m", api_key=Secret(KEY))

    with pytest.raises(TypeError, match=r"^Secret values cannot be pickled$"):
        pickle.dumps(config)


def test_config_without_a_key_is_picklable() -> None:
    config = LLMClientConfig("http://ollama:11434/v1", "m", total_timeout_s=30.0)

    assert pickle.loads(pickle.dumps(config)) == config  # noqa: S301 - own data


def test_config_with_a_key_can_be_copied() -> None:
    config = LLMClientConfig("https://api.example.com/v1", "m", api_key=Secret(KEY))

    for duplicate in (copy.copy(config), copy.deepcopy(config)):
        assert duplicate == config
        assert duplicate.api_key is config.api_key
