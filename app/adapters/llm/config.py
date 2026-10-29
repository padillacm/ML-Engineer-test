"""Settings of the OpenAI-compatible chat-completion client, validated when they are created."""

import math
from dataclasses import dataclass
from urllib.parse import urlsplit

from app.adapters.llm.secret import Secret
from app.domain.errors import ConfigurationError

_MAX_TEMPERATURE = 2.0

REASONING_EFFORTS: frozenset[str] = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
"""Accepted ``reasoning_effort`` values (the OpenAI set; Ollama's endpoint honours ``none``)."""

MAX_TOKENS_FIELDS: frozenset[str] = frozenset({"max_tokens", "max_completion_tokens"})
"""Names the request may give the reply limit: hosted reasoning models reject ``max_tokens``."""


@dataclass(frozen=True, slots=True)
class LLMClientConfig:
    """Where the model is served and how the client calls it.

    ``base_url`` is the OpenAI-compatible root (for Ollama ``http://ollama:11434/v1``); requests
    go to ``{base_url}/chat/completions``. It must be ``http``/``https`` with a host and no user
    info, query or fragment; trailing slashes are removed. ``api_key`` is sent as a bearer token
    when present and is masked in every text representation. ``attempts`` is the total number of
    tries (1 means no retry). Between tries the client waits for ``Retry-After`` (capped at
    ``retry_after_cap_s``) when the server sends one, otherwise for a random time in
    ``[0, min(backoff_cap_s, backoff_base_s * 2**(retry - 1))]`` (full jitter).

    ``total_timeout_s`` is the overall deadline of one ``complete()`` call, counted from its start
    across every attempt and wait; it must be at least ``connect_timeout_s``. No attempt starts and
    no wait begins once it has passed, a wait that would end after it is not taken, and each
    attempt's connect/read/write/pool timeouts are cut to the time left. Running out fails with
    ``DependencyTimeout`` (``llm_timeout``); a reply whose body completes after the deadline is
    dropped. Worst-case latency: httpx bounds each network operation (pool, connect, each write,
    each read) rather than the whole request, so the deadline is enforced at check points: before
    each attempt and wait, once the response headers arrive, and after every body chunk. The
    operations in flight when it passes (pool wait, connect, write, read) can overrun it by at most
    ``2 * connect_timeout_s + min(read_timeout_s, total_timeout_s)``, so a call returns or fails
    by ``total_timeout_s + 2 * connect_timeout_s + min(read_timeout_s, total_timeout_s)`` (160 s
    with the defaults), assuming the request goes out in one write and the headers in one read;
    a server that trickles its status line and headers byte by byte is not bounded by httpx.

    The request adapts to the model's parameters: ``reasoning_effort`` (one of
    ``REASONING_EFFORTS``) is sent only when set (``"none"`` turns thinking off on Ollama's
    OpenAI-compatible endpoint, where thinking models otherwise reason by default),
    ``max_tokens_field`` names the reply limit (``max_tokens``, or ``max_completion_tokens`` for
    hosted reasoning models, which reject ``max_tokens``), and ``send_temperature=False`` leaves
    ``temperature`` out for models that accept only their default. With the defaults the request
    is the plain chat-completions request. Invalid values raise ``ConfigurationError``.
    """

    base_url: str
    model: str
    api_key: Secret | None = None
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 60.0
    attempts: int = 3
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 8.0
    retry_after_cap_s: float = 30.0
    total_timeout_s: float = 90.0
    max_tokens: int = 512
    temperature: float = 0.0
    reasoning_effort: str | None = None
    max_tokens_field: str = "max_tokens"
    send_temperature: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _normalise_base_url(self.base_url))
        _require(
            isinstance(self.model, str) and bool(self.model.strip()),
            "model must be a non-blank string",
        )
        _require(
            self.api_key is None or isinstance(self.api_key, Secret),
            "api_key must be a Secret or None",
        )
        if self.api_key is not None and self.api_key.reveal() in self.base_url:
            raise ConfigurationError("base_url must not contain the API key")
        _require(_is_positive(self.connect_timeout_s), "connect_timeout_s must be positive")
        _require(_is_positive(self.read_timeout_s), "read_timeout_s must be positive")
        _require(
            _is_positive(self.total_timeout_s) and self.total_timeout_s >= self.connect_timeout_s,
            "total_timeout_s must be positive and at least connect_timeout_s",
        )
        _require(
            isinstance(self.attempts, int)
            and not isinstance(self.attempts, bool)
            and self.attempts >= 1,
            "attempts must be an integer of at least 1",
        )
        _require(_is_non_negative(self.backoff_base_s), "backoff_base_s must be non-negative")
        _require(
            _is_non_negative(self.backoff_cap_s) and self.backoff_cap_s >= self.backoff_base_s,
            "backoff_cap_s must be at least backoff_base_s",
        )
        _require(_is_non_negative(self.retry_after_cap_s), "retry_after_cap_s must be non-negative")
        _require(
            isinstance(self.max_tokens, int)
            and not isinstance(self.max_tokens, bool)
            and self.max_tokens >= 1,
            "max_tokens must be a positive integer",
        )
        _require(
            _is_non_negative(self.temperature) and self.temperature <= _MAX_TEMPERATURE,
            "temperature must be between 0 and 2",
        )
        _require(
            self.reasoning_effort is None
            or (
                isinstance(self.reasoning_effort, str)
                and self.reasoning_effort in REASONING_EFFORTS
            ),
            "reasoning_effort must be None or one of none, minimal, low, medium, high, xhigh",
        )
        _require(
            isinstance(self.max_tokens_field, str) and self.max_tokens_field in MAX_TOKENS_FIELDS,
            "max_tokens_field must be max_tokens or max_completion_tokens",
        )
        _require(isinstance(self.send_temperature, bool), "send_temperature must be a boolean")

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"


def _normalise_base_url(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("base_url must be a non-blank string")
    url = value.strip().rstrip("/")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as error:
        raise ConfigurationError("base_url is not a valid URL") from error
    del port
    _require(parts.scheme in {"http", "https"}, "base_url must use http or https")
    _require(bool(parts.hostname), "base_url must name a host")
    _require(
        parts.username is None and parts.password is None,
        "base_url must not contain credentials; use api_key",
    )
    _require(not parts.query and not parts.fragment, "base_url must not have a query or fragment")
    return url


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _is_positive(value: object) -> bool:
    return _is_number(value) and value > 0  # type: ignore[operator]


def _is_non_negative(value: object) -> bool:
    return _is_number(value) and value >= 0  # type: ignore[operator]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)
