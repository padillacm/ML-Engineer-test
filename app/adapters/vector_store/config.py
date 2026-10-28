"""Settings of the Qdrant vector store adapter."""

import re
from dataclasses import dataclass
from math import isfinite
from urllib.parse import urlsplit

from app.domain.errors import ConfigurationError

_PREFIX_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


@dataclass(frozen=True, slots=True, repr=False)
class QdrantStoreConfig:
    """Settings of ``QdrantVectorStore``.

    Exactly one of ``url`` (an ``http``/``https`` Qdrant server, the production and compose path)
    and ``location`` (``":memory:"``, qdrant-client's embedded local mode, for host tests) must be
    set. ``timeout_s`` bounds each HTTP call (qdrant-client rounds it up to whole seconds).
    ``attempts`` is the total number of tries of one call on connection errors, timeouts and HTTP
    502/503/504, with full-jitter exponential backoff
    ``random() * min(backoff_cap_s, backoff_base_s * 2**n)`` between them. ``deadline_s`` bounds
    the tries of one call: no new try starts once ``deadline_s`` has passed since the first one, so
    a call fails no later than ``deadline_s`` plus one ``timeout_s``. ``upsert_batch`` is the
    number of points per upsert request. ``api_key`` is never shown by ``repr``. Invalid values
    raise ``ConfigurationError``.
    """

    url: str | None = None
    location: str | None = None
    collection_prefix: str = "chunks"
    timeout_s: float = 10.0
    attempts: int = 3
    backoff_base_s: float = 0.2
    backoff_cap_s: float = 2.0
    deadline_s: float = 20.0
    upsert_batch: int = 128
    api_key: str | None = None

    def __post_init__(self) -> None:
        _require((self.url is None) != (self.location is None), "set exactly one of url, location")
        if self.url is not None:
            _require(_is_http_url(self.url), "url must be an http(s) URL with a host")
        if self.location is not None:
            _require(self.location == ":memory:", "location must be ':memory:'")
        _require(
            isinstance(self.collection_prefix, str)
            and _PREFIX_PATTERN.fullmatch(self.collection_prefix) is not None,
            "collection_prefix must be 1-64 letters, digits, '_' or '-', starting alphanumeric",
        )
        _require(_is_positive_number(self.timeout_s), "timeout_s must be a finite number > 0")
        _require(_is_positive_int(self.attempts), "attempts must be a positive integer")
        _require(
            _is_non_negative_number(self.backoff_base_s),
            "backoff_base_s must be a finite number >= 0",
        )
        _require(
            _is_non_negative_number(self.backoff_cap_s),
            "backoff_cap_s must be a finite number >= 0",
        )
        _require(_is_positive_number(self.deadline_s), "deadline_s must be a finite number > 0")
        _require(_is_positive_int(self.upsert_batch), "upsert_batch must be a positive integer")
        _require(
            self.api_key is None or (isinstance(self.api_key, str) and bool(self.api_key)),
            "api_key must be None or a non-empty string",
        )

    def __repr__(self) -> str:
        key = None if self.api_key is None else "***"
        return (
            f"QdrantStoreConfig(url={self.url!r}, location={self.location!r}, "
            f"collection_prefix={self.collection_prefix!r}, timeout_s={self.timeout_s!r}, "
            f"attempts={self.attempts!r}, backoff_base_s={self.backoff_base_s!r}, "
            f"backoff_cap_s={self.backoff_cap_s!r}, deadline_s={self.deadline_s!r}, "
            f"upsert_batch={self.upsert_batch!r}, "
            f"api_key={key!r})"
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def _is_http_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parts = urlsplit(value)
    return parts.scheme in {"http", "https"} and bool(parts.hostname)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_positive_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(value)
        and value > 0
    )


def _is_non_negative_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(value)
        and value >= 0
    )
