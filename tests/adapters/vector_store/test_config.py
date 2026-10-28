"""Validation and masking of ``QdrantStoreConfig``."""

from typing import Any

import pytest

from app.adapters.vector_store import QdrantStoreConfig
from app.domain.errors import ConfigurationError


def test_defaults() -> None:
    config = QdrantStoreConfig(url="http://qdrant:6333")

    assert (config.collection_prefix, config.timeout_s, config.attempts) == ("chunks", 10.0, 3)
    assert (config.backoff_base_s, config.backoff_cap_s, config.upsert_batch) == (0.2, 2.0, 128)
    assert config.deadline_s == 20.0
    assert config.api_key is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"url": "http://qdrant:6333", "location": ":memory:"},
        {"url": "qdrant:6333"},
        {"url": "ftp://qdrant"},
        {"url": "http://"},
        {"location": "qdrant-data"},
        {"location": ":memory:", "collection_prefix": ""},
        {"location": ":memory:", "collection_prefix": "has space"},
        {"location": ":memory:", "collection_prefix": "_leading"},
        {"location": ":memory:", "timeout_s": 0},
        {"location": ":memory:", "timeout_s": float("inf")},
        {"location": ":memory:", "timeout_s": True},
        {"location": ":memory:", "attempts": 0},
        {"location": ":memory:", "attempts": 1.5},
        {"location": ":memory:", "backoff_base_s": -0.1},
        {"location": ":memory:", "backoff_cap_s": float("nan")},
        {"location": ":memory:", "deadline_s": 0},
        {"location": ":memory:", "deadline_s": -1.0},
        {"location": ":memory:", "deadline_s": float("inf")},
        {"location": ":memory:", "deadline_s": "20"},
        {"location": ":memory:", "upsert_batch": 0},
        {"location": ":memory:", "api_key": ""},
    ],
)
def test_invalid_values_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        QdrantStoreConfig(**kwargs)


def test_repr_masks_the_api_key() -> None:
    config = QdrantStoreConfig(url="https://qdrant.example:6333", api_key="s3cr3t-key")

    assert "s3cr3t-key" not in repr(config)
    assert "api_key='***'" in repr(config)
    assert "api_key=None" in repr(QdrantStoreConfig(location=":memory:"))
