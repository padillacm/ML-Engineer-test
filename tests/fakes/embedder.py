"""Deterministic in-memory embedder based on hashed bag-of-words vectors."""

import hashlib
import math
import re
from collections.abc import Sequence

from app.domain.types import EmbeddingIdentity

_WORD = re.compile(r"\w+")


class FakeEmbedder:
    """Hashes each lowercase word to a vector slot and L2-normalises the counts.

    Texts that share words get a higher cosine than texts that do not; passages and queries
    live in the same space. Counts are non-negative, so cosines are always within 0..1.
    """

    def __init__(self, identity: EmbeddingIdentity | None = None) -> None:
        self._identity = identity or EmbeddingIdentity(
            model_id="fake-bag-of-words",
            dimension=64,
            prefix_scheme="none",
            normalized=True,
            metric="cosine",
        )
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.passage_calls.append(list(texts))
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._identity.dimension
        for word in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % len(vector)] += 1.0
        if not self._identity.normalized:
            return vector
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector
