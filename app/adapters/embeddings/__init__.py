"""Text embeddings with a local ``intfloat/multilingual-e5-base`` model."""

from app.adapters.embeddings.config import E5EmbedderConfig
from app.adapters.embeddings.embedder import E5Embedder

__all__ = ["E5Embedder", "E5EmbedderConfig"]
