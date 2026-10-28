"""Person-name recognition with a local ``transformers`` token-classification model."""

from app.adapters.ner.config import TransformersNerConfig
from app.adapters.ner.recognizer import TransformersNameRecognizer

__all__ = ["TransformersNameRecognizer", "TransformersNerConfig"]
