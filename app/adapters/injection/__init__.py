"""Prompt-injection detection: the Prompt Guard 2 adapter of the ``InjectionDetector`` port."""

from app.adapters.injection.config import InjectionDetectorConfig
from app.adapters.injection.detector import PromptGuardDetector

__all__ = ["InjectionDetectorConfig", "PromptGuardDetector"]
