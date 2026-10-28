"""Chat completions against any OpenAI-compatible server (Ollama or hosted providers)."""

from app.adapters.llm.config import LLMClientConfig
from app.adapters.llm.openai_compatible import OpenAICompatibleClient
from app.adapters.llm.secret import Secret

__all__ = ["LLMClientConfig", "OpenAICompatibleClient", "Secret"]
