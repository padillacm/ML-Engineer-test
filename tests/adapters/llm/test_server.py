"""One real completion against ``LLM_BASE_URL``/``LLM_MODEL`` (optional, marker ``llm_server``).

Skipped unless ``LLM_BASE_URL`` is set; ``LLM_API_KEY`` is sent when present.
"""

import os

import pytest

from app.adapters.llm import LLMClientConfig, OpenAICompatibleClient, Secret
from app.domain.types import ChatMessage

pytestmark = [
    pytest.mark.llm_server,
    pytest.mark.skipif(not os.environ.get("LLM_BASE_URL"), reason="LLM_BASE_URL is not set"),
]


def test_real_completion() -> None:
    key = os.environ.get("LLM_API_KEY")
    config = LLMClientConfig(
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ.get("LLM_MODEL", "qwen2.5:3b-instruct"),
        api_key=Secret(key) if key else None,
        max_tokens=16,
    )

    with OpenAICompatibleClient(config) as client:
        reply = client.complete(
            [
                ChatMessage("system", "Reply with one word."),
                ChatMessage("user", "What is the capital of France?"),
            ]
        )

    assert isinstance(reply, str)
    assert reply.strip()
