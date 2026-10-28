"""The OpenAI-compatible client against the ``LLMClient`` contract, over a mock transport."""

from collections.abc import Iterator

import httpx
import pytest

from app.adapters.llm import OpenAICompatibleClient
from app.application.ports import LLMClient
from tests.adapters.llm.helpers import KEY, ScriptedServer, make_client, ok
from tests.application.contracts import LLMClientContract


class TestOpenAICompatibleClientContract(LLMClientContract):
    @pytest.fixture
    def client(self) -> Iterator[LLMClient]:
        with make_client(ScriptedServer([ok()]), api_key=KEY) as client:
            yield client

    @pytest.fixture(
        params=[
            httpx.Response(503),
            httpx.Response(400),
            httpx.Response(200, text="{"),
            httpx.ReadTimeout("slow"),
            httpx.ConnectError("refused"),
        ],
        ids=["503", "400", "malformed", "timeout", "connect"],
    )
    def failing_client(self, request: pytest.FixtureRequest) -> Iterator[LLMClient]:
        with make_client(ScriptedServer([request.param]), api_key=KEY) as client:
            yield client


def test_client_type_satisfies_the_port() -> None:
    assert issubclass(OpenAICompatibleClient, LLMClient)
