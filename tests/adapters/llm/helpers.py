"""Scripted ``httpx.MockTransport`` servers for the LLM client tests."""

import json
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime

import httpx

from app.adapters.llm import LLMClientConfig, OpenAICompatibleClient, Secret

BASE_URL = "http://llm.test:11434/v1"
MODEL = "test-model"
KEY = "sk-very-secret-key-123"
NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

Step = httpx.Response | Exception | Callable[[httpx.Request], httpx.Response]


def ok(content: object = "Paris.") -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}
    )


class ScriptedServer:
    """Answers each request with the next step; the last step repeats once the script runs out.

    An exception step is raised with the request attached, as httpx does for transport errors.
    """

    def __init__(self, steps: Iterable[Step]) -> None:
        self._steps = list(steps)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self._steps[min(len(self.requests), len(self._steps)) - 1]
        if isinstance(step, httpx.RequestError):
            step.request = request
            raise step
        if isinstance(step, Exception):
            raise step
        if isinstance(step, httpx.Response):
            return step
        return step(request)

    @property
    def bodies(self) -> list[dict[str, object]]:
        return [json.loads(request.content) for request in self.requests]


class FakeClock:
    """Monotonic seconds that move only when a test (or the fake sleep) advances them."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingSleep:
    """Records each wait and moves its clock forward by it, as a real sleep would."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.clock = clock if clock is not None else FakeClock()
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)


def slow(response: httpx.Response, clock: FakeClock, seconds: float) -> Step:
    """A step that answers with ``response`` only after ``seconds`` of fake time."""

    def answer(request: httpx.Request) -> httpx.Response:
        clock.advance(seconds)
        return response

    return answer


class TricklingStream(httpx.SyncByteStream):
    """A body that yields one chunk every ``gap`` seconds of fake time."""

    def __init__(self, chunks: Iterable[bytes], clock: FakeClock, gap: float) -> None:
        self._chunks = list(chunks)
        self._clock = clock
        self._gap = gap
        self.sent = 0

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._chunks:
            self._clock.advance(self._gap)
            self.sent += 1
            yield chunk


def make_client(
    server: ScriptedServer,
    *,
    api_key: str | None = None,
    sleep: RecordingSleep | None = None,
    clock: FakeClock | None = None,
    rng_value: float = 0.5,
    **config: object,
) -> OpenAICompatibleClient:
    settings = LLMClientConfig(
        base_url=str(config.pop("base_url", BASE_URL)),
        model=MODEL,
        api_key=Secret(api_key) if api_key is not None else None,
        **config,  # type: ignore[arg-type]
    )
    if sleep is None:
        sleep = RecordingSleep(clock)
    return OpenAICompatibleClient(
        settings,
        transport=httpx.MockTransport(server),
        sleep=sleep,
        rng=lambda: rng_value,
        clock=lambda: NOW,
        monotonic=clock if clock is not None else sleep.clock,
    )
