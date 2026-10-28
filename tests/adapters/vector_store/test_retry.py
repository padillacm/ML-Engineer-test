"""Classification of qdrant-client, httpx and gRPC exceptions, and the backoff schedule."""

import grpc  # type: ignore[import-untyped]
import httpx
import pytest
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException

from app.adapters.vector_store.retry import Failure, Retrier, RetryPolicy, classify
from app.domain.errors import DependencyTimeout, DependencyUnavailable
from tests.adapters.vector_store.flaky_client import http_status


class _RpcError(grpc.RpcError):  # type: ignore[misc]
    def __init__(self, code: grpc.StatusCode) -> None:
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ResponseHandlingException(httpx.ConnectError("x")), Failure.CONNECTION),
        (ResponseHandlingException(httpx.ReadError("x")), Failure.CONNECTION),
        (ResponseHandlingException(httpx.RemoteProtocolError("x")), Failure.CONNECTION),
        (ResponseHandlingException(httpx.ConnectTimeout("x")), Failure.TIMEOUT),
        (ResponseHandlingException(httpx.ReadTimeout("x")), Failure.TIMEOUT),
        (ResponseHandlingException(httpx.PoolTimeout("x")), Failure.TIMEOUT),
        (ResponseHandlingException(ValueError("unparsable response")), Failure.REJECTED),
        (http_status(400), Failure.REJECTED),
        (http_status(404), Failure.REJECTED),
        (http_status(500), Failure.REJECTED),
        (http_status(502), Failure.CONNECTION),
        (http_status(503), Failure.CONNECTION),
        (http_status(504), Failure.CONNECTION),
        (ResourceExhaustedResponse("slow down", 1), Failure.REJECTED),
        (httpx.ConnectError("x"), Failure.CONNECTION),
        (httpx.WriteTimeout("x"), Failure.TIMEOUT),
        (ConnectionRefusedError(), Failure.CONNECTION),
        (TimeoutError(), Failure.TIMEOUT),
        (_RpcError(grpc.StatusCode.UNAVAILABLE), Failure.CONNECTION),
        (_RpcError(grpc.StatusCode.DEADLINE_EXCEEDED), Failure.TIMEOUT),
        (_RpcError(grpc.StatusCode.INVALID_ARGUMENT), Failure.REJECTED),
        (ValueError("bug"), Failure.OTHER),
        (KeyError("bug"), Failure.OTHER),
    ],
)
def test_classify(error: Exception, expected: Failure) -> None:
    assert classify(error) is expected


def test_full_jitter_backoff_doubles_up_to_the_cap() -> None:
    policy = RetryPolicy(attempts=6, backoff_base_s=0.2, backoff_cap_s=2.0)

    assert [policy.delay(n, 1.0) for n in range(6)] == pytest.approx([0.2, 0.4, 0.8, 1.6, 2.0, 2.0])
    assert policy.delay(3, 0.25) == pytest.approx(0.4)
    assert policy.delay(3, 0.0) == 0.0


def test_a_single_attempt_policy_never_sleeps() -> None:
    sleeps: list[float] = []
    retrier = Retrier(RetryPolicy(attempts=1), sleep=sleeps.append)
    calls = [0]

    def operation() -> str:
        calls[0] += 1
        raise ResponseHandlingException(httpx.ConnectError("x"))

    with pytest.raises(DependencyUnavailable):
        retrier.call(operation)
    assert calls == [1]
    assert sleeps == []


def test_the_last_failure_decides_between_timeout_and_unavailable() -> None:
    errors: list[Exception] = [
        ResponseHandlingException(httpx.ReadTimeout("x")),
        ResponseHandlingException(httpx.ConnectError("x")),
    ]
    retrier = Retrier(RetryPolicy(attempts=2), sleep=lambda _: None)

    def operation() -> str:
        raise errors.pop(0)

    with pytest.raises(DependencyUnavailable) as raised:
        retrier.call(operation)
    assert not isinstance(raised.value, DependencyTimeout)


class _Clock:
    """A fake monotonic clock that only ``sleep`` and ``advance`` move."""

    def __init__(self) -> None:
        self.now = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_no_attempt_starts_after_the_deadline() -> None:
    clock = _Clock()
    policy = RetryPolicy(attempts=100, backoff_base_s=0.5, backoff_cap_s=0.5, deadline_s=3.0)
    retrier = Retrier(policy, sleep=clock.sleep, uniform=lambda: 1.0, clock=lambda: clock.now)
    starts: list[float] = []

    def operation() -> str:
        starts.append(clock.now)
        clock.advance(0.5)  # each request takes 0.5 s to fail
        raise ResponseHandlingException(httpx.ConnectError("x"))

    with pytest.raises(DependencyTimeout) as raised:
        retrier.call(operation)

    assert raised.value.code == "vector_store_timeout"
    assert isinstance(raised.value.__cause__, ResponseHandlingException)
    assert starts == pytest.approx([0.0, 1.0, 2.0])  # a 4th try would start at 3.0
    assert all(start < policy.deadline_s for start in starts)


def test_a_call_succeeding_before_the_deadline_is_returned() -> None:
    clock = _Clock()
    retrier = Retrier(
        RetryPolicy(attempts=5, deadline_s=10.0),
        sleep=clock.sleep,
        uniform=lambda: 1.0,
        clock=lambda: clock.now,
    )
    outcomes: list[Exception | str] = [ResponseHandlingException(httpx.ReadTimeout("x")), "ok"]

    def operation() -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert retrier.call(operation) == "ok"


def test_the_attempt_limit_still_applies_within_the_deadline() -> None:
    clock = _Clock()
    retrier = Retrier(
        RetryPolicy(attempts=2, deadline_s=1000.0),
        sleep=clock.sleep,
        uniform=lambda: 0.0,
        clock=lambda: clock.now,
    )
    calls = [0]

    def operation() -> str:
        calls[0] += 1
        raise ResponseHandlingException(httpx.ConnectError("x"))

    with pytest.raises(DependencyUnavailable) as raised:
        retrier.call(operation)
    assert not isinstance(raised.value, DependencyTimeout)
    assert calls == [2]


def test_the_deadline_is_measured_per_call() -> None:
    clock = _Clock()
    retrier = Retrier(
        RetryPolicy(attempts=3, deadline_s=1.0), sleep=clock.sleep, clock=lambda: clock.now
    )
    clock.advance(100.0)  # time passing between calls does not count against the next one

    assert retrier.call(lambda: "ok") == "ok"
