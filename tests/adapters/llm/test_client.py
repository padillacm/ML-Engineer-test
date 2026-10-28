"""``OpenAICompatibleClient`` against scripted ``httpx.MockTransport`` servers (llm-client spec)."""

import traceback
from datetime import timedelta
from email.utils import format_datetime

import httpx
import pytest

from app.adapters.llm import LLMClientConfig, OpenAICompatibleClient
from app.domain.errors import (
    ConfigurationError,
    DependencyTimeout,
    DependencyUnavailable,
    ServiceError,
)
from app.domain.types import ChatMessage
from tests.adapters.llm.helpers import (
    BASE_URL,
    KEY,
    MODEL,
    NOW,
    FakeClock,
    RecordingSleep,
    ScriptedServer,
    TricklingStream,
    make_client,
    ok,
    slow,
)

PROMPT = "Who signed the confidential merger agreement?"
MESSAGES = [ChatMessage("system", "Answer from the context."), ChatMessage("user", PROMPT)]
BODY_SECRET = "server-side detail that must not leak"  # noqa: S105 - not a credential


def full_text(error: BaseException) -> str:
    """Everything a log line or traceback of the error would show."""
    return repr(error) + str(error) + "".join(traceback.format_exception(error))


# --- Any OpenAI-compatible endpoint can be used -----------------------------------------------


def test_local_model_without_key_sends_no_authorization_header() -> None:
    server = ScriptedServer([ok()])

    assert make_client(server).complete(MESSAGES) == "Paris."
    assert "authorization" not in server.requests[0].headers


def test_hosted_model_with_key_sends_bearer_token() -> None:
    server = ScriptedServer([ok()])

    make_client(server, api_key=KEY).complete(MESSAGES)

    assert server.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in str(server.requests[0].url)


def test_request_shape() -> None:
    server = ScriptedServer([ok()])

    make_client(server, max_tokens=99, temperature=0.2).complete(MESSAGES)

    request = server.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}/chat/completions"
    assert request.headers["content-type"] == "application/json"
    assert server.bodies[0] == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Answer from the context."},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0.2,
        "max_tokens": 99,
        "stream": False,
    }


def test_trailing_slash_in_base_url_gives_a_single_slash() -> None:
    server = ScriptedServer([ok()])

    make_client(server, base_url=f"{BASE_URL}/").complete(MESSAGES)

    assert str(server.requests[0].url) == f"{BASE_URL}/chat/completions"


@pytest.mark.parametrize("content", ["", "   ", "  padded answer \n"])
def test_content_is_returned_as_is(content: str) -> None:
    assert make_client(ScriptedServer([ok(content)])).complete(MESSAGES) == content


def test_timeouts_come_from_the_config() -> None:
    client = make_client(ScriptedServer([ok()]), connect_timeout_s=1.5, read_timeout_s=7.0)
    timeout = client._http.timeout

    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (1.5, 7.0, 7.0, 1.5)


# --- Calls are bounded in time and retried when transient -------------------------------------


def test_rate_limited_once_returns_the_second_reply() -> None:
    server = ScriptedServer([httpx.Response(429), ok("second")])
    sleep = RecordingSleep()

    assert make_client(server, sleep=sleep).complete(MESSAGES) == "second"
    assert len(server.requests) == 2
    assert sleep.calls == [0.5 * 0.5]  # rng 0.5 x min(cap, base * 2**0)


def test_backoff_is_exponential_with_full_jitter_and_capped() -> None:
    server = ScriptedServer([httpx.Response(503)])
    sleep = RecordingSleep()
    client = make_client(
        server, sleep=sleep, rng_value=0.5, attempts=6, backoff_base_s=1.0, backoff_cap_s=5.0
    )

    with pytest.raises(DependencyUnavailable):
        client.complete(MESSAGES)

    assert sleep.calls == [0.5, 1.0, 2.0, 2.5, 2.5]


def test_jitter_uses_the_injected_random_source() -> None:
    sleep = RecordingSleep()
    client = make_client(ScriptedServer([httpx.Response(500), ok()]), sleep=sleep, rng_value=0.0)

    client.complete(MESSAGES)

    assert sleep.calls == [0.0]


@pytest.mark.parametrize(("header", "expected"), [("2", 2.0), ("0.25", 0.25), ("0", 0.0)])
def test_retry_after_seconds_is_honoured(header: str, expected: float) -> None:
    sleep = RecordingSleep()
    server = ScriptedServer([httpx.Response(429, headers={"Retry-After": header}), ok()])

    make_client(server, sleep=sleep).complete(MESSAGES)

    assert sleep.calls == [expected]


def test_retry_after_date_is_honoured() -> None:
    sleep = RecordingSleep()
    when = format_datetime(NOW + timedelta(seconds=7), usegmt=True)
    server = ScriptedServer([httpx.Response(503, headers={"Retry-After": when}), ok()])

    make_client(server, sleep=sleep).complete(MESSAGES)

    assert sleep.calls == [7.0]


def test_retry_after_date_in_the_past_means_no_wait() -> None:
    sleep = RecordingSleep()
    when = format_datetime(NOW - timedelta(minutes=5), usegmt=True)
    server = ScriptedServer([httpx.Response(429, headers={"Retry-After": when}), ok()])

    make_client(server, sleep=sleep).complete(MESSAGES)

    assert sleep.calls == [0.0]


@pytest.mark.parametrize("header", ["3600", "1e9"])
def test_retry_after_seconds_is_capped(header: str) -> None:
    sleep = RecordingSleep()
    server = ScriptedServer([httpx.Response(429, headers={"Retry-After": header}), ok()])

    make_client(server, sleep=sleep, retry_after_cap_s=12.0).complete(MESSAGES)

    assert sleep.calls == [12.0]


def test_retry_after_date_is_capped() -> None:
    sleep = RecordingSleep()
    when = format_datetime(NOW + timedelta(hours=2), usegmt=True)
    server = ScriptedServer([httpx.Response(429, headers={"Retry-After": when}), ok()])

    make_client(server, sleep=sleep).complete(MESSAGES)

    assert sleep.calls == [30.0]


@pytest.mark.parametrize("header", ["soon", "", "nan", "inf"])
def test_unusable_retry_after_falls_back_to_backoff(header: str) -> None:
    sleep = RecordingSleep()
    server = ScriptedServer([httpx.Response(429, headers={"Retry-After": header}), ok()])

    make_client(server, sleep=sleep).complete(MESSAGES)

    assert sleep.calls == [0.25]


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_server_errors_are_retried(status: int) -> None:
    server = ScriptedServer([httpx.Response(status), httpx.Response(status), ok("third")])

    assert make_client(server).complete(MESSAGES) == "third"
    assert len(server.requests) == 3


@pytest.mark.parametrize("status", [500, 502, 503, 429])
def test_persistent_transient_status_fails_after_all_attempts(status: int) -> None:
    server = ScriptedServer([httpx.Response(status, text=BODY_SECRET)])
    sleep = RecordingSleep()

    with pytest.raises(DependencyUnavailable) as info:
        make_client(server, sleep=sleep, attempts=4).complete(MESSAGES)

    assert info.value.code == "llm_unavailable"
    assert f"status {status}" in info.value.message
    assert "4 attempts" in info.value.message
    assert BODY_SECRET not in full_text(info.value)
    assert len(server.requests) == 4
    assert len(sleep.calls) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_other_client_errors_fail_without_retrying(status: int) -> None:
    server = ScriptedServer([httpx.Response(status, text=BODY_SECRET), ok()])
    sleep = RecordingSleep()

    with pytest.raises(DependencyUnavailable) as info:
        make_client(server, sleep=sleep).complete(MESSAGES)

    assert info.value.code == "llm_unavailable"
    assert info.value.message == f"LLM request failed (status {status})"
    assert BODY_SECRET not in full_text(info.value)
    assert len(server.requests) == 1
    assert sleep.calls == []


def test_redirects_are_not_followed() -> None:
    server = ScriptedServer([httpx.Response(307, headers={"Location": "http://elsewhere/"})])

    with pytest.raises(DependencyUnavailable, match="status 307"):
        make_client(server).complete(MESSAGES)

    assert len(server.requests) == 1


@pytest.mark.parametrize(
    "timeout",
    [
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.WriteTimeout("write timed out"),
        httpx.PoolTimeout("pool timed out"),
    ],
)
def test_always_timing_out_fails_with_timeout_after_the_configured_attempts(
    timeout: httpx.TimeoutException,
) -> None:
    server = ScriptedServer([timeout])
    sleep = RecordingSleep()

    with pytest.raises(DependencyTimeout) as info:
        make_client(server, sleep=sleep, attempts=3).complete(MESSAGES)

    assert info.value.code == "llm_timeout"
    assert info.value.message == "LLM request timed out after 3 attempts"
    assert info.value.__cause__ is timeout
    assert len(server.requests) == 3
    assert len(sleep.calls) == 2


def test_timeout_then_success_returns_the_reply() -> None:
    server = ScriptedServer([httpx.ReadTimeout("slow"), ok("late")])

    assert make_client(server).complete(MESSAGES) == "late"


def test_timeout_then_server_error_reports_the_last_failure() -> None:
    server = ScriptedServer([httpx.ReadTimeout("slow"), httpx.Response(503)])

    with pytest.raises(DependencyUnavailable, match="status 503"):
        make_client(server, attempts=2).complete(MESSAGES)


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadError("connection reset"),
        httpx.WriteError("broken pipe"),
        httpx.RemoteProtocolError("server disconnected"),
    ],
)
def test_connection_errors_are_retried_then_unavailable(error: httpx.TransportError) -> None:
    server = ScriptedServer([error])
    sleep = RecordingSleep()

    with pytest.raises(DependencyUnavailable) as info:
        make_client(server, sleep=sleep, attempts=3).complete(MESSAGES)

    assert info.value.code == "llm_unavailable"
    assert info.value.message == (
        f"LLM request failed after 3 attempts (connection error: {type(error).__name__})"
    )
    assert info.value.__cause__ is error
    assert len(server.requests) == 3
    assert len(sleep.calls) == 2


def test_connection_error_then_success_returns_the_reply() -> None:
    server = ScriptedServer([httpx.ConnectError("refused"), ok("back")])

    assert make_client(server).complete(MESSAGES) == "back"


@pytest.mark.parametrize(
    "error", [httpx.UnsupportedProtocol("bad scheme"), httpx.LocalProtocolError("bad header")]
)
def test_request_errors_that_cannot_heal_fail_at_once(error: httpx.TransportError) -> None:
    server = ScriptedServer([error, ok()])

    with pytest.raises(DependencyUnavailable) as info:
        make_client(server).complete(MESSAGES)

    assert info.value.code == "llm_unavailable"
    assert len(server.requests) == 1


def test_single_attempt_does_not_retry_or_sleep() -> None:
    server = ScriptedServer([httpx.Response(503), ok()])
    sleep = RecordingSleep()

    with pytest.raises(DependencyUnavailable, match="after 1 attempts"):
        make_client(server, sleep=sleep, attempts=1).complete(MESSAGES)

    assert sleep.calls == []


# --- Overall deadline -------------------------------------------------------------------------

DEADLINE = "LLM request exceeded its deadline"


def timeout_of(request: httpx.Request) -> dict[str, float]:
    timeout: dict[str, float] = request.extensions["timeout"]
    return timeout


def assert_deadline_error(error: DependencyTimeout) -> None:
    assert error.code == "llm_timeout"
    assert error.message == DEADLINE


def test_per_attempt_timeouts_are_cut_to_the_time_left() -> None:
    clock = FakeClock()
    sleep = RecordingSleep(clock)
    server = ScriptedServer([slow(httpx.Response(503), clock, 7.0), ok()])
    client = make_client(
        server, sleep=sleep, connect_timeout_s=5.0, read_timeout_s=60.0, total_timeout_s=10.0
    )

    assert client.complete(MESSAGES) == "Paris."

    first, second = (timeout_of(request) for request in server.requests)
    assert first == {"connect": 5.0, "read": 10.0, "write": 10.0, "pool": 5.0}
    left = 10.0 - 7.0 - 0.25
    assert second == pytest.approx({"connect": left, "read": left, "write": left, "pool": left})


def test_configured_timeouts_apply_while_the_deadline_is_far() -> None:
    server = ScriptedServer([ok()])

    make_client(server, connect_timeout_s=1.5, read_timeout_s=7.0).complete(MESSAGES)

    assert timeout_of(server.requests[0]) == {
        "connect": 1.5,
        "read": 7.0,
        "write": 7.0,
        "pool": 1.5,
    }


def test_slow_failures_stop_at_the_deadline_whatever_the_attempts_left() -> None:
    clock = FakeClock()
    sleep = RecordingSleep(clock)
    server = ScriptedServer([slow(httpx.Response(503), clock, 6.0)])
    client = make_client(server, sleep=sleep, attempts=10, total_timeout_s=10.0)

    with pytest.raises(DependencyTimeout) as info:
        client.complete(MESSAGES)

    assert_deadline_error(info.value)
    assert len(server.requests) == 2
    assert sleep.calls == [0.25]


def test_slow_timeouts_stop_at_the_deadline_and_chain_the_last_timeout() -> None:
    clock = FakeClock()
    timeout = httpx.ReadTimeout("slow")

    def times_out(request: httpx.Request) -> httpx.Response:
        clock.advance(5.0)
        raise timeout

    server = ScriptedServer([times_out])
    client = make_client(server, sleep=RecordingSleep(clock), attempts=10, total_timeout_s=10.0)

    with pytest.raises(DependencyTimeout) as info:
        client.complete(MESSAGES)

    assert_deadline_error(info.value)
    assert info.value.__cause__ is timeout
    assert len(server.requests) == 2


def test_reply_arriving_after_the_deadline_is_dropped() -> None:
    clock = FakeClock()
    server = ScriptedServer([slow(ok(), clock, 11.0), ok("too late to ask")])

    with pytest.raises(DependencyTimeout) as info:
        make_client(server, sleep=RecordingSleep(clock), total_timeout_s=10.0).complete(MESSAGES)

    assert_deadline_error(info.value)
    assert len(server.requests) == 1


CHUNKS = [b'{"choices": ', b'[{"message": ', b'{"content": ', b'"Paris."', b"}}]}"]


def test_trickling_body_is_abandoned_at_the_deadline() -> None:
    clock = FakeClock()
    stream = TricklingStream(CHUNKS, clock, gap=4.0)
    server = ScriptedServer([httpx.Response(200, stream=stream)])

    with pytest.raises(DependencyTimeout) as info:
        make_client(server, sleep=RecordingSleep(clock), total_timeout_s=10.0).complete(MESSAGES)

    assert_deadline_error(info.value)
    assert stream.sent == 3
    assert len(server.requests) == 1


def test_trickling_body_within_the_deadline_is_returned() -> None:
    clock = FakeClock()
    server = ScriptedServer([httpx.Response(200, stream=TricklingStream(CHUNKS, clock, gap=1.0))])
    client = make_client(server, sleep=RecordingSleep(clock), total_timeout_s=10.0)

    assert client.complete(MESSAGES) == "Paris."


def test_retry_after_beyond_the_deadline_fails_without_waiting() -> None:
    clock = FakeClock()
    sleep = RecordingSleep(clock)
    server = ScriptedServer([httpx.Response(429, headers={"Retry-After": "20"}), ok()])

    with pytest.raises(DependencyTimeout) as info:
        make_client(server, sleep=sleep, total_timeout_s=10.0).complete(MESSAGES)

    assert_deadline_error(info.value)
    assert sleep.calls == []
    assert len(server.requests) == 1


def test_repeated_retry_after_stops_before_the_deadline() -> None:
    clock = FakeClock()
    sleep = RecordingSleep(clock)
    server = ScriptedServer([httpx.Response(429, headers={"Retry-After": "4"})])

    with pytest.raises(DependencyTimeout) as info:
        make_client(server, sleep=sleep, attempts=10, total_timeout_s=10.0).complete(MESSAGES)

    assert_deadline_error(info.value)
    assert sleep.calls == [4.0, 4.0]
    assert len(server.requests) == 3


@pytest.mark.parametrize(
    ("latency", "header"),
    [(0.0, "29"), (0.5, "3"), (2.9, None), (3.0, "1"), (9.9, None), (0.0, None), (0.01, "0")],
)
def test_worst_case_is_bounded_by_the_deadline(latency: float, header: str | None) -> None:
    """No attempt starts and no wait ends after the deadline; the overrun is one slow answer."""
    clock = FakeClock()
    start = clock.now
    sleep = RecordingSleep(clock)
    starts: list[float] = []
    headers = {"Retry-After": header} if header is not None else {}

    def answer(request: httpx.Request) -> httpx.Response:
        starts.append(clock.now)
        clock.advance(latency)
        return httpx.Response(503, headers=headers)

    client = make_client(
        ScriptedServer([answer]),
        sleep=sleep,
        attempts=1_000_000,
        backoff_base_s=1.0,
        backoff_cap_s=4.0,
        total_timeout_s=10.0,
    )
    with pytest.raises(DependencyTimeout, match=DEADLINE):
        client.complete(MESSAGES)

    deadline = start + 10.0
    assert starts
    assert all(at < deadline for at in starts)
    assert clock.now - start <= 10.0 + latency
    assert sum(sleep.calls) < 10.0


def test_deadline_counts_from_each_call() -> None:
    clock = FakeClock()
    server = ScriptedServer([slow(ok("a"), clock, 6.0), slow(ok("b"), clock, 6.0)])
    client = make_client(server, sleep=RecordingSleep(clock), total_timeout_s=10.0)

    assert [client.complete(MESSAGES), client.complete(MESSAGES)] == ["a", "b"]


# --- Unusable keys never reach httpx ----------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["sk-" + "".join(map(chr, (0x43A, 0x43B, 0x44E, 0x447))), "sk-key\r\nX-Injected: 1"]
)
def test_unusable_key_fails_at_configuration_before_any_request(key: str) -> None:
    server = ScriptedServer([ok()])

    with pytest.raises(ConfigurationError) as info:
        make_client(server, api_key=key)

    assert not isinstance(info.value, UnicodeError)
    assert key not in full_text(info.value)
    assert server.requests == []


# --- Malformed responses ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json at all"),
        httpx.Response(200, content=b"\xff\xfe\x00garbage"),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{}]}),
        httpx.Response(200, json={"choices": [{"message": None}]}),
        httpx.Response(200, json={"choices": [{"message": {"role": "assistant"}}]}),
        ok(None),
        ok(42),
        ok(["Paris"]),
        httpx.Response(204),
    ],
)
def test_malformed_success_response_is_unavailable(response: httpx.Response) -> None:
    server = ScriptedServer([response, ok()])

    with pytest.raises(DependencyUnavailable) as info:
        make_client(server).complete(MESSAGES)

    assert info.value.code == "llm_unavailable"
    assert "malformed" in info.value.message
    assert len(server.requests) == 1


# --- Errors never expose secrets or prompts ---------------------------------------------------


@pytest.mark.parametrize(
    "step",
    [
        httpx.Response(401, text=f"invalid key {KEY}"),
        httpx.Response(503, text=f"echo {PROMPT}"),
        httpx.Response(200, text=f"{KEY} {PROMPT}"),
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("refused"),
    ],
)
def test_key_and_prompt_never_leak(step: httpx.Response | Exception) -> None:
    client = make_client(ScriptedServer([step]), api_key=KEY)

    with pytest.raises(ServiceError) as info:
        client.complete(MESSAGES)

    assert isinstance(info.value, DependencyUnavailable | DependencyTimeout)
    for text in (str(info.value), repr(info.value), info.value.message):
        assert KEY not in text
        assert PROMPT not in text
    assert KEY not in repr(client)
    assert KEY not in str(client)
    assert KEY not in repr(client.config)


def test_cause_chain_does_not_carry_the_key() -> None:
    """The chained httpx error names the URL at most, and the key is never in the URL."""
    client = make_client(ScriptedServer([httpx.ConnectError("refused")]), api_key=KEY)

    with pytest.raises(DependencyUnavailable) as info:
        client.complete(MESSAGES)

    cause = info.value.__cause__
    assert isinstance(cause, httpx.ConnectError)
    assert KEY not in str(cause.request.url)
    assert KEY not in full_text(info.value)


def test_repr_names_endpoint_and_model_only() -> None:
    client = make_client(ScriptedServer([ok()]), api_key=KEY)

    assert repr(client) == (
        f"OpenAICompatibleClient(base_url='{BASE_URL}', model='{MODEL}', api_key=set)"
    )


# --- Lifecycle --------------------------------------------------------------------------------


def test_close_releases_the_http_client() -> None:
    client = make_client(ScriptedServer([ok()]))

    client.close()

    assert client._http.is_closed


def test_context_manager_closes_on_exit() -> None:
    server = ScriptedServer([ok()])

    with make_client(server) as client:
        assert client.complete(MESSAGES) == "Paris."

    assert client._http.is_closed


def test_one_http_client_serves_every_call() -> None:
    server = ScriptedServer([ok("a"), ok("b")])
    client = make_client(server)
    http = client._http

    assert [client.complete(MESSAGES), client.complete(MESSAGES)] == ["a", "b"]
    assert client._http is http


def test_default_clock_sleep_and_rng_are_usable() -> None:
    config = LLMClientConfig(BASE_URL, MODEL, backoff_base_s=0.0, backoff_cap_s=0.0)
    server = ScriptedServer([httpx.Response(503), ok()])

    with OpenAICompatibleClient(config, transport=httpx.MockTransport(server)) as client:
        assert client.complete(MESSAGES) == "Paris."
