"""Request options, provider error codes and cut-off replies (llm-client spec, add-deployment D9).

The request adapts to the model's parameters (``reasoning_effort``, ``max_tokens_field``,
``send_temperature``); a rejected request names the provider's error code when it is a short
identifier; an empty reply cut at the token limit is logged without its content.
"""

import json
import logging
from collections.abc import Iterator

import httpx
import pytest

from app.domain.errors import DependencyTimeout, DependencyUnavailable
from app.domain.types import ChatMessage
from app.observability import JsonFormatter
from tests.adapters.llm.helpers import (
    KEY,
    MODEL,
    FakeClock,
    RecordingSleep,
    ScriptedServer,
    TricklingStream,
    make_client,
    ok,
)

PROMPT = "Who signed the confidential merger agreement?"
MESSAGES = [ChatMessage("system", "Answer from the context."), ChatMessage("user", PROMPT)]
PROVIDER_MESSAGE = "Unsupported parameter: 'max_tokens' is not supported with this model."
LOGGER = "app.adapters.llm.openai_compatible"
EXPECTED_MESSAGES = [
    {"role": "system", "content": "Answer from the context."},
    {"role": "user", "content": PROMPT},
]


def openai_error(
    status: int, code: object = "unsupported_parameter", **error: object
) -> httpx.Response:
    fields: dict[str, object] = {
        "message": PROVIDER_MESSAGE,
        "type": "invalid_request_error",
        "param": None,
        "code": code,
        **error,
    }
    return httpx.Response(status, json={"error": fields})


def reply(content: object, finish_reason: object) -> httpx.Response:
    choice: dict[str, object] = {
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "finish_reason": finish_reason,
    }
    return httpx.Response(200, json={"choices": [choice]})


def rejection(response: httpx.Response, *, api_key: str | None = None) -> DependencyUnavailable:
    server = ScriptedServer([response, ok()])
    sleep = RecordingSleep()

    with pytest.raises(DependencyUnavailable) as info:
        make_client(server, api_key=api_key, sleep=sleep).complete(MESSAGES)

    assert info.value.code == "llm_unavailable"
    assert len(server.requests) == 1
    assert sleep.calls == []
    return info.value


def records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == LOGGER]


# --- The request adapts to the model's parameters --------------------------------------------


def test_default_request_is_unchanged_byte_for_byte() -> None:
    server = ScriptedServer([ok()])

    make_client(server).complete(MESSAGES)

    # The body the client sent before the options existed, encoded the way httpx encodes it.
    before = {
        "model": MODEL,
        "messages": EXPECTED_MESSAGES,
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False,
    }
    assert server.requests[0].content == httpx.Request("POST", "http://x", json=before).content


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh"])
def test_reasoning_effort_is_sent_when_set(effort: str) -> None:
    server = ScriptedServer([ok()])

    make_client(server, reasoning_effort=effort).complete(MESSAGES)

    body = server.bodies[0]
    assert body["reasoning_effort"] == effort
    assert (body["temperature"], body["max_tokens"]) == (0.0, 512)


def test_thinking_turned_off_on_a_local_server() -> None:
    server = ScriptedServer([ok()])

    make_client(server, reasoning_effort="none").complete(MESSAGES)

    assert server.bodies[0] == {
        "model": MODEL,
        "messages": EXPECTED_MESSAGES,
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False,
        "reasoning_effort": "none",
    }


def test_hosted_reasoning_model() -> None:
    server = ScriptedServer([ok()])

    make_client(
        server, max_tokens=300, max_tokens_field="max_completion_tokens", send_temperature=False
    ).complete(MESSAGES)

    assert server.bodies[0] == {
        "model": MODEL,
        "messages": EXPECTED_MESSAGES,
        "max_completion_tokens": 300,
        "stream": False,
    }


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("send_temperature", [True, False])
@pytest.mark.parametrize("effort", [None, "low"])
def test_every_combination_names_the_limit_once(
    field: str, send_temperature: bool, effort: str | None
) -> None:
    server = ScriptedServer([ok()])

    make_client(
        server,
        max_tokens=64,
        temperature=0.3,
        max_tokens_field=field,
        send_temperature=send_temperature,
        reasoning_effort=effort,
    ).complete(MESSAGES)

    body = server.bodies[0]
    expected_keys = {"model", "messages", field, "stream"}
    if send_temperature:
        expected_keys.add("temperature")
        assert body["temperature"] == 0.3
    if effort is not None:
        expected_keys.add("reasoning_effort")
        assert body["reasoning_effort"] == effort
    assert set(body) == expected_keys
    assert body[field] == 64


def test_options_apply_to_every_attempt() -> None:
    server = ScriptedServer([httpx.Response(503), ok()])

    make_client(server, reasoning_effort="none", max_tokens_field="max_completion_tokens").complete(
        MESSAGES
    )

    assert len(server.bodies) == 2
    assert server.bodies[0] == server.bodies[1]
    assert server.bodies[1]["reasoning_effort"] == "none"


# --- Rejected requests name the provider's error code -----------------------------------------


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "unsupported_parameter"),
        (400, "unsupported_value"),
        (401, "invalid_api_key"),
        (401, "expired_secret_key"),
        (404, "model_not_found"),
        (422, "Invalid.Value-2"),
        (400, "x" * 64),
    ],
)
def test_rejection_names_the_provider_code(status: int, code: str) -> None:
    error = rejection(openai_error(status, code))

    assert error.message == f"LLM request failed (status {status}, {code})"
    assert PROVIDER_MESSAGE not in str(error)
    assert "invalid_request_error" not in error.message


def test_unsupported_parameter_names_only_status_and_code() -> None:
    error = rejection(openai_error(400))

    assert error.message == "LLM request failed (status 400, unsupported_parameter)"
    for text in (str(error), repr(error), error.message):
        assert PROVIDER_MESSAGE not in text
        assert "max_tokens" not in text


@pytest.mark.parametrize("code", [None, "", 400, 1.5, True, ["unsupported"], {"id": "x"}])
def test_type_is_used_when_the_code_is_missing_or_unusable(code: object) -> None:
    assert rejection(openai_error(400, code)).message == (
        "LLM request failed (status 400, invalid_request_error)"
    )


def test_type_is_used_when_the_code_is_absent() -> None:
    response = httpx.Response(400, json={"error": {"type": "invalid_request_error"}})

    assert rejection(response).message == "LLM request failed (status 400, invalid_request_error)"


@pytest.mark.parametrize(
    "code",
    [
        "has space",
        "line\nbreak",
        "trailing\n",
        "x" * 65,
        "quote'd",
        "semi;colon",
        "Ünïcode",
        "{json}",
        "a/b",
        "<script>",
    ],
)
def test_unsafe_code_is_left_out(code: str) -> None:
    response = openai_error(400, code, type=code)

    error = rejection(response)

    assert error.message == "LLM request failed (status 400)"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, text="not json at all"),
        httpx.Response(400, text=PROVIDER_MESSAGE),
        httpx.Response(400, content=b"\xff\xfe\x00garbage"),
        httpx.Response(400),
        httpx.Response(400, json=None),
        httpx.Response(400, json=[{"error": {"code": "unsupported_parameter"}}]),
        httpx.Response(400, json="unsupported_parameter"),
        httpx.Response(400, json={"code": "unsupported_parameter"}),
        httpx.Response(400, json={"error": "unsupported_parameter"}),
        httpx.Response(400, json={"error": ["unsupported_parameter"]}),
        httpx.Response(400, json={"error": {"detail": {"code": "unsupported_parameter"}}}),
        httpx.Response(400, json={"error": {"code": {"code": "unsupported_parameter"}}}),
        httpx.Response(400, json={"error": {"message": "unsupported_parameter"}}),
        httpx.Response(400, json={"error": None}),
    ],
)
def test_bodies_without_a_usable_code_give_the_status_only(response: httpx.Response) -> None:
    assert rejection(response).message == "LLM request failed (status 400)"


def test_huge_body_is_not_read_past_the_limit() -> None:
    padding = " " * (64 * 1024)
    body = json.dumps({"error": {"code": "unsupported_parameter"}}) + padding
    response = httpx.Response(400, content=body.encode())

    assert rejection(response).message == "LLM request failed (status 400)"


def test_body_within_the_limit_is_parsed() -> None:
    body = json.dumps({"error": {"code": "unsupported_parameter", "message": "m" * 15_000}})
    assert len(body) <= 16 * 1024

    assert rejection(httpx.Response(400, content=body.encode())).message == (
        "LLM request failed (status 400, unsupported_parameter)"
    )


def test_huge_streamed_body_stops_reading_at_the_limit() -> None:
    clock = FakeClock()
    chunks = [b'{"error": {"code": "unsupported_parameter", "message": "'] + [b"m" * 4096] * 64
    stream = TricklingStream(chunks, clock, gap=0.0)
    server = ScriptedServer([httpx.Response(400, stream=stream)])

    with pytest.raises(DependencyUnavailable) as info:
        make_client(server, sleep=RecordingSleep(clock)).complete(MESSAGES)

    assert info.value.message == "LLM request failed (status 400)"
    assert stream.sent <= 6


def test_trickling_error_body_stops_at_the_deadline() -> None:
    clock = FakeClock()
    chunks = [b'{"error": ', b'{"code": ', b'"unsupported_parameter"', b"}}"]
    stream = TricklingStream(chunks, clock, gap=4.0)
    server = ScriptedServer([httpx.Response(400, stream=stream)])

    with pytest.raises(DependencyUnavailable) as info:
        make_client(server, sleep=RecordingSleep(clock), total_timeout_s=10.0).complete(MESSAGES)

    assert info.value.message == "LLM request failed (status 400)"
    assert stream.sent == 3
    assert len(server.requests) == 1


class _BrokenStream(httpx.SyncByteStream):
    def __init__(self, error: Exception) -> None:
        self._error = error

    def __iter__(self) -> Iterator[bytes]:
        yield b'{"error": '
        raise self._error


@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout("slow body"), httpx.ReadError("reset"), httpx.DecodingError("gz")]
)
def test_unreadable_error_body_still_fails_at_once_with_the_status(error: Exception) -> None:
    server = ScriptedServer([httpx.Response(400, stream=_BrokenStream(error)), ok()])
    sleep = RecordingSleep()

    with pytest.raises(DependencyUnavailable) as info:
        make_client(server, sleep=sleep).complete(MESSAGES)

    assert info.value.message == "LLM request failed (status 400)"
    assert len(server.requests) == 1
    assert sleep.calls == []


def test_rate_limit_and_server_errors_are_still_retried_without_a_code() -> None:
    server = ScriptedServer([openai_error(429, "rate_limit_exceeded"), openai_error(500, "boom")])

    with pytest.raises(DependencyUnavailable) as info:
        make_client(server, attempts=2).complete(MESSAGES)

    assert info.value.message == "LLM request failed after 2 attempts (status 500)"
    assert "boom" not in info.value.message
    assert "rate_limit_exceeded" not in info.value.message


def test_redirect_reports_the_status_only() -> None:
    response = httpx.Response(
        307, headers={"Location": "http://elsewhere/"}, json={"error": {"code": "moved"}}
    )

    assert rejection(response).message == "LLM request failed (status 307)"


@pytest.mark.parametrize(
    "code",
    [KEY, f"x{KEY}x", KEY[:8], KEY[3:20], KEY[-10:]],
)
def test_code_carrying_the_key_is_left_out(code: str) -> None:
    error = rejection(openai_error(401, code, type=code), api_key=KEY)

    assert error.message == "LLM request failed (status 401)"
    assert code not in str(error)


def test_short_code_that_happens_to_occur_in_the_key_is_kept() -> None:
    # Fragments shorter than 8 characters say nothing about the key ("key" is in KEY).
    error = rejection(openai_error(401, "key"), api_key=KEY)

    assert error.message == "LLM request failed (status 401, key)"


def test_code_is_kept_when_a_key_is_set_and_unrelated() -> None:
    error = rejection(openai_error(401, "invalid_api_key"), api_key=KEY)

    assert error.message == "LLM request failed (status 401, invalid_api_key)"


def test_rejection_is_logged_with_the_same_fixed_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)

    rejection(openai_error(400), api_key=KEY)

    (record,) = records(caplog)
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "LLM request failed (status 400, unsupported_parameter)"
    line = JsonFormatter().format(record)
    for secret in (KEY, PROMPT, PROVIDER_MESSAGE):
        assert secret not in line


def test_rejection_without_a_code_is_logged_with_the_status(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)

    rejection(httpx.Response(403, text=f"forbidden {KEY} {PROMPT}"), api_key=KEY)

    (record,) = records(caplog)
    assert record.getMessage() == "LLM request failed (status 403)"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": {"code": KEY, "message": f"bad key {KEY}"}}),
        httpx.Response(400, json={"error": {"code": PROMPT, "type": PROMPT}}),
        httpx.Response(400, json={"error": {"code": "unsupported_parameter", "message": PROMPT}}),
    ],
)
def test_key_and_prompt_never_reach_the_error_or_the_log(
    response: httpx.Response, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    error = rejection(response, api_key=KEY)

    text = repr(error) + str(error) + error.message
    text += "\n".join(JsonFormatter().format(record) for record in caplog.records)
    assert KEY not in text
    assert PROMPT not in text


# --- Empty replies at the token limit are logged ---------------------------------------------


@pytest.mark.parametrize("content", ["", "   ", "\n\t "])
def test_empty_reply_at_the_token_limit_is_logged_and_returned(
    content: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    server = ScriptedServer([reply(content, "length")])

    assert make_client(server, max_tokens=512).complete(MESSAGES) == content

    (record,) = records(caplog)
    assert record.levelno == logging.WARNING
    assert (
        record.getMessage() == "LLM reply was cut at the token limit (512 tokens) with no content"
    )
    line = json.loads(JsonFormatter().format(record))
    assert (line["model"], line["max_tokens"]) == (MODEL, 512)


def test_cut_off_warning_carries_no_content_or_prompt(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    reasoning = "The user asks who signed; the context says Jane Roe signed it."
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": " ", "reasoning": reasoning},
                    "finish_reason": "length",
                }
            ]
        },
    )

    make_client(ScriptedServer([response]), api_key=KEY).complete(MESSAGES)

    (record,) = records(caplog)
    line = JsonFormatter().format(record)
    for secret in (reasoning, "Jane Roe", PROMPT, KEY):
        assert secret not in line
    # Only the configured model name and token limit are added to the record.
    extras = set(record.__dict__) - set(logging.makeLogRecord({}).__dict__) - {"message"}
    assert extras == {"model", "max_tokens"}


@pytest.mark.parametrize(
    ("content", "finish_reason"),
    [
        ("Paris.", "length"),
        ("", "stop"),
        ("", None),
        ("Paris.", "stop"),
        ("", "content_filter"),
        ("", ["length"]),
    ],
)
def test_other_replies_are_not_logged(
    content: str, finish_reason: object, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    result = make_client(ScriptedServer([reply(content, finish_reason)])).complete(MESSAGES)

    assert result == content
    assert records(caplog) == []


def test_reply_without_a_finish_reason_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    assert make_client(ScriptedServer([ok("")])).complete(MESSAGES) == ""
    assert records(caplog) == []


def test_cut_off_reply_after_the_deadline_is_dropped_without_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    body = json.dumps({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
    stream = TricklingStream([body.encode()], clock, gap=11.0)
    server = ScriptedServer([httpx.Response(200, stream=stream)])

    with pytest.raises(DependencyTimeout):
        make_client(server, sleep=RecordingSleep(clock), total_timeout_s=10.0).complete(MESSAGES)

    assert records(caplog) == []
