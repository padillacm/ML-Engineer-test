"""The start-up check of the stack's own model (``app.llm_warmup``, spec: deployment).

Runs everywhere: the HTTP calls are scripted, except for ``urllib_http``, which is checked
against a local stdlib server.
"""

import http.server
import io
import json
import socket
import threading
import urllib.error
from collections.abc import Iterator

import pytest

from app.llm_warmup import (
    PROBE_TIMEOUT_S,
    SERVER_WAIT_S,
    WARMUP_TIMEOUT_S,
    WarmupError,
    main,
    urllib_http,
    uses_stack_ollama,
    warm_up,
)

OLLAMA = "http://10.213.47.250:11434"
WIRED = {"LLM_BASE_URL": f"{OLLAMA}/v1", "OLLAMA_URL": OLLAMA, "LLM_MODEL": "gemma4:e4b-it-qat"}

type Answer = tuple[int, bytes] | OSError


class ScriptedOllama:
    """Answers each ``(method, path)`` from a list of scripted answers (the last one repeats)."""

    def __init__(self, answers: dict[tuple[str, str], list[Answer]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str, object, float]] = []

    def __call__(
        self, method: str, url: str, body: bytes | None, timeout_s: float
    ) -> tuple[int, bytes]:
        path = url.removeprefix(OLLAMA)
        self.calls.append((method, path, None if body is None else json.loads(body), timeout_s))
        script = self.answers[(method, path)]
        answer = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(answer, OSError):
            raise answer
        return answer


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def healthy(**overrides: list[Answer]) -> ScriptedOllama:
    answers: dict[tuple[str, str], list[Answer]] = {
        ("GET", "/api/version"): [(200, b'{"version":"0.40.12"}')],
        ("POST", "/api/show"): [(200, b"{}")],
        ("POST", "/api/generate"): [(200, b'{"response":"OK"}')],
    }
    for key, value in overrides.items():
        method, path = key.split("_", 1)
        answers[(method.upper(), "/api/" + path)] = value
    return ScriptedOllama(answers)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        (WIRED, True),
        ({**WIRED, "LLM_BASE_URL": f"{OLLAMA}/v1/"}, True),
        ({**WIRED, "OLLAMA_URL": f"{OLLAMA}/"}, True),
        ({**WIRED, "LLM_BASE_URL": ""}, False),
        ({**WIRED, "LLM_BASE_URL": "https://api.openai.com/v1"}, False),
        ({**WIRED, "LLM_BASE_URL": "http://ollama:11434/v1"}, False),
        ({**WIRED, "OLLAMA_URL": ""}, False),
        ({}, False),
    ],
)
def test_uses_stack_ollama(environ: dict[str, str], expected: bool) -> None:
    assert uses_stack_ollama(environ) is expected


def test_a_service_without_the_stack_ollama_starts_at_once() -> None:
    ollama = healthy()

    outcome = warm_up({**WIRED, "LLM_BASE_URL": "https://api.openai.com/v1"}, ollama)

    assert outcome == "the service does not use the stack's Ollama; nothing to check"
    assert ollama.calls == []


def test_the_model_is_checked_then_asked_one_short_prompt() -> None:
    ollama = healthy()

    outcome = warm_up(WIRED, ollama, FakeClock(), lambda _: None)

    assert outcome.startswith("model gemma4:e4b-it-qat loaded; the warm-up prompt took ")
    assert [(method, path) for method, path, _, _ in ollama.calls] == [
        ("GET", "/api/version"),
        ("POST", "/api/show"),
        ("POST", "/api/generate"),
    ]
    _, _, show, show_timeout = ollama.calls[1]
    _, _, prompt, prompt_timeout = ollama.calls[2]
    assert show == {"model": "gemma4:e4b-it-qat"}
    assert prompt == {
        "model": "gemma4:e4b-it-qat",
        "prompt": "Reply with OK.",
        "stream": False,
        "options": {"num_predict": 16},
    }
    assert (show_timeout, prompt_timeout) == (PROBE_TIMEOUT_S, WARMUP_TIMEOUT_S)


def test_the_server_is_waited_for() -> None:
    ollama = healthy(get_version=[ConnectionRefusedError(), (503, b""), (200, b"{}")])
    clock = FakeClock()

    warm_up(WIRED, ollama, clock, clock.sleep)

    assert clock.now == 2.0
    assert [path for _, path, _, _ in ollama.calls].count("/api/version") == 3


def test_a_server_that_never_answers_fails_after_the_wait() -> None:
    ollama = healthy(get_version=[ConnectionRefusedError()])
    clock = FakeClock()

    with pytest.raises(WarmupError) as caught:
        warm_up(WIRED, ollama, clock, clock.sleep)

    assert str(caught.value) == (
        f"Ollama at {OLLAMA} did not answer within 30 s "
        "(see `docker compose logs ollama-pull ollama`)"
    )
    assert clock.now == SERVER_WAIT_S


def test_a_model_missing_from_the_volume_names_the_failed_download() -> None:
    ollama = healthy(post_show=[(404, b'{"error":"model \'gemma4:e4b-it-qat\' not found"}')])

    with pytest.raises(WarmupError) as caught:
        warm_up(WIRED, ollama, FakeClock(), lambda _: None)

    assert str(caught.value) == (
        "model gemma4:e4b-it-qat is not in Ollama's volume: its download failed "
        "(see `docker compose logs ollama-pull`)"
    )
    assert [path for _, path, _, _ in ollama.calls] == ["/api/version", "/api/show"]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"post_show": [(500, b'{"error":"bad manifest"}')]},
            "checking the model failed with HTTP 500: bad manifest",
        ),
        (
            {"post_generate": [(500, b'{"error":"CUDA error: out of memory"}')]},
            "the warm-up prompt failed with HTTP 500: CUDA error: out of memory",
        ),
        (
            {"post_generate": [(502, b"<html>bad gateway</html>")]},
            "the warm-up prompt failed with HTTP 502: <html>bad gateway</html>",
        ),
        ({"post_generate": [(500, b"")]}, "the warm-up prompt failed with HTTP 500: no message"),
        ({"post_generate": [(500, b'{"error": 3}')]}, 'failed with HTTP 500: {"error": 3}'),
        ({"post_generate": [TimeoutError()]}, "the warm-up prompt got no answer (TimeoutError)"),
        ({"post_show": [ConnectionResetError()]}, "checking the model got no answer"),
    ],
)
def test_failures_name_the_step_and_the_server_error(
    overrides: dict[str, list[Answer]], expected: str
) -> None:
    with pytest.raises(WarmupError) as caught:
        warm_up(WIRED, healthy(**overrides), FakeClock(), lambda _: None)

    assert expected in str(caught.value)


def test_a_long_server_error_is_shortened() -> None:
    ollama = healthy(post_generate=[(500, json.dumps({"error": "x" * 500}).encode())])

    with pytest.raises(WarmupError) as caught:
        warm_up(WIRED, ollama, FakeClock(), lambda _: None)

    assert str(caught.value) == "the warm-up prompt failed with HTTP 500: " + "x" * 200


def test_an_empty_model_name_fails_before_any_request() -> None:
    ollama = healthy()

    with pytest.raises(WarmupError, match="LLM_MODEL is empty"):
        warm_up({**WIRED, "LLM_MODEL": " "}, ollama)

    assert ollama.calls == []


def test_main_prints_the_outcome_and_exits_0_or_1() -> None:
    out = io.StringIO()
    assert main({}, healthy(), out) == 0
    assert (
        out.getvalue()
        == "llm-warmup: the service does not use the stack's Ollama; nothing to check\n"
    )

    out = io.StringIO()
    assert main({**WIRED, "LLM_MODEL": ""}, healthy(), out) == 1
    assert out.getvalue() == "llm-warmup: LLM_MODEL is empty\n"


def test_main_reads_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)

    assert main(http=healthy(), out=io.StringIO()) == 0


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._answer(200, b'{"version":"test"}')

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        if self.headers["Content-Type"] != "application/json":
            self._answer(415, b"")
        elif json.loads(body).get("model") == "missing":
            self._answer(404, b'{"error":"not found"}')
        else:
            self._answer(200, body)

    def _answer(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@pytest.fixture
def local_server() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_urllib_http_returns_the_status_and_body(local_server: str) -> None:
    assert urllib_http("GET", f"{local_server}/api/version", None, 5) == (
        200,
        b'{"version":"test"}',
    )
    assert urllib_http("POST", f"{local_server}/api/show", b'{"model":"m"}', 5) == (
        200,
        b'{"model":"m"}',
    )
    assert urllib_http("POST", f"{local_server}/api/show", b'{"model":"missing"}', 5) == (
        404,
        b'{"error":"not found"}',
    )


def test_urllib_http_raises_os_error_without_a_server() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(urllib.error.URLError):  # an OSError
        urllib_http("GET", f"http://127.0.0.1:{port}/api/version", None, 2)


def test_urllib_http_refuses_other_schemes() -> None:
    with pytest.raises(ValueError, match="only http and https"):
        urllib_http("GET", "file:///etc/passwd", None, 1)
