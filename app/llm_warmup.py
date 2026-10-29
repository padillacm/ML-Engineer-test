"""Start-up check of the stack's own language model (compose service ``llm-warmup``).

Compose runs ``python -m app.llm_warmup`` once before the service starts, with the service's
``LLM_BASE_URL`` and ``LLM_MODEL`` and the address of the stack's Ollama (``OLLAMA_URL``). When
the service is pointed at that Ollama (``LLM_BASE_URL`` is ``OLLAMA_URL`` + ``/v1``) it waits for
the server (at most ``SERVER_WAIT_S``), checks that the model is in the server's volume and asks
one short prompt, so the model is loaded and its GPU kernels compiled before the first question
(a fresh volume needs about 85 s for both). Any failure prints its cause and exits 1, and compose
then does not start the service: a failed model download (wrong name, full volume, no network)
is visible at ``docker compose up`` instead of leaving a service that reports a model it cannot
use. When the service uses another model, or none, it exits 0 at once.

Only the standard library is used (no proxy: the server is on the stack's internal network).
Server error texts are printed shortened; they never contain anything secret (the prompt is
fixed and no key is involved).
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import TextIO

SERVER_WAIT_S = 30.0
"""How long the server may take to answer at all. With the ``gpu`` profile compose starts this
check only once Ollama is healthy, so a longer wait only happens when Ollama did not start (its
model download failed) or is not part of the stack."""
WARMUP_TIMEOUT_S = 600.0
"""Longest wait for the warm-up answer: loading and compiling on a fresh volume took 69-100 s."""
PROBE_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 1.0
WARMUP_PROMPT = "Reply with OK."
WARMUP_TOKENS = 16
_HTTP_OK = 200
_HTTP_NOT_FOUND = 404
_ERROR_CHARS = 200

type Http = Callable[[str, str, bytes | None, float], tuple[int, bytes]]
"""``(method, url, json_body, timeout_s) -> (status, body)``; raises ``OSError`` when no answer."""


class WarmupError(Exception):
    """The stack's model cannot serve the service; the message says why."""


def urllib_http(method: str, url: str, body: bytes | None, timeout_s: float) -> tuple[int, bytes]:
    """One HTTP request without proxies; an error status is returned, not raised."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if not url.startswith(("http://", "https://")):
        raise ValueError("only http and https URLs are requested")
    request = urllib.request.Request(  # noqa: S310  # the scheme is checked above
        url, data=body, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return int(response.status), bytes(response.read())
    except urllib.error.HTTPError as error:
        with error:
            return int(error.code), bytes(error.read())


def uses_stack_ollama(environ: Mapping[str, str]) -> bool:
    """Whether the service's ``LLM_BASE_URL`` is the stack's Ollama (``OLLAMA_URL`` + ``/v1``)."""
    base_url = environ.get("LLM_BASE_URL", "").strip().rstrip("/")
    ollama_url = environ.get("OLLAMA_URL", "").strip().rstrip("/")
    return bool(base_url) and bool(ollama_url) and base_url == f"{ollama_url}/v1"


def warm_up(
    environ: Mapping[str, str],
    http: Http = urllib_http,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Check and load the stack's model for the service; returns what was done.

    Raises ``WarmupError`` naming the cause when the server does not answer, the model is not
    in its volume, or the warm-up prompt fails.
    """
    if not uses_stack_ollama(environ):
        return "the service does not use the stack's Ollama; nothing to check"
    server = environ["OLLAMA_URL"].strip().rstrip("/")
    model = environ.get("LLM_MODEL", "").strip()
    if not model:
        raise WarmupError("LLM_MODEL is empty")
    _wait_for_server(server, http, clock, sleep)
    payload = json.dumps({"model": model}).encode()
    status, body = _call(http, f"{server}/api/show", payload, PROBE_TIMEOUT_S, "checking the model")
    if status == _HTTP_NOT_FOUND:
        raise WarmupError(
            f"model {model} is not in Ollama's volume: its download failed "
            "(see `docker compose logs ollama-pull`)"
        )
    _expect_ok(status, body, "checking the model")
    started = clock()
    prompt = {
        "model": model,
        "prompt": WARMUP_PROMPT,
        "stream": False,
        "options": {"num_predict": WARMUP_TOKENS},
    }
    payload = json.dumps(prompt).encode()
    step = "the warm-up prompt"
    status, body = _call(http, f"{server}/api/generate", payload, WARMUP_TIMEOUT_S, step)
    _expect_ok(status, body, "the warm-up prompt")
    return f"model {model} loaded; the warm-up prompt took {clock() - started:.1f} s"


def _wait_for_server(
    server: str, http: Http, clock: Callable[[], float], sleep: Callable[[float], None]
) -> None:
    deadline = clock() + SERVER_WAIT_S
    while True:
        try:
            status, _ = http("GET", f"{server}/api/version", None, PROBE_TIMEOUT_S)
        except OSError:
            status = 0
        if status == _HTTP_OK:
            return
        if clock() >= deadline:
            raise WarmupError(
                f"Ollama at {server} did not answer within {SERVER_WAIT_S:.0f} s "
                "(see `docker compose logs ollama-pull ollama`)"
            )
        sleep(POLL_INTERVAL_S)


def _call(http: Http, url: str, body: bytes, timeout_s: float, step: str) -> tuple[int, bytes]:
    try:
        return http("POST", url, body, timeout_s)
    except OSError as error:
        raise WarmupError(f"{step} got no answer ({type(error).__name__})") from None


def _expect_ok(status: int, body: bytes, step: str) -> None:
    if status == _HTTP_OK:
        return
    raise WarmupError(f"{step} failed with HTTP {status}: {_error_text(body)}")


def _error_text(body: bytes) -> str:
    """Ollama's ``{"error": "..."}`` text, shortened; the raw body when it is not that."""
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    text = parsed.get("error") if isinstance(parsed, dict) else None
    message = text if isinstance(text, str) else body.decode("utf-8", "replace")
    return message.strip()[:_ERROR_CHARS] or "no message"


def main(
    environ: Mapping[str, str] | None = None,
    http: Http = urllib_http,
    out: TextIO | None = None,
) -> int:
    """Run ``warm_up`` and print the outcome: 0 when the service may start, 1 otherwise."""
    stream = sys.stderr if out is None else out
    try:
        outcome = warm_up(os.environ if environ is None else environ, http)
    except WarmupError as error:
        print(f"llm-warmup: {error}", file=stream)
        return 1
    print(f"llm-warmup: {outcome}", file=stream)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the compose stack
    sys.exit(main())
