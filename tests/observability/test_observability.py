"""Structured logs with request ids and no personal data (spec: observability)."""

import contextvars
import io
import json
import logging
import re
import sys
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import anyio
import anyio.to_thread
import pytest

from app.observability import (
    ALLOWED_FIELDS,
    REQUEST_ID_PATTERN,
    JsonFormatter,
    TextFormatter,
    accept_request_id,
    configure_logging,
    request_context,
    request_id_var,
)

LOGGER = "tests.observability"
GENERATED_ID = re.compile(r"[0-9a-f]{32}")
BASE_FIELDS = {"timestamp", "level", "logger", "message"}


@pytest.fixture
def capture() -> Iterator[Callable[[logging.Formatter], io.StringIO]]:
    """Attach a formatter to a private logger and return the stream it writes to."""
    logger = logging.getLogger(LOGGER)
    saved = (logger.handlers[:], logger.level, logger.propagate)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    def attach(formatter: logging.Formatter) -> io.StringIO:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        logger.handlers = [handler]
        return stream

    yield attach
    logger.handlers, level, logger.propagate = saved[0], saved[1], saved[2]
    logger.setLevel(level)


def _lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_allowed_fields_are_the_access_error_outcome_and_start_up_fields() -> None:
    expected = {
        "method", "route", "status", "duration_ms", "code", "category", "pages",
        "names_found", "matches", "requested_names", "size_bytes", "document_id",
        "chunks_stored", "chunks_flagged", "already_indexed", "hits", "hits_kept", "top_score",
        "refused", "refusal_reason", "injection_score", "support_share", "filter_reason",
        "llm_ms", "ocr_ms", "ner_ms", "embed_ms", "store_ms", "search_ms", "screen_ms",
        "sources", "check", "ready", "cpus_requested", "cpus_available", "torch_threads",
        "max_concurrent_jobs", "ner_pool_size", "embedding_pool_size", "ocr_workers",
        "injection_pool_size", "device", "gpu_name", "gpu_memory_mib", "model", "max_tokens",
    }  # fmt: skip
    assert set(ALLOWED_FIELDS) == expected


def test_allowed_fields_do_not_clash_with_log_record_attributes() -> None:
    # logging raises KeyError for an ``extra=`` key that is a LogRecord attribute.
    reserved = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
    assert not ALLOWED_FIELDS & reserved


def test_json_line_has_the_base_fields(capture: Callable[[logging.Formatter], io.StringIO]) -> None:
    stream = capture(JsonFormatter())
    before = datetime.now(UTC)

    logging.getLogger(LOGGER).warning("done %s", 3)

    (line,) = _lines(stream)
    assert set(line) == BASE_FIELDS
    assert line["level"] == "WARNING"
    assert line["logger"] == LOGGER
    assert line["message"] == "done 3"
    timestamp = datetime.fromisoformat(str(line["timestamp"]))
    assert timestamp.utcoffset() == timedelta(0)
    assert before - timedelta(seconds=1) <= timestamp <= datetime.now(UTC)


def test_one_json_object_per_line(capture: Callable[[logging.Formatter], io.StringIO]) -> None:
    stream = capture(JsonFormatter())
    logging.getLogger(LOGGER).info("first\nsecond")
    logging.getLogger(LOGGER).info("third")
    assert [line["message"] for line in _lines(stream)] == ["first\nsecond", "third"]


def test_request_id_is_present_inside_a_request_only(
    capture: Callable[[logging.Formatter], io.StringIO],
) -> None:
    stream = capture(JsonFormatter())
    logger = logging.getLogger(LOGGER)

    logger.info("before")
    with request_context("abc-123"):
        logger.info("inside")
    logger.info("after")

    before, inside, after = _lines(stream)
    assert "request_id" not in before
    assert inside["request_id"] == "abc-123"
    assert "request_id" not in after
    assert request_id_var.get() is None


def test_allowed_scalar_extras_are_kept(
    capture: Callable[[logging.Formatter], io.StringIO],
) -> None:
    stream = capture(JsonFormatter())
    extra = {
        "method": "POST",
        "route": "/api/extract",
        "status": 200,
        "duration_ms": 12.5,
        "refused": False,
        "document_id": None,
    }

    logging.getLogger(LOGGER).info("request", extra=extra)

    (line,) = _lines(stream)
    assert {key: line[key] for key in extra} == extra


def test_unknown_extras_are_dropped(capture: Callable[[logging.Formatter], io.StringIO]) -> None:
    stream = capture(JsonFormatter())

    logging.getLogger(LOGGER).info(
        "extracted",
        extra={
            "names": ["María González"],
            "text": "María González signed the contract",
            "question": "who signed?",
            "file_name": "contract.pdf",
            "api_key": "fake-key-0123456789abcdefghij",
            "pages": 2,
        },
    )

    output = stream.getvalue()
    (line,) = _lines(stream)
    assert set(line) == BASE_FIELDS | {"pages"}
    for fragment in ("Mar", "Gonz", "signed", "contract", "who", "fake-key"):
        assert fragment not in output


def test_non_scalar_values_are_dropped(capture: Callable[[logging.Formatter], io.StringIO]) -> None:
    stream = capture(JsonFormatter())

    logging.getLogger(LOGGER).info(
        "asked",
        extra={
            "matches": ["María"],
            "sources": {"text": "María"},
            "top_score": float("nan"),
            "llm_ms": float("inf"),
            "hits": b"bytes",
            "hits_kept": 1,
        },
    )

    (line,) = _lines(stream)
    assert set(line) == BASE_FIELDS | {"hits_kept"}
    assert "Mar" not in stream.getvalue()


def test_exception_is_rendered_as_text(
    capture: Callable[[logging.Formatter], io.StringIO],
) -> None:
    stream = capture(JsonFormatter())

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logging.getLogger(LOGGER).exception("unexpected", extra={"code": "internal_error"})

    (line,) = _lines(stream)
    assert set(line) == BASE_FIELDS | {"exception", "code"}
    exception = str(line["exception"])
    assert exception.startswith("Traceback (most recent call last):")
    assert "in test_exception_is_rendered_as_text" in exception
    assert exception.endswith("RuntimeError: <message redacted>")


def test_text_format(capture: Callable[[logging.Formatter], io.StringIO]) -> None:
    stream = capture(TextFormatter())
    logger = logging.getLogger(LOGGER)

    logger.info("outside", extra={"status": 415, "text": "María"})
    with request_context("req-1"):
        logger.error("inside", extra={"code": "not_a_pdf", "category": "invalid_input"})

    outside, inside = stream.getvalue().splitlines()
    assert outside.endswith(f" INFO {LOGGER} [-] outside status=415")
    assert inside.endswith(
        f' ERROR {LOGGER} [req-1] inside category="invalid_input" code="not_a_pdf"'
    )
    assert "María" not in stream.getvalue()
    datetime.fromisoformat(outside.split(" ", 1)[0])


def test_text_format_renders_the_exception(
    capture: Callable[[logging.Formatter], io.StringIO],
) -> None:
    stream = capture(TextFormatter())
    try:
        raise ValueError("bad")
    except ValueError:
        logging.getLogger(LOGGER).exception("failed")

    first, *rest = stream.getvalue().splitlines()
    assert first.endswith("[-] failed")
    assert rest[0] == "Traceback (most recent call last):"
    assert rest[-1] == "ValueError: <message redacted>"


@pytest.mark.parametrize("header", ["abc-123", "A.b_c-9", "x", "a" * 64, "0123456789abcdef"])
def test_well_formed_request_id_is_kept(header: str) -> None:
    assert accept_request_id(header) == header
    assert REQUEST_ID_PATTERN.fullmatch(header)


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "a" * 65,
        "a" * 200,
        "abc\n",
        "abc\ndef",
        "abc\r\nX-Injected: 1",
        "abc def",
        "ñandú",
        "id" + chr(0x0661),  # an Arabic-Indic digit is not an ASCII digit
        "a/b",
        "{json}",
    ],
)
def test_malformed_request_id_is_replaced(header: str | None) -> None:
    generated = accept_request_id(header)
    assert generated != header
    assert GENERATED_ID.fullmatch(generated)
    assert REQUEST_ID_PATTERN.fullmatch(generated)


def test_generated_request_ids_differ() -> None:
    assert len({accept_request_id(None) for _ in range(100)}) == 100


def test_request_id_follows_copy_context_into_a_worker_thread(
    capture: Callable[[logging.Formatter], io.StringIO],
) -> None:
    stream = capture(JsonFormatter())
    logger = logging.getLogger(LOGGER)

    with request_context("ctx-1"), ThreadPoolExecutor(max_workers=1) as pool:
        context = contextvars.copy_context()
        pool.submit(context.run, logger.info, "in worker").result()
        pool.submit(logger.info, "without context").result()

    in_worker, without_context = _lines(stream)
    assert in_worker["request_id"] == "ctx-1"
    assert "request_id" not in without_context


def test_request_id_follows_anyio_to_thread(
    capture: Callable[[logging.Formatter], io.StringIO],
) -> None:
    stream = capture(JsonFormatter())
    logger = logging.getLogger(LOGGER)
    threads: list[int] = []

    def work() -> None:
        threads.append(threading.get_ident())
        logger.info("sync handler")

    async def handle() -> None:
        with request_context("anyio-7"):
            await anyio.to_thread.run_sync(work)

    anyio.run(handle)

    (line,) = _lines(stream)
    assert line["request_id"] == "anyio-7"
    assert threads != [threading.get_ident()]


@pytest.fixture
def logging_state() -> Iterator[None]:
    """Restore the root and uvicorn loggers after ``configure_logging``."""
    names = ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore")
    root = logging.getLogger()
    saved_root = (root.handlers[:], root.level)
    saved = {
        name: (lg.handlers[:], lg.level, lg.propagate, lg.disabled)
        for name in names
        for lg in [logging.getLogger(name)]
    }
    yield
    for handler in root.handlers[:]:
        if handler not in saved_root[0]:
            root.removeHandler(handler)
    root.handlers = saved_root[0]
    root.setLevel(saved_root[1])
    for name, (handlers, level, propagate, disabled) in saved.items():
        lg = logging.getLogger(name)
        lg.handlers, lg.propagate, lg.disabled = handlers, propagate, disabled
        lg.setLevel(level)


def _service_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if type(h).__name__ == "_ServiceHandler"]


@pytest.mark.usefixtures("logging_state")
def test_configure_logging_is_idempotent() -> None:
    other = logging.NullHandler()
    logging.getLogger().addHandler(other)

    configure_logging("INFO", "json")
    configure_logging("DEBUG", "text")
    configure_logging("warning", "json")

    handlers = _service_handlers()
    assert len(handlers) == 1
    assert isinstance(handlers[0].formatter, JsonFormatter)
    assert other in logging.getLogger().handlers
    assert logging.getLogger().level == logging.WARNING


@pytest.mark.usefixtures("logging_state")
def test_configure_logging_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    assert _service_handlers()[0].stream is sys.stdout  # type: ignore[attr-defined]

    with request_context("out-1"):
        logging.getLogger("app.api").info("request", extra={"status": 200})
    logging.getLogger("app.api").debug("hidden")

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines == [
        {
            "timestamp": lines[0]["timestamp"],
            "level": "INFO",
            "logger": "app.api",
            "message": "request",
            "request_id": "out-1",
            "status": 200,
        }
    ]


@pytest.mark.usefixtures("logging_state")
def test_configure_logging_text_format(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG", "text")
    logging.getLogger("app.api").debug("plain", extra={"pages": 3})
    assert capsys.readouterr().out.rstrip("\n").endswith(" DEBUG app.api [-] plain pages=3")


@pytest.mark.usefixtures("logging_state")
def test_uvicorn_access_is_disabled_and_error_kept(capsys: pytest.CaptureFixture[str]) -> None:
    logging.getLogger("uvicorn.error").addHandler(logging.NullHandler())
    logging.getLogger("uvicorn.error").propagate = False

    configure_logging("INFO", "json")
    logging.getLogger("uvicorn.access").warning("GET / 200")
    logging.getLogger("uvicorn.error").error("server error")

    access = logging.getLogger("uvicorn.access")
    assert access.disabled
    assert access.level == logging.WARNING
    assert not access.propagate
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(line["logger"], line["message"]) for line in lines] == [
        ("uvicorn.error", "server error")
    ]


@pytest.mark.usefixtures("logging_state")
def test_http_client_request_lines_are_not_logged(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG", "json")
    logging.getLogger("httpx").info("HTTP Request: POST http://user:fake-key-x@llm/v1 200")
    logging.getLogger("httpcore.connection").debug("connect_tcp.started host='llm'")
    logging.getLogger("httpx").warning("client warning")

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(line["logger"], line["message"]) for line in lines] == [("httpx", "client warning")]


@pytest.mark.usefixtures("logging_state")
@pytest.mark.parametrize(("level", "fmt"), [("INFO", "xml"), ("VERBOSE", "json"), ("", "text")])
def test_configure_logging_rejects_bad_values(level: str, fmt: str) -> None:
    with pytest.raises(ValueError, match="log"):
        configure_logging(level, fmt)
    assert not _service_handlers()
