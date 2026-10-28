"""Exceptions are logged with their frames but never their messages (spec: observability).

Exception messages can hold document text, questions or credentials (``RuntimeError(f"...
{text}")``, a URL with a password), so both formatters render every exception of a traceback,
chained and grouped ones included, as its type and frames with the message replaced.
"""

import io
import json
import logging
import sys
from collections.abc import Callable, Iterator

import pytest

from app.domain.errors import DependencyUnavailable
from app.observability import (
    REDACTED_MESSAGE,
    JsonFormatter,
    TextFormatter,
    configure_logging,
    format_exception_redacted,
)

LOGGER = "tests.observability.redaction"
# Built at run time, so the source lines a traceback shows never contain them.
DOCUMENT_TEXT = " ".join(("Ximena", "Quintanilla", "signed", "the", "lease"))
CREDENTIAL_URL = "http://admin:" + "-".join(("s3cr3t", "key")) + "@qdrant:6333"
SECRETS = ("Ximena", "Quintanilla", "s3cr3t", "admin:")
CAUSE = "The above exception was the direct cause of the following exception:"
CONTEXT = "During handling of the above exception, another exception occurred:"

type Capture = Callable[[logging.Formatter], io.StringIO]


@pytest.fixture
def capture() -> Iterator[Capture]:
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
    logger.handlers, logger.propagate = saved[0], saved[2]
    logger.setLevel(saved[1])


FORMATTERS = pytest.mark.parametrize(
    "formatter", [JsonFormatter(), TextFormatter()], ids=["json", "text"]
)


def _raise_with_document_text() -> None:
    raise RuntimeError(f"tokenizer choked on: {DOCUMENT_TEXT}")


def _raise_chained() -> None:
    try:
        _raise_with_document_text()
    except RuntimeError as error:
        try:
            raise ConnectionError(f"connect to {CREDENTIAL_URL} refused") from error
        except ConnectionError:
            raise KeyError(DOCUMENT_TEXT)  # noqa: B904 - the implicit context is the point


def _logged(stream: io.StringIO, raiser: Callable[[], None]) -> str:
    """Log what ``raiser`` raises; return the exception text of the line."""
    try:
        raiser()
    except BaseException:
        logging.getLogger(LOGGER).exception("unexpected error")
    output = stream.getvalue()
    for secret in SECRETS:
        assert secret not in output, f"{secret!r} reached the log"
    if output.startswith("{"):
        return str(json.loads(output)["exception"])
    return output.rstrip("\n")


@FORMATTERS
def test_message_is_redacted_and_frames_are_kept(
    capture: Capture, formatter: logging.Formatter
) -> None:
    text = _logged(capture(formatter), _raise_with_document_text)

    assert "Traceback (most recent call last):" in text
    assert "test_redaction.py" in text
    assert "in _raise_with_document_text" in text
    assert text.endswith(f"RuntimeError: {REDACTED_MESSAGE}")


@FORMATTERS
def test_cause_and_context_are_redacted_oldest_first(
    capture: Capture, formatter: logging.Formatter
) -> None:
    text = _logged(capture(formatter), _raise_chained)

    positions = [
        text.index(f"RuntimeError: {REDACTED_MESSAGE}"),
        text.index(CAUSE),
        text.index(f"ConnectionError: {REDACTED_MESSAGE}"),
        text.index(CONTEXT),
        text.index(f"KeyError: {REDACTED_MESSAGE}"),
    ]
    assert positions == sorted(positions)
    assert text.count("Traceback (most recent call last):") == 3


@FORMATTERS
def test_exception_group_members_are_redacted(
    capture: Capture, formatter: logging.Formatter
) -> None:
    def grouped() -> None:
        raise ExceptionGroup(
            f"batch {DOCUMENT_TEXT}",
            [ValueError(DOCUMENT_TEXT), ExceptionGroup("inner", [OSError(CREDENTIAL_URL)])],
        )

    text = _logged(capture(formatter), grouped)

    assert f"ExceptionGroup: {REDACTED_MESSAGE}" in text
    assert f"| ValueError: {REDACTED_MESSAGE}" in text
    assert f"| | OSError: {REDACTED_MESSAGE}" in text


def test_service_error_keeps_its_code_only(capture: Capture) -> None:
    code = "vector_store_unavailable"

    def failing() -> None:
        raise DependencyUnavailable(f"cannot reach {CREDENTIAL_URL}", code=code)

    text = _logged(capture(JsonFormatter()), failing)

    assert text.endswith(
        f"app.domain.errors.DependencyUnavailable: {REDACTED_MESSAGE} [code={code}]"
    )


def test_unknown_code_attribute_is_not_shown(capture: Capture) -> None:
    class CodedError(Exception):
        code = DOCUMENT_TEXT

    def failing() -> None:
        raise CodedError

    text = _logged(capture(JsonFormatter()), failing)

    assert text.endswith(f"CodedError: {REDACTED_MESSAGE}")


def test_notes_and_suppressed_context_are_not_shown(capture: Capture) -> None:
    def failing() -> None:
        try:
            _raise_with_document_text()
        except RuntimeError:
            error = ValueError("wrapped")
            error.add_note(DOCUMENT_TEXT)
            raise error from None

    text = _logged(capture(TextFormatter()), failing)

    assert "RuntimeError" not in text
    assert text.endswith(f"ValueError: {REDACTED_MESSAGE}")


def test_str_is_never_called(capture: Capture) -> None:
    class Exploding(Exception):
        def __str__(self) -> str:
            raise AssertionError("the formatter rendered the message")

    def failing() -> None:
        raise Exploding

    assert _logged(capture(JsonFormatter()), failing).endswith(f"Exploding: {REDACTED_MESSAGE}")


def test_context_cycle_terminates() -> None:
    first, second = ValueError(DOCUMENT_TEXT), KeyError(DOCUMENT_TEXT)
    first.__context__, second.__context__ = second, first

    text = format_exception_redacted((ValueError, first, None))

    assert text.count(REDACTED_MESSAGE) == 2
    assert "Quintanilla" not in text


def test_no_exception() -> None:
    assert format_exception_redacted((None, None, None)) == "NoneType: None"


@pytest.fixture
def server_logging() -> Iterator[None]:
    """Restore the root and Uvicorn loggers after ``configure_logging``."""
    names = ("uvicorn", "uvicorn.error", "uvicorn.access")
    root = logging.getLogger()
    saved_root = (root.handlers[:], root.level)
    saved = {
        name: (lg.handlers[:], lg.level, lg.propagate, lg.disabled)
        for name in names
        for lg in [logging.getLogger(name)]
    }
    yield
    root.handlers = saved_root[0]
    root.setLevel(saved_root[1])
    for name, (handlers, level, propagate, disabled) in saved.items():
        lg = logging.getLogger(name)
        lg.handlers, lg.propagate, lg.disabled = handlers, propagate, disabled
        lg.setLevel(level)


@pytest.mark.usefixtures("server_logging")
@pytest.mark.parametrize("fmt", ["json", "text"])
def test_uvicorn_error_traceback_is_redacted(capsys: pytest.CaptureFixture[str], fmt: str) -> None:
    configure_logging("INFO", fmt)
    try:
        _raise_chained()
    except KeyError:
        # What Uvicorn logs when an exception escapes the application.
        logging.getLogger("uvicorn.error").error("Exception in ASGI application\n", exc_info=True)
    sys.stdout.flush()

    out = capsys.readouterr().out
    assert "Exception in ASGI application" in out
    assert "in _raise_with_document_text" in out
    for secret in SECRETS:
        assert secret not in out


def test_source_lines_are_not_shown(capture: Capture) -> None:
    def failing() -> None:
        raise RuntimeError("literal-in-the-source")

    text = _logged(capture(JsonFormatter()), failing)

    assert "literal-in-the-source" not in text
    assert "raise RuntimeError" not in text
    assert ", in failing" in text


def test_recursion_is_collapsed() -> None:
    def recurse(depth: int) -> None:
        if depth == 0:
            raise RecursionError(DOCUMENT_TEXT)
        recurse(depth - 1)

    try:
        recurse(50)
    except RecursionError:
        text = format_exception_redacted(sys.exc_info())

    frames = [line for line in text.splitlines() if line.endswith(", in recurse")]
    assert len(frames) == 4  # three of the repeated call, then the one that raised
    assert "  [Previous line repeated 47 more times]" in text


# --- LOG_EXCEPTION_MESSAGES: the opt-in for local debugging -------------------------------


@pytest.mark.usefixtures("server_logging")
@pytest.mark.parametrize("fmt", ["json", "text"])
@pytest.mark.parametrize("enabled", [False, True], ids=["default", "enabled"])
def test_exception_messages_are_logged_only_when_enabled(
    capsys: pytest.CaptureFixture[str], fmt: str, enabled: bool
) -> None:
    if enabled:
        configure_logging("INFO", fmt, exception_messages=True)
    else:
        configure_logging("INFO", fmt)  # the default
    try:
        _raise_chained()
    except KeyError:
        logging.getLogger("app.api.errors").exception("unexpected error")
    sys.stdout.flush()

    out = capsys.readouterr().out
    assert "in _raise_with_document_text" in out  # frames either way
    assert (DOCUMENT_TEXT in out) is enabled
    assert ("s3cr3t-key" in out) is enabled
    assert (REDACTED_MESSAGE in out) is not enabled


@pytest.mark.parametrize("formatter_type", [JsonFormatter, TextFormatter])
def test_formatters_redact_unless_told_otherwise(
    capture: Capture, formatter_type: type[JsonFormatter | TextFormatter]
) -> None:
    assert formatter_type().exception_messages is False
    stream = capture(formatter_type(exception_messages=True))
    try:
        _raise_with_document_text()
    except RuntimeError:
        logging.getLogger(LOGGER).exception("unexpected error")

    assert DOCUMENT_TEXT in stream.getvalue()
