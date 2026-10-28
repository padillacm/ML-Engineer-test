"""Logging of the service: JSON (or text) lines on stdout carrying the request id.

The request id lives in ``request_id_var``, set by the HTTP middleware for the duration of a
request. Context variables follow ``contextvars.copy_context().run`` (which is how AnyIO's
``to_thread`` runs sync handlers), so lines logged from the worker thread carry it too.

Documents carry personal data, so a log line holds only the fixed fields (timestamp, level,
logger, message, request id, exception) plus the ``extra=`` keys named in ``ALLOWED_FIELDS``
whose values are JSON scalars (str, int, finite float, bool, None). Any other ``extra=`` key
or value is dropped. Callers still decide what goes into an allowed field: counts, sizes,
durations, codes, ids and scores, never names, questions, answers or document text.

Exceptions are rendered redacted (``format_exception_redacted``): the type and the stack frames
(file, line number, function) of every exception in the chain, cause, context and
exception-group members included, but never an exception's message or notes, which can hold
document text, questions or credentials, nor the source lines (a literal in the code that
raised can be as sensitive as the message). A ``ServiceError`` keeps its code (one of
``KNOWN_CODES``). Both formatters render exceptions this way, so the rule holds for every line
written through the service's handler, Uvicorn's error log and ``logger.exception`` included.
``LOG_EXCEPTION_MESSAGES=true`` (``configure_logging(..., exception_messages=True)``) renders the
standard traceback instead, messages and source lines included: for local debugging only, since
those messages can hold personal data and secrets.
"""

import json
import logging
import math
import re
import sys
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from types import TracebackType
from typing import Literal, TextIO

from app.domain.errors import KNOWN_CODES

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
"""Accepted ``X-Request-ID`` values: 1-64 ASCII letters, digits, ``.``, ``_`` or ``-``."""

ALLOWED_FIELDS: frozenset[str] = frozenset(
    {
        "method",
        "route",
        "status",
        "duration_ms",
        "code",
        "category",
        "pages",
        "names_found",
        "matches",
        "requested_names",
        "size_bytes",
        "document_id",
        "chunks_stored",
        "already_indexed",
        "hits",
        "hits_kept",
        "top_score",
        "refused",
        "refusal_reason",
        "llm_ms",
        "ocr_ms",
        "ner_ms",
        "embed_ms",
        "store_ms",
        "search_ms",
        "sources",
        "check",
        "ready",
        "cpus_requested",
        "cpus_available",
    }
)
"""``extra=`` keys a log line may carry; every other key is dropped.

Each holds a count, size, duration (``*_ms``), status, code, id, score, boolean or a short
fixed enum (``category``, ``refusal_reason``, ``check``): the access line (``app.access``), the
error lines, the one outcome line each use case writes (``app.application.*``) and the
start-up CPU warning (``app.bootstrap``).
"""

LogFormat = Literal["json", "text"]

_FORMATS = ("json", "text")
_ACCESS_LOGGER = "uvicorn.access"
_SERVER_LOGGERS = ("uvicorn", "uvicorn.error")
# HTTP client libraries log every request at INFO with its full URL ('HTTP Request: POST
# <url>'), message text the allow-list cannot filter; their warnings and errors are kept.
_CLIENT_LOGGERS = ("httpx", "httpcore")

type Scalar = str | int | float | bool | None
type ExcInfo = tuple[type[BaseException], BaseException, TracebackType | None]

REDACTED_MESSAGE = "<message redacted>"
"""Shown in place of every exception message in a formatted traceback."""

_TRACEBACK_HEADER = "Traceback (most recent call last):"
_CAUSE_HEADER = "The above exception was the direct cause of the following exception:"
_CONTEXT_HEADER = "During handling of the above exception, another exception occurred:"
_CHAIN_LIMIT = 32
_GROUP_LIMIT = 15
_REPEAT_LIMIT = 3


def accept_request_id(header: str | None) -> str:
    """The incoming ``X-Request-ID`` when well formed, otherwise a new random id (hex)."""
    if header is not None and REQUEST_ID_PATTERN.fullmatch(header) is not None:
        return header
    return uuid.uuid4().hex


@contextmanager
def request_context(request_id: str) -> Iterator[str]:
    """Set ``request_id_var`` for the duration of the block and restore it afterwards."""
    token = request_id_var.set(request_id)
    try:
        yield request_id
    finally:
        request_id_var.reset(token)


def allowed_extras(record: logging.LogRecord) -> dict[str, Scalar]:
    """The allow-listed ``extra=`` fields of ``record`` with scalar values, in sorted order."""
    fields: dict[str, Scalar] = {}
    for key in sorted(ALLOWED_FIELDS):
        if key in record.__dict__:
            value = record.__dict__[key]
            if _is_scalar(value):
                fields[key] = value
    return fields


def _is_scalar(value: object) -> bool:
    if value is None or isinstance(value, str | int | bool):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _timestamp(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds")


def format_exception_redacted(exc_info: ExcInfo | tuple[None, None, None]) -> str:
    """The traceback of ``exc_info`` without any exception message.

    Like ``traceback.format_exception``: the chain (``__cause__`` / ``__context__``) oldest
    first with the usual separators, exception-group members indented below their group, the
    frames of each exception, and a last line ``module.Type: <message redacted>`` (with
    `` [code=...]`` for a ``ServiceError`` whose code is known).
    """
    error = exc_info[1]
    if error is None:
        return "NoneType: None"
    return "\n".join(_format_chain(error, set()))


def _format_chain(error: BaseException, seen: set[int]) -> list[str]:
    """``error`` and the exceptions it was raised from or while handling, oldest first."""
    chain: list[tuple[BaseException, str | None]] = [(error, None)]
    seen.add(id(error))
    current = error
    while len(chain) < _CHAIN_LIMIT:
        if current.__cause__ is not None:
            following, header = current.__cause__, _CAUSE_HEADER
        elif current.__context__ is not None and not current.__suppress_context__:
            following, header = current.__context__, _CONTEXT_HEADER
        else:
            break
        if id(following) in seen:
            break
        seen.add(id(following))
        chain.append((following, header))  # the header goes between it and the newer one
        current = following
    lines: list[str] = []
    for exception, separator in reversed(chain):
        lines.extend(_format_single(exception, seen))
        if separator is not None:
            lines.extend(["", separator, ""])
    return lines


def _format_single(error: BaseException, seen: set[int]) -> list[str]:
    lines: list[str] = []
    if error.__traceback__ is not None:
        lines.append(_TRACEBACK_HEADER)
        lines.extend(_format_frames(error.__traceback__))
    lines.append(_exception_line(error))
    if isinstance(error, BaseExceptionGroup):
        members = error.exceptions
        for index, member in enumerate(members[:_GROUP_LIMIT], start=1):
            lines.append(f"+---------------- {index} ----------------")
            nested = _format_chain(member, seen) if id(member) not in seen else ["<cycle>"]
            lines.extend(f"| {line}" for line in nested)
        if len(members) > _GROUP_LIMIT:
            lines.append(f"+---------------- and {len(members) - _GROUP_LIMIT} more exceptions")
        lines.append("+------------------------------------")
    return lines


def _format_frames(tb: TracebackType) -> list[str]:
    """One line per frame, without its source line; a run of identical frames (recursion) is
    collapsed like the standard traceback does."""
    lines: list[str] = []
    previous: str | None = None
    repeats = 0
    frames = traceback.StackSummary.extract(traceback.walk_tb(tb), lookup_lines=False)
    for frame in frames:
        line = f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}'
        if line == previous:
            repeats += 1
            if repeats >= _REPEAT_LIMIT:
                continue
        else:
            if repeats >= _REPEAT_LIMIT:
                lines.append(_repeated(repeats - _REPEAT_LIMIT + 1))
            previous, repeats = line, 0
        lines.append(line)
    if repeats >= _REPEAT_LIMIT:
        lines.append(_repeated(repeats - _REPEAT_LIMIT + 1))
    return lines


def _repeated(count: int) -> str:
    return f"  [Previous line repeated {count} more time{'s' if count > 1 else ''}]"


def _exception_line(error: BaseException) -> str:
    kind = type(error)
    name = kind.__qualname__
    if kind.__module__ not in ("builtins", "__main__"):
        name = f"{kind.__module__}.{name}"
    code = getattr(error, "code", None)
    known = isinstance(code, str) and code in KNOWN_CODES
    return f"{name}: {REDACTED_MESSAGE}" + (f" [code={code}]" if known else "")


class _ServiceFormatter(logging.Formatter):
    """Renders exceptions redacted, or in full when ``exception_messages`` is set."""

    def __init__(self, *, exception_messages: bool = False) -> None:
        super().__init__()
        self.exception_messages = exception_messages

    def formatException(self, ei: ExcInfo | tuple[None, None, None]) -> str:
        if self.exception_messages:
            return super().formatException(ei)
        return format_exception_redacted(ei)


class JsonFormatter(_ServiceFormatter):
    """One JSON object per line: timestamp (ISO-8601, UTC), level, logger, message, request
    id (inside a request), allow-listed extras and, when present, the exception (redacted
    unless ``exception_messages``)."""

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, object] = {
            "timestamp": _timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id is not None:
            line["request_id"] = request_id
        line.update(allowed_extras(record))
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            line["stack"] = self.formatStack(record.stack_info)
        return json.dumps(line, allow_nan=False)


class TextFormatter(_ServiceFormatter):
    """Human-readable line: ``timestamp LEVEL logger [request id] message key=value ...``,
    followed by the exception (redacted unless ``exception_messages``) on the next lines."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = request_id_var.get() or "-"
        parts = [_timestamp(record), record.levelname, record.name, f"[{request_id}]"]
        parts.append(record.getMessage())
        parts.extend(f"{key}={json.dumps(value)}" for key, value in allowed_extras(record).items())
        text = " ".join(parts)
        if record.exc_info:
            text = f"{text}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            text = f"{text}\n{self.formatStack(record.stack_info)}"
        return text


class _ServiceHandler(logging.StreamHandler[TextIO]):
    """The stdout handler installed by ``configure_logging`` (recognised when replacing it)."""


def configure_logging(level: str, fmt: str, *, exception_messages: bool = False) -> None:
    """Send every log line to stdout in ``fmt`` (``json`` or ``text``) from ``level`` up.

    Exceptions are rendered redacted; ``exception_messages=True`` (``LOG_EXCEPTION_MESSAGES``,
    local debugging only) renders them with their messages.

    Idempotent: a second call replaces the handler of the first, so the root logger keeps one
    service handler (handlers installed by others, such as pytest's capture, are left alone).
    Uvicorn's access log is disabled because the service writes its own access line; its error
    log is kept and goes through the same handler. The HTTP client libraries log from WARNING
    up, so their per-request lines (with full URLs) are not written.
    """
    if fmt not in _FORMATS:
        raise ValueError(f"log format must be one of {', '.join(_FORMATS)}")
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError("log level must be a standard level name")

    handler = _ServiceHandler(sys.stdout)
    formatter = JsonFormatter if fmt == "json" else TextFormatter
    handler.setFormatter(formatter(exception_messages=exception_messages))

    root = logging.getLogger()
    for existing in [h for h in root.handlers if isinstance(h, _ServiceHandler)]:
        root.removeHandler(existing)
        existing.close()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    for name in _SERVER_LOGGERS:
        server_logger = logging.getLogger(name)
        server_logger.handlers.clear()
        server_logger.propagate = True
    access = logging.getLogger(_ACCESS_LOGGER)
    access.handlers.clear()
    access.setLevel(logging.WARNING)
    access.propagate = False
    access.disabled = True
    for name in _CLIENT_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
