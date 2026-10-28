"""Form fields of the upload endpoints, parsed into core values with core errors.

The PDF is read with at most ``max_bytes + 1`` bytes (enough to tell that it is too large) and
judged by its content (``validate_pdf_bytes``); its name and declared content type are ignored.
The upload is closed whatever happens. ``names`` is a JSON array of ``{first_name, last_name}``
objects whose names are non-blank and at most ``MAX_NAME_CHARS`` characters each; ``language``
an optional supported language code. Messages say which entry or rule is wrong but never repeat
the submitted values, which are personal data.
"""

import json

from fastapi import UploadFile
from pydantic import ValidationError

from app.domain.errors import InvalidInput
from app.domain.language import validate_language
from app.domain.types import NamePair
from app.domain.uploads import validate_pdf_bytes
from app.models import schemas

NAMES_SHAPE = 'names must be a JSON array of {"first_name": ..., "last_name": ...} objects'

MAX_NAME_CHARS = 100
"""Longest ``first_name`` or ``last_name`` accepted, in characters after stripping whitespace;
real names are far shorter, and the bound keeps the ``names`` field small."""


def read_pdf(upload: UploadFile, max_bytes: int) -> bytes:
    """The uploaded PDF's bytes; raise ``InvalidInput`` (``empty_file``, ``too_large``,
    ``not_a_pdf``) when it cannot be a PDF of at most ``max_bytes`` bytes.

    Reads synchronously (the handlers run in the thread pool) and always closes the upload.
    """
    try:
        data = upload.file.read(max_bytes + 1)
    finally:
        upload.file.close()
    validate_pdf_bytes(data, max_bytes)
    return data


def parse_names(raw: str, max_names: int) -> list[NamePair]:
    """The requested names in ``raw`` (a JSON array), as core ``NamePair`` values.

    Raises ``InvalidInput`` (``invalid_names``) when ``raw`` is not a JSON array, holds more
    than ``max_names`` entries, or an entry lacks a non-blank ``first_name`` or ``last_name`` of
    at most ``MAX_NAME_CHARS`` characters; the message names the entry by its 0-based index.
    """
    try:
        entries = json.loads(raw)
    except (ValueError, RecursionError):
        raise _invalid(f"{NAMES_SHAPE}; the value is not valid JSON") from None
    if not isinstance(entries, list):
        raise _invalid(NAMES_SHAPE)
    if len(entries) > max_names:
        raise _invalid(f"names has {len(entries)} entries; at most {max_names} are accepted")
    return [_name_pair(index, entry) for index, entry in enumerate(entries)]


def parse_language(raw: str | None) -> str | None:
    """The language hint as a supported code, or ``None`` when absent or blank.

    Raises ``InvalidInput`` (``unsupported_language``) for any other code.
    """
    if raw is None or not raw.strip():
        return None
    return validate_language(raw)


def _name_pair(index: int, entry: object) -> NamePair:
    if not isinstance(entry, dict):
        raise _invalid(f"names[{index}] must be an object with first_name and last_name")
    try:
        pair = schemas.NamePair.model_validate(entry)
    except ValidationError as error:
        raise _invalid(f"names[{index}]: {_problems(error)}") from None
    blank = [field for field in ("first_name", "last_name") if not getattr(pair, field).strip()]
    if blank:
        raise _invalid(f"names[{index}]: {' and '.join(blank)} must not be blank")
    long = [
        field
        for field in ("first_name", "last_name")
        if len(getattr(pair, field).strip()) > MAX_NAME_CHARS
    ]
    if long:
        verb = "must" if len(long) == 1 else "must each"
        raise _invalid(
            f"names[{index}]: {' and '.join(long)} {verb} be at most {MAX_NAME_CHARS} characters"
        )
    return NamePair(first_name=pair.first_name, last_name=pair.last_name)


def _problems(error: ValidationError) -> str:
    """Which fields are wrong and how, without the submitted values."""
    problems: list[str] = []
    for detail in error.errors(include_url=False, include_input=False, include_context=False):
        field = ".".join(str(part) for part in detail["loc"]) or "entry"
        problems.append(f"{field} {_rule(detail['type'])}")
    return "; ".join(problems)


def _rule(error_type: str) -> str:
    if error_type == "missing":
        return "is required"
    return "must be a non-empty string"


def _invalid(message: str) -> InvalidInput:
    return InvalidInput(message, code="invalid_names")
