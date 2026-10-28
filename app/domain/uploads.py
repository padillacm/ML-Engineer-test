"""Content checks of an uploaded PDF, made before anything tries to open it.

The file is judged by its bytes only; its name and declared content type are ignored. The PDF
header ``%PDF-`` is accepted when it starts within the first ``HEADER_SEARCH_BYTES`` bytes, the
tolerance of Acrobat and PDFium, so files with a little leading junk that PDF readers open are
not refused here; whether the rest is a readable PDF is decided when it is opened
(``pdf_unreadable``). Error messages never repeat any of the file's content.
"""

from app.domain.errors import ConfigurationError, InvalidInput

PDF_HEADER = b"%PDF-"
HEADER_SEARCH_BYTES = 1024
"""The header must start at an offset below this value."""


def validate_pdf_bytes(data: bytes, max_bytes: int) -> None:
    """Raise ``InvalidInput`` unless ``data`` may be a PDF of at most ``max_bytes`` bytes.

    Checks run in this order: empty (code ``empty_file``), longer than ``max_bytes`` (code
    ``too_large``), no ``PDF_HEADER`` starting within the first ``HEADER_SEARCH_BYTES`` bytes
    (code ``not_a_pdf``). A ``max_bytes`` that is not an integer >= 1 raises
    ``ConfigurationError``.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ConfigurationError(f"max_bytes must be an integer >= 1, got {max_bytes!r}")
    if not data:
        raise InvalidInput("the file is empty", code="empty_file")
    if len(data) > max_bytes:
        raise InvalidInput(f"the file is larger than {max_bytes} bytes", code="too_large")
    if not has_pdf_header(data):
        raise InvalidInput(
            f"the file has no PDF header within its first {HEADER_SEARCH_BYTES} bytes",
            code="not_a_pdf",
        )


def has_pdf_header(data: bytes) -> bool:
    """Whether ``PDF_HEADER`` starts at an offset below ``HEADER_SEARCH_BYTES`` in ``data``."""
    # The search end is exclusive: a header starting at the last allowed offset ends at
    # HEADER_SEARCH_BYTES - 1 + len(PDF_HEADER), and one starting any later does not fit.
    return data.find(PDF_HEADER, 0, HEADER_SEARCH_BYTES - 1 + len(PDF_HEADER)) != -1
