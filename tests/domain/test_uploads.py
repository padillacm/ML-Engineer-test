import pytest

from app.domain.errors import ConfigurationError, InvalidInput
from app.domain.uploads import HEADER_SEARCH_BYTES, PDF_HEADER, has_pdf_header, validate_pdf_bytes

LIMIT = 4096
PDF = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n%%EOF\n"


def _code(data: bytes, max_bytes: int = LIMIT) -> str:
    with pytest.raises(InvalidInput) as caught:
        validate_pdf_bytes(data, max_bytes)
    return caught.value.code


def test_constants() -> None:
    assert PDF_HEADER == b"%PDF-"
    assert HEADER_SEARCH_BYTES == 1024


def test_minimal_pdf_is_accepted() -> None:
    validate_pdf_bytes(PDF, LIMIT)


def test_empty_file_is_rejected() -> None:
    assert _code(b"") == "empty_file"


def test_exactly_the_limit_is_accepted() -> None:
    data = PDF + b"\0" * (LIMIT - len(PDF))
    assert len(data) == LIMIT

    validate_pdf_bytes(data, LIMIT)


def test_one_byte_over_the_limit_is_rejected() -> None:
    data = PDF + b"\0" * (LIMIT + 1 - len(PDF))

    assert _code(data) == "too_large"


def test_limit_of_one_byte() -> None:
    assert _code(b"%", 1) == "not_a_pdf"
    assert _code(b"%P", 1) == "too_large"


@pytest.mark.parametrize("offset", [0, 1, 512, HEADER_SEARCH_BYTES - 1])
def test_header_starting_within_the_first_1024_bytes_is_accepted(offset: int) -> None:
    data = b"j" * offset + PDF

    validate_pdf_bytes(data, LIMIT)
    assert has_pdf_header(data)


@pytest.mark.parametrize("offset", [HEADER_SEARCH_BYTES, HEADER_SEARCH_BYTES + 1])
def test_header_starting_at_1024_or_later_is_rejected(offset: int) -> None:
    data = b"j" * offset + PDF

    assert _code(data) == "not_a_pdf"
    assert not has_pdf_header(data)


def test_header_cut_by_the_end_of_the_file_is_rejected() -> None:
    assert _code(b"j" * 10 + b"%PDF") == "not_a_pdf"


def test_header_only_after_two_kilobytes_is_rejected() -> None:
    data = b"\x00" * 2048 + PDF

    assert _code(data) == "not_a_pdf"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\0" * 32, id="png"),
        pytest.param(b"PK\x03\x04\x14\x00\x00\x00\x08\x00" + b"\0" * 32, id="zip"),
        pytest.param(b"<!DOCTYPE html><html><body>%PDF</body></html>", id="html"),
        pytest.param(b"just some plain text\nwith two lines\n", id="text"),
        pytest.param(b"%pdf-1.7\n", id="lowercase-header"),
    ],
)
def test_other_formats_are_rejected(data: bytes) -> None:
    assert _code(data) == "not_a_pdf"


def test_checks_run_in_order_empty_size_header() -> None:
    assert _code(b"", 1) == "empty_file"
    assert _code(b"not a pdf at all", 4) == "too_large"
    assert _code(b"not a pdf", 100) == "not_a_pdf"


def test_messages_do_not_echo_the_content() -> None:
    secret = b"Maria Gonzalez 4111111111111111"
    for data, limit in ((secret, LIMIT), (secret, 8)):
        with pytest.raises(InvalidInput) as caught:
            validate_pdf_bytes(data, limit)
        assert b"Maria" not in str(caught.value).encode()
        assert "Maria" not in repr(caught.value)


def test_messages_state_the_limits() -> None:
    with pytest.raises(InvalidInput) as too_large:
        validate_pdf_bytes(PDF, 8)
    with pytest.raises(InvalidInput) as not_a_pdf:
        validate_pdf_bytes(b"text", LIMIT)

    assert "8 bytes" in too_large.value.message
    assert "1024 bytes" in not_a_pdf.value.message


@pytest.mark.parametrize("max_bytes", [0, -1, True, 1.5, "10", None])
def test_invalid_limit_is_a_configuration_error(max_bytes: object) -> None:
    with pytest.raises(ConfigurationError) as caught:
        validate_pdf_bytes(PDF, max_bytes)  # type: ignore[arg-type]

    assert caught.value.code == "invalid_configuration"
    assert "max_bytes" in caught.value.message
