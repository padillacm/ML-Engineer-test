"""``POST /api/extract`` (spec: http-api)."""

import pytest
from fastapi.testclient import TestClient

from app.api.uploads import MAX_NAME_CHARS
from app.domain.cancellation import NEVER_CANCELLED, Cancellation
from app.domain.errors import InvalidInput
from app.domain.types import OcrPage
from tests.api.conftest import ClientFactory
from tests.api.support import MIB, PDF, PNG, Fakes, names_json
from tests.fakes.ocr import FakeOcrEngine


def test_extraction_returns_boxes_in_pdf_points_and_matches(client: TestClient) -> None:
    response = client.post(
        "/api/extract",
        files={"pdf_file": ("scan.pdf", PDF, "application/pdf")},
        data={"names": names_json(("Maria", "Gonzalez"))},
    )

    assert response.status_code == 200
    body = response.json()
    (extracted,) = body["extracted_names"]
    assert extracted["name"] == "María González"
    # The fake page lays words out left to right, 6 points per character and per space.
    assert extracted["bounding_box"] == {
        "page_number": 0,
        "x": 50.0 + 6.0 * (len("The contract was signed by") + 1),
        "y": 50.0,
        "width": 6.0 * len("María González"),
        "height": 12.0,
    }
    (match,) = body["fuzzy_matches"]
    assert match["extracted_name"] == "María González"
    assert match["matched_name"] == "Maria Gonzalez"
    assert 0.0 <= match["score"] <= 1.0


def test_names_default_to_none_requested(client: TestClient) -> None:
    response = client.post("/api/extract", files={"pdf_file": ("a.pdf", PDF, "application/pdf")})

    assert response.status_code == 200
    assert response.json()["fuzzy_matches"] == []
    assert len(response.json()["extracted_names"]) == 1


def test_language_hint_reaches_the_ocr_engine(client: TestClient, fakes: Fakes) -> None:
    response = client.post(
        "/api/extract",
        files={"pdf_file": ("a.pdf", PDF, "application/pdf")},
        data={"language": " ES "},
    )

    assert response.status_code == 200
    assert isinstance(fakes.ocr, FakeOcrEngine)
    assert fakes.ocr.calls == [(PDF, "es")]


def test_png_renamed_pdf_is_415(client: TestClient, fakes: Fakes) -> None:
    response = client.post("/api/extract", files={"pdf_file": ("scan.pdf", PNG, "application/pdf")})

    assert response.status_code == 415
    assert response.json()["code"] == "not_a_pdf"
    assert isinstance(fakes.ocr, FakeOcrEngine)
    assert fakes.ocr.calls == []


def test_empty_file_is_400(client: TestClient) -> None:
    response = client.post("/api/extract", files={"pdf_file": ("scan.pdf", b"", "application/pdf")})

    assert response.status_code == 400
    assert response.json()["code"] == "empty_file"


def test_missing_file_is_422_invalid_request(client: TestClient) -> None:
    response = client.post("/api/extract", data={"names": "[]"})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_request"
    assert isinstance(body["detail"], list)
    assert body["detail"][0]["loc"] == ["body", "pdf_file"]
    assert body["request_id"] == response.headers["X-Request-ID"]


class _Refusing:
    """OCR engine that refuses every document with a scripted core error."""

    def __init__(self, error: InvalidInput) -> None:
        self.error = error

    def recognize(
        self, pdf: bytes, language: str | None = None, *, cancel: Cancellation = NEVER_CANCELLED
    ) -> list[OcrPage]:
        raise self.error


@pytest.mark.parametrize(("code", "status"), [("too_many_pages", 413), ("pdf_unreadable", 422)])
def test_ocr_refusals_map_to_their_status(
    make_client: ClientFactory, fakes: Fakes, code: str, status: int
) -> None:
    fakes.ocr = _Refusing(InvalidInput("refused by the engine", code=code))
    client = make_client()

    response = client.post("/api/extract", files={"pdf_file": ("a.pdf", PDF, "application/pdf")})

    assert response.status_code == status
    assert response.json()["code"] == code


@pytest.mark.parametrize(
    ("names", "entry"),
    [
        ("not json", None),
        ('{"first_name": "Ana", "last_name": "Ruiz"}', None),
        ('[{"first_name": "Ana", "last_name": "Ruiz"}, {"first_name": "Bo"}]', "names[1]"),
        ('[{"first_name": "Ana", "last_name": ""}]', "names[0]"),
        ('[{"first_name": "Ana", "last_name": "   "}]', "names[0]"),
        ('[{"first_name": 3, "last_name": "Ruiz"}]', "names[0]"),
        ('[["Ana", "Ruiz"]]', "names[0]"),
        ("[" * 100_000 + "]" * 100_000, None),
    ],
    ids=["not-json", "object", "missing-last", "empty", "blank", "number", "array", "deep"],
)
def test_malformed_names_are_422_naming_the_entry(
    client: TestClient, fakes: Fakes, names: str, entry: str | None
) -> None:
    response = client.post(
        "/api/extract",
        files={"pdf_file": ("a.pdf", PDF, "application/pdf")},
        data={"names": names},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_names"
    if entry is not None:
        assert entry in body["detail"]
    assert "Ana" not in body["detail"]
    assert "Ruiz" not in body["detail"]
    assert isinstance(fakes.ocr, FakeOcrEngine)
    assert fakes.ocr.calls == []


def test_missing_last_name_is_reported_as_required(client: TestClient) -> None:
    response = client.post(
        "/api/extract",
        files={"pdf_file": ("a.pdf", PDF, "application/pdf")},
        data={"names": '[{"first_name": "Ana"}]'},
    )

    assert response.json()["detail"] == "names[0]: last_name is required"


@pytest.mark.parametrize(
    ("pairs", "detail"),
    [
        (
            [("Ana", "Ruiz"), ("Ana" * 34, "Ruiz")],
            "names[1]: first_name must be at most 100 characters",
        ),
        (
            [("Ana" * 34, "Ruiz" * 26)],
            "names[0]: first_name and last_name must each be at most 100 characters",
        ),
    ],
    ids=["first", "both"],
)
def test_names_over_100_characters_are_422_naming_the_entry_not_the_value(
    client: TestClient, fakes: Fakes, pairs: list[tuple[str, str]], detail: str
) -> None:
    response = client.post(
        "/api/extract",
        files={"pdf_file": ("a.pdf", PDF, "application/pdf")},
        data={"names": names_json(*pairs)},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_names"
    assert response.json()["detail"] == detail
    assert isinstance(fakes.ocr, FakeOcrEngine)
    assert fakes.ocr.calls == []


def test_100_characters_after_stripping_are_accepted(client: TestClient) -> None:
    longest = names_json(("  " + "a" * MAX_NAME_CHARS + "  ", "é" * MAX_NAME_CHARS))

    response = client.post(
        "/api/extract",
        files={"pdf_file": ("a.pdf", PDF, "application/pdf")},
        data={"names": longest},
    )

    assert response.status_code == 200


def test_too_many_names_is_422(client: TestClient) -> None:
    six = names_json(*[("Ana", f"Ruiz{index}") for index in range(6)])

    response = client.post(
        "/api/extract", files={"pdf_file": ("a.pdf", PDF, "application/pdf")}, data={"names": six}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_names"
    assert "at most 5" in response.json()["detail"]


def test_unsupported_language_is_422(client: TestClient) -> None:
    response = client.post(
        "/api/extract",
        files={"pdf_file": ("a.pdf", PDF, "application/pdf")},
        data={"language": "it"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "unsupported_language"


def test_file_one_byte_over_the_limit_is_413_and_no_ocr_runs(
    client: TestClient, fakes: Fakes
) -> None:
    oversized = PDF + b"0" * (MIB + 1 - len(PDF))

    response = client.post(
        "/api/extract", files={"pdf_file": ("a.pdf", oversized, "application/pdf")}
    )

    assert response.status_code == 413
    assert response.json()["code"] == "too_large"
    assert isinstance(fakes.ocr, FakeOcrEngine)
    assert fakes.ocr.calls == []


def test_file_at_the_limit_is_accepted(client: TestClient) -> None:
    exact = PDF + b"0" * (MIB - len(PDF))

    response = client.post("/api/extract", files={"pdf_file": ("a.pdf", exact, "application/pdf")})

    assert response.status_code == 200
