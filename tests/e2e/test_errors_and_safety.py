"""Errors and safety of the audit viewer in headless Chromium (spec: audit-viewer).

API errors from the real service (a file that is not a PDF, an oversized upload, names the API
rejects), names JSON the page blocks before sending, server errors and a network failure
injected by route interception, a canned response whose names carry markup (shown as text, no
element created, no script run), and the request and CSP record of a full audit, with a control
proving that the CSP listener reports a violation.
"""

import json
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("playwright.sync_api", reason="Playwright is installed only in the e2e image")

from playwright.sync_api import Route

from tests.e2e.viewer import Session, response_json

pytestmark = pytest.mark.e2e

MEMO = "company_memo.pdf"
MAX_UPLOAD_MB = 20  # the service's default (MAX_UPLOAD_MB); the stack runs with it
MARKUP_NAMES = (
    "<img src=x onerror=window.__xss=1>",
    "<script>window.__xss=2</script>",
    '"><svg onload=window.__xss=3>',
)
NO_VALUE = chr(0x2013)  # the page shows an en dash for "none"
MARKUP_MATCH = "<b>bold</b><img src=y onerror=window.__xss=4>"


def _error(session: Session) -> tuple[str | None, str]:
    page = session.page
    assert page.get_attribute("body", "data-state") == "error"
    assert page.locator("#error").is_visible()
    return page.get_attribute("#error", "data-code"), page.text_content("#error") or ""


def _no_result(session: Session) -> None:
    assert session.page.locator(".box").count() == 0
    assert session.page.locator(".page").count() == 0


def _posts(session: Session) -> list[str]:
    return [url for method, url in session.requests if method == "POST"]


def test_a_file_that_is_not_a_pdf(open_viewer: Callable[..., Session]) -> None:
    session = open_viewer()

    response = session.extract(b"just some text, not a PDF\n", filename="notes.pdf")

    body = response_json(response)
    assert response.status == 415
    assert body["code"] == "not_a_pdf"
    code, text = _error(session)
    assert code == "not_a_pdf"
    assert text == f"not_a_pdf: {body['detail']}"
    assert session.page.text_content("#request-id") == response.headers["x-request-id"]
    assert not session.page.locator("#download-json").is_disabled()
    _no_result(session)
    session.check(http_errors=True)


def test_an_oversized_upload(open_viewer: Callable[..., Session]) -> None:
    session = open_viewer()
    data = b"%PDF-1.7\n" + b"0" * (MAX_UPLOAD_MB * 1024 * 1024 + 512 * 1024)

    response = session.extract(data, filename="large.pdf")

    assert response.status == 413
    code, text = _error(session)
    assert code == "too_large"
    assert text.startswith("too_large: ")
    _no_result(session)
    session.check(http_errors=True)


def test_names_the_api_rejects_are_shown_with_its_code(
    open_viewer: Callable[..., Session], samples: dict[str, bytes]
) -> None:
    session = open_viewer()

    response = session.extract(samples[MEMO], '[{"first_name": " ", "last_name": "Chen"}]')

    assert 400 <= response.status < 500
    code, _ = _error(session)
    assert code == "invalid_names"
    _no_result(session)
    session.check(http_errors=True)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ('[{"first_name": "Ana", "last_name": ', "Not valid JSON"),
        ('{"first_name": "Ana", "last_name": "Diaz"}', "The JSON must be an array"),
        ("Madonna", "needs a first and a last name"),
    ],
)
def test_invalid_names_are_blocked_before_sending(
    open_viewer: Callable[..., Session], samples: dict[str, bytes], text: str, problem: str
) -> None:
    session = open_viewer()
    page = session.page

    session.choose(samples[MEMO], text)

    assert page.locator("#extract").is_disabled()
    assert page.get_attribute("#names", "aria-invalid") == "true"
    assert problem in (page.text_content("#names-problems") or "")
    assert page.text_content("#names-preview") == NO_VALUE
    page.locator("#extract-form").evaluate("form => form.requestSubmit()")
    page.wait_for_timeout(300)
    assert _posts(session) == []
    assert page.get_attribute("body", "data-state") == "idle"
    session.check()


def _fulfil(
    status: int, body: str, content_type: str, request_id: str | None
) -> Callable[[Route], None]:
    headers = {"X-Request-ID": request_id} if request_id else {}

    def handle(route: Route) -> None:
        route.fulfill(status=status, body=body, content_type=content_type, headers=headers)

    return handle


def test_a_server_error_with_a_code(
    open_viewer: Callable[..., Session], samples: dict[str, bytes]
) -> None:
    session = open_viewer()
    body = {
        "detail": "the service is busy, retry later",
        "code": "service_busy",
        "request_id": "rid-503",
    }
    session.page.route(
        "**/api/extract", _fulfil(503, json.dumps(body), "application/json", "rid-503")
    )

    session.extract(samples[MEMO])

    code, text = _error(session)
    assert code == "service_busy"
    assert text == "service_busy: the service is busy, retry later"
    assert session.page.text_content("#request-id") == "rid-503"
    assert "HTTP 503" in (session.page.text_content("#status") or "")
    _no_result(session)
    session.check(http_errors=True)


def test_a_server_error_without_json(
    open_viewer: Callable[..., Session], samples: dict[str, bytes]
) -> None:
    session = open_viewer()
    session.page.route(
        "**/api/extract", _fulfil(502, "<html><b>Bad gateway</b></html>", "text/html", None)
    )

    session.extract(samples[MEMO])

    code, text = _error(session)
    assert code is None
    assert text == "HTTP 502"
    assert session.page.locator("#download-json").is_disabled()
    assert session.page.locator("#error b").count() == 0
    _no_result(session)
    session.check(http_errors=True)


def test_a_network_failure(open_viewer: Callable[..., Session], samples: dict[str, bytes]) -> None:
    session = open_viewer()
    session.page.route("**/api/extract", lambda route: route.abort("connectionreset"))
    session.choose(samples[MEMO])

    session.page.click("#extract")
    session.page.wait_for_selector("body[data-state=error]")

    _, text = _error(session)
    assert text.startswith("Network error: ")
    _no_result(session)
    session.check(http_errors=True)


def _markup_response() -> dict[str, Any]:
    names = [
        {
            "name": name,
            "bounding_box": {
                "page_number": 0,
                "x": 72.0,
                "y": 100.0 + 40 * i,
                "width": 160.0,
                "height": 14.0,
            },
        }
        for i, name in enumerate(MARKUP_NAMES)
    ]
    return {
        "extracted_names": names,
        "fuzzy_matches": [
            {"extracted_name": MARKUP_NAMES[0], "matched_name": MARKUP_MATCH, "score": 0.95}
        ],
    }


def test_names_with_markup_are_shown_as_text(
    open_viewer: Callable[..., Session], samples: dict[str, bytes]
) -> None:
    session = open_viewer()
    page = session.page
    canned = _markup_response()
    page.route("**/api/extract", _fulfil(200, json.dumps(canned), "application/json", "rid-markup"))

    session.extract(samples[MEMO])
    page.click('#name-list li[data-box-id="box-0"] button')
    page.hover("#box-1")
    page.wait_for_timeout(500)

    assert page.get_attribute("body", "data-state") == "done"
    assert page.locator("#name-list .item-name").all_text_contents() == list(MARKUP_NAMES)
    assert (
        page.locator("#name-list .item-match").first.text_content()
        == f"matched: {MARKUP_MATCH} (0.950)"
    )
    assert page.text_content("#details h3") == MARKUP_NAMES[1]
    assert (page.get_attribute("#box-0", "aria-label") or "").startswith(
        f"{MARKUP_NAMES[0]}; page 0; "
    )
    created = page.evaluate(
        "() => document.querySelectorAll('body img, body svg, body b, body script').length"
    )
    assert created == 0
    assert page.evaluate("() => window.__xss === undefined")
    session.check()


def test_a_full_audit_reaches_only_the_app(
    open_viewer: Callable[..., Session], samples: dict[str, bytes]
) -> None:
    session = open_viewer()

    response = session.extract(samples["meeting_minutes.pdf"], "Elizabeth Park")
    session.page.evaluate("() => window.__auditViewer.scrollAllPages()")

    assert response.status == 200
    paths = {(method, url.removeprefix(session.origin)) for method, url in session.requests}
    assert {
        ("GET", "/audit"),
        ("GET", "/audit/static/app.mjs"),
        ("GET", "/audit/static/viewer.css"),
        ("GET", "/audit/vendor/pdfjs/pdf.min.mjs"),
        ("GET", "/audit/vendor/pdfjs/pdf.worker.min.mjs"),
        ("POST", "/api/extract"),
    } <= paths
    assert session.page.locator('.page[data-rendered="true"]').count() >= 1
    page_response = session.page.request.get(session.url)
    assert "script-src 'self'" in page_response.headers["content-security-policy"]
    session.check()


def test_the_csp_listener_reports_a_violation(open_viewer: Callable[..., Session]) -> None:
    """Control: an inline script is refused by the page's CSP and the listener records it."""
    session = open_viewer()

    session.page.evaluate(
        "() => { const s = document.createElement('script'); s.textContent = 'window.__ran = 1';"
        " document.body.append(s); }"
    )
    session.page.wait_for_function("() => window.__cspViolations.length > 0")

    assert session.page.evaluate("() => window.__ran === undefined")
    assert session.csp_violations()[0]["directive"].startswith("script-src")


def test_a_validation_list_of_unusual_shape_is_shown_as_text(
    open_viewer: Callable[..., Session], samples: dict[str, bytes]
) -> None:
    """``detail`` entries that are not ``{loc: [...], msg: "..."}`` and a numeric request id in
    the body (no header): read defensively, never a page error."""
    session = open_viewer()
    body = {
        "detail": [
            {"loc": "body", "msg": 5},
            "plain text",
            None,
            {"loc": ["body", 3, "names"], "msg": "bad"},
            {"loc": None},
        ],
        "code": "invalid_names",
        "request_id": 12345,
    }
    session.page.route("**/api/extract", _fulfil(422, json.dumps(body), "application/json", None))

    session.extract(samples[MEMO])

    code, text = _error(session)
    assert code == "invalid_names"
    assert text == "invalid_names: 5; plain text; body.3.names: bad"
    assert session.page.text_content("#request-id") == "12345"
    with session.page.expect_download() as info:
        session.page.click("#download-json")
    assert info.value.suggested_filename == "extract-12345.json"
    _no_result(session)
    session.check(http_errors=True)


def test_a_body_with_unexpected_types_falls_back_to_the_status(
    open_viewer: Callable[..., Session], samples: dict[str, bytes]
) -> None:
    session = open_viewer()
    body = {"detail": {"nested": 1}, "code": 3, "request_id": {"x": 1}}
    session.page.route("**/api/extract", _fulfil(500, json.dumps(body), "application/json", None))

    session.extract(samples[MEMO])

    code, text = _error(session)
    assert code is None
    assert text == "HTTP 500"
    assert session.page.text_content("#request-id") == NO_VALUE
    with session.page.expect_download() as info:
        session.page.click("#download-json")
    assert info.value.suggested_filename == "extract-response.json"
    _no_result(session)
    session.check(http_errors=True)


def test_a_reload_revalidates_the_page_files_with_304(open_viewer: Callable[..., Session]) -> None:
    """The page's own files are ``no-cache`` with an ``ETag``: a reload asks again and gets 304
    (no body); the page itself is ``no-store`` and comes back whole."""
    session = open_viewer()
    statuses: dict[str, int] = {}
    session.page.on(
        "response",
        lambda response: statuses.setdefault(
            response.url.removeprefix(session.origin), response.status
        ),
    )

    session.page.reload(wait_until="load")
    session.page.evaluate("() => window.__auditViewer.ready")

    assert statuses["/audit"] == 200
    static = {
        path: status for path, status in statuses.items() if path.startswith("/audit/static/")
    }
    assert static
    assert set(static.values()) == {304}, static
    assert session.page.get_attribute("body", "data-state") == "idle"
    session.check()
