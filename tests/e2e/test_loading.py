"""The audit viewer while an extraction is in progress (spec: audit-viewer).

The API is held by route interception until the test answers, as a slow service would be. The
live status is set at the start and the end only (the ticking time is in an element that is not
announced); the names' problems are announced once typing pauses, not on every key; the file,
names and language are locked while loading and the result names the file it belongs to; and
Cancel aborts the request and returns the page to idle.
"""

import contextlib
import json
from collections.abc import Callable

import pytest

pytest.importorskip("playwright.sync_api", reason="Playwright is installed only in the e2e image")

from tests.e2e import frames
from tests.e2e.viewer import Session

pytestmark = pytest.mark.e2e

HOLD_MS = 1_300
"""Long enough for five ticks of the 250 ms timer."""
ANNOUNCE_WAIT_MS = 1_600
"""Past the page's 1 s pause before it announces the names' problems."""
LOCKED = ("#pdf-file", "#names", "#language", "#extract")


@pytest.fixture(scope="module")
def one_page() -> frames.FrameDocument:
    return frames.build(frames.PAGES[:1])


def _canned(document: frames.FrameDocument) -> str:
    return json.dumps(frames.canned_response(document))


def _start(session: Session, document: frames.FrameDocument, filename: str) -> None:
    session.choose(document.pdf, frames.NAMES_TEXT, filename=filename)
    session.page.click("#extract")
    session.page.wait_for_selector("body[data-state=loading]")


def test_only_the_start_and_the_end_are_announced(
    open_viewer: Callable[..., Session], one_page: frames.FrameDocument
) -> None:
    session = open_viewer()
    page = session.page
    held = session.hold_extract()
    session.watch_announcements()

    _start(session, one_page, "slow.pdf")
    held.wait()
    first_tick = page.text_content("#elapsed")
    page.wait_for_timeout(HOLD_MS)
    later_tick = page.text_content("#elapsed")
    held.fulfil(200, _canned(one_page), "rid-slow")
    page.wait_for_selector("body[data-state=done]")

    assert first_tick != later_tick  # the time is shown and ticks...
    status = session.announcements("status")
    assert len(status) == 2, status  # ... but only the start and the end reach the live region
    assert status[0].startswith("Extracting slow.pdf, ")
    assert status[1].startswith("Done in ")
    assert "(slow.pdf)" in status[1]
    assert page.text_content("#elapsed") == ""
    assert page.evaluate(
        "() => document.getElementById('elapsed')"
        ".closest('[aria-live], [role=status], [role=alert]') === null"
    )
    session.check()


def test_the_names_problems_are_announced_once_typing_pauses(
    open_viewer: Callable[..., Session],
) -> None:
    session = open_viewer()
    page = session.page
    session.watch_announcements()

    page.locator("#names").press_sequentially("Madonna", delay=40)
    shown_at_once = page.text_content("#names-problems") or ""
    announced_while_typing = session.announcements("names-announce")
    page.wait_for_timeout(ANNOUNCE_WAIT_MS)
    after_pause = session.announcements("names-announce")
    page.locator("#names").press_sequentially(" Ciccone", delay=40)
    page.wait_for_timeout(ANNOUNCE_WAIT_MS)

    assert "needs a first and a last name" in shown_at_once
    assert announced_while_typing == []
    assert len(after_pause) == 1
    assert after_pause[0].startswith("Names: Line 1: needs a first and a last name")
    assert session.announcements("names-announce") == [after_pause[0], "Names: no problems"]
    assert page.get_attribute("#names-problems", "aria-live") is None
    assert session.announcements("status") == []
    session.check()


def test_the_form_is_locked_while_loading_and_the_result_names_its_file(
    open_viewer: Callable[..., Session], one_page: frames.FrameDocument
) -> None:
    session = open_viewer()
    page = session.page
    held = session.hold_extract()

    _start(session, one_page, "first.pdf")
    held.wait()
    locked = {selector: page.locator(selector).is_disabled() for selector in LOCKED}
    cancel_ready = page.locator("#cancel").is_visible() and page.locator("#cancel").is_enabled()
    # A disabled input may still take files from automation (or refuse them, which is also
    # correct); the result must belong to the file sent.
    with contextlib.suppress(Exception):
        page.set_input_files(
            "#pdf-file",
            files=[{"name": "second.pdf", "mimeType": "application/pdf", "buffer": b"%PDF-"}],
            timeout=2_000,
        )
    still_locked = page.locator("#extract").is_disabled()
    held.fulfil(200, _canned(one_page), "rid-first")
    page.wait_for_selector("body[data-state=done]")

    assert locked == dict.fromkeys(LOCKED, True)
    assert cancel_ready
    assert still_locked
    status = page.text_content("#status") or ""
    assert "(first.pdf)" in status
    assert "second.pdf" not in status
    assert page.locator(".box").count() == len(one_page.names)
    for selector in ("#pdf-file", "#names", "#language"):
        assert page.locator(selector).is_enabled(), selector
    assert page.locator("#cancel").is_hidden()
    session.check()


def test_cancel_aborts_the_request_and_returns_to_idle(
    open_viewer: Callable[..., Session], one_page: frames.FrameDocument
) -> None:
    session = open_viewer()
    page = session.page
    held = session.hold_extract()

    _start(session, one_page, "cancelled.pdf")
    held.wait()
    page.click("#cancel")
    page.wait_for_selector("body[data-state=idle]")
    # The browser has dropped the request; answering it (which may be refused) changes nothing.
    with contextlib.suppress(Exception):
        held.fulfil(200, _canned(one_page), "rid-late")
    page.wait_for_timeout(500)

    status = page.text_content("#status") or ""
    assert status.startswith("Cancelled after ")
    assert "(cancelled.pdf)" in status
    assert page.get_attribute("body", "data-state") == "idle"
    assert page.locator("#error").is_hidden()
    assert page.locator(".box").count() == 0
    assert page.locator(".page").count() == 0
    assert page.text_content("#request-id") == chr(0x2013)
    assert page.locator("#cancel").is_hidden()
    for selector in LOCKED:
        assert page.locator(selector).is_enabled(), selector

    held.release()
    canned = _canned(one_page)
    page.route(
        "**/api/extract",
        lambda route: route.fulfill(status=200, body=canned, content_type="application/json"),
    )
    session.extract(one_page.pdf, frames.NAMES_TEXT, filename="again.pdf")
    assert page.get_attribute("body", "data-state") == "done"
    assert page.locator(".box").count() == len(one_page.names)
    session.check(http_errors=True)  # Chromium logs the aborted request
