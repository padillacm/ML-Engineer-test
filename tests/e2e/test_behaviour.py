"""Interaction with the audit viewer in headless Chromium (spec: audit-viewer).

One real extraction of the frame PDF (seven pages, matched and unmatched names), then: the
response drawn with its request id, selection from the list, the filter, zoom, keyboard order
and accessible labels, hover details and the raw JSON download. Every test ends with the
session's safety check (no CSP violation, no request outside the app, no console error).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("playwright.sync_api", reason="Playwright is installed only in the e2e image")

from tests.e2e import frames
from tests.e2e.viewer import Session, position_errors, response_json

pytestmark = pytest.mark.e2e

SELECTED_PAGE = 3
NO_VALUE = chr(0x2013)  # the page shows an en dash for "none"


@pytest.fixture(scope="module")
def audited(
    open_viewer: Callable[..., Session], frame_document: frames.FrameDocument
) -> tuple[Session, dict[str, Any], str]:
    session = open_viewer()
    response = session.extract(frame_document.pdf, frames.NAMES_TEXT, "en", filename="frames.pdf")
    assert response.status == 200
    return session, response_json(response), response.body().decode("utf-8")


def _points(value: float) -> str:
    return f"{value:.1f}"


def _matches(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for match in body["fuzzy_matches"]:
        known = best.get(match["extracted_name"])
        if known is None or match["score"] > known["score"]:
            best[match["extracted_name"]] = match
    return best


def test_the_response_is_drawn_with_its_request_id(
    audited: tuple[Session, dict[str, Any], str],
) -> None:
    session, body, _ = audited
    page = session.page

    assert page.get_attribute("body", "data-state") == "done"
    assert page.locator(".box").count() == len(body["extracted_names"])
    assert page.locator("#name-list li[data-box-id]").count() == len(body["extracted_names"])
    headings = page.locator("#name-list li.list-page")
    assert headings.count() == len(frames.PAGES)
    request_id = page.text_content("#request-id")
    assert request_id
    assert request_id != NO_VALUE
    assert page.locator("#error").is_hidden()
    assert "Done in" in (page.text_content("#status") or "")
    session.check()


def test_selecting_a_name_in_the_list_scrolls_to_its_box(
    audited: tuple[Session, dict[str, Any], str],
) -> None:
    session, body, _ = audited
    page = session.page
    session.zoom("2")
    page.evaluate("() => window.scrollTo(0, 0)")
    number = next(
        i
        for i, e in enumerate(body["extracted_names"])
        if e["bounding_box"]["page_number"] == SELECTED_PAGE
    )
    entry = body["extracted_names"][number]

    page.click(f'#name-list li[data-box-id="box-{number}"] button')

    box = page.locator(f"#box-{number}")
    item = page.locator(f'#name-list li[data-box-id="box-{number}"]')
    assert "selected" in (box.get_attribute("class") or "")
    assert box.get_attribute("aria-current") == "true"
    assert "selected" in (item.get_attribute("class") or "")
    assert page.locator(".box.selected").count() == 1
    assert page.get_attribute("#details", "data-box-id") == f"box-{number}"
    assert page.text_content("#details h3") == entry["name"]
    rect = box.bounding_box()
    viewport = page.viewport_size
    assert rect is not None
    assert viewport is not None
    assert rect["y"] >= 0
    assert rect["y"] + rect["height"] <= viewport["height"]
    assert rect["x"] >= 0
    assert rect["x"] + rect["width"] <= viewport["width"]
    session.check()


@pytest.mark.parametrize("choice", ["matched", "unmatched", "all"])
def test_the_filter_shows_the_chosen_names(
    audited: tuple[Session, dict[str, Any], str], choice: str
) -> None:
    session, body, _ = audited
    page = session.page
    best = _matches(body)
    expected = [
        i
        for i, e in enumerate(body["extracted_names"])
        if choice == "all" or (choice == "matched") == (e["name"] in best)
    ]

    session.filter(choice)

    visible_boxes = page.locator(".box:not([hidden])")
    visible_items = page.locator("#name-list li[data-box-id]:not([hidden])")
    assert sorted(visible_boxes.evaluate_all("els => els.map(e => e.id)")) == sorted(
        f"box-{i}" for i in expected
    )
    assert visible_items.count() == len(expected)
    assert page.locator("#name-list li.list-page:not([hidden])").count() == len(
        {body["extracted_names"][i]["bounding_box"]["page_number"] for i in expected}
    )
    if choice == "matched":
        assert len(expected) == len(frames.LISTED_NAMES)
    session.filter("all")
    session.check()


@pytest.mark.parametrize("zoom", ["1", "1.5", "2", "fit"])
def test_zoom_keeps_the_boxes_aligned(
    audited: tuple[Session, dict[str, Any], str], zoom: str
) -> None:
    session, body, _ = audited

    session.zoom(zoom)

    assert position_errors(session.box_rects(), body) == []
    session.check()


def test_tab_moves_through_the_boxes_in_page_order(
    audited: tuple[Session, dict[str, Any], str],
) -> None:
    session, body, _ = audited
    page = session.page
    best = _matches(body)
    order = sorted(
        range(len(body["extracted_names"])),
        key=lambda i: (body["extracted_names"][i]["bounding_box"]["page_number"], i),
    )
    page.locator("#name-list li[data-box-id] button").last.focus()

    visited = []
    for _ in order:
        page.keyboard.press("Tab")
        focused = page.evaluate("() => document.activeElement.id")
        visited.append(focused)
        assert page.get_attribute("#details", "data-box-id") == focused

    assert visited == [f"box-{i}" for i in order]
    for i in order:
        entry = body["extracted_names"][i]
        box = entry["bounding_box"]
        label = page.get_attribute(f"#box-{i}", "aria-label") or ""
        assert label.startswith(f"{entry['name']}; page {box['page_number']}; ")
        coordinates = (
            f"box x {_points(box['x'])}, y {_points(box['y'])}, width {_points(box['width'])}, "
            f"height {_points(box['height'])} points"
        )
        assert coordinates in label
        match = best.get(entry["name"])
        if match:
            assert f"matched to {match['matched_name']}, score {match['score']:.3f}" in label
        else:
            assert "; no match; " in label
        assert page.get_attribute(f"#box-{i}", "role") == "button"
        assert page.get_attribute(f"#box-{i}", "tabindex") == "0"
    session.check()


def test_hover_shows_the_details(audited: tuple[Session, dict[str, Any], str]) -> None:
    session, body, _ = audited
    page = session.page
    session.zoom("1")
    best = _matches(body)
    number = next(i for i, e in enumerate(body["extracted_names"]) if e["name"] in best)
    entry = body["extracted_names"][number]
    page.locator(f"#box-{number}").scroll_into_view_if_needed()

    page.hover(f"#box-{number}")

    assert page.get_attribute("#details", "data-box-id") == f"box-{number}"
    assert page.text_content("#details h3") == entry["name"]
    values = page.locator("#details dd").all_text_contents()
    box = entry["bounding_box"]
    assert values[:5] == [
        "matched",
        str(box["page_number"]),
        f"x {_points(box['x'])}, y {_points(box['y'])}, "
        f"w {_points(box['width'])}, h {_points(box['height'])}",
        best[entry["name"]]["matched_name"],
        f"{best[entry['name']]['score']:.3f}",
    ]
    session.check()


def test_download_is_the_exact_response(audited: tuple[Session, dict[str, Any], str]) -> None:
    session, _, raw = audited
    page = session.page

    with page.expect_download() as info:
        page.click("#download-json")
    download = info.value

    assert Path(download.path()).read_bytes().decode("utf-8") == raw
    request_id = page.text_content("#request-id")
    assert download.suggested_filename == f"extract-{request_id}.json"
    session.check()
