"""The documentation: no broken local link, a pinned site, no file name that only differs in case.

Links are checked as GitHub resolves them (relative to the file that holds them); the docsify
site maps the same links onto its routes (``docs/index.html``) and ``scripts/serve_docs.py`` must
serve every page the site links. Links inside code blocks and inline code are not links.
"""

import re
from collections import defaultdict
from pathlib import Path

import pytest

from scripts.serve_docs import File, resolve

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SOURCES = sorted(
    [*DOCS.glob("*.md"), ROOT / "README.md", ROOT / "DESIGN.md", ROOT / "DECISIONS.md"]
)
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
EXTERNAL = ("http://", "https://", "mailto:", "#")
pytestmark = pytest.mark.skipif(
    not (DOCS / "index.html").is_file(), reason="docs/ is not in the images (.dockerignore)"
)
PINNED_ASSET = re.compile(
    r"<(?:script|link)\b[^>]*?(?:src|href)=\"(https://[^\"]+)\"[^>]*>", re.IGNORECASE
)


def _links(path: Path) -> list[str]:
    text = INLINE_CODE.sub("", FENCE.sub("", path.read_text(encoding="utf-8")))
    return [target for target in LINK.findall(text) if not target.startswith(EXTERNAL)]


def _target(source: Path, link: str) -> Path:
    return (source.parent / link.split("#", 1)[0].split("?", 1)[0]).resolve()


def test_the_documents_to_check_exist() -> None:
    assert {path.name for path in SOURCES} >= {
        "README.md",
        "DESIGN.md",
        "DECISIONS.md",
        "architecture.md",
        "getting-started.md",
        "benchmark-summary.md",
        "BENCHMARKS.md",
        "SECURITY.md",
        "KNOWN_LIMITATIONS.md",
    }


@pytest.mark.parametrize("source", [*SOURCES, DOCS / "_sidebar.md"], ids=lambda p: p.name)
def test_every_local_link_points_to_an_existing_file(source: Path) -> None:
    missing = [link for link in _links(source) if not _target(source, link).is_file()]
    assert not missing, f"{source.relative_to(ROOT)} links to missing files: {missing}"


@pytest.mark.parametrize("source", [*DOCS.glob("*.md"), DOCS / "_sidebar.md"], ids=lambda p: p.name)
def test_every_page_the_site_links_is_served(source: Path) -> None:
    """A Markdown page linked from the site must be reachable through the docs server."""
    pages = [link for link in _links(source) if link.split("#", 1)[0].endswith(".md")]
    for link in pages:
        target = _target(source, link)
        url = "/" + target.relative_to(ROOT).as_posix()
        assert isinstance(resolve(url, ROOT), File), f"{source.name}: {link} is not served"


def test_the_deliverables_are_at_the_root_and_in_the_sidebar() -> None:
    sidebar = _links(DOCS / "_sidebar.md")
    assert "../DESIGN.md" in sidebar
    assert "../DECISIONS.md" in sidebar
    for name in ("DESIGN.md", "DECISIONS.md"):
        assert (ROOT / name).is_file()


def test_no_two_files_differ_only_in_case() -> None:
    """``benchmarks.md`` and ``BENCHMARKS.md`` are one file on Windows and macOS."""
    seen: defaultdict[tuple[Path, str], list[str]] = defaultdict(list)
    for path in [*DOCS.rglob("*"), *ROOT.glob("*")]:
        seen[(path.parent, path.name.casefold())].append(path.name)
    clashes = [names for names in seen.values() if len(names) > 1]
    assert not clashes, clashes


def test_every_external_asset_is_pinned_with_integrity() -> None:
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    tags = [match.group(0) for match in PINNED_ASSET.finditer(html)]
    assert len(tags) >= 4, "docsify, its theme, search and mermaid"
    for tag in tags:
        url = PINNED_ASSET.match(tag)
        assert url is not None
        assert url.group(1).startswith("https://cdn.jsdelivr.net/npm/"), tag
        assert re.search(r"@\d+\.\d+\.\d+/", url.group(1)), f"not an exact version: {tag}"
        assert re.search(r'integrity="sha384-[A-Za-z0-9+/]{64}"', tag), f"no integrity: {tag}"
        assert 'crossorigin="anonymous"' in tag, tag


def test_the_site_maps_the_root_deliverables() -> None:
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert '"/DESIGN": "../DESIGN.md"' in html
    assert '"/DECISIONS": "../DECISIONS.md"' in html


FENCE_LINE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$")
MERMAID_BLOCK = re.compile(r"^```mermaid\n(.*?)^```", re.MULTILINE | re.DOTALL)
DIAGRAM_TYPES = ("flowchart ", "sequenceDiagram")


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_every_code_fence_is_well_formed(source: Path) -> None:
    """A fence is alone on its line and every block is closed: a fence written inside a line
    (or a block collapsed onto one line) renders as text, not as code or a diagram."""
    lines = source.read_text(encoding="utf-8").splitlines()
    fences = [number for number, line in enumerate(lines, 1) if "```" in line]
    broken = [number for number in fences if not FENCE_LINE.match(lines[number - 1])]
    assert not broken, f"{source.name}: fence not alone on its line at lines {broken}"
    assert len(fences) % 2 == 0, f"{source.name}: a code block is not closed"


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_every_diagram_starts_with_a_diagram_type(source: Path) -> None:
    for block in MERMAID_BLOCK.findall(source.read_text(encoding="utf-8")):
        first = block.lstrip().splitlines()[0] if block.strip() else ""
        assert first.startswith(DIAGRAM_TYPES), f"{source.name}: mermaid block starts {first!r}"
        assert len(block.strip().splitlines()) > 1, f"{source.name}: one-line mermaid block"


def test_the_configuration_page_lists_every_setting() -> None:
    from app.settings import RETIRED_VARIABLES, Settings  # noqa: PLC0415 - only this test

    page = (DOCS / "configuration.md").read_text(encoding="utf-8")
    missing = [
        name.upper()
        for name in Settings.model_fields
        if name not in RETIRED_VARIABLES and f"`{name.upper()}`" not in page
    ]
    assert not missing, f"docs/configuration.md does not list {missing}"
