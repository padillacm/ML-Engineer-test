"""The core imports no vendor library and no outer layer (spec: architecture-boundaries).

The scan is static: dynamic imports (``importlib``, ``__import__``) are not detected.
"""

from pathlib import Path

import pytest

from tests.architecture.scan import Violation, collect_imports, find_violations, is_under

APP = Path(__file__).resolve().parents[2] / "app"
DOMAIN = APP / "domain"
APPLICATION = APP / "application"
OUTER_LAYERS = (
    "app.adapters",
    "app.models",
    "app.api",
    "app.services",
    "app.main",
)


def _report(violations: list[Violation]) -> str:
    return "\n".join(f"  {v.file.relative_to(APP.parent)}:{v.line}  {v.module}" for v in violations)


def test_scanning_a_missing_directory_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_violations(tmp_path / "nope", "app.nope", frozenset(), ("app.nope",))
    with pytest.raises(FileNotFoundError):
        collect_imports(tmp_path / "nope", "app.nope")


def test_scanning_a_directory_without_python_files_fails(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("import requests", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        find_violations(tmp_path, "app.empty", frozenset(), ("app.empty",))


def test_scanner_reports_a_forbidden_import(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("import requests\nfrom os import path\n", encoding="utf-8")

    violations = find_violations(tmp_path, "app.bad", frozenset(), ("app.bad",))

    assert [(v.line, v.module) for v in violations] == [(1, "requests")]


@pytest.mark.parametrize(
    ("directory", "name", "expected"),
    [
        (
            DOMAIN,
            "app.domain",
            {"types.py", "errors.py", "geometry.py", "matching.py", "chunking.py"},
        ),
        (APPLICATION, "app.application", {"ports.py"}),
    ],
)
def test_scan_covers_the_expected_files(directory: Path, name: str, expected: set[str]) -> None:
    assert collect_imports(directory, name)
    assert expected <= {file.name for file in directory.rglob("*.py")}


def test_domain_imports_only_stdlib_rapidfuzz_and_domain() -> None:
    violations = find_violations(DOMAIN, "app.domain", frozenset({"rapidfuzz"}), ("app.domain",))
    assert not violations, f"app/domain has forbidden imports:\n{_report(violations)}"


def test_application_imports_only_stdlib_domain_and_application() -> None:
    violations = find_violations(
        APPLICATION, "app.application", frozenset(), ("app.domain", "app.application")
    )
    assert not violations, f"app/application has forbidden imports:\n{_report(violations)}"


def test_domain_does_not_import_application() -> None:
    offenders = [
        imp
        for imp in collect_imports(DOMAIN, "app.domain")
        if is_under(imp.module, ("app.application",))
    ]
    assert not offenders, f"app/domain imports app.application:\n{_report(offenders)}"


@pytest.mark.parametrize(
    ("directory", "name"), [(DOMAIN, "app.domain"), (APPLICATION, "app.application")]
)
def test_core_does_not_import_outer_layers(directory: Path, name: str) -> None:
    offenders = [
        imp for imp in collect_imports(directory, name) if is_under(imp.module, OUTER_LAYERS)
    ]
    assert not offenders, f"{name} imports an outer layer:\n{_report(offenders)}"
