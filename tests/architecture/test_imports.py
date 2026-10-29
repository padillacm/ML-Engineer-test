"""The core imports no vendor library and no outer layer, the HTTP layer only the framework,
the use cases and core types, and only the composition root imports adapters (spec:
architecture-boundaries).

The scan is static: dynamic imports (``importlib``, ``__import__``) are not detected.
"""

import sys
from pathlib import Path

import pytest

from tests.architecture.scan import Violation, collect_imports, find_violations, is_under

APP = Path(__file__).resolve().parents[2] / "app"
DOMAIN = APP / "domain"
APPLICATION = APP / "application"
API = APP / "api"
OUTER_LAYERS = (
    "app.adapters",
    "app.models",
    "app.api",
    "app.bootstrap",
    "app.main",
    "app.settings",
    "app.observability",
)
API_THIRD_PARTY = frozenset({"fastapi", "starlette", "pydantic", "anyio"})
API_INTERNAL = (
    "app.api",
    "app.application",
    "app.domain",
    "app.models",
    "app.settings",
    "app.observability",
)
COMPOSITION_ROOT = APP / "bootstrap.py"


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


def test_api_imports_only_the_framework_use_cases_and_core() -> None:
    violations = find_violations(API, "app.api", API_THIRD_PARTY, API_INTERNAL)
    assert not violations, f"app/api has forbidden imports:\n{_report(violations)}"


def test_api_guard_reports_an_adapter_or_vendor_import(tmp_path: Path) -> None:
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "router.py").write_text(
        "from fastapi import APIRouter\n"
        "from app.application.extract_names import ExtractNames\n"
        "from app.adapters.ocr.engine import TesseractOcrEngine\n"
        "import torch\n"
        "from app.bootstrap import build_container\n",
        encoding="utf-8",
    )

    violations = find_violations(tmp_path, "app.api", API_THIRD_PARTY, API_INTERNAL)

    assert [(v.file.name, v.line, v.module) for v in violations] == [
        ("router.py", 3, "app.adapters.ocr.engine"),
        ("router.py", 4, "torch"),
        ("router.py", 5, "app.bootstrap"),
    ]


def test_the_viewer_imports_only_the_framework_and_the_http_layer() -> None:
    """The audit viewer serves files: no use case, adapter, settings or composition root
    (design D5); the page calls ``/api/extract`` like any client."""
    viewer = API / "viewer.py"
    imports = [imp for imp in collect_imports(API, "app.api") if imp.file == viewer]
    assert imports, "app/api/viewer.py has no imports to check"

    offenders = [
        imp
        for imp in imports
        if not (
            is_under(imp.module, ("fastapi", "starlette"))
            or imp.module.split(".", maxsplit=1)[0] in sys.stdlib_module_names
        )
    ]
    assert not offenders, f"app/api/viewer.py has forbidden imports:\n{_report(offenders)}"


def test_only_the_composition_root_imports_adapters() -> None:
    importers = {
        imp.file
        for imp in collect_imports(APP, "app")
        if is_under(imp.module, ("app.adapters",)) and not imp.file.is_relative_to(APP / "adapters")
    }
    assert importers == {COMPOSITION_ROOT}
