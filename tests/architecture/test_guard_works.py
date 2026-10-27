"""Prove the import guard reports violations, so a passing architecture test means something."""

from pathlib import Path

from tests.architecture.scan import find_violations

NO_THIRD_PARTY: frozenset[str] = frozenset()


def _write(root: Path, name: str, source: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _scan(
    root: Path,
    third_party: frozenset[str] = NO_THIRD_PARTY,
    internal: tuple[str, ...] = ("pkg",),
) -> list[tuple[str, int, str]]:
    violations = find_violations(root, "pkg", third_party, internal)
    return [(v.file.name, v.line, v.module) for v in violations]


def test_forbidden_imports_are_reported_with_file_and_line(tmp_path: Path) -> None:
    _write(tmp_path, "__init__.py", "")
    _write(tmp_path, "web.py", "import os\n\nimport fastapi\n")
    _write(tmp_path, "schema.py", "from pydantic import BaseModel\n")
    _write(tmp_path, "adapter.py", "import json\nimport app.adapters.x\n")
    _write(tmp_path, "models.py", "x = 1\nfrom app.models.schemas import BoundingBox\n")
    _write(tmp_path, "heavy.py", "import numpy\n")

    assert _scan(tmp_path) == [
        ("adapter.py", 2, "app.adapters.x"),
        ("heavy.py", 1, "numpy"),
        ("models.py", 2, "app.models.schemas"),
        ("schema.py", 1, "pydantic"),
        ("web.py", 3, "fastapi"),
    ]


def test_clean_modules_are_not_reported(tmp_path: Path) -> None:
    _write(tmp_path, "__init__.py", "")
    _write(tmp_path, "a.py", "from __future__ import annotations\n\nimport math\nimport re\n")
    _write(tmp_path, "b.py", "from collections.abc import Sequence\nfrom pkg.a import x\n")
    _write(tmp_path, "c.py", "from . import a\nfrom .b import y\nimport pkg.sub.z\n")
    _write(tmp_path, "sub/__init__.py", "from .z import q\nfrom .. import a\n")
    _write(tmp_path, "sub/z.py", "from ..a import x\n")

    assert _scan(tmp_path) == []


def test_relative_import_escaping_the_package_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "__init__.py", "")
    _write(tmp_path, "mod.py", "from ..outside import thing\n")
    _write(tmp_path, "sub/deep.py", "from ....elsewhere import thing\n")

    assert _scan(tmp_path) == [
        ("mod.py", 1, "outside"),
        ("deep.py", 1, "....elsewhere"),
    ]


def test_nested_imports_are_reported(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "late.py",
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n    import httpx\n\n"
        "def f() -> None:\n    import chromadb\n",
    )

    assert _scan(tmp_path) == [("late.py", 4, "httpx"), ("late.py", 7, "chromadb")]


def test_third_party_is_accepted_only_when_listed(tmp_path: Path) -> None:
    _write(tmp_path, "fuzzy.py", "from rapidfuzz import fuzz\n")

    assert _scan(tmp_path) == [("fuzzy.py", 1, "rapidfuzz")]
    assert _scan(tmp_path, third_party=frozenset({"rapidfuzz"})) == []


def test_internal_prefix_matches_whole_segments_only(tmp_path: Path) -> None:
    _write(tmp_path, "m.py", "import app.domain.types\nimport app.domainx\n")

    assert _scan(tmp_path, internal=("app.domain",)) == [("m.py", 2, "app.domainx")]
