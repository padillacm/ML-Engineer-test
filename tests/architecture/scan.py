"""AST-based import scanner used by the architecture tests."""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Violation:
    """One import statement, located by file and line."""

    file: Path
    line: int
    module: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line} imports {self.module}"


def _module_parts(file: Path, package_dir: Path, package_name: str) -> list[str]:
    relative = file.relative_to(package_dir).with_suffix("")
    parts = [*package_name.split("."), *relative.parts]
    return parts[:-1] if parts[-1] == "__init__" else parts


def _resolve(node: ast.ImportFrom, module_parts: list[str], is_init: bool) -> str:
    """Return the absolute dotted name of a ``from`` import, resolving relative levels."""
    if node.level == 0:
        return node.module or ""
    base = module_parts if is_init else module_parts[:-1]
    drop = node.level - 1
    if drop > len(base):
        return "." * node.level + (node.module or "")
    kept = base[: len(base) - drop]
    name = ".".join([*kept, *([node.module] if node.module else [])])
    return name or "." * node.level


def collect_imports(package_dir: Path, package_name: str) -> list[Violation]:
    """Return every import found in the ``.py`` files under ``package_dir``.

    Only static ``import`` and ``from`` statements are seen; dynamic imports (``importlib``,
    ``__import__``) are not detected.

    Raises ``FileNotFoundError`` when the directory is missing or holds no ``.py`` file, so a
    mistyped path can never pass the architecture checks vacuously.
    """
    if not package_dir.is_dir():
        raise FileNotFoundError(f"directory to scan does not exist: {package_dir}")
    files = sorted(package_dir.rglob("*.py"))
    if not files:
        raise FileNotFoundError(f"no .py files to scan under: {package_dir}")
    found: list[Violation] = []
    for file in files:
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        parts = _module_parts(file, package_dir, package_name)
        is_init = file.name == "__init__.py"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.extend(Violation(file, node.lineno, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                found.append(Violation(file, node.lineno, _resolve(node, parts, is_init)))
    return sorted(found, key=lambda v: (str(v.file), v.line, v.module))


def is_under(module: str, prefixes: tuple[str, ...]) -> bool:
    """Return whether ``module`` equals or sits below one of the dotted ``prefixes``."""
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _is_allowed(
    module: str, allowed_third_party: frozenset[str], allowed_internal_prefixes: tuple[str, ...]
) -> bool:
    if is_under(module, allowed_internal_prefixes):
        return True
    top = module.split(".", maxsplit=1)[0]
    return top == "__future__" or top in sys.stdlib_module_names or top in allowed_third_party


def find_violations(
    package_dir: Path,
    package_name: str,
    allowed_third_party: frozenset[str],
    allowed_internal_prefixes: tuple[str, ...],
) -> list[Violation]:
    """Return the imports under ``package_dir`` that are not on the allowlist.

    Allowed: the standard library, ``__future__``, the listed third-party top-level names and
    any module equal to or under one of the internal prefixes. Raises ``FileNotFoundError`` like
    ``collect_imports``.
    """
    return [
        imp
        for imp in collect_imports(package_dir, package_name)
        if not _is_allowed(imp.module, allowed_third_party, allowed_internal_prefixes)
    ]
