"""Validate a commit message against Conventional Commits 1.0.0 plus the house conventions.

Usage: python validate_commit_msg.py <message-file>
Exits 0 when the message is valid, 1 otherwise (problems are printed to stderr).
Git comment lines (starting with '#') are ignored, so it also works as a commit-msg hook.
"""

import re
import sys
from pathlib import Path

TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
)
MAX_HEADER = 72
MAX_BODY_LINE = 100  # 72 is the target; URLs and tables get some slack
BREAKING_PREFIXES = ("BREAKING CHANGE: ", "BREAKING-CHANGE: ")

HEADER_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[a-z0-9][a-z0-9._/-]*)\))?(?P<bang>!)?: (?P<desc>.+)$"
)
FOOTER_RE = re.compile(r"^(?:BREAKING CHANGE|BREAKING-CHANGE|[A-Za-z][A-Za-z0-9-]*)(?:: | #)")
LOOSE_BREAKING_RE = re.compile(r"^breaking[ -]change\b", re.IGNORECASE)
PAST_TENSE_RE = re.compile(
    r"^(added|fixed|updated|removed|changed|created|implemented|refactored|renamed)\b"
)


def check_header(match: re.Match[str]) -> list[str]:
    errors: list[str] = []
    type_, scope, desc = match["type"], match["scope"], match["desc"]
    if type_ not in TYPES:
        errors.append(f"unknown type {type_!r}; use one of: {', '.join(TYPES)}")
    if scope == type_:
        errors.append("scope repeats the type; name a code area or drop it")
    if len(match.string) > MAX_HEADER:
        errors.append(f"header is {len(match.string)} chars; keep it <= {MAX_HEADER}")
    if desc != desc.strip():
        errors.append("description has leading or trailing whitespace")
    if desc[:1].isupper() and not desc.split()[0].isupper():
        errors.append("description must start lowercase (acronyms are fine)")
    if desc.endswith("."):
        errors.append("description must not end with a period")
    if PAST_TENSE_RE.match(desc.lower()):
        errors.append("description must be imperative ('add', not 'added')")
    return errors


def check_rest(rest: list[str], *, bang: bool) -> list[str]:
    """Check everything after the header: blank separator, body width, footers."""
    errors: list[str] = []
    if rest and rest[0]:
        errors.append("leave one blank line between the header and the body/footers")

    last_paragraph = "\n".join(rest).strip("\n").split("\n\n")[-1].splitlines() if rest else []
    errors += [
        "breaking-change footer must be exactly 'BREAKING CHANGE: <description>'"
        for line in last_paragraph
        if LOOSE_BREAKING_RE.match(line) and not line.startswith(BREAKING_PREFIXES)
    ]
    is_footer_block = bool(last_paragraph) and FOOTER_RE.match(last_paragraph[0]) is not None
    has_breaking = is_footer_block and any(
        line.startswith(BREAKING_PREFIXES) for line in last_paragraph
    )
    if bang and not has_breaking:
        errors.append("'!' used: add a 'BREAKING CHANGE: <what breaks and how to migrate>' footer")
    if has_breaking and not bang:
        errors.append("BREAKING CHANGE footer present: also add '!' before the ':' in the header")

    errors += [
        f"body line {number} is {len(line)} chars; wrap at ~72"
        for number, line in enumerate(rest, start=2)
        if len(line) > MAX_BODY_LINE and not re.search(r"https?://", line)
    ]
    return errors


def validate(message: str) -> list[str]:
    lines = [line.rstrip() for line in message.splitlines() if not line.startswith("#")]
    while lines and not lines[-1]:
        lines.pop()
    if not lines or not lines[0]:
        return ["message is empty or starts with a blank line"]

    match = HEADER_RE.match(lines[0])
    if not match:
        return [f"header must look like 'type(scope)!: description', got: {lines[0]!r}"]
    return check_header(match) + check_rest(lines[1:], bang=bool(match["bang"]))


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    errors = validate(Path(argv[0]).read_text(encoding="utf-8"))
    for error in errors:
        print(f"commit message: {error}", file=sys.stderr)
    if not errors:
        print("commit message OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
