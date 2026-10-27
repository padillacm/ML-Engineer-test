"""In-memory person-name recogniser driven by a configured list of names."""

import re
from collections.abc import Sequence

from app.domain.types import TextSpan


class FakeNameRecognizer:
    """Finds the configured names in text as whole-word, case-sensitive matches."""

    def __init__(self, names: Sequence[str] = ()) -> None:
        # Longest first so "Anna Maria Lopez" wins over "Anna Maria" on the same text.
        self._patterns = [
            re.compile(rf"(?<!\w){re.escape(name)}(?!\w)")
            for name in sorted(set(names), key=len, reverse=True)
        ]
        self.calls: list[str] = []

    def find_persons(self, text: str) -> list[TextSpan]:
        self.calls.append(text)
        spans: list[TextSpan] = []
        for pattern in self._patterns:
            for match in pattern.finditer(text):
                span = TextSpan(match.start(), match.end())
                if not any(span.start < other.end and other.start < span.end for other in spans):
                    spans.append(span)
        return sorted(spans, key=lambda span: span.start)
