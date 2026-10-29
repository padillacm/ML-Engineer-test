"""The rule-based injection detector and the composite that joins it with the model (D5b).

``PatternDetector`` implements ``InjectionDetector`` over ``domain.injection_patterns``: a text
holding an instruction pattern scores 1.0, any other text (blank included) 0.0. It never fails
and needs no model, so it is always ``configured``. ``CompositeDetector`` screens with several
detectors and keeps, per text, the highest score, so the composition root can put the patterns
next to Prompt Guard (or next to ``NoDetector`` when no model is set) without the use cases
knowing: a failure of any member propagates unchanged.
"""

from collections.abc import Sequence

from app.application.ports import InjectionDetector
from app.domain.injection_patterns import matches


class PatternDetector:
    """``InjectionDetector`` scoring 1.0 for a text with an instruction pattern, else 0.0."""

    @property
    def configured(self) -> bool:
        return True

    def screen(self, texts: Sequence[str]) -> list[float]:
        """One score per text: 1.0 when ``injection_patterns.matches`` it, 0.0 otherwise."""
        return [1.0 if matches(text) else 0.0 for text in texts]

    def screen_query(self, text: str) -> float:
        """1.0 when ``injection_patterns.matches`` the question, 0.0 otherwise."""
        return 1.0 if matches(text) else 0.0


class CompositeDetector:
    """``InjectionDetector`` returning, per text, the highest score of its ``detectors``.

    ``configured`` when any member is. With no member every score is 0 and it is not
    configured, like ``NoDetector``.
    """

    def __init__(self, detectors: Sequence[InjectionDetector]) -> None:
        self._detectors = tuple(detectors)

    @property
    def detectors(self) -> tuple[InjectionDetector, ...]:
        return self._detectors

    @property
    def configured(self) -> bool:
        return any(detector.configured for detector in self._detectors)

    def screen(self, texts: Sequence[str]) -> list[float]:
        """The element-wise maximum of every member's scores, in text order."""
        scores = [0.0 for _ in texts]
        for detector in self._detectors:
            scores = [max(a, b) for a, b in zip(scores, detector.screen(texts), strict=True)]
        return scores

    def screen_query(self, text: str) -> float:
        """The highest of every member's question score."""
        return max((detector.screen_query(text) for detector in self._detectors), default=0.0)
