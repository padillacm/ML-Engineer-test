"""The injection-detector port of a service started without a detector model."""

from collections.abc import Sequence


class NoDetector:
    """``InjectionDetector`` that scores every text 0: nothing is flagged or refused.

    It stands in for the detector when ``INJECTION_MODEL_PATH`` is unset, so the use cases
    always have a detector and the other guardrails (delimited context, support check, output
    filter) stay in force.
    """

    @property
    def configured(self) -> bool:
        return False

    def screen(self, texts: Sequence[str]) -> list[float]:
        """Return a 0 for each text; nothing is inspected."""
        return [0.0 for _ in texts]

    def screen_query(self, text: str) -> float:  # noqa: ARG002 - the port's signature
        """Return 0; nothing is inspected."""
        return 0.0
