"""Fixtures of the adversarial suite.

The HTTP fixtures come from ``tests/api`` (imported, not copied): ``make_client`` and ``logs``
from its ``conftest``, ``clock`` and ``make_limited`` (the fake-clock application) from the
rate-limit tests. ``fakes`` is overridden here with ``GuardedFakes``, so every application the
suite starts screens questions and pages with a fake detector that knows the corpus markers.
``real_detector`` loads the Prompt Guard model once per session for the ``injection_model``
tests and ``combined_detector`` puts the instruction patterns next to it, as the composition
root does by default; the suite-wide hook skips those tests when ``INJECTION_MODEL_PATH`` is
unset.
"""

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from app.application.pattern_detector import CompositeDetector, PatternDetector
from tests.adversarial.support import GuardedFakes
from tests.api.conftest import logs, make_client, restore_logging
from tests.api.test_rate_limit import clock, make_limited

if TYPE_CHECKING:
    from app.adapters.injection import PromptGuardDetector

__all__ = [
    "clock",
    "combined_detector",
    "fakes",
    "logs",
    "make_client",
    "make_limited",
    "real_detector",
    "restore_logging",
]


@pytest.fixture
def fakes() -> GuardedFakes:
    return GuardedFakes()


@pytest.fixture(scope="session")
def real_detector() -> Iterator["PromptGuardDetector"]:
    """The real detector over the model at ``INJECTION_MODEL_PATH`` (loaded lazily: the host
    run without the model never imports torch for this suite)."""
    import torch  # noqa: PLC0415 - heavy, only for the real-model tests

    from app.adapters.injection import (  # noqa: PLC0415
        InjectionDetectorConfig,
        PromptGuardDetector,
    )

    threads = torch.get_num_threads()
    detector = PromptGuardDetector(InjectionDetectorConfig(os.environ["INJECTION_MODEL_PATH"]))
    yield detector
    detector.close()
    torch.set_num_threads(threads)


@pytest.fixture(scope="session")
def combined_detector(real_detector: "PromptGuardDetector") -> CompositeDetector:
    """The detector the composition root builds by default: the patterns next to the model."""
    return CompositeDetector([PatternDetector(), real_detector])
