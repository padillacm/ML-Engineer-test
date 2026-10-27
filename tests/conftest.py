"""Suite-wide hooks: tests marked ``tesseract`` run only where the executable is installed."""

import shutil

import pytest

_NO_TESSERACT = pytest.mark.skip(
    reason=(
        "tesseract executable not on PATH; run the container suite: "
        "docker build --target test -t docintel-test . && docker run --rm docintel-test"
    )
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``tesseract``-marked tests when the executable cannot be found."""
    if shutil.which("tesseract") is not None:
        return
    for item in items:
        if "tesseract" in item.keywords:
            item.add_marker(_NO_TESSERACT)
