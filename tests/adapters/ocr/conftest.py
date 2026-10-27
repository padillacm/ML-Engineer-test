from collections.abc import Callable
from dataclasses import dataclass, field

import pypdfium2 as pdfium
import pytest

from app.adapters.ocr import rendering
from app.adapters.ocr.rendering import RenderedPage


@pytest.fixture
def opened_documents(monkeypatch: pytest.MonkeyPatch) -> list[pdfium.PdfDocument]:
    """Every PDFium document opened by the adapter during the test, in opening order.

    A closed document has ``raw is None``.
    """
    opened: list[pdfium.PdfDocument] = []
    real_open: Callable[..., pdfium.PdfDocument] = pdfium.PdfDocument

    def spy(*args: object, **kwargs: object) -> pdfium.PdfDocument:
        document = real_open(*args, **kwargs)
        opened.append(document)
        return document

    monkeypatch.setattr(pdfium, "PdfDocument", spy)
    return opened


@dataclass
class RenderSpy:
    """Counts the pages rendered by the adapter, in rendering order."""

    indexes: list[int] = field(default_factory=list)


@pytest.fixture
def rendered_pages(monkeypatch: pytest.MonkeyPatch) -> RenderSpy:
    """Record the index of every page the adapter renders."""
    spy = RenderSpy()
    real_render: Callable[..., RenderedPage] = rendering._render

    def render(*args: object, **kwargs: object) -> RenderedPage:
        page = real_render(*args, **kwargs)
        spy.indexes.append(page.index)
        return page

    monkeypatch.setattr(rendering, "_render", render)
    return spy
