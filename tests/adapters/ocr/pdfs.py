"""PDFs built in memory for the OCR adapter tests."""

from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf

A4 = (595.0, 842.0)
LETTER = (612.0, 792.0)
SMALL = (200.0, 300.0)

ENGLISH_LINES = (
    "The quarterly review of the supplier contract",
    "was signed by Maria Ramirez and John Carter",
    "on behalf of the company. The board has approved",
    "the budget for the next year and the plan for",
    "the new office in the city centre.",
)


@dataclass(frozen=True, slots=True)
class PageSpec:
    """One page: its size in points, its lines of text (none for a blank page), its rotation."""

    size: tuple[float, float] = SMALL
    lines: Sequence[str] = ("Hello",)
    rotation: int = 0


def build_pdf(pages: Sequence[PageSpec]) -> bytes:
    """Return a PDF with vector text drawn at 12 pt from the top-left of each page."""
    with pymupdf.open() as document:
        for spec in pages:
            page = document.new_page(width=spec.size[0], height=spec.size[1])
            for number, line in enumerate(spec.lines):
                page.insert_text((20, 30 + 16 * number), line, fontsize=12)
            if spec.rotation:
                page.set_rotation(spec.rotation)
        data: bytes = document.tobytes()
        return data


def image_only_pdf(pages: Sequence[PageSpec], dpi: int = 150) -> bytes:
    """Return a scanned-like PDF: every page holds only a raster image of ``build_pdf``'s page."""
    with (
        pymupdf.open(stream=build_pdf(pages), filetype="pdf") as source,
        pymupdf.open() as scanned,
    ):
        for page in source:
            pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
            rect = page.rect
            target = scanned.new_page(width=float(rect.width), height=float(rect.height))
            target.insert_image(target.rect, pixmap=pixmap)
        data: bytes = scanned.tobytes()
        return data


def encrypted_pdf() -> bytes:
    """Return a one-page PDF that needs a user password to open."""
    with pymupdf.open(stream=build_pdf([PageSpec()]), filetype="pdf") as document:
        data: bytes = document.tobytes(
            encryption=pymupdf.mupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user"
        )
        return data
