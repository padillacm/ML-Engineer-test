"""PDFs built in memory for the OCR adapter tests."""

import re
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


def declare_page_count(pdf: bytes, count: int) -> bytes:
    """Return ``pdf`` (as written by ``build_pdf``) with its page tree declaring ``count`` pages.

    Only the root ``/Pages`` dictionary's ``/Count`` changes; the cross-reference table is
    rewritten so every object offset stays right and PDFium does not have to repair the file.
    """
    patched, replaced = re.subn(
        rb"/Type/Pages/Count \d+", b"/Type/Pages/Count %d" % count, pdf, count=1
    )
    assert replaced == 1, "no root page tree found"
    body = patched[: patched.rindex(b"\nxref\n") + 1]
    offsets = {int(m.group(1)): m.start() for m in re.finditer(rb"(?m)^(\d+) 0 obj\b", body)}
    size = max(offsets) + 1
    rows = [b"0000000000 65535 f \n"]
    rows += [b"%010d 00000 n \n" % offsets[number] for number in range(1, size)]
    trailer = patched[patched.rindex(b"trailer") : patched.rindex(b"startxref")]
    return (
        body
        + b"xref\n0 %d\n" % size
        + b"".join(rows)
        + trailer
        + b"startxref\n%d\n%%%%EOF\n" % len(body)
    )


PAGE_DICT = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>"
"""A blank 200 x 200 pt page whose parent is object 2, the root of ``page_tree_pdf``."""


def page_tree_pdf(
    kids: Sequence[bytes],
    count: int | None = None,
    tree_entries: bytes = b"",
    extra: Sequence[bytes] = (),
) -> bytes:
    """Return a PDF whose root page tree (object 2) has ``kids`` as objects 3, 4, ...

    A kid is the body of its object: a page dictionary such as ``PAGE_DICT``, or anything else
    (``b"42"``) for a damaged entry. ``count`` defaults to the number of kids; ``tree_entries``
    are added to the page tree dictionary (inherited attributes); ``extra`` objects follow the
    kids. The cross-reference table is exact, so PDFium does not repair the file.
    """
    references = b" ".join(b"%d 0 R" % (3 + index) for index in range(len(kids)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [%s] /Count %d %s >>"
        % (references, len(kids) if count is None else count, tree_entries),
        *kids,
        *extra,
    ]
    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)
