"""Generators of hostile PDFs, built in memory for the adversarial suite (test-only code).

Each generator names the bound it attacks and the code the page-ocr spec expects for it; the
tests in ``test_hostile_pdfs`` assert that code and the time the refusal takes. The files are
written by hand with ``page_tree_pdf`` (an exact cross-reference table, so PDFium never has to
repair them) or with PyMuPDF, a test-only library that ``app/`` never imports. Nothing here is
committed as a binary: every file is a few kilobytes built on demand.

- ``many_images_pdf``: one page drawing thousands of copies of a large image; the declared
  pixels exceed the image budget, so the pre-check refuses it ``too_large`` without decoding
  anything. With the pixel budget switched off, the drawing itself exceeds the time budget
  (``pdf_unreadable``): PDFium pauses between the bands of each image.
- ``many_paths_pdf``: one page whose content stream draws a million tiny paths, each its own
  page object; the render budget stops it ``pdf_unreadable`` (PDFium pauses every hundred
  objects).
- ``declared_million_pages_pdf``: a page tree whose ``/Count`` claims a million pages. With
  more real pages than ``max_pages``, the one lookup of page index ``max_pages`` says
  ``too_many_pages``; with a single real page the declared count is simply wrong and the one
  page is accepted. Neither case looks up a million pages.
- ``million_kids_pdf``: a page tree with ``kids`` references to the same page object (a six
  megabyte file); ``too_many_pages`` after one lookup.
- ``nested_forms_pdf``: a chain of form XObjects, each invoking the next, with the drawing at
  the innermost level. PDFium stops descending at its own nesting limit (about forty levels):
  a hundred levels would render as a blank page. Below that limit a form nested deeper than
  eight levels is drawn as one operation of the progressive renderer, without a pause poll
  (``NESTED_FORM_POLL_DEPTH``), so the time budget could not bound it: the renderer's
  pre-check refuses any page whose forms nest deeper than that, before anything is drawn.
- ``huge_mediabox_pdf``: a page of a million points a side; the pixel budget refuses it
  ``too_large`` before anything is loaded.
- ``compression_bomb_pdf``: a content stream of a few dozen kilobytes that inflates to tens of
  megabytes of graphics-state operators (a thousand to one). PDFium inflates and parses the
  stream when the page is loaded, before the first pause poll, so that part is not under the
  budget (about a second for 64 MB on the measuring machine); the drawing then stops at the
  budget (``pdf_unreadable``).
- ``encrypted_pdf``: a file that needs a user password (``pdf_unreadable``, from the OCR
  adapter tests).
- ``truncated_pdf``: a valid file cut in the middle of its objects (``pdf_unreadable``) and
  ``headerless_bytes``: a file without the ``%PDF-`` header (``not_a_pdf`` at the HTTP edge).
"""

import zlib

import pymupdf

from tests.adapters.ocr.pdfs import PAGE_DICT, PageSpec, build_pdf, page_tree_pdf

__all__ = [
    "NESTED_FORM_POLL_DEPTH",
    "PDFIUM_FORM_NESTING_LIMIT",
    "compression_bomb_pdf",
    "declared_million_pages_pdf",
    "encrypted_pdf",
    "headerless_bytes",
    "huge_mediabox_pdf",
    "many_images_pdf",
    "many_paths_pdf",
    "million_kids_pdf",
    "nested_forms_pdf",
    "truncated_pdf",
]

MILLION = 1_000_000
STROKE = b"0 0 m 200 200 l S\n"
"""One diagonal line across the 200 x 200 pt page of ``PAGE_DICT``: one page object."""

NESTED_FORM_POLL_DEPTH = 8
"""Deepest form nesting at which the progressive renderer still polls the pause callback
between the objects of the form (measured: a form at depth 9 draws as one operation)."""

PDFIUM_FORM_NESTING_LIMIT = 48
"""Form nesting from which PDFium draws nothing (measured: 40 levels draw, 48 do not)."""


def _flate_stream(content: bytes, dictionary: bytes = b"") -> bytes:
    packed = zlib.compress(content, 9)
    return (
        b"<< /Length %d /Filter /FlateDecode %s>>\nstream\n" % (len(packed), dictionary)
        + packed
        + b"\nendstream"
    )


def _plain_stream(content: bytes, dictionary: bytes = b"") -> bytes:
    return b"<< /Length %d %s>>\nstream\n" % (len(content), dictionary) + content + b"\nendstream"


def _page_with(resources: bytes, contents_object: int) -> bytes:
    return (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << %s >>"
        b" /Contents %d 0 R >>" % (resources, contents_object)
    )


def many_images_pdf(images: int = 3000, side: int = 2000) -> bytes:
    """One page drawing ``images`` copies of a ``side`` x ``side`` gray image (object 4).

    The image data is a flate stream of one gray value, a few kilobytes however large the
    image; the declared pixels are ``images x side x side`` (12 thousand million by default).
    """
    raw = zlib.compress(b"\x80" * (side * side), 9)
    image = (
        b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray"
        b" /BitsPerComponent 8 /Filter /FlateDecode /Length %d >>\nstream\n"
        % (side, side, len(raw))
        + raw
        + b"\nendstream"
    )
    content = b"".join(
        b"q 200 0 0 200 %d %d cm /Im0 Do Q\n" % (index % 7, index % 11) for index in range(images)
    )
    page = _page_with(b"/XObject << /Im0 4 0 R >>", 5)
    return page_tree_pdf([page], extra=[image, _flate_stream(content)])


def many_paths_pdf(paths: int = MILLION) -> bytes:
    """One page whose content stream strokes ``paths`` lines, each its own page object."""
    page = PAGE_DICT.replace(b" >>", b" /Contents 4 0 R >>")
    return page_tree_pdf([page], extra=[_flate_stream(STROKE * paths)])


def declared_million_pages_pdf(real_pages: int = 1) -> bytes:
    """A page tree of ``real_pages`` blank pages whose ``/Count`` declares a million."""
    return page_tree_pdf([PAGE_DICT] * real_pages, count=MILLION)


def million_kids_pdf(kids: int = MILLION) -> bytes:
    """A page tree whose ``/Kids`` references one blank page object ``kids`` times."""
    references = b" ".join([b"3 0 R"] * kids)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [%s] /Count %d >>" % (references, kids),
        PAGE_DICT,
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


def nested_forms_pdf(depth: int = 100, strokes: int = 1) -> bytes:
    """One page invoking a chain of ``depth`` form XObjects; the innermost strokes ``strokes``
    lines (objects 4 .. 3 + depth are the forms, the page content follows them)."""
    first_form = 4
    page = _page_with(b"/XObject << /F0 %d 0 R >>" % first_form, first_form + depth)
    forms: list[bytes] = []
    for level in range(depth):
        number = first_form + level
        if level < depth - 1:
            body = b"q /Fx Do Q"
            resources = b"/Resources << /XObject << /Fx %d 0 R >> >>" % (number + 1)
        else:
            body = STROKE * strokes
            resources = b""
        forms.append(
            _flate_stream(
                body, b"/Type /XObject /Subtype /Form /BBox [0 0 200 200] %s " % resources
            )
        )
    return page_tree_pdf([page], extra=[*forms, _plain_stream(b"q /F0 Do Q")])


def huge_mediabox_pdf(side_pt: int = MILLION) -> bytes:
    """One blank page of ``side_pt`` points a side (the PDF limit is 14,400)."""
    return page_tree_pdf(
        [b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] >>" % (side_pt, side_pt)]
    )


def compression_bomb_pdf(inflated_bytes: int = 64 * 1024 * 1024) -> bytes:
    """One page whose flate content stream inflates to ``inflated_bytes`` of ``q Q`` pairs."""
    operators = b"q Q\n"
    page = PAGE_DICT.replace(b" >>", b" /Contents 4 0 R >>")
    return page_tree_pdf(
        [page], extra=[_flate_stream(operators * (inflated_bytes // len(operators)))]
    )


def encrypted_pdf() -> bytes:
    """A one-page PDF that needs a user password to open."""
    with pymupdf.open(stream=build_pdf([PageSpec()]), filetype="pdf") as document:
        data: bytes = document.tobytes(
            encryption=pymupdf.mupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user"
        )
        return data


def truncated_pdf() -> bytes:
    """A valid two-page PDF cut in the middle of its objects (no cross-reference table)."""
    whole = build_pdf([PageSpec(), PageSpec()])
    return whole[: len(whole) // 2]


def headerless_bytes() -> bytes:
    """A kilobyte of PostScript-like text without the ``%PDF-`` header anywhere."""
    return b"%!PS-Adobe-3.0\n%%Title: not a pdf\n" + b"% padding\n" * 120
