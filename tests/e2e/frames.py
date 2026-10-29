"""Synthetic scanned PDFs for the frame tests of the audit viewer (spec: audit-viewer).

Every page is an image-only page, as a scanner produces it, whose DISPLAYED text is upright:
the page is drawn upright, then its pixels are turned into the orientation of the page's user
space and ``/Rotate`` turns them back for display (a page scanned sideways and flagged with
``/Rotate 90``). A page with a crop box gets an image of its media box, with the visible text
inside the crop box and one more name outside it, which neither renderer may show. Pages of
different sizes share one file.

Three more pages exercise what the JPEG pages do not: a page with ``/UserUnit 2`` (pdf.js
applies it, the service's renderer PDFium ignores it, so the response is in user units and the
viewer must draw in that frame and warn), and pages whose scanned image is JPEG 2000
(``JPXDecode``) and CCITT G4 (``CCITTFaxDecode``), which pdf.js decodes with the JavaScript
fallback decoders the viewer serves. pypdfium2 cannot write those, so each is written as a
one-page PDF by ``_raw_pdf`` and imported (PDFium keeps the page dictionary and the streams).

Made with Pillow (the bundled Aileron font, so the result is the same on every machine) and
pypdfium2, both runtime dependencies of the service: no PDF library is added for the tests.
Each page records its displayed size and, for every expected name, the box of its glyphs in
points of the displayed page (top-left origin; user units on the ``/UserUnit`` page, the
service's frame), which the tests compare with the response.
"""

import io
from dataclasses import dataclass, replace
from typing import Any, Literal

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont, TiffImagePlugin

DPI = 150
PX_PER_PT = DPI / 72
FONT_PT = 20.0
LINE_PITCH_PT = 64.0
LEFT_PT = 56.0
TOP_PT = 96.0
JPEG_QUALITY = 92

A4 = (595.0, 842.0)
LETTER = (612.0, 792.0)
A5_LANDSCAPE = (595.0, 420.0)
# Crop margins (left, top, right, bottom) in displayed points: asymmetric, so an offset taken
# from the wrong corner or side of the media box shows up as a shift.
CROP_MARGINS = (44.0, 72.0, 28.0, 52.0)

ROTATIONS = (0, 90, 180, 270)
USER_UNIT = 2.0

type Codec = Literal["jpeg", "jpx", "ccitt"]


@dataclass(frozen=True)
class NameBox:
    """An expected person name and its glyph box in displayed points (top-left origin)."""

    name: str
    page: int
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class PageSpec:
    """One page: displayed size in points, ``/Rotate``, crop margins and its text lines.

    Each line is ``(prefix, name, suffix)``; ``outside`` is a line drawn outside the crop box.
    ``user_unit`` other than 1 and a ``codec`` other than JPEG need an upright page without a
    crop box (written by ``_raw_pdf``); ``size_pt`` is then in user units.
    """

    label: str
    size_pt: tuple[float, float]
    rotate: int
    lines: tuple[tuple[str, str, str], ...]
    crop: bool = False
    outside: tuple[str, str, str] | None = None
    user_unit: float = 1.0
    codec: Codec = "jpeg"
    rotate_entry: str | None = None
    """A ``/Rotate`` value written as it is (for example the string ``(90)``), raw pages only."""

    @property
    def raw(self) -> bool:
        return self.user_unit != 1.0 or self.codec != "jpeg" or self.rotate_entry is not None


@dataclass(frozen=True)
class FrameDocument:
    pdf: bytes
    pages: tuple[PageSpec, ...]
    displayed_sizes: tuple[tuple[float, float], ...]
    names: tuple[NameBox, ...]
    hidden_names: tuple[str, ...]


PAGES: tuple[PageSpec, ...] = (
    PageSpec(
        "A4, no rotation",
        A4,
        0,
        (
            ("Memo from ", "Maria Gonzalez", " to the board."),
            ("Approved by ", "John Smith", " in March."),
        ),
    ),
    PageSpec(
        "A4 scanned sideways, /Rotate 90",
        A4,
        90,
        (
            ("Report written by ", "Hans Weber", " for the team."),
            ("Reviewed by ", "Laura Chen", " on Monday."),
        ),
    ),
    PageSpec(
        "Letter scanned upside down, /Rotate 180",
        LETTER,
        180,
        (
            ("Letter signed by ", "Pedro Alvarez", " in Lisbon."),
            ("Copy sent to ", "Emma Brown", " last week."),
        ),
    ),
    PageSpec(
        "A4 scanned sideways, /Rotate 270",
        A4,
        270,
        (
            ("Minutes taken by ", "Lucas Martin", " at noon."),
            ("Meeting chaired by ", "Sofia Rossi", " today."),
        ),
    ),
    PageSpec(
        "Letter, crop box offset from the media box",
        LETTER,
        0,
        (
            ("Contract signed by ", "David Miller", " in Boston."),
            ("Witnessed by ", "Anna Kowalski", " that day."),
        ),
        crop=True,
        outside=("Draft notes by ", "Oliver Grant", " only."),
    ),
    PageSpec(
        "A5 landscape",
        A5_LANDSCAPE,
        0,
        (
            ("Invoice issued by ", "Peter Novak", " in May."),
            ("Paid by ", "Julia Fischer", " in June."),
        ),
    ),
    PageSpec(
        "A4 scanned sideways with a crop box, /Rotate 90",
        A4,
        90,
        (
            ("Statement made by ", "Carlos Diaz", " in court."),
            ("Translated by ", "Nora Jensen", " for us."),
        ),
        crop=True,
        outside=("Archive copy for ", "Victor Hale", " only."),
    ),
    PageSpec(
        "A5 landscape in user units of 2 pt, /UserUnit 2",
        A5_LANDSCAPE,
        0,
        (
            ("Drawing approved by ", "Robert Klein", " today."),
            ("Checked by ", "Helena Costa", " in July."),
        ),
        user_unit=USER_UNIT,
    ),
    PageSpec(
        "A4 scanned as JPEG 2000",
        A4,
        0,
        (
            ("Receipt issued by ", "Thomas Becker", " in Bonn."),
            ("Signed for by ", "Clara Moreno", " at once."),
        ),
        codec="jpx",
    ),
    PageSpec(
        "A4 scanned as CCITT G4 (black and white)",
        A4,
        0,
        (
            ("Fax sent by ", "Daniel Hughes", " at noon."),
            ("Received by ", "Marta Silva", " in Porto."),
        ),
        codec="ccitt",
    ),
)

# Not in PAGES: a page whose /Rotate is the string (90). PDFium (the service) reads it as 0,
# pdf.js keeps the string and turns the page by 90 degrees, so the two frames differ and the
# viewer flags the page (KNOWN_LIMITATIONS V5).
STRING_ROTATE_PAGE = PageSpec(
    "A4, /Rotate written as the string (90)",
    A4,
    0,
    (("Note written by ", "Irene Walsh", " on Friday."),),
    rotate_entry="(90)",
)

# The names list the tests send: the first name of every page (matched boxes), the second name
# of every page left out (unmatched boxes), and a decoy that is in no page.
LISTED_NAMES = tuple(page.lines[0][1] for page in PAGES)
DECOY_NAME = "Zebulon Quackenbush"
NAMES_TEXT = "\n".join((*LISTED_NAMES, DECOY_NAME))


def build(pages: tuple[PageSpec, ...] = PAGES) -> FrameDocument:
    """The PDF of ``pages`` with its expected displayed sizes and name boxes."""
    font = ImageFont.load_default(size=round(FONT_PT * PX_PER_PT))
    assert isinstance(font, ImageFont.FreeTypeFont), "Pillow without FreeType"
    pdf = pdfium.PdfDocument.new()
    sizes: list[tuple[float, float]] = []
    names: list[NameBox] = []
    hidden: list[str] = []
    try:
        for index, spec in enumerate(pages):
            size, boxes = _add_page(pdf, index, spec, font)
            sizes.append(size)
            names.extend(boxes)
            if spec.outside is not None:
                hidden.append(spec.outside[1])
        buffer = io.BytesIO()
        pdf.save(buffer)
    finally:
        pdf.close()
    return FrameDocument(buffer.getvalue(), pages, tuple(sizes), tuple(names), tuple(hidden))


def _px(value_pt: float) -> int:
    return round(value_pt * PX_PER_PT)


def _add_page(
    pdf: pdfium.PdfDocument, index: int, spec: PageSpec, font: ImageFont.FreeTypeFont
) -> tuple[tuple[float, float], list[NameBox]]:
    left, top, right, bottom = CROP_MARGINS if spec.crop else (0.0, 0.0, 0.0, 0.0)
    crop_w_px, crop_h_px = _px(spec.size_pt[0]), _px(spec.size_pt[1])
    full = Image.new(
        "L", (_px(left) + crop_w_px + _px(right), _px(top) + crop_h_px + _px(bottom)), 255
    )
    draw = ImageDraw.Draw(full)
    origin = (_px(left), _px(top))
    boxes = []
    for number, line in enumerate(spec.lines):
        y_pt = TOP_PT + number * LINE_PITCH_PT
        box = _draw_line(draw, font, line, origin, (LEFT_PT, y_pt))
        boxes.append(NameBox(line[1], index, *box))
    if spec.outside is not None:
        # In the top margin, outside the crop box.
        _draw_line(draw, font, spec.outside, (_px(left), 0), (LEFT_PT, (top - FONT_PT) / 2))
    if spec.raw:
        if spec.rotate or spec.crop:
            raise ValueError(f"{spec.label}: a raw page is upright and has no crop box")
        source = pdfium.PdfDocument(_raw_pdf(full, spec))
        try:
            pdf.import_pages(source)
        finally:
            source.close()
        return (crop_w_px / PX_PER_PT, crop_h_px / PX_PER_PT), boxes
    content = _to_user_space(full, spec.rotate)
    width_pt, height_pt = content.width / PX_PER_PT, content.height / PX_PER_PT
    page = pdf.new_page(width_pt, height_pt)
    try:
        _place_image(pdf, page, content, width_pt, height_pt)
        page.set_rotation(spec.rotate)
        if spec.crop:
            crop = (
                _px(left) / PX_PER_PT,
                _px(top) / PX_PER_PT,
                crop_w_px / PX_PER_PT,
                crop_h_px / PX_PER_PT,
            )
            page.set_cropbox(*_crop_box(crop, spec.rotate, width_pt, height_pt))
    finally:
        page.close()
    return (crop_w_px / PX_PER_PT, crop_h_px / PX_PER_PT), boxes


def _draw_line(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    line: tuple[str, str, str],
    origin_px: tuple[int, int],
    at_pt: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Draw ``line`` at ``at_pt`` (relative to ``origin_px``); the name's glyph box in points."""
    prefix, name, suffix = line
    x_px = origin_px[0] + _px(at_pt[0])
    y_px = origin_px[1] + _px(at_pt[1])
    draw.text((x_px, y_px), prefix + name + suffix, fill=0, font=font)
    name_x = x_px + draw.textlength(prefix, font=font)
    left, top, right, bottom = draw.textbbox((name_x, y_px), name, font=font)
    return (
        (left - origin_px[0]) / PX_PER_PT,
        (top - origin_px[1]) / PX_PER_PT,
        (right - left) / PX_PER_PT,
        (bottom - top) / PX_PER_PT,
    )


def _to_user_space(displayed: Image.Image, rotate: int) -> Image.Image:
    """The pixels as stored in the page's user space: ``/Rotate`` (clockwise) undoes this."""
    if rotate == 0:
        return displayed
    return displayed.rotate(rotate, expand=True)  # Pillow rotates counter-clockwise


def _place_image(
    pdf: pdfium.PdfDocument,
    page: pdfium.PdfPage,
    image: Image.Image,
    width_pt: float,
    height_pt: float,
) -> None:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    buffer.seek(0)
    picture = pdfium.PdfImage.new(pdf)
    picture.load_jpeg(buffer, inline=True)
    picture.set_matrix(pdfium.PdfMatrix().scale(width_pt, height_pt))
    page.insert_obj(picture)
    page.gen_content()


def _crop_box(
    displayed: tuple[float, float, float, float], rotate: int, width_pt: float, height_pt: float
) -> tuple[float, float, float, float]:
    """``(left, bottom, right, top)`` in PDF user space of a displayed ``(x, y, w, h)`` rect.

    ``width_pt`` x ``height_pt`` is the media box (user space, unrotated); displayed
    coordinates have their origin at the top-left of the page as shown, y down.
    """
    x, y, w, h = displayed
    corners = [
        _to_user(cx, cy, rotate, width_pt, height_pt) for cx in (x, x + w) for cy in (y, y + h)
    ]
    us = [u for u, _ in corners]
    vs = [v for _, v in corners]
    # (u, v): top-left origin of the unrotated page; PDF user space has y up.
    return min(us), height_pt - max(vs), max(us), height_pt - min(vs)


def _to_user(
    x: float, y: float, rotate: int, width_pt: float, height_pt: float
) -> tuple[float, float]:
    """Displayed (x, y) -> unrotated top-left (u, v) on a ``width_pt`` x ``height_pt`` page."""
    if rotate == 0:
        return x, y
    if rotate == 90:
        return y, height_pt - x
    if rotate == 180:
        return width_pt - x, height_pt - y
    if rotate == 270:
        return width_pt - y, x
    raise ValueError(f"unsupported rotation: {rotate}")


def _encoded(image: Image.Image, codec: Codec) -> tuple[str, bytes]:
    """The image dictionary entries (without ``/Length``) and the stream of ``image``."""
    width, height = image.size
    buffer = io.BytesIO()
    if codec == "ccitt":
        # Group 4 from a one-strip TIFF: the strip is the CCITT stream PDF wants.
        bitonal = image.point(lambda value: 255 if value > 128 else 0).convert("1")
        bitonal.save(buffer, format="TIFF", compression="group4", tiffinfo={278: height})
        tiff = Image.open(io.BytesIO(buffer.getvalue()))
        assert isinstance(tiff, TiffImagePlugin.TiffImageFile)
        (offset,), (count,) = tiff.tag_v2[273], tiff.tag_v2[279]
        black_is_1 = "true" if tiff.tag_v2.get(262) == 1 else "false"
        data = buffer.getvalue()[offset : offset + count]
        parms = f"/K -1 /Columns {width} /Rows {height} /BlackIs1 {black_is_1}"
        entries = f"/BitsPerComponent 1 /Filter /CCITTFaxDecode /DecodeParms << {parms} >>"
        return f"/Width {width} /Height {height} /ColorSpace /DeviceGray {entries}", data
    if codec == "jpx":
        image.save(buffer, format="JPEG2000", irreversible=False)
        filter_name = "/JPXDecode"
    else:
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        filter_name = "/DCTDecode"
    return (
        f"/Width {width} /Height {height} /ColorSpace /DeviceGray /BitsPerComponent 8 "
        f"/Filter {filter_name}",
        buffer.getvalue(),
    )


def _raw_pdf(image: Image.Image, spec: PageSpec) -> bytes:
    """A one-page PDF showing ``image`` over the whole page of ``spec`` (its ``codec`` and
    ``/UserUnit``), written object by object with a classic cross-reference table."""
    width_pt, height_pt = image.width / PX_PER_PT, image.height / PX_PER_PT
    image_entries, data = _encoded(image, spec.codec)
    content = f"q {width_pt:.4f} 0 0 {height_pt:.4f} 0 0 cm /Im0 Do Q".encode()
    unit = f" /UserUnit {spec.user_unit:g}" if spec.user_unit != 1.0 else ""
    if spec.rotate_entry is not None:
        unit += f" /Rotate {spec.rotate_entry}"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width_pt:.4f} {height_pt:.4f}]{unit} "
            "/Contents 4 0 R /Resources << /XObject << /Im0 5 0 R >> >> >>"
        ).encode(),
        _stream(b"", content),
        _stream(f"/Type /XObject /Subtype /Image {image_entries} ".encode(), data),
    ]
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def _stream(entries: bytes, data: bytes) -> bytes:
    return b"<< " + entries + b"/Length %d >>\nstream\n" % len(data) + data + b"\nendstream"


def repeat(document: FrameDocument, times: int) -> FrameDocument:
    """``document``'s pages ``times`` over in one PDF, with the expected names of every copy."""
    count = len(document.pages)
    source = pdfium.PdfDocument(document.pdf)
    pdf = pdfium.PdfDocument.new()
    try:
        for _ in range(times):
            pdf.import_pages(source)
        buffer = io.BytesIO()
        pdf.save(buffer)
    finally:
        pdf.close()
        source.close()
    names = tuple(
        replace(name, page=name.page + copy * count)
        for copy in range(times)
        for name in document.names
    )
    return FrameDocument(
        buffer.getvalue(),
        document.pages * times,
        document.displayed_sizes * times,
        names,
        document.hidden_names * times,
    )


def canned_response(
    document: FrameDocument, listed: tuple[str, ...] = LISTED_NAMES
) -> dict[str, Any]:
    """An extraction response with every expected name at its drawn glyph box (the service's
    frame), and a perfect match for each listed name: for tests that intercept the API."""
    return {
        "extracted_names": [
            {
                "name": name.name,
                "bounding_box": {
                    "page_number": name.page,
                    "x": name.x,
                    "y": name.y,
                    "width": name.width,
                    "height": name.height,
                },
            }
            for name in document.names
        ],
        "fuzzy_matches": [
            {"extracted_name": name.name, "matched_name": name.name, "score": 1.0}
            for name in document.names
            if name.name in listed
        ],
    }
