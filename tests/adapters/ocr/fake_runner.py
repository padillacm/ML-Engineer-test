"""A stand-in for the Tesseract process that answers with canned TSV."""

import io
import threading
import time
from collections.abc import Callable

import numpy as np
from PIL import Image

from app.domain.errors import ProcessingFailed

TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
)


def tsv(words: list[tuple[str, tuple[int, int, int, int]]]) -> str:
    """Build Tesseract TSV with one word row per ``(text, (left, top, width, height))``."""
    rows = [TSV_HEADER, "1\t1\t0\t0\t0\t0\t0\t0\t1\t1\t-1\t"]
    rows += [
        f"5\t1\t1\t1\t1\t{number}\t{left}\t{top}\t{width}\t{height}\t90.0\t{text}"
        for number, (text, (left, top, width, height)) in enumerate(words, start=1)
    ]
    return "\n".join(rows) + "\n"


def decode(png: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png)) as image:
        assert image.mode == "L"
        return np.asarray(image)


class FakeTesseract:
    """Records every call and answers like Tesseract would for the rendered image.

    A blank (all white) image yields no words. Any other image yields a tag word
    ``"<languages>:<image width>"`` at pixel box ``(0, 0, 8, 8)`` followed by the words of
    ``page_text[width]`` when that width is listed, else of ``text(languages)``.
    ``fail_width`` makes the image of that width fail; ``delay(width)`` returns seconds to sleep
    before answering, to shuffle completion order between workers. ``widths`` lists the image
    width of every call in call order; ``max_active`` is the largest number of overlapping calls.
    """

    def __init__(
        self,
        text: Callable[[str], str] = lambda languages: "",
        *,
        fail_width: int | None = None,
        delay: Callable[[int], float] = lambda width: 0.0,
        page_text: dict[int, str] | None = None,
    ) -> None:
        self._text = text
        self._fail_width = fail_width
        self._delay = delay
        self._page_text = page_text or {}
        self._lock = threading.Lock()
        self.calls: list[str] = []
        self.pngs: list[bytes] = []
        self.widths: list[int] = []
        self.active = 0
        self.max_active = 0

    def __call__(self, png: bytes, languages: str) -> str:
        image = decode(png)
        width = image.shape[1]
        with self._lock:
            self.calls.append(languages)
            self.pngs.append(png)
            self.widths.append(width)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self._delay(width))
            if width == self._fail_width:
                raise ProcessingFailed("tesseract exited with code 1", "ocr_failed")
            if image.min() == 255:
                return tsv([])
            box = (0, 0, 8, 8)
            text = self._page_text.get(width)
            text = self._text(languages) if text is None else text
            words = [f"{languages}:{width}", *text.split()]
            return tsv([(word, box) for word in words])
        finally:
            with self._lock:
                self.active -= 1
