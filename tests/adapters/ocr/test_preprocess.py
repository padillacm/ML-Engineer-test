import time
import tracemalloc

import numpy as np
import pytest

from app.adapters.ocr.preprocess import (
    flatten_illumination,
    gaussian_blur_u8,
    gaussian_kernel_u8,
)

# Reference values produced by OpenCV 5.0 (the benchmark's cv2.GaussianBlur / ocr_run.preprocess
# 'flat'); OpenCV itself is not a dependency. Kernel taps are the centre and right half.
OPENCV_TAPS_RIGHT_HALF = {
    dpi: [int(tap) for tap in taps.split()]
    for dpi, taps in {
        72: "8 9 8 8 9 7 8 7 7 6 6 6 5 5 4 4 4 3 3 2 2 2 2 1 1 1 1 0 1 1 0 0 0 1 0 0 0",
        200: "4 3 3 3 3 3 3 3 3 3 3 3 3 2 3 3 3 2 3 3 2 3 2 3 2 2 3 2 2 2 2 2 2 2 2 2 1 2 1 2 "
        "2 1 1 2 1 1 1 2 1 1 1 1 1 0 1 1 1 0 1 1 0 1 1 0 1 0 0 1 0 1 0 0 1 0 0 0 1 0 0 0 "
        "0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0",
    }.items()
}
OPENCV_SMALL_PAGE = {  # 5x7 image at 72 DPI: kernel (73 taps) far wider than the image
    "image": [
        [95, 130, 194, 217, 207, 235, 15],
        [163, 33, 215, 217, 130, 248, 189],
        [16, 69, 184, 232, 205, 78, 169],
        [61, 125, 10, 29, 240, 66, 19],
        [182, 39, 59, 4, 59, 81, 222],
    ],
    "blur": [
        [131, 130, 130, 130, 131, 130, 131],
        [131, 130, 130, 130, 130, 130, 131],
        [131, 131, 130, 130, 131, 130, 131],
        [131, 131, 130, 130, 131, 130, 131],
        [131, 131, 130, 130, 131, 130, 131],
    ],
    "flat": [
        [146, 255, 255, 255, 255, 255, 255],
        [146, 255, 255, 255, 255, 255, 255],
        [45, 69, 236, 255, 255, 255, 255],
        [45, 45, 40, 40, 99, 106, 106],
        [236, 40, 0, 39, 39, 106, 106],
    ],
}


def _sigma(dpi: int) -> float:
    return 25.0 * dpi / 150.0


def _naive_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Direct integer fixed-point blur: rows then columns, reflect-101 border, round half up."""
    half = len(kernel) // 2
    height, width = image.shape
    padded = np.pad(image.astype(np.int64), half, mode="reflect")  # numpy 'reflect' == reflect-101
    rows = np.zeros((height + 2 * half, width), dtype=np.int64)
    for i, tap in enumerate(kernel):
        rows += int(tap) * padded[:, i : i + width]
    columns = np.zeros((height, width), dtype=np.int64)
    for i, tap in enumerate(kernel):
        columns += int(tap) * rows[i : i + height]
    blurred: np.ndarray = ((columns + (1 << 15)) >> 16).astype(np.uint8)
    return blurred


def _gradient_lit_page(height: int = 400, width: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """A page lit from the left (paper 250 -> 110) with dark text-like bars; returns the mask."""
    lighting = np.linspace(250.0, 110.0, width, dtype=np.float32)
    paper = np.tile(lighting, (height, 1))
    ink = np.zeros((height, width), dtype=bool)
    for top in range(40, height - 40, 30):
        ink[top : top + 8, 30 : width - 30] = True
    page = np.where(ink, paper * 0.2, paper)
    return page.astype(np.uint8), ink


@pytest.mark.parametrize("dpi", sorted(OPENCV_TAPS_RIGHT_HALF))
def test_kernel_matches_opencv_bit_exact_taps(dpi: int) -> None:
    kernel = gaussian_kernel_u8(_sigma(dpi))

    right = OPENCV_TAPS_RIGHT_HALF[dpi]
    assert kernel.tolist() == right[:0:-1] + right


@pytest.mark.parametrize("dpi", [1, 50, 72, 96, 150, 200, 300, 600])
def test_kernel_is_symmetric_sums_to_one_and_spans_three_sigma(dpi: int) -> None:
    sigma = _sigma(dpi)
    kernel = gaussian_kernel_u8(sigma)

    assert len(kernel) == round(sigma * 6 + 1) | 1
    assert int(kernel.sum()) == 256
    assert np.array_equal(kernel, kernel[::-1])
    assert (kernel >= 0).all()


def test_impulse_response_is_the_rounded_kernel_outer_product() -> None:
    kernel = gaussian_kernel_u8(_sigma(150))
    size = 2 * len(kernel) + 1
    image = np.zeros((size, size), dtype=np.uint8)
    image[size // 2, size // 2] = 255

    blurred = gaussian_blur_u8(image, _sigma(150))

    expected = np.zeros((size, size), dtype=np.int64)
    half = len(kernel) // 2
    centre = slice(size // 2 - half, size // 2 + half + 1)
    expected[centre, centre] = 255 * np.outer(kernel, kernel)
    assert np.array_equal(blurred, ((expected + (1 << 15)) >> 16).astype(np.uint8))


@pytest.mark.parametrize(
    ("shape", "dpi"),
    [((60, 45), 30), ((23, 17), 72), ((1, 40), 30), ((40, 1), 30), ((1, 1), 72), ((120, 90), 40)],
)
def test_blur_equals_direct_fixed_point_convolution_with_reflect_101(
    shape: tuple[int, int], dpi: int
) -> None:
    image = np.random.default_rng(3).integers(0, 256, size=shape, dtype=np.uint8)

    blurred = gaussian_blur_u8(image, _sigma(dpi))

    assert np.array_equal(blurred, _naive_blur(image, gaussian_kernel_u8(_sigma(dpi))))


def test_small_page_matches_opencv_reference() -> None:
    image = np.array(OPENCV_SMALL_PAGE["image"], dtype=np.uint8)

    assert gaussian_blur_u8(image, _sigma(72)).tolist() == OPENCV_SMALL_PAGE["blur"]
    assert flatten_illumination(image, dpi=72).tolist() == OPENCV_SMALL_PAGE["flat"]


def test_uneven_lighting_is_flattened() -> None:
    page, ink = _gradient_lit_page()
    # Background far from the bars, so the 3x3 median does not mix ink into it.
    near_ink = ink.copy()
    for shift in range(-3, 4):
        near_ink |= np.roll(ink, shift, axis=0) | np.roll(ink, shift, axis=1)
    background = ~near_ink

    flat = flatten_illumination(page, dpi=150)

    assert page[background].std() > 30
    assert flat[background].std() < page[background].std() / 5
    assert flat[ink].mean() < flat[background].mean() - 100


@pytest.mark.parametrize(("shape", "dpi"), [((400, 300), 150), ((257, 131), 200), ((5, 7), 72)])
def test_shape_and_dtype_are_preserved(shape: tuple[int, int], dpi: int) -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=shape, dtype=np.uint8)

    flat = flatten_illumination(image, dpi=dpi)

    assert flat.shape == shape
    assert flat.dtype == np.uint8


def test_blank_page_stays_white() -> None:
    blank = np.full((120, 80), 255, dtype=np.uint8)

    flat = flatten_illumination(blank, dpi=200)

    assert np.array_equal(flat, blank)
    assert flat.flags.writeable


def test_contrast_is_stretched_to_the_full_range() -> None:
    page, _ = _gradient_lit_page()

    flat = flatten_illumination(page, dpi=150)

    assert int(flat.min()) == 0
    assert int(flat.max()) == 255


def test_input_is_not_modified() -> None:
    page, _ = _gradient_lit_page()
    original = page.copy()

    flatten_illumination(page, dpi=150)

    assert np.array_equal(page, original)


def _large_page(height: int, width: int) -> np.ndarray:
    page = np.full((height, width), 235, dtype=np.uint8)
    page[:, : width // 2] = 180
    page[::40, :] = 20
    return page


def test_a4_page_at_200_dpi_is_fast() -> None:
    page = _large_page(2339, 1654)
    flatten_illumination(page, dpi=200)

    start = time.perf_counter()
    flatten_illumination(page, dpi=200)

    # ~0.14 s in the test container; a generous bound that still catches a per-tap convolution.
    assert time.perf_counter() - start < 2.0


def test_memory_stays_a_few_bytes_per_pixel() -> None:
    page = _large_page(4000, 3000)  # 12 M pixels

    tracemalloc.start()
    try:
        flatten_illumination(page, dpi=200)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Full-size buffers: uint16 row pass + uint8 background/flat/output; FFT strips are bounded.
    # A float32 full-size temporary alone would add 4 B/pixel.
    assert peak < 4 * page.size + 48 * 2**20
