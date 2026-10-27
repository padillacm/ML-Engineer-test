"""Illumination flattening of a grayscale page image, with numpy and Pillow only.

Reproduces, pixel for pixel, the benchmarked ``flat`` preprocessing (OpenCV): divide the image
by a heavily blurred copy of itself (the background), clip, remove speckle with a 3x3 median and
stretch the contrast from the 1st percentile up to white.

The background blur is OpenCV's ``GaussianBlur(image, (0, 0), sigma)`` for 8-bit images, which is
*not* a float Gaussian: OpenCV builds a bit-exact kernel quantised to 1/256 (error diffusion from
the tails, centre tap = 256 - rest), sizes it ``round(6 * sigma + 1) | 1``, filters rows then
columns in integer fixed point with ``BORDER_REFLECT_101`` and rounds half up once at the end.
Every intermediate is an integer, so the two passes are done here as float64 FFT convolutions
whose results are exact integers after rounding (all values stay far below 2**53), in strips that
bound the temporary memory. The per-pixel division and the contrast stretch are lookup tables
built with the benchmark's own float arithmetic.
"""

import math
from functools import cache

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFilter

from app.adapters.ocr.rendering import GrayImage

_SIGMA_AT_150_DPI = 25.0
_MEDIAN_SIZE = 3
_DARK_PERCENTILE = 1.0
_WHITE = 255.0
_KERNEL_BITS = 8  # OpenCV's ufixedpoint16 kernel taps: 8 fractional bits
_ONE = 1 << _KERNEL_BITS
_STRIP_CELLS = 1 << 20  # FFT samples per strip (~25 MB of float64/complex temporaries)

type _Kernel = npt.NDArray[np.int64]


def flatten_illumination(image: GrayImage, dpi: int) -> GrayImage:
    """Return ``image`` (2-D ``uint8``) evenly lit, with the same shape and dtype.

    The background blur scales with the resolution so the same physical size is smoothed at
    every DPI. A blank white image is returned white.
    """
    sigma = _SIGMA_AT_150_DPI * dpi / 150.0
    background = gaussian_blur_u8(image, sigma)
    flat = _divide(image, background)
    del background
    flat = np.asarray(Image.fromarray(flat).filter(ImageFilter.MedianFilter(_MEDIAN_SIZE)))
    low = np.percentile(flat, _DARK_PERCENTILE)
    if low >= _WHITE:
        # An evenly white (blank) page has nothing to stretch; stretching would turn it black.
        return np.array(flat)
    result: GrayImage = _stretch_table(low)[flat]
    return result


def gaussian_blur_u8(image: GrayImage, sigma: float) -> GrayImage:
    """Bit-exact ``cv2.GaussianBlur(image, (0, 0), sigma)`` for a 2-D ``uint8`` image."""
    kernel = gaussian_kernel_u8(sigma)
    # Rows: sum of uint8 * 8-bit taps, an exact integer below 2**16 (8 fractional bits).
    rows = _convolve(image, kernel, axis=1, out_dtype=np.uint16, shift=0)
    # Columns: sum of the above * 8-bit taps (< 2**24), rounded half up from 16 fractional bits.
    blurred: GrayImage = _convolve(rows, kernel, axis=0, out_dtype=np.uint8, shift=2 * _KERNEL_BITS)
    return blurred


def gaussian_kernel_u8(sigma: float) -> _Kernel:
    """OpenCV's 8-bit Gaussian kernel for ``ksize=(0, 0)``: integer taps summing to 256."""
    size = _round_half_even(sigma * 3 * 2 + 1) | 1
    half = size // 2
    # getGaussianKernelBitExact: exp(-d^2 / (2 sigma^2)) for the tail taps, centre 1, normalised.
    scale = -0.125 / (sigma * sigma)
    values = []
    total = 0.0
    for x in range(1 - size, 0, 2):
        value = math.exp(float(x * x) * scale)
        values.append(value)
        total += value
    total = total * 2.0 + 1.0
    inverse = 1.0 / total
    # getGaussianKernelFixedPoint_ED: error diffusion from the tail inwards, centre takes the rest.
    taps = np.zeros(size, dtype=np.int64)
    error = 0.0
    for index, value in enumerate(values):
        adjusted = value * inverse * _ONE + error
        tap = _round_half_even(adjusted)
        error = adjusted - tap
        taps[index] = taps[size - 1 - index] = tap
    taps[half] = _ONE - 2 * int(taps[:half].sum())
    return taps


def _round_half_even(value: float) -> int:
    return round(value)


def _convolve[T: np.unsignedinteger](
    image: npt.NDArray[np.unsignedinteger],
    kernel: _Kernel,
    axis: int,
    out_dtype: type[T],
    shift: int,
) -> npt.NDArray[T]:
    """Integer convolution of every line along ``axis`` with a symmetric kernel, reflect-101 border.

    The exact integer sums are divided by ``2**shift`` rounding half up (as OpenCV's fixed point
    does), strip by strip, so no full-size wide temporary is ever allocated.
    """
    length = image.shape[axis]
    half = len(kernel) // 2
    source = _reflect_101(np.arange(-half, length + half), length)
    fft_size = _fast_size(len(source))
    spectrum = np.fft.rfft(kernel.astype(np.float64), fft_size)
    out = np.empty(image.shape, dtype=out_dtype)
    other = image.shape[1 - axis]
    step = max(1, _STRIP_CELLS // fft_size)
    for start in range(0, other, step):
        stop = min(start + step, other)
        if axis == 1:
            lines = image[start:stop].take(source, axis=1)
        else:
            lines = image[:, start:stop].take(source, axis=0)
        filtered = np.fft.irfft(
            np.fft.rfft(lines, fft_size, axis=axis) * _along(spectrum, axis), fft_size, axis=axis
        )
        del lines
        # The symmetric kernel makes correlation = convolution; output x sits at index x + 2*half.
        window = (
            filtered[:, 2 * half : 2 * half + length]
            if axis == 1
            else filtered[2 * half : 2 * half + length]
        )
        target = out[start:stop] if axis == 1 else out[:, start:stop]
        np.rint(window, out=window)
        if shift:
            # Exact in float64: integers < 2**24 plus a half, divided by a power of two.
            window += 1 << (shift - 1)
            window /= 1 << shift
            np.floor(window, out=window)
        target[...] = window
        del filtered
    return out


def _along(spectrum: npt.NDArray[np.complex128], axis: int) -> npt.NDArray[np.complex128]:
    return spectrum[np.newaxis, :] if axis == 1 else spectrum[:, np.newaxis]


def _reflect_101(index: npt.NDArray[np.intp], length: int) -> npt.NDArray[np.intp]:
    """OpenCV's BORDER_REFLECT_101 (``dcb|abcd|cba``), repeated for kernels wider than the image."""
    if length == 1:
        return np.zeros_like(index)
    period = 2 * (length - 1)
    index = np.mod(index, period)
    return np.where(index < length, index, period - index)


def _fast_size(minimum: int) -> int:
    """Smallest 2^a 3^b 5^c >= ``minimum`` (a size numpy's pocketfft handles fastest)."""
    best = 1 << max(minimum - 1, 0).bit_length()
    power5 = 1
    while power5 < best:
        power35 = power5
        while power35 < best:
            candidate = power35
            while candidate < minimum:
                candidate *= 2
            best = min(best, candidate)
            power35 *= 3
        power5 *= 5
    return best


def _divide(image: GrayImage, background: GrayImage) -> GrayImage:
    """``clip(image / max(background, 1) * 255)`` in float32, as the benchmark computes it."""
    table = _division_table()
    flat = np.empty_like(image)
    rows = max(1, _STRIP_CELLS // max(image.shape[1], 1))
    for start in range(0, image.shape[0], rows):
        stop = start + rows
        index = background[start:stop].astype(np.uint16)
        index <<= 8
        index |= image[start:stop]
        flat[start:stop] = table[index]
    return flat


@cache
def _division_table() -> npt.NDArray[np.uint8]:
    """Flat 64 K table indexed by ``background << 8 | pixel``."""
    values = np.arange(256, dtype=np.float32)
    ratio = (
        values[np.newaxis, :]
        / np.maximum(values[:, np.newaxis], np.float32(1.0))
        * np.float32(_WHITE)
    )
    table: npt.NDArray[np.uint8] = np.clip(ratio, 0.0, _WHITE).astype(np.uint8).ravel()
    table.flags.writeable = False
    return table


def _stretch_table(low: np.floating) -> npt.NDArray[np.uint8]:
    """``(value - low) * 255 / (255 - low)`` clipped, in the benchmark's float64 arithmetic."""
    values = np.arange(256, dtype=np.float32).astype(np.float64)
    stretched = (values - low) * _WHITE / np.maximum(_WHITE - low, 1.0)
    table: npt.NDArray[np.uint8] = np.clip(stretched, 0.0, _WHITE).astype(np.uint8)
    return table
