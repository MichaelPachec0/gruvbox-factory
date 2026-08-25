"""Pure pixel mapping: arrays in, arrays out. No I/O, no palette loading.

Two kernels, one interface. ``map_rgb_packed`` uniques the colors actually
present in the frame and does an L1 argmin over that much smaller set;
``map_rgb_lut`` indexes a prebuilt table covering all 2**24 colors. They are
required to agree exactly, and a test asserts it.

Measured on a 3490x984 RGBA image with 116,135 unique colors:

    packed kernel        0.83s
    lut apply            0.29s
    lut build (once)    13.0s

so the table only pays off across many frames. Do not replace the packed
kernel with a table build for stills.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np

from factory.palette import Palette

# Pixels this transparent are left exactly as they are. The value comes from
# image-go-nord and is kept so that behaviour on transparent regions does not
# change; only the palette-mutation defect does.
TRANSPARENCY_TOLERANCE = 190

# Rows of the (chunk x K x 3) distance matrix built at a time. At K=131 that
# matrix is 52MB, which is the peak working set regardless of image size.
UNIQUE_CHUNK = 65536

# 13.542s to build the table, against 0.452s - 0.055s saved per 1080p frame.
LUT_BREAK_EVEN_FRAMES = 34

# What select_kernel may return, and what bind accepts. "auto" is a policy
# rather than a kernel, so it appears in KERNELS but never in RESOLVED_KERNELS.
RESOLVED_KERNELS = ("lut", "packed")
KERNELS = ("auto", *RESOLVED_KERNELS)

RgbKernel = Callable[[np.ndarray], np.ndarray]


class UnknownKernelError(ValueError):
    """A kernel name that does not exist."""

    def __init__(self, mode: str, choices: tuple[str, ...] = KERNELS) -> None:
        super().__init__(f"unknown kernel {mode!r}; choose from {', '.join(choices)}")


class MissingTableError(ValueError):
    """The lut kernel was requested without a table to index."""

    def __init__(self) -> None:
        super().__init__("the lut kernel needs a table; build one with lut.get")


def pack(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 -> (N,) uint32 keys, laid out as r << 16 | g << 8 | b."""
    flat = rgb.reshape(-1, 3).astype(np.uint32)
    return (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]


def unpack(key: np.ndarray) -> np.ndarray:
    """(N,) uint32 keys -> (N, 3) int16 colors."""
    channels = ((key >> 16) & 255, (key >> 8) & 255, key & 255)
    return np.stack(channels, axis=1).astype(np.int16)


def nearest(
    colors: np.ndarray, palette: Palette, *, chunk: int = UNIQUE_CHUNK
) -> np.ndarray:
    """(N, 3) int16 -> (N,) uint8 index of the L1-nearest palette entry.

    Ties fall to the lowest index. Palette sorts its names ascending on load,
    so the lowest index is the lexicographically smallest '#rrggbb' name,
    which is the documented tie-break.

    The sum accumulates in int16 deliberately: three channel deltas cap at
    765, well inside int16, and letting numpy promote to int64 would quadruple
    the largest allocation here for nothing.
    """
    out = np.empty(len(colors), dtype=np.uint8)
    entries = palette.rgb
    for start in range(0, len(colors), chunk):
        stop = start + chunk
        block = colors[start:stop]
        distance = np.abs(block[:, None, :] - entries[None, :, :]).sum(
            axis=-1, dtype=np.int16
        )
        out[start:stop] = distance.argmin(axis=1).astype(np.uint8)
    return out


def map_rgb_packed(
    rgb: np.ndarray, palette: Palette, *, chunk: int = UNIQUE_CHUNK
) -> np.ndarray:
    """Map (H, W, 3) uint8 through the palette, uniquing the colors present."""
    unique, inverse = np.unique(pack(rgb), return_inverse=True)
    index = nearest(unpack(unique), palette, chunk=chunk)
    return palette.rgb[index].astype(np.uint8)[inverse].reshape(rgb.shape)


def map_rgb_lut(rgb: np.ndarray, palette: Palette, table: np.ndarray) -> np.ndarray:
    """Map (H, W, 3) uint8 through a prebuilt 2**24-entry index table.

    The table is trusted. lut.load_cached is the only supported way to obtain
    one and it rejects anything of the wrong size, so validating again here
    would cost a branch on every frame to re-check an invariant already held.
    """
    return palette.rgb[table[pack(rgb)]].astype(np.uint8).reshape(rgb.shape)


def bind(palette: Palette, name: str, table: np.ndarray | None = None) -> RgbKernel:
    """Turn a resolved kernel name into a callable taking only pixels.

    This is the one place that knows a lut kernel needs a table and a packed
    one does not, so callers hold a plain (H, W, 3) -> (H, W, 3) function.
    """
    if name not in RESOLVED_KERNELS:
        raise UnknownKernelError(name, RESOLVED_KERNELS)
    if name == "lut":
        if table is None:
            raise MissingTableError
        return partial(map_rgb_lut, palette=palette, table=table)
    return partial(map_rgb_packed, palette=palette)


def apply_rgba(rgba: np.ndarray, map_rgb: RgbKernel) -> np.ndarray:
    """Apply a bound RGB kernel to (H, W, 4) uint8, honouring the alpha rule.

    Sufficiently transparent pixels keep their original color, and every pixel
    keeps its original alpha. Alpha never enters the distance metric.
    """
    out = rgba.copy()
    opaque = rgba[..., 3] >= TRANSPARENCY_TOLERANCE
    out[opaque, :3] = map_rgb(rgba[opaque, :3])
    return out


def select_kernel(mode: str, n_frames: int, cache_hit: bool) -> str:
    """Resolve 'auto' against frame count and cache state. Pure decision logic."""
    if mode not in KERNELS:
        raise UnknownKernelError(mode)
    if mode != "auto":
        return mode
    if cache_hit:
        return "lut"
    return "lut" if n_frames >= LUT_BREAK_EVEN_FRAMES else "packed"
