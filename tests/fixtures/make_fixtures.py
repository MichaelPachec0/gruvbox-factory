"""Regenerate the binary fixtures in this directory.

Deterministic by construction: no RNG, no clock, no locale, no float
arithmetic. Run it from the repository root with

    nix develop --command python tests/fixtures/make_fixtures.py

A slow-marked test asserts that the committed PNG still matches this code, so
the fixture can never drift away from its own provenance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

WIDTH, HEIGHT = 200, 120
STILL_NAME = "still_rgba.png"

GIF_WIDTH, GIF_HEIGHT = 64, 48
GIF_FRAMES = 4
GIF_DURATIONS = [40, 80, 120, 160]
GIF_DISPOSALS = [0, 2, 1, 2]
ANIMATED_NAME = "animated.gif"

# The fixture's own palette, unrelated to gruvbox. 24 colors leaves index 24
# free for transparency and keeps the file honest GIF input rather than
# something already sitting in the output palette.
_GIF_PALETTE = [
    [red, green, blue]
    for red in (0, 85, 170, 255)
    for green in (0, 128, 255)
    for blue in (0, 255)
]


def build_still() -> np.ndarray:
    """A 200x120 RGBA array exercising every clause of the kernel contract.

    Red ramps left to right, green top to bottom, blue along x+y. That yields
    23,998 distinct colors in 24,000 pixels, so the unique-color path does
    real work instead of collapsing to a handful of entries.

    Alpha ramps 128..255 down the rows and reaches TRANSPARENCY_TOLERANCE
    (190) exactly at y=59, putting the boundary on a pixel rather than between
    two. 11,796 pixels land below it and must survive untouched; the other
    12,204 must be converted.

    Row 0, columns 0 to 3 are opaque controls: pure black, pure white,
    #928374 (a member of all three shipped palettes, so it must map to
    itself) and #808080 (a mid-grey probe).
    """
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    rgba[..., 0] = columns * 255 // (WIDTH - 1)
    rgba[..., 1] = rows * 255 // (HEIGHT - 1)
    rgba[..., 2] = (columns + rows) * 255 // (WIDTH + HEIGHT - 2)
    rgba[..., 3] = 128 + rows * 127 // (HEIGHT - 1)
    rgba[0, :4, 3] = 255
    rgba[0, 0, :3] = (0x00, 0x00, 0x00)
    rgba[0, 1, :3] = (0xFF, 0xFF, 0xFF)
    rgba[0, 2, :3] = (0x92, 0x83, 0x74)
    rgba[0, 3, :3] = (0x80, 0x80, 0x80)
    return rgba


def build_animated() -> list[np.ndarray]:
    """Four 64x48 RGBA frames with a transparent square that moves.

    Frame content shifts enough between frames that Pillow's identical-frame
    merging never triggers, so the frame count survives a round trip. Pillow
    merges consecutive identical frames and sums their durations, which would
    otherwise make a converted GIF come out shorter than its input.
    """
    rows, columns = np.mgrid[0:GIF_HEIGHT, 0:GIF_WIDTH]
    frames = []
    for index in range(GIF_FRAMES):
        rgba = np.zeros((GIF_HEIGHT, GIF_WIDTH, 4), dtype=np.uint8)
        rgba[..., 0] = (columns * 4 + index * 37) % 256
        rgba[..., 1] = (rows * 5 + index * 61) % 256
        rgba[..., 2] = (columns + rows + index * 23) % 256
        rgba[..., 3] = 255
        top = 4 + index * 6
        left = 8 + index * 10
        rgba[top : top + 12, left : left + 12, 3] = 0
        frames.append(rgba)
    return frames


def _to_paletted(rgba: np.ndarray, colors: np.ndarray, index: int) -> Image.Image:
    """Quantize onto an explicit palette, reserving `index` for alpha == 0."""
    flat = rgba[..., :3].reshape(-1, 3).astype(np.int16)
    distance = np.abs(flat[:, None, :] - colors[None, :, :]).sum(axis=-1)
    table = distance.argmin(axis=1).astype(np.uint8).reshape(rgba.shape[:2])
    table[rgba[..., 3] == 0] = index
    image = Image.fromarray(table, mode="P")
    entries = list(colors.astype(np.uint8).reshape(-1))
    image.putpalette(entries + [0] * (768 - len(entries)))
    return image


def write_animated(destination: Path) -> None:
    colors = np.array(_GIF_PALETTE, dtype=np.int16)
    transparent = len(colors)
    pages = [_to_paletted(f, colors, transparent) for f in build_animated()]
    pages[0].save(
        destination,
        save_all=True,
        append_images=pages[1:],
        duration=GIF_DURATIONS,
        disposal=GIF_DISPOSALS,
        loop=0,
        transparency=transparent,
        optimize=False,
    )


def main() -> None:
    here = Path(__file__).parent
    still = here / STILL_NAME
    Image.fromarray(build_still(), "RGBA").save(still, optimize=True)
    print(f"wrote {still}")
    animated = here / ANIMATED_NAME
    write_animated(animated)
    print(f"wrote {animated}")


if __name__ == "__main__":
    main()
