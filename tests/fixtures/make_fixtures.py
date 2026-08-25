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


def main() -> None:
    destination = Path(__file__).parent / STILL_NAME
    Image.fromarray(build_still(), "RGBA").save(destination, optimize=True)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
