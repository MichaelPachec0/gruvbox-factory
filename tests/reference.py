"""The reference conversion: slow, obvious, and deliberately not shared.

This is a faithful port of image-go-nord's ``converted_loop`` with defect D1
fixed. Its value is that it was written from the upstream algorithm rather
than from factory/kernel.py, so agreement between the two is evidence and not
a tautology. It uses np.unique(axis=0) and plain int64 accumulation -- both
choices the real kernel rejects on performance grounds.

The same port with D1 *replicated* reproduces the library's own
d1964b3f03fd0542 for example.png byte for byte, which is what established that
the port is faithful. With D1 fixed it produces 4ff69519c583a3d5.

Do not optimise this file. Its slowness is the point.
"""

from __future__ import annotations

import numpy as np

TRANSPARENCY_TOLERANCE = 190


def convert(rgba: np.ndarray, entries: np.ndarray) -> np.ndarray:
    """Map an (H, W, 4) uint8 image through a (K, 3) palette by L1 distance."""
    rgb = rgba[..., :3].astype(np.int16)
    alpha = rgba[..., 3]
    flat = rgb.reshape(-1, 3)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    distance = np.abs(unique[:, None, :] - entries[None, :, :]).sum(-1)
    nearest = entries[distance.argmin(1)].astype(np.uint8)
    mapped = nearest[inverse].reshape(rgba.shape[0], rgba.shape[1], 3)
    out = np.dstack([mapped, alpha])
    transparent = alpha < TRANSPARENCY_TOLERANCE
    out[transparent] = rgba[transparent]
    return out
