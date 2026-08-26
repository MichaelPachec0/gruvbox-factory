"""Decode and encode. Every format difference lives here and nowhere else.

Each adapter is a context manager, so file handles, ffmpeg subprocesses and
temporary files are released on every exit path, including an abort part-way
through a video. The palette and the metadata policy are bound at
``open_source``, so ``write`` takes only frames and a destination and callers
never have to know which formats need what.

Dispatch is by suffix; the actual decoded format comes from Pillow's own
sniffing and drives metadata transport. A mislabelled file therefore still
converts correctly and keeps the name the user asked for.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, Self

import numpy as np
from PIL import Image

from factory import metadata
from factory.palette import Palette

STILL_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"})
GIF_SUFFIXES = frozenset({".gif"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


class UnsupportedFormatError(ValueError):
    """A path whose suffix matches no adapter."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"unsupported file type: {path.name}")


class UnopenedSourceError(RuntimeError):
    """An adapter used before its context manager was entered.

    A dedicated error rather than an assert: this is a programming mistake the
    caller should see named, and asserts vanish under python -O.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(f"{path.name}: use the source as a context manager")


class Source(Protocol):
    """One input file, decoded to frames and re-encoded in the same format."""

    n_frames: int | None

    def __enter__(self) -> Self: ...

    def __exit__(self, *exc: object) -> None: ...

    def frames(self) -> Iterator[np.ndarray]: ...

    def write(self, frames: Iterator[np.ndarray], dest: Path) -> None: ...


def _has_alpha(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA", "PA"}:
        return True
    return image.mode == "P" and "transparency" in image.info


class StillSource:
    """A single image. One frame in, one file out."""

    def __init__(self, path: Path, palette: Palette, *, keep_metadata: bool) -> None:
        self.path = path
        self.palette = palette
        self.keep_metadata = keep_metadata
        self.n_frames: int | None = 1
        self._image: Image.Image | None = None
        self._metadata = metadata.Metadata()
        self._format = "PNG"
        self._mode = "RGB"

    def __enter__(self) -> Self:
        opened = Image.open(self.path)
        # Capture before anything rewrites the image: format and info are both
        # lost the moment the pixels are touched.
        self._format = opened.format or "PNG"
        self._metadata = metadata.capture(opened)
        self._mode = "RGBA" if _has_alpha(opened) else "RGB"
        self._image = metadata.upright(opened).convert(self._mode)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._image is not None:
            self._image.close()
            self._image = None

    def frames(self) -> Iterator[np.ndarray]:
        if self._image is None:
            raise UnopenedSourceError(self.path)
        yield np.array(self._image)

    def write(self, frames: Iterator[np.ndarray], dest: Path) -> None:
        array = next(iter(frames))
        out = Image.fromarray(array, self._mode)
        out.save(
            dest,
            **metadata.save_kwargs(
                self._metadata, self._format, keep=self.keep_metadata
            ),
        )


def open_source(
    path: Path | str, palette: Palette, *, keep_metadata: bool = False
) -> Source:
    """Pick an adapter by suffix.

    Only selects; it does not open. A missing or unreadable file is reported
    by Pillow or ffmpeg inside the context manager, which avoids duplicating a
    check the opener performs anyway.
    """
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix in STILL_SUFFIXES:
        return StillSource(resolved, palette, keep_metadata=keep_metadata)
    raise UnsupportedFormatError(resolved)
