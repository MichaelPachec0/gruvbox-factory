"""Build, cache and evict the 24-bit palette-index lookup table.

The table is 2**24 uint8 entries, exactly 16 MiB, mapping a packed RGB key to
a *palette index* rather than to a color. Storing the index keeps it a third
the size and means one table serves any palette whose canonical form matches.

The cache key is the palette's content hash, never its filename. A filename
key would serve a stale table after a palette edit, which is the single most
likely thing to get wrong in this design.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from factory.kernel import nearest
from factory.palette import Palette

LUT_SIZE = 1 << 24
CACHE_LIMIT_BYTES = 100 * 1024 * 1024

# One red value covers 65536 (green, blue) pairs, so the table is built as 256
# planes of that size. This bounds the distance matrix the same way
# kernel.UNIQUE_CHUNK does for the packed kernel.
_PLANE = 1 << 16


def build(palette: Palette) -> np.ndarray:
    """Compute the whole table. Roughly 13 seconds, largely independent of K."""
    table = np.empty(LUT_SIZE, dtype=np.uint8)
    axis = np.arange(256, dtype=np.int16)
    green_blue = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)
    plane = np.empty((_PLANE, 3), dtype=np.int16)
    plane[:, 1:] = green_blue.reshape(-1, 2)
    for red in range(256):
        plane[:, 0] = red
        table[red * _PLANE : (red + 1) * _PLANE] = nearest(
            plane, palette, chunk=_PLANE
        )
    return table


def cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(root) / "gruvbox-factory"


def cache_path(palette: Palette, directory: Path | None = None) -> Path:
    return (directory or cache_dir()) / f"{palette.content_hash}.lut"


def load_cached(palette: Palette, directory: Path | None = None) -> np.ndarray | None:
    """Return the cached table, or None if it is absent or the wrong size.

    Size is the only validation a headerless file allows, and it is enough:
    the write is atomic, so a wrong size means a crashed or foreign writer.

    The mtime is stamped on every hit because that is what eviction sorts on.
    atime would be free, but relatime updates it at most once a day and
    noatime never does, so recency has to be recorded deliberately.
    """
    path = cache_path(palette, directory)
    try:
        if path.stat().st_size != LUT_SIZE:
            return None
        os.utime(path)
    except OSError:
        return None
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(LUT_SIZE,))


def store(palette: Palette, table: np.ndarray, directory: Path | None = None) -> Path:
    """Write the table atomically, then evict down to the cap.

    Two processes may build the same table concurrently and both write it.
    That wastes work but cannot corrupt anything: the content is deterministic,
    so whichever rename lands last leaves a correct file, and no reader ever
    observes a partial one.

    There is no fsync. This is a cache; an entry lost to a crash costs 13
    seconds to rebuild, which is not worth stalling every miss to prevent.
    """
    target = directory or cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(table.tobytes())
        path = cache_path(palette, target)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    evict(target)
    return path


def evict(directory: Path, limit: int = CACHE_LIMIT_BYTES) -> list[Path]:
    """Delete least-recently-used tables until the directory fits the cap.

    Only writes can grow the cache, and every write calls this, so reads are
    left alone rather than paying a directory scan to enforce an invariant
    they cannot break.
    """
    entries = []
    for path in directory.glob("*.lut"):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
    entries.sort()
    total = sum(size for _, size, _ in entries)
    removed = []
    for _, size, path in entries:
        if total <= limit:
            break
        path.unlink(missing_ok=True)
        total -= size
        removed.append(path)
    return removed


def get(palette: Palette, directory: Path | None = None) -> np.ndarray:
    """The cached table for this palette, building and storing it on a miss."""
    table = load_cached(palette, directory)
    if table is None:
        table = build(palette)
        store(palette, table, directory)
    return table
