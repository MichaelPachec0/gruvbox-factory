"""Load ``#rrggbb`` palette files into an immutable, canonical Palette.

Canonical means sorted ascending by hex name, deduplicated, lowercased and
``#``-prefixed. Sorting is not cosmetic: the kernel resolves distance ties with
a plain ``argmin``, which returns the first minimum, so sorted order is exactly
what makes ties fall to the lexicographically smallest name.

``content_hash`` covers the canonical form rather than the raw file bytes, so
two files differing only in order, case, blank lines or a duplicated entry
describe one palette and share one cached lookup table.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib.resources import files

import numpy as np

BUILTIN = ("pink", "white", "mix")

_HEX = re.compile(r"\A#?([0-9a-fA-F]{6})\Z")


class PaletteError(ValueError):
    """A palette that cannot be loaded.

    Subclasses build their own message so that raise sites stay short. They
    remain ValueError, so callers may catch either name.
    """


class PaletteSyntaxError(PaletteError):
    """A line that is not a six-digit hex color."""

    def __init__(self, source: str, lineno: int, line: str) -> None:
        super().__init__(f"{source}:{lineno}: expected #rrggbb, got {line!r}")


class PaletteEncodingError(PaletteError):
    """A palette file containing bytes outside ASCII."""

    def __init__(self, source: str) -> None:
        super().__init__(f"{source}: palette must be ASCII")


class EmptyPaletteError(PaletteError):
    """A palette with no colors in it."""

    def __init__(self, source: str) -> None:
        super().__init__(f"{source}: palette contains no colors")


class UnknownPaletteError(PaletteError):
    """A built-in palette name that does not ship with the package."""

    def __init__(self, name: str) -> None:
        super().__init__(f"unknown palette {name!r}; choose from {', '.join(BUILTIN)}")


@dataclass(frozen=True, slots=True, eq=False)
class Palette:
    """An ordered, immutable set of colors.

    ``eq=False`` because the generated ``__eq__`` would compare ``rgb``
    element-wise and raise "truth value of an array is ambiguous". Identity
    comparison is what callers want; ``content_hash`` is the value comparison.
    """

    names: tuple[str, ...]
    rgb: np.ndarray
    content_hash: str

    def __post_init__(self) -> None:
        # Defect D1 was the palette being mutated through an alias during
        # conversion. Marking the array read-only turns that from a bug that
        # has to be tested for into one that cannot be written.
        self.rgb.flags.writeable = False

    def __len__(self) -> int:
        return len(self.names)


def parse(text: str, source: str = "<string>") -> Palette:
    """Parse palette text. Raise PaletteError on anything unparseable."""
    seen: dict[str, None] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        match = _HEX.match(stripped)
        if match is None:
            raise PaletteSyntaxError(source, lineno, line)
        seen.setdefault(match.group(1).lower(), None)
    if not seen:
        raise EmptyPaletteError(source)
    names = tuple(f"#{digits}" for digits in sorted(seen))
    rgb = np.array(
        [
            [int(name[1:3], 16), int(name[3:5], 16), int(name[5:7], 16)]
            for name in names
        ],
        dtype=np.int16,
    )
    canonical = "".join(f"{name}\n" for name in names)
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return Palette(names=names, rgb=rgb, content_hash=digest)


def builtin(name: str) -> Palette:
    """Load one of the palettes shipped inside the package."""
    if name not in BUILTIN:
        raise UnknownPaletteError(name)
    source = f"gruvbox-{name}.txt"
    resource = files("factory") / source
    try:
        text = resource.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise PaletteEncodingError(source) from exc
    return parse(text, source=source)
