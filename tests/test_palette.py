"""palette.py: canonicalisation, immutability and error reporting."""

from __future__ import annotations

import numpy as np
import pytest

from factory import palette as palette_module
from factory.palette import (
    BUILTIN,
    EmptyPaletteError,
    PaletteError,
    PaletteSyntaxError,
    UnknownPaletteError,
    builtin,
    parse,
)

PINK_HASH = "4442dfd45f29546bc39dfa38eca4056a027326c6e463077ecac3974c37d9a629"
WHITE_HASH = "3016db39a333504f6d7be5aaafa112b00cae939a04cff04212e8e05a8de950e6"
MIX_HASH = "ed225e3d09de888615f8ba880020504173959cb11a5f892d7822a46237e9e168"

EXPECTED = {
    "pink": (27, PINK_HASH),
    "white": (36, WHITE_HASH),
    "mix": (131, MIX_HASH),
}


def test_builtin_names_are_exactly_the_shipped_files() -> None:
    assert BUILTIN == ("pink", "white", "mix")


@pytest.mark.parametrize("name", BUILTIN)
def test_builtin_loads_with_expected_size_and_hash(name: str) -> None:
    size, digest = EXPECTED[name]
    loaded = builtin(name)
    assert len(loaded) == size
    assert loaded.content_hash == digest


@pytest.mark.parametrize("name", BUILTIN)
def test_names_are_sorted_unique_and_lowercase(name: str) -> None:
    loaded = builtin(name)
    assert list(loaded.names) == sorted(loaded.names)
    assert len(set(loaded.names)) == len(loaded.names)
    assert all(n == n.lower() and n.startswith("#") for n in loaded.names)
    assert all(len(n) == 7 for n in loaded.names)


def test_pink_deduplicates_its_repeated_entry() -> None:
    """gruvbox-pink.txt lists #928374 on both line 6 and line 7."""
    pink = builtin("pink")
    assert len(pink) == 27
    assert pink.names.count("#928374") == 1


def test_last_color_survives_a_missing_trailing_newline() -> None:
    """gruvbox-pink.txt and gruvbox-white.txt both end without a newline."""
    assert parse("#aabbcc\n#001122").names == ("#001122", "#aabbcc")
    assert len(parse("#aabbcc\n#001122")) == 2


def test_shipped_files_without_a_trailing_newline_are_complete() -> None:
    """A loader using split("\\n") would silently drop one of these."""
    assert len(builtin("pink")) == 27
    assert len(builtin("white")) == 36


@pytest.mark.parametrize("name", BUILTIN)
def test_rgb_array_matches_names(name: str) -> None:
    loaded = builtin(name)
    assert loaded.rgb.shape == (len(loaded), 3)
    assert loaded.rgb.dtype == np.int16
    for index, hex_name in enumerate(loaded.names):
        expected = [int(hex_name[i : i + 2], 16) for i in (1, 3, 5)]
        assert list(loaded.rgb[index]) == expected


@pytest.mark.parametrize("name", BUILTIN)
def test_rgb_array_is_read_only(name: str) -> None:
    """D1 made structurally impossible rather than merely tested for."""
    loaded = builtin(name)
    assert not loaded.rgb.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        loaded.rgb[0, 0] = 0


def test_palette_fields_cannot_be_rebound() -> None:
    loaded = builtin("pink")
    with pytest.raises(AttributeError):
        loaded.names = ()


def test_content_hash_ignores_order_case_and_blank_lines() -> None:
    a = parse("#AABBCC\n#001122\n")
    b = parse("\n  #001122  \n\n#aabbcc")
    assert a.content_hash == b.content_hash
    assert a.names == ("#001122", "#aabbcc")


def test_content_hash_ignores_a_duplicated_entry() -> None:
    assert parse("#aabbcc\n").content_hash == parse("#aabbcc\n#aabbcc\n").content_hash


def test_content_hash_changes_when_a_color_changes() -> None:
    a = parse("#aabbcc\n#001122\n")
    b = parse("#aabbcd\n#001122\n")
    assert a.content_hash != b.content_hash


def test_bare_hex_without_hash_is_accepted() -> None:
    assert parse("aabbcc").names == ("#aabbcc",)


@pytest.mark.parametrize(
    "text",
    ["#abc\n", "#gggggg\n", "#aabbccdd\n", "not a color\n", "#aabbcc extra\n"],
)
def test_malformed_line_raises_with_its_line_number(text: str) -> None:
    with pytest.raises(PaletteSyntaxError, match=r"palette\.txt:1: expected #rrggbb"):
        parse(text, source="palette.txt")


def test_line_number_points_at_the_offending_line() -> None:
    with pytest.raises(PaletteSyntaxError, match=r":3: expected"):
        parse("#aabbcc\n#001122\nnope\n", source="p")


def test_empty_palette_is_rejected() -> None:
    with pytest.raises(EmptyPaletteError, match="no colors"):
        parse("\n\n   \n", source="p")


def test_unknown_builtin_names_the_alternatives() -> None:
    with pytest.raises(UnknownPaletteError, match="unknown palette 'nord'"):
        builtin("nord")
    with pytest.raises(UnknownPaletteError, match="pink, white, mix"):
        builtin("nord")


@pytest.mark.parametrize(
    "error", [PaletteSyntaxError, EmptyPaletteError, UnknownPaletteError]
)
def test_every_palette_error_is_a_value_error(error: type) -> None:
    """Callers may catch PaletteError, or stay generic with ValueError."""
    assert issubclass(error, PaletteError)
    assert issubclass(error, ValueError)


def test_module_exposes_no_mutable_globals() -> None:
    """A module-level list would be the same class of bug as D1."""
    assert isinstance(palette_module.BUILTIN, tuple)
