"""kernel.py: the correctness contract, the goldens, and kernel selection."""

from __future__ import annotations

import numpy as np
import pytest
import reference
from conftest import pixel_hash

from factory import kernel
from factory.kernel import MissingTableError, UnknownKernelError
from factory.palette import BUILTIN, builtin, parse

EXAMPLE_GOLDEN = {
    "pink": "4ff69519c583a3d5",
    "white": "ef78a8c4874656ba",
    "mix": "d0fa46851e5ddfe5",
}

FIXTURE_GOLDEN = {
    "pink": "c9d5498fddf138f2",
    "white": "0283d4384da8e93d",
    "mix": "66c5f8a96586f55e",
}


def packed(rgba: np.ndarray, name: str) -> np.ndarray:
    return kernel.apply_rgba(rgba, kernel.bind(builtin(name), "packed"))


# --- pack and unpack -----------------------------------------------------


def test_pack_round_trips_through_unpack() -> None:
    colors = np.array(
        [[0, 0, 0], [255, 255, 255], [1, 2, 3], [146, 131, 116]], dtype=np.uint8
    )
    assert np.array_equal(kernel.unpack(kernel.pack(colors)), colors.astype(np.int16))


def test_pack_is_the_documented_bit_layout() -> None:
    key = kernel.pack(np.array([[0x12, 0x34, 0x56]], dtype=np.uint8))
    assert int(key[0]) == 0x123456


# --- nearest -------------------------------------------------------------


def test_exact_palette_color_maps_to_itself() -> None:
    palette = builtin("pink")
    index = kernel.nearest(np.asarray(palette.rgb), palette)
    assert np.array_equal(index, np.arange(len(palette), dtype=np.uint8))


def test_ties_resolve_to_the_smallest_hex_name() -> None:
    """#010101 is L1 distance 3 from both entries. #000000 sorts first."""
    palette = parse("#020202\n#000000\n")
    assert palette.names == ("#000000", "#020202")
    probe = np.array([[1, 1, 1]], dtype=np.int16)
    assert int(kernel.nearest(probe, palette)[0]) == 0


def test_chunk_size_does_not_change_the_result(still_rgba: np.ndarray) -> None:
    palette = builtin("mix")
    colors = kernel.unpack(np.unique(kernel.pack(still_rgba[..., :3])))
    assert np.array_equal(
        kernel.nearest(colors, palette), kernel.nearest(colors, palette, chunk=7)
    )


def test_distance_ignores_alpha() -> None:
    """Two pixels with identical RGB and different alpha map identically."""
    rgba = np.array([[[100, 120, 140, 255], [100, 120, 140, 190]]], dtype=np.uint8)
    out = packed(rgba, "pink")
    assert np.array_equal(out[0, 0, :3], out[0, 1, :3])


# --- D1 regression -------------------------------------------------------


@pytest.mark.parametrize("name", BUILTIN)
def test_palette_is_not_mutated_by_conversion(
    name: str, still_rgba: np.ndarray
) -> None:
    """Defect D1: image-go-nord appended alpha onto the shared palette rows."""
    palette = builtin(name)
    before = np.array(palette.rgb, copy=True)
    kernel.apply_rgba(still_rgba, kernel.bind(palette, "packed"))
    assert np.array_equal(np.asarray(palette.rgb), before)
    assert palette.rgb.shape == (len(palette), 3)


def test_repeated_conversion_is_bit_identical(still_rgba: np.ndarray) -> None:
    """The D1 symptom was output changing between runs. It must not."""
    assert pixel_hash(packed(still_rgba, "pink")) == pixel_hash(
        packed(still_rgba, "pink")
    )


# --- the alpha rule ------------------------------------------------------


def test_alpha_below_tolerance_passes_through_untouched(
    still_rgba: np.ndarray,
) -> None:
    out = packed(still_rgba, "pink")
    transparent = still_rgba[..., 3] < kernel.TRANSPARENCY_TOLERANCE
    assert transparent.any()
    assert np.array_equal(out[transparent], still_rgba[transparent])


def test_alpha_exactly_at_tolerance_is_converted(still_rgba: np.ndarray) -> None:
    """Row 59 is alpha 190 exactly; row 58 is 189."""
    out = packed(still_rgba, "pink")
    assert int(still_rgba[58, 0, 3]) == 189
    assert int(still_rgba[59, 0, 3]) == 190
    assert np.array_equal(out[58, 0], still_rgba[58, 0])
    assert not np.array_equal(out[59, 0, :3], still_rgba[59, 0, :3])


def test_alpha_is_preserved_everywhere(still_rgba: np.ndarray) -> None:
    out = packed(still_rgba, "pink")
    assert np.array_equal(out[..., 3], still_rgba[..., 3])


def test_output_colors_are_all_drawn_from_the_palette(
    still_rgba: np.ndarray,
) -> None:
    palette = builtin("pink")
    out = packed(still_rgba, "pink")
    opaque = still_rgba[..., 3] >= kernel.TRANSPARENCY_TOLERANCE
    produced = set(map(tuple, out[opaque][:, :3].tolist()))
    allowed = set(map(tuple, np.asarray(palette.rgb).tolist()))
    assert produced <= allowed


def test_mask_first_matches_convert_everything(still_rgba: np.ndarray) -> None:
    """apply_rgba skips transparent pixels; doing them and restoring agrees."""
    palette = builtin("pink")
    convert_all = still_rgba.copy()
    convert_all[..., :3] = kernel.map_rgb_packed(still_rgba[..., :3], palette)
    transparent = still_rgba[..., 3] < kernel.TRANSPARENCY_TOLERANCE
    convert_all[transparent] = still_rgba[transparent]
    assert pixel_hash(convert_all) == pixel_hash(packed(still_rgba, "pink"))


def test_input_array_is_not_modified(still_rgba: np.ndarray) -> None:
    before = still_rgba.copy()
    packed(still_rgba, "pink")
    assert np.array_equal(still_rgba, before)


# --- goldens -------------------------------------------------------------


@pytest.mark.parametrize("name", BUILTIN)
def test_fixture_golden_matches_reference_and_constant(
    name: str, still_rgba: np.ndarray
) -> None:
    expected = reference.convert(still_rgba, np.asarray(builtin(name).rgb))
    assert (
        pixel_hash(packed(still_rgba, name))
        == pixel_hash(expected)
        == FIXTURE_GOLDEN[name]
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", BUILTIN)
def test_example_golden_matches_reference_and_constant(
    name: str, example_rgba: np.ndarray
) -> None:
    """The faithful-port evidence. pink must be 4ff69519c583a3d5."""
    expected = reference.convert(example_rgba, np.asarray(builtin(name).rgb))
    assert (
        pixel_hash(packed(example_rgba, name))
        == pixel_hash(expected)
        == EXAMPLE_GOLDEN[name]
    )


# --- bind ----------------------------------------------------------------


def test_bind_packed_matches_a_direct_call(still_rgba: np.ndarray) -> None:
    palette = builtin("pink")
    rgb = still_rgba[..., :3]
    assert np.array_equal(
        kernel.bind(palette, "packed")(rgb), kernel.map_rgb_packed(rgb, palette)
    )


def test_bind_lut_without_a_table_is_rejected() -> None:
    with pytest.raises(MissingTableError, match="needs a table"):
        kernel.bind(builtin("pink"), "lut")


def test_bind_rejects_auto() -> None:
    """auto is a policy, not a kernel. select_kernel resolves it first."""
    with pytest.raises(UnknownKernelError, match="lut, packed"):
        kernel.bind(builtin("pink"), "auto")


def test_bind_rejects_an_unknown_name() -> None:
    with pytest.raises(UnknownKernelError):
        kernel.bind(builtin("pink"), "quantize")


# --- select_kernel -------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "frames", "cache_hit", "expected"),
    [
        ("packed", 1, False, "packed"),
        ("packed", 10_000, True, "packed"),
        ("lut", 1, False, "lut"),
        ("lut", 1, True, "lut"),
        ("auto", 1, True, "lut"),
        ("auto", 1, False, "packed"),
        ("auto", 33, False, "packed"),
        ("auto", 34, False, "lut"),
        ("auto", 35, False, "lut"),
        ("auto", 0, False, "packed"),
    ],
)
def test_select_kernel_truth_table(
    mode: str, frames: int, cache_hit: bool, expected: str
) -> None:
    assert kernel.select_kernel(mode, frames, cache_hit) == expected


def test_select_kernel_rejects_an_unknown_mode() -> None:
    with pytest.raises(UnknownKernelError, match="unknown kernel 'fast'"):
        kernel.select_kernel("fast", 1, False)


def test_select_kernel_always_returns_something_bindable() -> None:
    for mode in kernel.KERNELS:
        assert kernel.select_kernel(mode, 1, False) in kernel.RESOLVED_KERNELS


def test_break_even_constant_matches_its_derivation() -> None:
    """13.542s build / (0.452s - 0.055s) per frame, rounded down."""
    assert kernel.LUT_BREAK_EVEN_FRAMES == int(13.542 / (0.452 - 0.055))
