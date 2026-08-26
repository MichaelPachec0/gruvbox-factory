"""sources.GifSource: timing, disposal, transparency and determinism."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import pixel_hash, read_gif
from PIL import Image

from factory import kernel
from factory.palette import builtin, parse
from factory.sources import GifSource, PaletteTooLargeError, open_source

GOLDEN = {
    "pink": "6c92e7a20eec69b5",
    "white": "8735aad4595f84da",
    "mix": "56255634ab06c536",
}


def convert(src: Path, dest: Path, name: str = "pink") -> None:
    palette = builtin(name)
    bound = kernel.bind(palette, "packed")
    with open_source(src, palette) as source:
        source.write((kernel.apply_rgba(f, bound) for f in source.frames()), dest)


# --- protocol ------------------------------------------------------------


def test_gif_source_reports_its_frame_count(animated_gif_path: Path) -> None:
    with open_source(animated_gif_path, builtin("pink")) as source:
        assert isinstance(source, GifSource)
        assert source.n_frames == 4


def test_every_frame_arrives_as_rgba(animated_gif_path: Path) -> None:
    """Pillow hands out frame 0 as P and the rest as RGBA, having composited."""
    with open_source(animated_gif_path, builtin("pink")) as source:
        assert [f.shape for f in source.frames()] == [(48, 64, 4)] * 4


def test_a_palette_with_no_free_index_is_rejected() -> None:
    """GIF has 256 entries and one must be reserved for transparency."""
    oversized = parse("\n".join(f"#{value:06x}" for value in range(256)))
    assert len(oversized) == 256
    with pytest.raises(PaletteTooLargeError, match="256"):
        GifSource(Path("x.gif"), oversized, keep_metadata=False)


# --- goldens -------------------------------------------------------------


@pytest.mark.parametrize("name", ["pink", "white", "mix"])
def test_golden_round_trip(animated_gif_path: Path, tmp_path: Path, name: str) -> None:
    dest = tmp_path / f"{name}.gif"
    convert(animated_gif_path, dest, name)
    frames, _, _, _ = read_gif(dest)
    assert pixel_hash(np.concatenate(frames)) == GOLDEN[name]


# --- preservation --------------------------------------------------------


def test_timing_and_disposal_survive(animated_gif_path: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out.gif"
    convert(animated_gif_path, dest)
    _, durations, disposals, loop = read_gif(dest)
    assert durations == [40, 80, 120, 160]
    assert disposals == [0, 2, 1, 2]
    assert loop == 0


def test_total_duration_is_preserved(animated_gif_path: Path, tmp_path: Path) -> None:
    """Assert total, not count: Pillow merges identical consecutive frames."""
    dest = tmp_path / "out.gif"
    convert(animated_gif_path, dest)
    _, before, _, _ = read_gif(animated_gif_path)
    _, after, _, _ = read_gif(dest)
    assert sum(after) == sum(before)


def test_transparency_is_preserved(animated_gif_path: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out.gif"
    convert(animated_gif_path, dest)
    frames, _, _, _ = read_gif(dest)
    assert [int((f[..., 3] == 0).sum()) for f in frames] == [144, 12, 144, 12]


def test_opaque_output_uses_only_palette_colors(
    animated_gif_path: Path, tmp_path: Path
) -> None:
    """This is what would break if optimize=True remapped the palette."""
    dest = tmp_path / "out.gif"
    convert(animated_gif_path, dest)
    frames, _, _, _ = read_gif(dest)
    allowed = set(map(tuple, np.asarray(builtin("pink").rgb).tolist()))
    opaque = np.concatenate([f[f[..., 3] != 0][:, :3] for f in frames])
    assert set(map(tuple, np.unique(opaque, axis=0).tolist())) <= allowed


def test_encoding_is_deterministic(animated_gif_path: Path, tmp_path: Path) -> None:
    """No adaptive quantiser runs, so the same input gives the same bytes."""
    first, second = tmp_path / "a.gif", tmp_path / "b.gif"
    convert(animated_gif_path, first)
    convert(animated_gif_path, second)
    assert first.read_bytes() == second.read_bytes()


# --- single frame --------------------------------------------------------


def test_a_single_frame_gif_uses_the_same_path(tmp_path: Path) -> None:
    src = tmp_path / "one.gif"
    Image.new("P", (8, 8)).save(src)
    dest = tmp_path / "one_out.gif"
    convert(src, dest)
    frames, _, _, _ = read_gif(dest)
    assert len(frames) == 1


def test_a_still_gif_does_not_gain_a_loop(tmp_path: Path) -> None:
    """A source with no loop must not come out as a 1-frame infinite loop."""
    src = tmp_path / "one.gif"
    Image.new("P", (8, 8)).save(src)
    assert "loop" not in Image.open(src).info
    dest = tmp_path / "one_out.gif"
    convert(src, dest)
    assert "loop" not in Image.open(dest).info
