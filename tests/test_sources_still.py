"""sources.StillSource: mode promotion, orientation, metadata, round trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import EXAMPLE_PNG, pixel_hash
from PIL import Image

from factory import kernel, sources
from factory.palette import builtin
from factory.sources import UnopenedSourceError, UnsupportedFormatError, open_source

PINK_FIXTURE_GOLDEN = "c9d5498fddf138f2"


def convert(path: Path, dest: Path, name: str = "pink", *, keep: bool = False) -> None:
    palette = builtin(name)
    bound = kernel.bind(palette, "packed")
    with open_source(path, palette, keep_metadata=keep) as source:
        frames = (
            kernel.apply_rgba(f, bound) if f.shape[-1] == 4 else bound(f)
            for f in source.frames()
        )
        source.write(frames, dest)


# --- protocol ------------------------------------------------------------


def test_still_source_reports_one_frame(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    Image.fromarray(np.zeros((4, 4, 3), np.uint8), "RGB").save(path)
    with open_source(path, builtin("pink")) as source:
        assert source.n_frames == 1
        assert len(list(source.frames())) == 1


def test_frames_outside_the_context_manager_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(path)
    source = open_source(path, builtin("pink"))
    with pytest.raises(UnopenedSourceError, match="context manager"):
        list(source.frames())


def test_unsupported_suffix_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("not an image")
    with pytest.raises(UnsupportedFormatError, match="a.txt"):
        open_source(path, builtin("pink"))


def test_suffix_tables_do_not_overlap() -> None:
    tables = (sources.STILL_SUFFIXES, sources.GIF_SUFFIXES, sources.VIDEO_SUFFIXES)
    assert len(set().union(*tables)) == sum(len(t) for t in tables)


def test_a_missing_file_fails_on_enter_not_on_dispatch(tmp_path: Path) -> None:
    """open_source only picks an adapter; the opener reports a missing file."""
    source = open_source(tmp_path / "nope.png", builtin("pink"))
    with pytest.raises(FileNotFoundError), source:
        pass


# --- mode promotion ------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "channels"),
    [("RGB", 3), ("RGBA", 4), ("L", 3), ("LA", 4)],
)
def test_mode_promotion_preserves_whether_alpha_existed(
    tmp_path: Path, mode: str, channels: int
) -> None:
    """Today these raise TypeError: 'int' object is not iterable."""
    path = tmp_path / f"{mode}.png"
    Image.new(mode, (6, 5)).save(path)
    with open_source(path, builtin("pink")) as source:
        assert next(iter(source.frames())).shape == (5, 6, channels)


def test_opaque_palette_image_becomes_rgb(tmp_path: Path) -> None:
    path = tmp_path / "p.png"
    Image.new("P", (4, 4)).save(path)
    with open_source(path, builtin("pink")) as source:
        assert next(iter(source.frames())).shape[-1] == 3


def test_transparent_palette_image_becomes_rgba(tmp_path: Path) -> None:
    path = tmp_path / "pt.png"
    Image.new("P", (4, 4)).save(path, transparency=0)
    with open_source(path, builtin("pink")) as source:
        assert next(iter(source.frames())).shape[-1] == 4


# --- orientation ---------------------------------------------------------


def test_orientation_is_applied_to_pixels(tmp_path: Path) -> None:
    source_image = Image.new("RGB", (3, 2))
    exif = source_image.getexif()
    exif[0x0112] = 6
    path = tmp_path / "rot.jpg"
    source_image.save(path, exif=exif.tobytes())
    with open_source(path, builtin("pink")) as opened:
        assert next(iter(opened.frames())).shape[:2] == (3, 2)


# --- round trip ----------------------------------------------------------


def test_round_trip_matches_the_part_one_golden(
    still_rgba: np.ndarray, tmp_path: Path
) -> None:
    """Same fixture and palette the kernel goldens use."""
    src = tmp_path / "in.png"
    Image.fromarray(still_rgba, "RGBA").save(src)
    dest = tmp_path / "out.png"
    convert(src, dest)
    assert pixel_hash(np.array(Image.open(dest))) == PINK_FIXTURE_GOLDEN


def test_an_opaque_source_stays_opaque(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    Image.new("RGB", (8, 6), (200, 100, 50)).save(src)
    dest = tmp_path / "out.png"
    convert(src, dest)
    assert Image.open(dest).mode == "RGB"


# --- metadata ------------------------------------------------------------


def test_default_output_drops_provenance(tmp_path: Path) -> None:
    dest = tmp_path / "out.png"
    convert(EXAMPLE_PNG, dest)
    back = Image.open(dest)
    assert dict(back.getexif()) == {}
    assert "XML:com.adobe.xmp" not in back.info
    assert "Raw profile type iptc" not in back.info
    assert back.info.get("icc_profile") is not None


def test_keep_metadata_carries_provenance_and_stamp(tmp_path: Path) -> None:
    from factory import metadata as md

    dest = tmp_path / "out.png"
    convert(EXAMPLE_PNG, dest, keep=True)
    back = Image.open(dest)
    assert back.getexif()[md.STAMP_TAG] == md.STAMP
    assert back.info["Raw profile type iptc"] is not None
    assert back.info["XML:com.adobe.xmp"].startswith("<?xpacket")
