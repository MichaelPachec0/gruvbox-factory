"""metadata.py: capture, privacy defaults, per-format transport, orientation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import EXAMPLE_PNG
from PIL import Image

from factory import metadata as md

PIXELS = np.zeros((4, 4, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def example_metadata() -> md.Metadata:
    return md.capture(Image.open(EXAMPLE_PNG))


def save(meta: md.Metadata, path: Path, *, keep: bool) -> Image.Image:
    image = Image.fromarray(PIXELS, "RGB")
    image.save(path, **md.save_kwargs(meta, "PNG", keep=keep))
    return Image.open(path)


# --- capture -------------------------------------------------------------


def test_capture_finds_every_block_in_example_png(
    example_metadata: md.Metadata,
) -> None:
    assert len(example_metadata.icc_profile) == 672
    assert example_metadata.dpi == (299.9994, 299.9994)
    assert len(example_metadata.exif) == 2704
    assert len(example_metadata.iptc) == 109
    assert len(example_metadata.xmp) == 3957


def test_capture_reads_exif_from_the_legacy_raw_profile(
    example_metadata: md.Metadata,
) -> None:
    """PNG has no eXIf chunk here; the payload is an ImageMagick text chunk."""
    assert example_metadata.exif.startswith(b"Exif\x00\x00")


def test_capture_of_a_bare_image_is_all_none() -> None:
    assert md.capture(Image.fromarray(PIXELS, "RGB")) == md.Metadata()


# --- the privacy default -------------------------------------------------


def test_default_keeps_only_icc_and_dpi(example_metadata: md.Metadata) -> None:
    assert sorted(md.save_kwargs(example_metadata, "PNG", keep=False)) == [
        "dpi",
        "icc_profile",
    ]


def test_default_output_carries_no_provenance(
    example_metadata: md.Metadata, tmp_path: Path
) -> None:
    back = save(example_metadata, tmp_path / "d.png", keep=False)
    assert dict(back.getexif()) == {}
    assert "Raw profile type iptc" not in back.info
    assert "XML:com.adobe.xmp" not in back.info
    assert back.info["dpi"] == (299.9994, 299.9994)
    assert back.info["icc_profile"] == example_metadata.icc_profile


def test_default_output_is_not_stamped(
    example_metadata: md.Metadata, tmp_path: Path
) -> None:
    """A privacy-first default adds nothing as well as inheriting nothing."""
    back = save(example_metadata, tmp_path / "d.png", keep=False)
    assert back.getexif().get(md.STAMP_TAG) is None


# --- keep ----------------------------------------------------------------


def test_keep_merges_source_exif_with_the_stamp(
    example_metadata: md.Metadata, tmp_path: Path
) -> None:
    exif = save(example_metadata, tmp_path / "k.png", keep=True).getexif()
    assert exif[md.STAMP_TAG] == md.STAMP
    assert exif[0x0131] == "GIMP 2.10.22"
    assert exif[0x0132] == "2021:03:06 17:15:35"


def test_keep_emits_one_exif_chunk_not_two(
    example_metadata: md.Metadata, tmp_path: Path
) -> None:
    """The legacy raw-profile chunk is dropped in favour of standard eXIf.

    A PNG holding both would report different EXIF depending on which chunk
    the reader looks at.
    """
    back = save(example_metadata, tmp_path / "k.png", keep=True)
    assert "Raw profile type exif" not in back.info
    assert "exif" in back.info


def test_keep_drops_the_source_orientation(tmp_path: Path) -> None:
    """upright already rotated the pixels; carrying the tag would re-rotate."""
    source = md.Metadata(exif=_exif_with_orientation())
    back = save(source, tmp_path / "o.png", keep=True)
    assert back.getexif().get(0x0112) is None


def test_keep_preserves_iptc_byte_exactly(
    example_metadata: md.Metadata, tmp_path: Path
) -> None:
    back = save(example_metadata, tmp_path / "k.png", keep=True)
    assert back.info["Raw profile type iptc"] == example_metadata.iptc


def test_keep_preserves_xmp_as_a_png_text_chunk(
    example_metadata: md.Metadata, tmp_path: Path
) -> None:
    """save(xmp=...) is accepted and silently discarded for PNG."""
    back = save(example_metadata, tmp_path / "k.png", keep=True)
    assert back.info["XML:com.adobe.xmp"].startswith("<?xpacket")


def test_non_png_formats_use_the_xmp_kwarg(example_metadata: md.Metadata) -> None:
    kwargs = md.save_kwargs(example_metadata, "JPEG", keep=True)
    assert kwargs["xmp"] == example_metadata.xmp
    assert "pnginfo" not in kwargs


def test_stamp_is_written_even_with_no_source_exif(tmp_path: Path) -> None:
    assert save(md.Metadata(), tmp_path / "s.png", keep=True).getexif()[
        md.STAMP_TAG
    ] == md.STAMP


def test_stamp_carries_no_version() -> None:
    """Output bytes must stay stable across releases."""
    assert md.STAMP == "gruvbox-factory"


def test_absent_blocks_are_simply_omitted() -> None:
    assert md.save_kwargs(md.Metadata(), "PNG", keep=False) == {}


# --- orientation ---------------------------------------------------------


def _exif_with_orientation(value: int = 6) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = value
    return exif.tobytes()


def test_upright_bakes_rotation_into_pixels(tmp_path: Path) -> None:
    source = Image.fromarray(np.arange(6, dtype=np.uint8).reshape(2, 3) * 40, "L")
    path = tmp_path / "rot.jpg"
    source.convert("RGB").save(path, exif=_exif_with_orientation())

    loaded = Image.open(path)
    assert loaded.size == (3, 2)
    assert loaded.getexif()[0x0112] == 6

    fixed = md.upright(loaded)
    assert fixed.size == (2, 3)
    assert fixed.getexif().get(0x0112) is None


def test_upright_leaves_an_unrotated_image_alone() -> None:
    source = Image.fromarray(PIXELS, "RGB")
    assert md.upright(source).size == source.size
