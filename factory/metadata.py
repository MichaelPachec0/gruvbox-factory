"""What an image carries alongside its pixels, and what the output may keep.

The default is privacy-first: ICC and DPI travel, everything else is dropped.
EXIF is the obvious carrier of GPS coordinates, camera serials and original
timestamps, but IPTC and XMP matter too. Modern tools mirror EXIF into the XMP
packet, so a file can hold exif:GPSLatitude inside XMP long after the EXIF
block is gone; dropping EXIF while keeping XMP would leak exactly what
dropping EXIF was for.

ICC and DPI stay because neither can carry personal data. ICC is a colour
transform and DPI is a print-size hint, so losing them changes how the image
renders for no privacy gain.

This module also owns orientation. EXIF Orientation instructs a viewer to
rotate and decoders do not apply it, so dropping EXIF from a portrait phone
photo would leave upright-looking pixels with the rotation instruction
removed, and it would display sideways. ``upright`` bakes the rotation in
before anything else runs, which is the only arrangement where stripping
metadata cannot change how an image looks.

Formats disagree about transport and Pillow hides that badly. Two traps, both
of which look like success:

- PNG has no EXIF or IPTC chunk in the files seen here. Both arrive as
  ImageMagick ``Raw profile type ...`` text chunks holding hex payloads, and
  ``save(xmp=...)`` is accepted and silently discarded for PNG. XMP only
  survives as an ``XML:com.adobe.xmp`` text chunk.
- IPTC is PNG-only. Pillow reads a JPEG's APP13 block into ``info["photoshop"]``
  but has no way to write one back: ``save(photoshop=...)`` is accepted and
  dropped. A JPEG's IPTC therefore cannot survive a round trip, with or
  without ``keep``.

``save_kwargs`` takes the target format for that reason, and returns only what
that format actually honours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageOps, PngImagePlugin

# EXIF ProcessingSoftware. Constant, with no version, so output bytes stay
# stable across releases.
STAMP_TAG = 0x000B
STAMP = "gruvbox-factory"

_ORIENTATION = 0x0112
_RAW_EXIF = "Raw profile type exif"
_RAW_IPTC = "Raw profile type iptc"
_PNG_XMP = "XML:com.adobe.xmp"


def _unwrap(raw: str) -> bytes:
    """Decode an ImageMagick raw profile: newline, name, length, hex payload."""
    return bytes.fromhex("".join(raw.split("\n", 3)[3].split()))


@dataclass(frozen=True, slots=True)
class Metadata:
    """Everything worth carrying from a source image."""

    icc_profile: bytes | None = None
    dpi: tuple[float, float] | None = None
    exif: bytes | None = None
    iptc: str | None = None
    xmp: bytes | None = None


def upright(image: Image.Image) -> Image.Image:
    """Apply EXIF Orientation to the pixels and drop the tag."""
    return ImageOps.exif_transpose(image)


def capture(image: Image.Image) -> Metadata:
    """Read every supported block out of an opened image."""
    info = image.info
    exif = info.get("exif")
    if exif is None and _RAW_EXIF in info:
        exif = _unwrap(info[_RAW_EXIF])
    return Metadata(
        icc_profile=info.get("icc_profile"),
        dpi=info.get("dpi"),
        exif=exif,
        iptc=info.get(_RAW_IPTC),
        xmp=info.get("xmp"),
    )


def save_kwargs(metadata: Metadata, fmt: str, *, keep: bool) -> dict[str, Any]:
    """Save keywords for ``fmt``, honouring the privacy default.

    With ``keep`` false the result holds only ICC and DPI, and no stamp: the
    output carries nothing inherited and nothing added.

    With ``keep`` true the source EXIF is merged with the stamp and re-emitted
    as a single standard block. The legacy raw-profile chunk is deliberately
    not also copied, because a PNG holding both would report different EXIF
    depending on which chunk the reader looks at.
    """
    out: dict[str, Any] = {}
    if metadata.icc_profile:
        out["icc_profile"] = metadata.icc_profile
    if metadata.dpi:
        out["dpi"] = metadata.dpi
    if not keep:
        return out

    exif = Image.Exif()
    if metadata.exif:
        exif.load(metadata.exif)
    # upright already rotated the pixels. Carrying the tag forward would make
    # any viewer that honours it rotate them a second time.
    exif.pop(_ORIENTATION, None)
    exif[STAMP_TAG] = STAMP
    out["exif"] = exif.tobytes()

    if fmt == "PNG":
        png = PngImagePlugin.PngInfo()
        if metadata.iptc:
            png.add_text(_RAW_IPTC, metadata.iptc)
        if metadata.xmp:
            png.add_text(_PNG_XMP, metadata.xmp.decode("utf-8", "replace"))
        out["pnginfo"] = png
    elif metadata.xmp:
        out["xmp"] = metadata.xmp
    return out
