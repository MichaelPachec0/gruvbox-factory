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

import contextlib
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import ffmpeg
import numpy as np
from PIL import Image

from factory import kernel, metadata
from factory.palette import Palette

STILL_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"})
GIF_SUFFIXES = frozenset({".gif"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})

# GIF holds 256 palette entries and one must be reserved for transparency.
_GIF_MAX_COLORS = 255


class UnsupportedFormatError(ValueError):
    """A path whose suffix matches no adapter."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"unsupported file type: {path.name}")


class PaletteTooLargeError(ValueError):
    """A palette leaving no free GIF index for transparency."""

    def __init__(self, size: int) -> None:
        super().__init__(f"palette has {size} colors; GIF allows at most 255")


class OddDimensionsError(ValueError):
    """Video dimensions the yuv420p pixel format cannot encode."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__(f"{width}x{height}: yuv420p needs even dimensions")


class UnknownFrameRateError(ValueError):
    """A stream whose frame rate cannot be determined."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"{path.name}: no usable frame rate")


class EncoderError(RuntimeError):
    """ffmpeg exited non-zero while encoding."""

    def __init__(self, dest: Path, detail: str = "") -> None:
        suffix = f": {detail}" if detail else ""
        super().__init__(f"encoder failed writing {dest.name}{suffix}")


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


class GifSource:
    """An animated or single-frame GIF.

    ``keep_metadata`` is accepted and ignored. GIF stores none of what the
    metadata policy covers: Pillow accepts icc_profile, dpi, exif and xmp on a
    GIF save and discards all four silently, so passing them through would
    look like preservation that is not happening.
    """

    def __init__(self, path: Path, palette: Palette, *, keep_metadata: bool) -> None:
        if len(palette) > _GIF_MAX_COLORS:
            raise PaletteTooLargeError(len(palette))
        self.path = path
        self.palette = palette
        self.n_frames: int | None = None
        self._image: Image.Image | None = None
        self._durations: list[int] = []
        self._disposals: list[int] = []
        self._loop: int | None = None

    def __enter__(self) -> Self:
        self._image = Image.open(self.path)
        self.n_frames = self._image.n_frames
        # A still GIF carries no loop. Defaulting it to 0 would mean "loop
        # forever" and turn a static image into a one-frame animation.
        self._loop = self._image.info.get("loop")
        for index in range(self.n_frames):
            self._image.seek(index)
            self._durations.append(self._image.info.get("duration", 100))
            # Not frame.info["disposal"], which Pillow leaves as None on every
            # frame. The real value is on the image object after seek.
            self._disposals.append(self._image.disposal_method)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._image is not None:
            self._image.close()
            self._image = None

    def frames(self) -> Iterator[np.ndarray]:
        if self._image is None:
            raise UnopenedSourceError(self.path)
        for index in range(self._image.n_frames):
            self._image.seek(index)
            # Frame 0 arrives as P and later frames as RGBA, because Pillow
            # composites. Convert every frame rather than trusting the first.
            yield np.array(self._image.convert("RGBA"))

    def _paletted(self, rgba: np.ndarray) -> Image.Image:
        """Map onto an explicit P palette, reserving one index for alpha.

        An explicit table means no adaptive quantiser runs, so the encode is a
        pure lookup and the output is byte-deterministic.
        """
        transparent = len(self.palette)
        flat = rgba[..., :3].reshape(-1, 3).astype(np.int16)
        table = kernel.nearest(flat, self.palette).reshape(rgba.shape[:2]).copy()
        table[rgba[..., 3] == 0] = transparent
        page = Image.fromarray(table, mode="P")
        entries = list(np.asarray(self.palette.rgb).astype(np.uint8).reshape(-1))
        page.putpalette(entries + [0] * (768 - len(entries)))
        return page

    def write(self, frames: Iterator[np.ndarray], dest: Path) -> None:
        # Accumulates rather than streams: Pillow's GIF encoder has no
        # streaming interface, save_all needs every page at once.
        pages = [self._paletted(frame) for frame in frames]
        single = len(pages) == 1
        options: dict[str, object] = {
            "save_all": True,
            "append_images": pages[1:],
            # Pillow's single-frame path writes the local header directly and
            # does int(duration / 10) on whatever it is given, so a one-page
            # save needs scalars where a multi-page save needs lists.
            "duration": self._durations[0] if single else self._durations,
            "disposal": self._disposals[0] if single else self._disposals,
            "transparency": len(self.palette),
            "optimize": True,
        }
        if self._loop is not None:
            options["loop"] = self._loop
        pages[0].save(dest, **options)


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int | None
    has_audio: bool


def probe(path: Path | str) -> VideoInfo:
    """Stream geometry and frame count.

    nb_frames is absent in some containers, so fall back to duration times
    frame rate and finally to None. Both consumers, the kernel selector and
    the progress bar, tolerate an estimate: a wrong count changes how smooth a
    bar looks and nothing else.

    An unusable frame rate is fatal rather than defaulted. Re-encoding at a
    guessed rate would produce a video that plays at the wrong speed, which is
    the quiet corruption this project exists to remove.
    """
    resolved = Path(path)
    data = ffmpeg.probe(str(resolved))
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    numerator, _, denominator = video.get("avg_frame_rate", "0/0").partition("/")
    fps = float(numerator) / float(denominator) if float(denominator or 0) else 0.0
    if not fps:
        raise UnknownFrameRateError(resolved)
    reported = video.get("nb_frames")
    if reported is not None and str(reported).isdigit():
        count: int | None = int(reported)
    elif video.get("duration"):
        count = round(float(video["duration"]) * fps)
    else:
        count = None
    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        n_frames=count,
        has_audio=any(s["codec_type"] == "audio" for s in data["streams"]),
    )


class VideoSource:
    """Video through an ffmpeg rawvideo pipe, with the audio remuxed back.

    Frames arrive as RGB24 and carry no alpha, so the transparency rule never
    applies. ``keep_metadata`` is accepted and ignored: none of the metadata
    policy survives a video container.

    Any failure aborts. A partially converted video is the same class of
    silent corruption as the defect this project started from, so the
    temporary file and both subprocesses are released on every exit path.
    """

    def __init__(self, path: Path, palette: Palette, *, keep_metadata: bool) -> None:
        self.path = path
        self.palette = palette
        self.info: VideoInfo | None = None
        self.n_frames: int | None = None
        self._decoder: Any = None
        self._encoder: Any = None
        self._temporary: Path | None = None

    def __enter__(self) -> Self:
        info = probe(self.path)
        if info.width % 2 or info.height % 2:
            raise OddDimensionsError(info.width, info.height)
        self.info = info
        self.n_frames = info.n_frames
        return self

    def __exit__(self, *exc: object) -> None:
        # Close the encoder's stdin before killing it. Left open, the pipe is
        # finalised during garbage collection and raises BrokenPipeError,
        # which prints an ignored-exception traceback that reads like a crash.
        if self._encoder is not None and self._encoder.stdin is not None:
            with contextlib.suppress(BrokenPipeError, OSError):
                self._encoder.stdin.close()
        for process in (self._decoder, self._encoder):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
        self._decoder = self._encoder = None
        if self._temporary is not None and self._temporary.exists():
            self._temporary.unlink()
        self._temporary = None

    def frames(self) -> Iterator[np.ndarray]:
        if self.info is None:
            raise UnopenedSourceError(self.path)
        self._decoder = (
            ffmpeg.input(str(self.path))
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run_async(pipe_stdout=True, quiet=True)
        )
        size = self.info.width * self.info.height * 3
        while True:
            raw = self._decoder.stdout.read(size)
            if len(raw) < size:
                break
            yield np.frombuffer(raw, np.uint8).reshape(
                self.info.height, self.info.width, 3
            )
        self._decoder.wait()

    def _stderr_tail(self, lines: int = 2) -> str:
        """The last few lines of the encoder's stderr, for an error message."""
        if self._encoder is None or self._encoder.stderr is None:
            return ""
        captured = self._encoder.stderr.read().decode("utf-8", "replace").strip()
        return " | ".join(captured.splitlines()[-lines:])

    def write(self, frames: Iterator[np.ndarray], dest: Path) -> None:
        if self.info is None:
            raise UnopenedSourceError(self.path)
        dest = Path(dest)
        # The temporary lives beside the destination because os.replace is
        # only atomic within one filesystem, and the system temp dir often is
        # not the same one.
        # The suffix matters: ffmpeg picks its muxer from the filename, and a
        # generic .tmp makes it exit 234 having written nothing.
        handle, name = tempfile.mkstemp(
            prefix=".gruvbox-", suffix=dest.suffix, dir=dest.parent
        )
        os.close(handle)
        self._temporary = Path(name)
        self._encoder = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{self.info.width}x{self.info.height}",
                framerate=self.info.fps,
            )
            .output(str(self._temporary), pix_fmt="yuv420p", vcodec="libx264")
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True)
        )
        try:
            for frame in frames:
                self._encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
            self._encoder.stdin.close()
        except BrokenPipeError as exc:
            # The encoder died mid-stream. Its stderr says why; without this
            # the caller would only see a pipe error from inside a generator.
            self._encoder.wait()
            raise EncoderError(dest, self._stderr_tail()) from exc
        if self._encoder.wait() != 0:
            raise EncoderError(dest, self._stderr_tail())
        if self.info.has_audio:
            ffmpeg.output(
                ffmpeg.input(str(self._temporary)),
                ffmpeg.input(str(self.path)).audio,
                str(dest),
                vcodec="copy",
                acodec="copy",
            ).overwrite_output().run(quiet=True)
            self._temporary.unlink()
        else:
            os.replace(self._temporary, dest)
        self._temporary = None


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
    if suffix in GIF_SUFFIXES:
        return GifSource(resolved, palette, keep_metadata=keep_metadata)
    if suffix in VIDEO_SUFFIXES:
        return VideoSource(resolved, palette, keep_metadata=keep_metadata)
    raise UnsupportedFormatError(resolved)
