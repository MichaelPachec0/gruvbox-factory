"""sources.VideoSource: probing, round trip, audio remux and abort cleanup."""

from __future__ import annotations

import threading
from pathlib import Path

import ffmpeg
import numpy as np
import pytest

from factory import kernel
from factory.palette import builtin
from factory.pipeline import map_frames
from factory.sources import (
    OddDimensionsError,
    UnknownFrameRateError,
    VideoSource,
    open_source,
    probe,
)

SENTINEL = "badframe"


def run(src: Path, dest: Path, work=None, workers: int = 2) -> None:
    palette = builtin("pink")
    bound = work or kernel.bind(palette, "packed")
    with open_source(src, palette) as source:
        source.write(map_frames(source.frames(), bound, workers=workers), dest)


def encode(dest: Path, size: str = "64x48", *, audio: bool = False) -> Path:
    video = ffmpeg.input(f"testsrc=size={size}:rate=12:duration=1", f="lavfi")
    streams = [video]
    options: dict[str, object] = {"vcodec": "libx264", "pix_fmt": "yuv420p"}
    if audio:
        streams.append(ffmpeg.input("sine=frequency=440:duration=1", f="lavfi"))
        options |= {"acodec": "aac", "shortest": None}
    ffmpeg.output(*streams, str(dest), **options).overwrite_output().run(quiet=True)
    return dest


# --- probe ---------------------------------------------------------------


def test_probe_reads_the_stream(sample_video: Path) -> None:
    info = probe(sample_video)
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(12.0)
    assert info.n_frames == 12
    assert info.has_audio is True


def test_probe_reports_no_audio_when_there_is_none(tmp_path: Path) -> None:
    assert probe(encode(tmp_path / "silent.mp4")).has_audio is False


# --- guards --------------------------------------------------------------


def test_odd_dimensions_are_rejected_with_a_clear_error(tmp_path: Path) -> None:
    """yuv420p needs even dimensions; ffmpeg's own message blames bit_rate."""
    src = tmp_path / "odd.mp4"
    ffmpeg.output(
        ffmpeg.input("testsrc=size=65x49:rate=12:duration=1", f="lavfi"),
        str(src),
        vcodec="libx264",
        pix_fmt="yuv444p",
    ).overwrite_output().run(quiet=True)
    with pytest.raises(OddDimensionsError, match="65x49"), open_source(
        src, builtin("pink")
    ):
        pass


def test_errors_are_value_errors() -> None:
    assert issubclass(OddDimensionsError, ValueError)
    assert issubclass(UnknownFrameRateError, ValueError)


# --- decode --------------------------------------------------------------


def test_frames_decode_as_rgb_without_alpha(sample_video: Path) -> None:
    with open_source(sample_video, builtin("pink")) as source:
        assert isinstance(source, VideoSource)
        assert source.n_frames == 12
        assert [f.shape for f in source.frames()] == [(48, 64, 3)] * 12


def test_converted_frames_use_only_palette_colors(sample_video: Path) -> None:
    """Asserted on pipeline output, not the file: H.264 yuv420p is lossy."""
    palette = builtin("pink")
    allowed = set(map(tuple, np.asarray(palette.rgb).tolist()))
    with open_source(sample_video, palette) as source:
        for frame in map_frames(
            source.frames(), kernel.bind(palette, "packed"), workers=2
        ):
            found = set(map(tuple, np.unique(frame.reshape(-1, 3), axis=0).tolist()))
            assert found <= allowed


# --- round trip ----------------------------------------------------------


def test_round_trip_preserves_container_properties(
    sample_video: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "out.mp4"
    run(sample_video, dest)
    after = probe(dest)
    assert (after.width, after.height) == (64, 48)
    assert after.n_frames == 12
    assert after.fps == pytest.approx(12.0)


def test_audio_stream_survives(sample_video: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out.mp4"
    run(sample_video, dest)
    assert probe(dest).has_audio is True


def test_a_video_without_audio_round_trips(tmp_path: Path) -> None:
    src = encode(tmp_path / "silent.mp4")
    dest = tmp_path / "out.mp4"
    run(src, dest)
    assert probe(dest).n_frames == 12
    assert probe(dest).has_audio is False


def test_no_temporary_file_is_left_behind(sample_video: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out.mp4"
    run(sample_video, dest)
    assert [p.name for p in tmp_path.iterdir()] == ["out.mp4"]


# --- abort ---------------------------------------------------------------


def test_a_failing_frame_leaves_no_output_and_no_temp(
    sample_video: Path, tmp_path: Path
) -> None:
    """Decision 13: abort the whole video rather than write a partial one."""
    palette = builtin("pink")
    bound = kernel.bind(palette, "packed")
    dest = tmp_path / "aborted.mp4"
    seen: list[int] = []
    # Increment and compare under a lock. Without it two workers can append
    # back to back and both read a count past the trigger, so the failure
    # never fires -- which is exactly what the free-threaded lane caught.
    lock = threading.Lock()

    def boom(frame: np.ndarray) -> np.ndarray:
        with lock:
            seen.append(1)
            tripped = len(seen) == 5
        if tripped:
            raise RuntimeError(SENTINEL)
        return bound(frame)

    with pytest.raises(RuntimeError, match=SENTINEL):
        run(sample_video, dest, work=boom)

    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_video_ignores_the_metadata_policy(sample_video: Path, tmp_path: Path) -> None:
    palette = builtin("pink")
    dest = tmp_path / "out.mp4"
    with open_source(sample_video, palette, keep_metadata=True) as source:
        source.write(
            map_frames(source.frames(), kernel.bind(palette, "packed"), workers=2),
            dest,
        )
    assert dest.exists()
