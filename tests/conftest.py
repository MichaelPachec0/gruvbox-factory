"""Shared fixtures.

Every test in this suite is hermetic: no network, and no writes anywhere
except pytest's own tmp_path. The isolated_cache fixture is autouse and not
optional -- lut.py reads XDG_CACHE_HOME, and a test that forgot to override it
would silently read and evict the developer's real cache.

tests/ has no __init__.py on purpose. Pytest's prepend import mode therefore
puts tests/ itself on sys.path, so these modules are imported as top-level
names (``from conftest import ...``, ``import reference``) and ``factory``
resolves to the installed virtualenv rather than to the source tree beside it.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import ffmpeg
import numpy as np
import pytest
from PIL import Image

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
REPO_ROOT = TESTS_DIR.parent
EXAMPLE_PNG = REPO_ROOT / "example.png"
ANIMATED_GIF = FIXTURES_DIR / "animated.gif"


def pixel_hash(array: np.ndarray) -> str:
    """The golden hash: sha256 over decoded pixels, first 16 hex digits.

    Never hash file bytes. Those depend on the zlib bundled in the Pillow
    wheel and on PNG chunk ordering; pixel values are integer-exact and stable
    across Pillow releases.
    """
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


@pytest.fixture(scope="session")
def still_rgba() -> np.ndarray:
    return np.array(Image.open(FIXTURES_DIR / "still_rgba.png"))


@pytest.fixture(scope="session")
def example_rgba() -> np.ndarray:
    return np.array(Image.open(EXAMPLE_PNG))


@pytest.fixture(scope="session")
def fixture_generator() -> ModuleType:
    """Load make_fixtures.py by path, so tests/fixtures need not be a package."""
    path = FIXTURES_DIR / "make_fixtures.py"
    spec = importlib.util.spec_from_file_location("make_fixtures", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point XDG_CACHE_HOME at tmp_path for every test, without exception."""
    cache_home = tmp_path / "xdg-cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return cache_home


def read_gif(path: Path) -> tuple[list[np.ndarray], list[int], list[int], int]:
    """Frames, durations, disposals and loop from a GIF.

    Disposal comes from image.disposal_method after seek, never from
    frame.info["disposal"], which Pillow leaves as None on every frame.
    Frames are converted explicitly because Pillow hands out frame 0 as P and
    later frames as RGBA, having composited them.
    """
    image = Image.open(path)
    frames, durations, disposals = [], [], []
    for index in range(image.n_frames):
        image.seek(index)
        frames.append(np.array(image.convert("RGBA")))
        durations.append(image.info.get("duration", 100))
        disposals.append(image.disposal_method)
    return frames, durations, disposals, image.info.get("loop", 0)


@pytest.fixture(scope="session")
def animated_gif_path() -> Path:
    return ANIMATED_GIF


@pytest.fixture(scope="session")
def animated_frames() -> list[np.ndarray]:
    return read_gif(ANIMATED_GIF)[0]


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A one second 64x48 12fps H.264 clip with an AAC track.

    Encoded at test time rather than committed: libx264 output is not stable
    across ffmpeg versions or build flags, so a committed MP4 could never be
    asserted against its own generator the way the PNG and GIF fixtures are.

    Built through ffmpeg-python, the same binding the video adapter uses, so a
    broken binding fails here with a clear message instead of deep inside an
    adapter test. If ffmpeg is missing this raises rather than skipping: a
    skip would let a check derivation that lost ffmpeg report green while
    silently testing no video at all.
    """
    destination = tmp_path_factory.mktemp("video") / "sample.mp4"
    video = ffmpeg.input("testsrc=size=64x48:rate=12:duration=1", f="lavfi")
    audio = ffmpeg.input("sine=frequency=440:duration=1", f="lavfi")
    (
        ffmpeg.output(
            video,
            audio,
            str(destination),
            vcodec="libx264",
            pix_fmt="yuv420p",
            acodec="aac",
            shortest=None,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return destination
