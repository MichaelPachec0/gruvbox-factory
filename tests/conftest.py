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

import numpy as np
import pytest
from PIL import Image

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
REPO_ROOT = TESTS_DIR.parent
EXAMPLE_PNG = REPO_ROOT / "example.png"


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
