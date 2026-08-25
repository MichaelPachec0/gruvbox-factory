"""The harness proving itself, before any product code depends on it."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from conftest import pixel_hash

FIXTURE_SOURCE_HASH = "1868cdf9393475de"
EXAMPLE_SHAPE = (984, 3490, 4)
FIXTURE_SHAPE = (120, 200, 4)


def test_fixture_matches_its_committed_hash(still_rgba: np.ndarray) -> None:
    assert still_rgba.shape == FIXTURE_SHAPE
    assert still_rgba.dtype == np.uint8
    assert pixel_hash(still_rgba) == FIXTURE_SOURCE_HASH


def test_fixture_straddles_the_transparency_boundary(still_rgba: np.ndarray) -> None:
    alpha = still_rgba[..., 3]
    assert int(alpha[58, 0]) == 189
    assert int(alpha[59, 0]) == 190
    assert int((alpha < 190).sum()) == 11796
    assert int((alpha >= 190).sum()) == 12204


def test_fixture_has_enough_distinct_colors(still_rgba: np.ndarray) -> None:
    flat = still_rgba[..., :3].reshape(-1, 3)
    assert len(np.unique(flat, axis=0)) == 23998


def test_example_png_is_present_and_rgba(example_rgba: np.ndarray) -> None:
    assert example_rgba.shape == EXAMPLE_SHAPE


def test_cache_is_isolated(isolated_cache: Path) -> None:
    assert os.environ["XDG_CACHE_HOME"] == str(isolated_cache)
    assert str(Path.home()) not in os.environ["XDG_CACHE_HOME"]


@pytest.mark.slow
def test_committed_fixture_matches_its_generator(
    still_rgba: np.ndarray, fixture_generator: ModuleType
) -> None:
    """The PNG in git is exactly what make_fixtures.py produces today."""
    assert np.array_equal(still_rgba, fixture_generator.build_still())
