"""lut.py: cache keying, atomic writes, LRU eviction and the real table."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from conftest import pixel_hash

from factory import kernel, lut
from factory.palette import Palette, builtin, parse

MEGABYTE = 1024 * 1024
FIXTURE_PINK_GOLDEN = "c9d5498fddf138f2"


@pytest.fixture(scope="session")
def pink_table() -> np.ndarray:
    """One real 2**24 table, built once. About 13 seconds.

    Several slow tests only consume a table. Building it per test would be
    roughly 65 seconds of identical work. Tests that care about how or when
    build is called deliberately do not use this.
    """
    return lut.build(builtin("pink"))


@pytest.fixture
def cache(isolated_cache: Path) -> Iterator[Path]:
    directory = lut.cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    yield directory


def blank_table() -> np.ndarray:
    return np.zeros(lut.LUT_SIZE, dtype=np.uint8)


def fill(directory: Path, count: int, size: int) -> list[Path]:
    """Write `count` stale cache entries, oldest first by mtime."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for index in range(count):
        path = directory / (f"{index:x}" * 64 + ".lut")
        path.write_bytes(b"\0" * size)
        os.utime(path, (1000, 1000 + index))
        written.append(path)
    return written


# --- cache keying --------------------------------------------------------


def test_cache_path_is_the_content_hash() -> None:
    palette = builtin("pink")
    assert lut.cache_path(palette).name == f"{palette.content_hash}.lut"


def test_cache_dir_follows_xdg_cache_home(isolated_cache: Path) -> None:
    assert lut.cache_dir() == isolated_cache / "gruvbox-factory"


def test_cache_dir_falls_back_to_home_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert lut.cache_dir() == tmp_path / ".cache" / "gruvbox-factory"


def test_editing_a_palette_changes_the_key() -> None:
    """The single most likely thing to get wrong in this design."""
    before = parse("#aabbcc\n#001122\n")
    after = parse("#aabbcd\n#001122\n")
    assert lut.cache_path(before) != lut.cache_path(after)


def test_reordering_a_palette_keeps_the_key() -> None:
    """Same colors, same table. Keying on raw bytes would store it twice."""
    a = parse("#aabbcc\n#001122\n")
    b = parse("#001122\n#AABBCC\n")
    assert lut.cache_path(a) == lut.cache_path(b)


@pytest.mark.parametrize("name", ["pink", "white", "mix"])
def test_every_builtin_gets_a_distinct_key(name: str) -> None:
    others = {lut.cache_path(builtin(o)) for o in ("pink", "white", "mix") if o != name}
    assert lut.cache_path(builtin(name)) not in others


# --- store and load ------------------------------------------------------


def test_store_then_load_round_trips(cache: Path) -> None:
    palette = builtin("pink")
    table = blank_table()
    table[:1000] = np.arange(1000, dtype=np.uint8)
    path = lut.store(palette, table)
    assert path.stat().st_size == lut.LUT_SIZE
    loaded = lut.load_cached(palette)
    assert loaded is not None
    assert np.array_equal(np.asarray(loaded), table)


def test_load_returns_none_on_a_cold_cache() -> None:
    assert lut.load_cached(builtin("mix")) is None


def test_truncated_cache_entry_is_rejected(cache: Path) -> None:
    palette = builtin("pink")
    path = lut.store(palette, blank_table())
    path.write_bytes(b"short")
    assert lut.load_cached(palette) is None


def test_oversized_cache_entry_is_rejected(cache: Path) -> None:
    palette = builtin("pink")
    path = lut.store(palette, blank_table())
    with path.open("ab") as handle:
        handle.write(b"\0")
    assert lut.load_cached(palette) is None


def test_store_leaves_no_temporary_files(cache: Path) -> None:
    lut.store(builtin("pink"), blank_table())
    assert list(cache.glob("*.tmp")) == []


def test_store_creates_the_cache_directory(isolated_cache: Path) -> None:
    assert not lut.cache_dir().exists()
    lut.store(builtin("pink"), blank_table())
    assert lut.cache_dir().is_dir()


def test_a_cache_hit_refreshes_mtime(cache: Path) -> None:
    palette = builtin("pink")
    path = lut.store(palette, blank_table())
    os.utime(path, (1000, 1000))
    assert lut.load_cached(palette) is not None
    assert path.stat().st_mtime > 1000


def test_a_cached_table_is_read_only(cache: Path) -> None:
    palette = builtin("pink")
    lut.store(palette, blank_table())
    loaded = lut.load_cached(palette)
    assert loaded is not None
    with pytest.raises(ValueError, match="read-only|assignment"):
        loaded[0] = 1


# --- eviction ------------------------------------------------------------


def test_eviction_removes_the_oldest_first(tmp_path: Path) -> None:
    stale = fill(tmp_path, 6, 20 * MEGABYTE)
    removed = lut.evict(tmp_path)
    assert set(removed) == {stale[0]}
    assert {p.name for p in tmp_path.glob("*.lut")} == {p.name for p in stale[1:]}


def test_eviction_stops_at_the_cap(tmp_path: Path) -> None:
    fill(tmp_path, 10, 20 * MEGABYTE)
    lut.evict(tmp_path)
    total = sum(p.stat().st_size for p in tmp_path.glob("*.lut"))
    assert total <= lut.CACHE_LIMIT_BYTES


def test_eviction_under_the_cap_removes_nothing(tmp_path: Path) -> None:
    fill(tmp_path, 2, 20 * MEGABYTE)
    assert lut.evict(tmp_path) == []


def test_eviction_ignores_unrelated_files(tmp_path: Path) -> None:
    fill(tmp_path, 6, 20 * MEGABYTE)
    keep = tmp_path / "notes.txt"
    keep.write_text("not a table")
    lut.evict(tmp_path)
    assert keep.exists()


def test_store_never_evicts_what_it_just_wrote(tmp_path: Path) -> None:
    """The new table is the newest by mtime, so LRU must spare it."""
    fill(tmp_path, 6, 20 * MEGABYTE)
    path = lut.store(builtin("pink"), blank_table(), tmp_path)
    assert path.exists()
    total = sum(p.stat().st_size for p in tmp_path.glob("*.lut"))
    assert total <= lut.CACHE_LIMIT_BYTES


def test_eviction_order_is_by_recorded_recency(tmp_path: Path) -> None:
    """Touching the oldest entry saves it and condemns the next one."""
    stale = fill(tmp_path, 6, 20 * MEGABYTE)
    os.utime(stale[0], (2000, 2000))
    removed = lut.evict(tmp_path)
    assert set(removed) == {stale[1]}


# --- the real table ------------------------------------------------------


@pytest.mark.slow
def test_build_produces_a_full_uint8_table(pink_table: np.ndarray) -> None:
    assert pink_table.shape == (lut.LUT_SIZE,)
    assert pink_table.dtype == np.uint8
    assert int(pink_table.max()) < len(builtin("pink"))


@pytest.mark.slow
def test_every_palette_color_maps_to_its_own_index(pink_table: np.ndarray) -> None:
    palette = builtin("pink")
    keys = kernel.pack(np.asarray(palette.rgb).astype(np.uint8))
    assert np.array_equal(pink_table[keys], np.arange(len(palette), dtype=np.uint8))


@pytest.mark.slow
def test_lut_and_packed_kernels_agree(
    still_rgba: np.ndarray, pink_table: np.ndarray
) -> None:
    palette = builtin("pink")
    through_lut = kernel.apply_rgba(
        still_rgba, kernel.bind(palette, "lut", pink_table)
    )
    through_packed = kernel.apply_rgba(still_rgba, kernel.bind(palette, "packed"))
    assert (
        pixel_hash(through_lut)
        == pixel_hash(through_packed)
        == FIXTURE_PINK_GOLDEN
    )


@pytest.mark.slow
def test_a_memmapped_table_converts_identically(
    cache: Path, still_rgba: np.ndarray, pink_table: np.ndarray
) -> None:
    palette = builtin("pink")
    lut.store(palette, pink_table)
    mapped = lut.load_cached(palette)
    assert mapped is not None
    assert pixel_hash(
        kernel.apply_rgba(still_rgba, kernel.bind(palette, "lut", mapped))
    ) == pixel_hash(
        kernel.apply_rgba(still_rgba, kernel.bind(palette, "lut", pink_table))
    )


@pytest.mark.slow
def test_get_builds_once_then_reuses(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deliberately does not use pink_table: counting builds is the point."""
    palette = builtin("pink")
    calls: list[Palette] = []
    real_build = lut.build

    def counting_build(target: Palette) -> np.ndarray:
        calls.append(target)
        return real_build(target)

    monkeypatch.setattr(lut, "build", counting_build)
    first = lut.get(palette)
    second = lut.get(palette)
    assert len(calls) == 1
    assert np.array_equal(np.asarray(first), np.asarray(second))
