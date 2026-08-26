"""pipeline.py: ordering, boundedness, concurrency and abort."""

from __future__ import annotations

import threading
import time

import pytest

from factory.pipeline import WindowSizeError, WorkerCountError, map_frames


def square(value: int) -> int:
    return value * value + 1


# Sentinel messages are one word on purpose: TRY003 flags any raise whose
# message contains whitespace, and it is enforced on tests as well as code.
SENTINEL = "badframe"


def counting_stream(total: int, seen: list[int]):
    for value in range(total):
        seen.append(value)
        yield value


# --- ordering ------------------------------------------------------------


def test_output_order_matches_input_order() -> None:
    assert list(map_frames(range(50), lambda x: x * 2, workers=8)) == [
        x * 2 for x in range(50)
    ]


@pytest.mark.parametrize("workers", [1, 4, 8])
def test_results_match_sequential_execution(workers: int) -> None:
    """D3 regression: worker count must not change the result."""
    assert list(map_frames(range(40), square, workers=workers)) == [
        square(x) for x in range(40)
    ]


def test_empty_input_yields_nothing() -> None:
    assert list(map_frames([], lambda x: x, workers=4)) == []


# --- boundedness ---------------------------------------------------------


def test_input_is_not_consumed_ahead_of_the_window() -> None:
    """ThreadPoolExecutor.map would pull all 1000 before yielding anything."""
    seen: list[int] = []
    stream = map_frames(counting_stream(1000, seen), lambda x: x, workers=4, window=8)
    next(stream)
    assert len(seen) == 8
    stream.close()


def test_window_defaults_to_twice_the_workers() -> None:
    seen: list[int] = []
    stream = map_frames(counting_stream(1000, seen), lambda x: x, workers=3)
    next(stream)
    assert len(seen) == 6
    stream.close()


def test_closing_the_stream_early_is_clean() -> None:
    seen: list[int] = []
    stream = map_frames(counting_stream(1000, seen), lambda x: x, workers=2, window=4)
    next(stream)
    stream.close()
    assert len(seen) == 4


# --- concurrency ---------------------------------------------------------


def test_work_actually_runs_concurrently() -> None:
    live = peak = 0
    lock = threading.Lock()

    def slow(value: int) -> int:
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.02)
        with lock:
            live -= 1
        return value

    list(map_frames(range(32), slow, workers=8))
    assert peak > 1


# --- abort ---------------------------------------------------------------


def test_worker_exception_propagates() -> None:
    """D2 regression: a failure must surface, not truncate the output."""

    def boom(value: int) -> int:
        if value == 5:
            raise RuntimeError(SENTINEL)
        return value

    with pytest.raises(RuntimeError, match=SENTINEL):
        list(map_frames(range(500), boom, workers=4, window=8))


def test_abort_does_not_process_the_whole_stream() -> None:
    """Cancelling pending futures is what makes an abort an abort.

    Measured 8 to 14 frames started across worker and window combinations, so
    the bound is loose on purpose: an exact figure depends on thread timing.
    Without the cancellation the executor's __exit__ would wait for every
    submitted frame.
    """
    started: list[int] = []

    def boom(value: int) -> int:
        started.append(value)
        if value == 5:
            raise RuntimeError(SENTINEL)
        time.sleep(0.01)
        return value

    with pytest.raises(RuntimeError):
        list(map_frames(range(500), boom, workers=4, window=8))
    assert len(started) < 50


# --- validation ----------------------------------------------------------


def test_window_of_one_is_sequential_and_correct() -> None:
    assert list(map_frames(range(20), lambda x: x + 1, workers=4, window=1)) == list(
        range(1, 21)
    )


@pytest.mark.parametrize("workers", [0, -1])
def test_worker_count_is_validated(workers: int) -> None:
    with pytest.raises(WorkerCountError, match="at least 1"):
        list(map_frames(range(3), lambda x: x, workers=workers))


@pytest.mark.parametrize("window", [0, -1])
def test_window_size_is_validated(window: int) -> None:
    with pytest.raises(WindowSizeError, match="at least 1"):
        list(map_frames(range(3), lambda x: x, workers=4, window=window))


def test_errors_are_value_errors() -> None:
    assert issubclass(WorkerCountError, ValueError)
    assert issubclass(WindowSizeError, ValueError)
