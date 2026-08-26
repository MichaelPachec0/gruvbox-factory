"""Map work over a frame stream with bounded memory and ordered output.

This is where defects D2 and D3 are answered. D2 was a ``join(timeout=30)``
with no liveness check, which silently returned a half-converted image; here a
worker exception surfaces through ``Future.result()`` and cannot be swallowed.
D3 was threading inside a single image, which the GIL made slower than serial;
here the unit of work is a whole frame, so numpy releases the GIL for the
duration and there is no shared mutable state to race over.

``ThreadPoolExecutor.map`` is deliberately not used. It submits every future
eagerly, so a 300 frame 1080p video would materialise 1.9GB before the first
result came back. A sliding window of futures keeps the same ordering
guarantee with a memory ceiling that does not depend on stream length.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor


class WorkerCountError(ValueError):
    """A worker count that cannot run anything."""

    def __init__(self, workers: int) -> None:
        super().__init__(f"workers must be at least 1, got {workers}")


class WindowSizeError(ValueError):
    """An in-flight window that cannot hold anything."""

    def __init__(self, window: int) -> None:
        super().__init__(f"window must be at least 1, got {window}")


def map_frames[T, R](
    frames: Iterable[T],
    work: Callable[[T], R],
    *,
    workers: int,
    window: int | None = None,
) -> Iterator[R]:
    """Apply ``work`` to each frame, yielding results in input order.

    At most ``window`` frames are in flight, so a generator input is consumed
    lazily and peak memory is bounded regardless of how long the stream is.
    The default of ``workers * 2`` keeps every worker fed with one queued
    behind it: at 8 workers and 1080p RGB that is 16 frames, about 100MB.

    ``workers`` is required rather than defaulting to ``os.cpu_count()``.
    Picking a thread count from whichever machine happens to be running is a
    policy decision, and keeping it out means behaviour here is fully
    determined by the arguments.

    ``workers=1`` still goes through the pool. One code path means the
    single-worker tests exercise the same machinery as the eight-worker ones,
    rather than the most-tested configuration being the one that skips it.

    If ``work`` raises, pending futures are cancelled and the exception
    propagates. Without the cancellation the executor's ``__exit__`` would
    call ``shutdown(wait=True)`` and quietly finish every already-submitted
    frame before the error surfaced.
    """
    if workers < 1:
        raise WorkerCountError(workers)
    limit = workers * 2 if window is None else window
    if limit < 1:
        raise WindowSizeError(limit)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque[Future[R]] = deque()
        try:
            for frame in frames:
                pending.append(pool.submit(work, frame))
                if len(pending) >= limit:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
