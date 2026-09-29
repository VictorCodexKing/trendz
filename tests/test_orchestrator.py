"""Offline tests for the bounded worker pool, the concurrency core of the design."""

from __future__ import annotations

import asyncio

import pytest

from trendz.concurrency import bounded_map
from trendz.orchestrator import bounded_map as orchestrator_bounded_map


async def test_bounded_map_preserves_input_order() -> None:
    """Results come back in input order regardless of completion order."""

    async def slow_for_small(n: int) -> int:
        # Smaller inputs sleep longer, so completion order is the reverse of
        # input order; the result list must still follow input order.
        await asyncio.sleep((5 - n) * 0.01)
        return n * n

    result = await bounded_map(slow_for_small, [1, 2, 3, 4], concurrency=4)

    assert result == [1, 4, 9, 16]


async def test_bounded_map_caps_concurrency() -> None:
    """No more than ``concurrency`` coroutines run simultaneously."""
    live = 0
    high_water = 0
    concurrency = 3

    async def worker(item: int) -> int:
        nonlocal live, high_water
        live += 1
        high_water = max(high_water, live)
        # Hold the slot long enough for the pool to saturate.
        await asyncio.sleep(0.02)
        live -= 1
        return item

    items = list(range(12))
    result = await bounded_map(worker, items, concurrency=concurrency)

    assert result == items
    assert high_water <= concurrency
    # With 12 items and a cap of 3, the pool should actually reach the cap.
    assert high_water == concurrency


async def test_bounded_map_rejects_non_positive_concurrency() -> None:
    """A concurrency below 1 is a programming error and is rejected."""

    async def identity(n: int) -> int:
        return n

    with pytest.raises(ValueError):
        await bounded_map(identity, [1, 2], concurrency=0)


async def test_bounded_map_handles_empty_items() -> None:
    """An empty input yields an empty result without running the worker."""

    async def boom(n: int) -> int:
        raise AssertionError("worker should not be called for empty input")

    assert await bounded_map(boom, [], concurrency=2) == []


def test_orchestrator_reexports_bounded_map() -> None:
    """The orchestrator re-exports the shared helper as its concurrency core."""
    assert orchestrator_bounded_map is bounded_map
