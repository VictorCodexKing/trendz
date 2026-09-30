"""Offline tests for the bounded worker pool and the end-to-end pipeline run."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import get_args

import pytest

from trendz.concurrency import bounded_map
from trendz.contracts import Platform, PublishPlan, RunConfig
from trendz.orchestrator import Orchestrator
from trendz.orchestrator import bounded_map as orchestrator_bounded_map
from trendz.sources.mock_source import MockTrendSource


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


async def test_orchestrator_run_returns_publish_plan_by_platform() -> None:
    """A full offline run (stages 1-5) returns a per-platform PublishPlan.

    Built with the MockTrendSource exactly as ``python -m trendz`` does, this
    stays fully offline and deterministic. It asserts relative spacing and
    ordering rather than absolute wall-clock times so it cannot be flaky.
    """
    config = RunConfig()
    orchestrator = Orchestrator(config=config, sources=[MockTrendSource()])

    plan = await orchestrator.run()

    assert isinstance(plan, PublishPlan)
    # The plan is organized by platform, and every platform is a known one.
    known_platforms = set(get_args(Platform))
    assert set(plan.platforms) <= known_platforms
    # platforms is sorted and distinct.
    assert list(plan.platforms) == sorted(set(plan.platforms))

    # slices() covers exactly the platforms property, in the same order, and
    # every post routes to the platform it is filed under.
    assert [platform for platform, _ in plan.slices()] == list(plan.platforms)
    total_posts = 0
    gap = timedelta(minutes=config.min_post_gap_minutes)
    for platform, posts in plan.slices():
        assert posts == plan.for_platform(platform)
        assert all(post.platform == platform for post in posts)
        total_posts += len(posts)
        # Within a platform, posts are in scheduled order and respect spacing.
        for earlier, later in zip(posts, posts[1:], strict=False):
            assert later.scheduled_at >= earlier.scheduled_at
            assert later.scheduled_at - earlier.scheduled_at >= gap

    # slices() partitions all posts with no leftovers.
    assert total_posts == len(plan)


async def test_orchestrator_run_is_deterministic() -> None:
    """Two full offline runs with the same config produce equal plans (barring run id)."""
    config = RunConfig()
    first = await Orchestrator(config=config, sources=[MockTrendSource()]).run()
    second = await Orchestrator(config=config, sources=[MockTrendSource()]).run()

    # run_id differs per run; the scheduled posts must be identical.
    assert first.posts == second.posts
