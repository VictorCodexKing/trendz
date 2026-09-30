"""Offline tests for the bounded worker pool and the end-to-end pipeline run."""

from __future__ import annotations

import asyncio
from typing import get_args

import pytest

from trendz.concurrency import bounded_map
from trendz.contracts import Platform, PostResults, RunConfig
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


async def test_orchestrator_run_returns_post_results_by_platform() -> None:
    """A full offline run (stages 1-6) returns per-platform PostResults.

    Built with the MockTrendSource exactly as ``python -m trendz`` does, this
    stays fully offline and deterministic. It asserts per-platform partitioning
    and attribution rather than absolute wall-clock times so it cannot be flaky.
    """
    config = RunConfig()
    orchestrator = Orchestrator(config=config, sources=[MockTrendSource()])

    results = await orchestrator.run()

    assert isinstance(results, PostResults)
    # The results are organized by platform, and every platform is a known one.
    known_platforms = set(get_args(Platform))
    assert set(results.platforms) <= known_platforms
    # platforms is sorted and distinct.
    assert list(results.platforms) == sorted(set(results.platforms))

    # slices() covers exactly the platforms property, in the same order, and
    # every result routes to the platform it is filed under; each is attributed.
    assert [platform for platform, _ in results.slices()] == list(results.platforms)
    total = 0
    for platform, platform_results in results.slices():
        assert platform_results == results.for_platform(platform)
        assert all(r.platform == platform for r in platform_results)
        for result in platform_results:
            assert result.clip_id
            assert result.brief_id
            assert result.trend_id
            assert result.idempotency_key
        total += len(platform_results)

    # slices() partitions all results with no leftovers, and succeeded/failed
    # partition them too.
    assert total == len(results)
    assert len(results.succeeded) + len(results.failed) == len(results)


async def test_orchestrator_run_is_deterministic() -> None:
    """Two full offline runs with the same config produce equal results (barring run id).

    The Publisher derives each post's idempotency key (and thus its stub
    post_id/post_url) from the run id, which is a fresh UUID per run, so those
    fields legitimately differ between runs. Normalising the run id out shows the
    rest of every result is byte-identical, proving the pipeline is deterministic.
    """
    config = RunConfig()
    first = await Orchestrator(config=config, sources=[MockTrendSource()]).run()
    second = await Orchestrator(config=config, sources=[MockTrendSource()]).run()

    def _normalise(results: PostResults) -> list[dict[str, object]]:
        normalised: list[dict[str, object]] = []
        for result in results.results:
            data = result.model_dump()
            for key in ("idempotency_key", "post_id", "post_url"):
                value = data[key]
                if isinstance(value, str):
                    data[key] = value.replace(results.run_id, "<run_id>")
            normalised.append(data)
        return normalised

    assert _normalise(first) == _normalise(second)
