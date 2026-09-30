"""Offline tests for the bounded worker pool and the end-to-end pipeline run."""

from __future__ import annotations

import asyncio
from typing import get_args

import pytest

from trendz.concurrency import bounded_map
from trendz.contracts import PerformanceReports, Platform, RunConfig
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


async def test_orchestrator_run_returns_performance_reports_by_platform() -> None:
    """A full offline run (stages 1-7) returns per-platform PerformanceReports.

    Built with the MockTrendSource exactly as ``python -m trendz`` does, this
    stays fully offline and deterministic. It asserts per-platform partitioning
    and attribution rather than absolute wall-clock times so it cannot be flaky.
    """
    config = RunConfig()
    orchestrator = Orchestrator(config=config, sources=[MockTrendSource()])

    reports = await orchestrator.run()

    assert isinstance(reports, PerformanceReports)
    # The reports are organized by platform, and every platform is a known one.
    known_platforms = set(get_args(Platform))
    assert set(reports.platforms) <= known_platforms
    # platforms is sorted and distinct.
    assert list(reports.platforms) == sorted(set(reports.platforms))

    # slices() covers exactly the platforms property, in the same order, and
    # every report routes to the platform it is filed under; each is attributed.
    assert [platform for platform, _ in reports.slices()] == list(reports.platforms)
    total = 0
    for platform, platform_reports in reports.slices():
        assert platform_reports == reports.for_platform(platform)
        assert all(r.platform == platform for r in platform_reports)
        for report in platform_reports:
            assert report.clip_id
            assert report.brief_id
            assert report.trend_id
            assert report.post_id
        total += len(platform_reports)

    # slices() partitions all reports with no leftovers.
    assert total == len(reports)


async def test_orchestrator_run_is_deterministic() -> None:
    """Two full offline runs with the same config produce equal reports (barring run id).

    The Publisher derives each post's idempotency key (and thus its stub
    post_id/post_url) from the run id, which is a fresh UUID per run, so those
    fields legitimately differ between runs. Normalising the run id out shows the
    rest of every report is byte-identical, proving the pipeline is deterministic.
    """
    config = RunConfig()
    first = await Orchestrator(config=config, sources=[MockTrendSource()]).run()
    second = await Orchestrator(config=config, sources=[MockTrendSource()]).run()

    def _normalise(reports: PerformanceReports) -> list[dict[str, object]]:
        # The stub post_id embeds the per-run idempotency key (which embeds the
        # fresh-per-run UUID), and the stub metrics - and thus the derived score
        # - are a pure function of that post_id, so post_id/metrics/score
        # legitimately differ between runs. Normalising the run id out of post_id
        # and dropping the post-id-derived metrics/score shows the rest of every
        # report (the full attribution + the deterministic collected_at) is
        # byte-identical, proving the pipeline is deterministic.
        normalised: list[dict[str, object]] = []
        for report in reports.reports:
            data = report.model_dump()
            value = data["post_id"]
            if isinstance(value, str):
                data["post_id"] = value.replace(reports.run_id, "<run_id>")
            data.pop("metrics", None)
            data.pop("performance_score", None)
            normalised.append(data)
        return normalised

    assert _normalise(first) == _normalise(second)
