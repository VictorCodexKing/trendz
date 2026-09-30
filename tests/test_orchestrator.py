"""Offline tests for the bounded worker pool and the end-to-end pipeline run."""

from __future__ import annotations

import asyncio
from typing import get_args

import pytest

from trendz.agents.content_strategist import ContentStrategist
from trendz.agents.quality_gate import QualityGate
from trendz.agents.scheduler import Scheduler
from trendz.agents.trend_scout import ScoringWeights, TrendScout
from trendz.checkers.stub_checker import StubClipChecker
from trendz.concurrency import bounded_map
from trendz.contracts import (
    LearningState,
    PerformanceReports,
    Platform,
    RunConfig,
    RunContext,
    RunResult,
)
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


async def test_orchestrator_run_returns_run_result_bundle() -> None:
    """A full offline run (stages 1-8) returns a RunResult bundling reports + learnings.

    Built with the MockTrendSource exactly as ``python -m trendz`` does, this
    stays fully offline and deterministic. It asserts the new return type carries
    both the stage-7 PerformanceReports and the stage-8 LearningState (with the
    priors updated away from their zeroed defaults and non-empty bandit state).
    """
    config = RunConfig()
    orchestrator = Orchestrator(config=config, sources=[MockTrendSource()])

    result = await orchestrator.run()

    assert isinstance(result, RunResult)
    assert result.run_id == result.reports.run_id == result.learnings.run_id

    reports = result.reports
    assert isinstance(reports, PerformanceReports)

    learnings = result.learnings
    assert isinstance(learnings, LearningState)
    # A non-empty run applied one update, so the version/counter advanced past
    # the default state and the bandit + aggregates carry real state.
    assert learnings.version == 1
    assert learnings.updates_applied == 1
    assert learnings.arms
    assert learnings.trend_aggregates
    # The learned scoring weights are a usable ScoringWeights and the timing
    # hints are the provider's slot shape.
    assert isinstance(learnings.as_scoring_weights(), ScoringWeights)
    assert learnings.as_timing_slots()
    # Format priors were nudged for every observed platform (off their 0.0 seed).
    assert any(prior > 0.0 for prior in learnings.format_priors.values())

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
    first = (await Orchestrator(config=config, sources=[MockTrendSource()]).run()).reports
    second = (await Orchestrator(config=config, sources=[MockTrendSource()]).run()).reports

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


async def test_orchestrator_feedback_path_maps_learnings_into_next_run_configs() -> None:
    """The stage-8 learnings map back into the earlier agents' tunable configs.

    Exercises ``Orchestrator.next_run_agents``: after a full offline run, the
    produced LearningState is fed back into fresh agents/providers. This proves
    the loop is closed - the learned ScoringWeights, timing slots, format priors,
    and engagement threshold flow into real constructors the agents expose, not a
    parallel/cosmetic shape.
    """
    config = RunConfig()
    result = await Orchestrator(config=config, sources=[MockTrendSource()]).run()
    learnings = result.learnings

    trend_scout, content_strategist, quality_gate, scheduler = Orchestrator.next_run_agents(
        learnings
    )

    # Each learning fed a real agent/provider that accepts it.
    assert isinstance(trend_scout, TrendScout)
    assert isinstance(content_strategist, ContentStrategist)
    assert isinstance(quality_gate, QualityGate)
    assert isinstance(scheduler, Scheduler)

    # The Trend Scout is seeded with the learned weights (real ScoringWeights).
    learned_weights = learnings.as_scoring_weights()
    assert isinstance(learned_weights, ScoringWeights)
    assert trend_scout._weights == learned_weights

    # The Scheduler's timing provider carries the learned per-platform slots.
    learned_slots = learnings.as_timing_slots()
    ctx = RunContext.new(run_id="feedback-test", config=config)
    for platform, slots in learned_slots.items():
        assert scheduler._provider.optimal_minutes_of_day(ctx, platform) == slots

    # The Quality Gate's checker is seeded with the learned engagement threshold
    # on its REAL engagement gate (min_engagement_score), not the duration floor.
    checker = quality_gate._checker
    assert isinstance(checker, StubClipChecker)
    assert checker._min_engagement_score == learnings.engagement_threshold
