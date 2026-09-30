"""Offline tests for the Performance Analyst agent and its per-post bounded fan-out."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from trendz.agents.performance_analyst import PerformanceAnalyst, _performance_score
from trendz.analytics.base import MetricsProvider
from trendz.analytics.stub_provider import StubMetricsProvider
from trendz.contracts import (
    PerformanceReports,
    Platform,
    PostMetrics,
    PostResult,
    PostResults,
    RunConfig,
    RunContext,
)

_BASE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _result(
    idx: int,
    platform: Platform = "tiktok",
    variant: int = 0,
    succeeded: bool = True,
    minute_offset: int = 0,
) -> PostResult:
    """Build a deterministic PostResult for tests (succeeded by default)."""
    if succeeded:
        return PostResult(
            clip_id=f"clip-b{idx}",
            brief_id=f"b{idx}",
            trend_id=f"t{idx}",
            platform=platform,
            variant=variant,
            idempotency_key=f"run:clip-b{idx}:{platform}:{variant}",
            succeeded=True,
            post_id=f"post-{idx}",
            post_url=f"stub://{platform}/posts/{idx}",
            error=None,
            published_at=_BASE.replace(minute=minute_offset % 60),
        )
    return PostResult(
        clip_id=f"clip-b{idx}",
        brief_id=f"b{idx}",
        trend_id=f"t{idx}",
        platform=platform,
        variant=variant,
        idempotency_key=f"run:clip-b{idx}:{platform}:{variant}",
        succeeded=False,
        post_id=None,
        post_url=None,
        error="error: boom",
        published_at=None,
    )


def _results(run_id: str, results: tuple[PostResult, ...]) -> PostResults:
    """Build a PostResults from post results."""
    return PostResults(run_id=run_id, results=results)


async def test_happy_path_every_succeeded_post_yields_a_report(ctx: RunContext) -> None:
    """Every succeeded post yields a PostPerformance with correct attribution."""
    results = (
        _result(0, "tiktok", minute_offset=0),
        _result(1, "instagram", minute_offset=5),
        _result(2, "youtube_shorts", minute_offset=10),
        _result(3, "facebook", minute_offset=15),
    )
    payload = _results(ctx.run_id, results)
    analyst = PerformanceAnalyst(StubMetricsProvider())

    reports = await analyst.run(ctx, payload)

    assert isinstance(reports, PerformanceReports)
    assert reports.run_id == ctx.run_id
    assert len(reports) == len(results)
    by_clip = {report.clip_id: report for report in reports.reports}
    for result in results:
        report = by_clip[result.clip_id]
        assert report.brief_id == result.brief_id
        assert report.trend_id == result.trend_id
        assert report.platform == result.platform
        assert report.variant == result.variant
        assert report.post_id == result.post_id
        # Metrics are all non-negative.
        assert report.metrics.views >= 0
        assert report.metrics.watch_time_seconds >= 0
        assert report.metrics.likes >= 0
        assert report.metrics.shares >= 0
        assert report.metrics.comments >= 0
        assert report.metrics.follows >= 0
        assert report.performance_score >= 0
        assert result.published_at is not None
        expected = result.published_at + timedelta(hours=ctx.config.dwell_hours)
        assert report.collected_at == expected


async def test_failed_posts_are_excluded_from_collection(ctx: RunContext) -> None:
    """A PostResults mixing succeeded + failed yields reports only for the succeeded."""
    results = (
        _result(0, "tiktok", succeeded=True),
        _result(1, "instagram", succeeded=False),
        _result(2, "facebook", succeeded=True),
        _result(3, "youtube_shorts", succeeded=False),
    )
    payload = _results(ctx.run_id, results)

    reports = await PerformanceAnalyst(StubMetricsProvider()).run(ctx, payload)

    assert len(reports) == 2
    reported_clips = {report.clip_id for report in reports.reports}
    assert reported_clips == {"clip-b0", "clip-b2"}
    # No report carries a post_id from a failed post.
    assert all(report.post_id for report in reports.reports)


async def test_performance_score_is_deterministic_and_reflects_metrics(ctx: RunContext) -> None:
    """The per-post score is derived deterministically and increases with engagement."""
    results = (_result(0, "tiktok"),)
    payload = _results(ctx.run_id, results)

    first = await PerformanceAnalyst(StubMetricsProvider()).run(ctx, payload)
    second = await PerformanceAnalyst(StubMetricsProvider()).run(ctx, payload)

    # Same input -> same score.
    assert first.reports[0].performance_score == second.reports[0].performance_score

    # The score reflects the metrics: more engagement scores strictly higher.
    low = PostMetrics(
        views=100, watch_time_seconds=700.0, likes=10, shares=2, comments=1, follows=0
    )
    high = PostMetrics(
        views=200, watch_time_seconds=1400.0, likes=20, shares=4, comments=2, follows=1
    )
    assert _performance_score(low) < _performance_score(high)
    # And it matches the score the report carries for the actual metrics.
    report = first.reports[0]
    assert report.performance_score == _performance_score(report.metrics)


async def test_reports_grouped_per_platform(ctx: RunContext) -> None:
    """for_platform / slices partition reports and cover reports.platforms."""
    results = (
        _result(0, "tiktok"),
        _result(1, "tiktok", variant=1),
        _result(2, "instagram"),
        _result(3, "facebook"),
    )
    payload = _results(ctx.run_id, results)

    reports = await PerformanceAnalyst(StubMetricsProvider()).run(ctx, payload)

    assert set(reports.platforms) == {"tiktok", "instagram", "facebook"}
    assert list(reports.platforms) == sorted(set(reports.platforms))
    assert [platform for platform, _ in reports.slices()] == list(reports.platforms)
    total = 0
    for platform, platform_reports in reports.slices():
        assert platform_reports == reports.for_platform(platform)
        assert all(r.platform == platform for r in platform_reports)
        total += len(platform_reports)
    assert total == len(reports)


async def test_attribution_grouped_by_trend(ctx: RunContext) -> None:
    """by_trend groups reports by originating trend in sorted trend order."""
    results = (
        _result(0, "tiktok"),  # trend t0
        _result(0, "instagram"),  # same clip/trend t0, different platform
        _result(1, "facebook"),  # trend t1
    )
    payload = _results(ctx.run_id, results)

    reports = await PerformanceAnalyst(StubMetricsProvider()).run(ctx, payload)

    by_trend = reports.by_trend()
    trend_ids = [trend_id for trend_id, _ in by_trend]
    assert trend_ids == sorted(trend_ids)
    grouped = {trend_id: group for trend_id, group in by_trend}
    assert set(grouped) == {"t0", "t1"}
    assert len(grouped["t0"]) == 2
    assert len(grouped["t1"]) == 1
    # Every report in a group actually belongs to that trend, and the groups
    # partition all reports.
    for trend_id, group in by_trend:
        assert all(r.trend_id == trend_id for r in group)
    assert sum(len(group) for _, group in by_trend) == len(reports)


async def test_top_ranks_reports_by_performance_score(ctx: RunContext) -> None:
    """top(n) returns the highest-scoring reports, not publish order.

    On a scored container ``top`` ranks by ``performance_score`` descending
    (mirroring TrendResults.top), so it must not simply echo the stored publish
    order when that order is not score-sorted.
    """
    results = (
        _result(0, "tiktok"),
        _result(1, "instagram"),
        _result(2, "youtube_shorts"),
        _result(3, "facebook"),
    )
    payload = _results(ctx.run_id, results)

    reports = await PerformanceAnalyst(StubMetricsProvider()).run(ctx, payload)

    ranked = sorted(reports.reports, key=lambda r: r.performance_score, reverse=True)
    # top(n) is score-ranked, not stored/publish order.
    assert reports.top(2) == tuple(ranked[:2])
    assert reports.top(len(reports)) == tuple(ranked)
    # Descending by score, and a partition of all reports.
    scores = [r.performance_score for r in reports.top(len(reports))]
    assert scores == sorted(scores, reverse=True)
    assert set(reports.top(len(reports))) == set(reports.reports)
    # Guard is preserved.
    try:
        reports.top(-1)
    except ValueError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("top(-1) must raise ValueError")


class _ConcurrencyProbeProvider(MetricsProvider):
    """A provider that tracks the high-water mark of concurrent collect() calls."""

    def __init__(self) -> None:
        self.live = 0
        self.high_water = 0

    @property
    def name(self) -> str:
        return "concurrency_probe"

    async def collect(self, ctx: RunContext, result: PostResult) -> PostMetrics:
        self.live += 1
        self.high_water = max(self.high_water, self.live)
        # Hold the slot long enough for the per-post pool to saturate.
        await asyncio.sleep(0.02)
        self.live -= 1
        return PostMetrics(
            views=100, watch_time_seconds=700.0, likes=10, shares=2, comments=1, follows=0
        )


async def test_fan_out_reaches_but_never_exceeds_concurrency_cap() -> None:
    """The per-post fan-out runs concurrently up to the cap, never beyond.

    A PostResults with more posts than the cap: the high-water mark must reach
    the cap (proving real concurrency across per-post collections) but never
    exceed it (proving bounded_map governs the per-post fan-out).
    """
    concurrency = 3
    config = RunConfig(concurrency_degree=concurrency)
    ctx = RunContext.new(run_id="concurrency-run", config=config)
    results = tuple(_result(i, "tiktok", variant=i) for i in range(12))
    payload = _results(ctx.run_id, results)
    probe = _ConcurrencyProbeProvider()

    reports = await PerformanceAnalyst(probe).run(ctx, payload)

    assert len(reports) == 12
    assert probe.high_water <= concurrency
    # Twelve posts with a cap of 3: the pool should actually reach the cap.
    assert probe.high_water == concurrency


async def test_empty_results_yields_empty_reports_without_dispatch(ctx: RunContext) -> None:
    """An empty PostResults yields an empty PerformanceReports without collecting."""
    probe = _ConcurrencyProbeProvider()

    reports = await PerformanceAnalyst(probe).run(ctx, PostResults(run_id=ctx.run_id))

    assert isinstance(reports, PerformanceReports)
    assert len(reports) == 0
    assert reports.run_id == ctx.run_id
    assert probe.high_water == 0


async def test_all_failed_results_yields_empty_reports_without_dispatch(ctx: RunContext) -> None:
    """An all-failed PostResults yields empty reports without collecting."""
    results = (
        _result(0, "tiktok", succeeded=False),
        _result(1, "instagram", succeeded=False),
    )
    payload = _results(ctx.run_id, results)
    probe = _ConcurrencyProbeProvider()

    reports = await PerformanceAnalyst(probe).run(ctx, payload)

    assert len(reports) == 0
    assert probe.high_water == 0


async def test_analysis_is_deterministic(ctx: RunContext) -> None:
    """Two runs over the same PostResults produce equal PerformanceReports."""
    results = (
        _result(0, "tiktok"),
        _result(1, "instagram"),
        _result(2, "youtube_shorts"),
        _result(3, "facebook"),
    )
    payload = _results(ctx.run_id, results)

    first = await PerformanceAnalyst(StubMetricsProvider()).run(ctx, payload)
    second = await PerformanceAnalyst(StubMetricsProvider()).run(ctx, payload)

    assert first == second


async def test_default_provider_is_stub() -> None:
    """The Performance Analyst defaults to the deterministic offline stub provider."""
    analyst = PerformanceAnalyst()
    assert isinstance(analyst._provider, StubMetricsProvider)
