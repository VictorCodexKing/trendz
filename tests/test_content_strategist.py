"""Offline tests for the Content Strategist agent."""

from __future__ import annotations

from trendz.agents.content_strategist import (
    DEFAULT_STRATEGY,
    PLATFORM_FORMATS,
    ContentStrategist,
    StrategyConfig,
)
from trendz.contracts import (
    ClipBriefList,
    RunConfig,
    RunContext,
    Trend,
    TrendList,
)


def _trend(id: str, title: str) -> Trend:
    """Build a minimal scored trend for planning."""
    return Trend(id=id, title=title, source="mock", velocity=0.5, score=0.5)


def _trend_list(run_id: str, *trends: Trend) -> TrendList:
    return TrendList(run_id=run_id, trends=tuple(trends))


async def test_empty_trend_list_yields_empty_brief_list(ctx: RunContext) -> None:
    """An empty TrendList produces an empty ClipBriefList, no error."""
    strategist = ContentStrategist()

    result = await strategist.run(ctx, _trend_list(ctx.run_id))

    assert isinstance(result, ClipBriefList)
    assert result.run_id == ctx.run_id
    assert len(result) == 0


async def test_briefs_link_back_to_input_trend_ids(ctx: RunContext) -> None:
    """Every brief's trend_id refers to a trend present in the input."""
    trends = _trend_list(ctx.run_id, _trend("t1", "AI cooking hacks"), _trend("t2", "Retro gaming"))
    strategist = ContentStrategist()

    result = await strategist.run(ctx, trends)

    input_ids = {t.id for t in trends.trends}
    assert len(result) > 0
    for brief in result.briefs:
        assert brief.trend_id in input_ids


async def test_total_count_respects_target_clip_count() -> None:
    """The number of briefs never exceeds ctx.config.target_clip_count."""
    config = RunConfig(target_clip_count=4)
    ctx = RunContext.new(run_id="cap-run", config=config)
    # 3 trends x 3 platforms = 9 possible briefs; the cap of 4 must bound it.
    trends = _trend_list(
        ctx.run_id,
        _trend("t1", "AI cooking hacks"),
        _trend("t2", "Retro gaming speedruns"),
        _trend("t3", "Sustainable fashion tips"),
    )
    strategist = ContentStrategist()

    result = await strategist.run(ctx, trends)

    assert len(result) == 4


async def test_zero_target_clip_count_yields_no_briefs() -> None:
    """A zero clip budget produces no briefs even with trends present."""
    config = RunConfig(target_clip_count=0)
    ctx = RunContext.new(run_id="zero-run", config=config)
    trends = _trend_list(ctx.run_id, _trend("t1", "AI cooking hacks"))
    strategist = ContentStrategist()

    result = await strategist.run(ctx, trends)

    assert len(result) == 0


async def test_planning_is_deterministic(ctx: RunContext) -> None:
    """Same input yields identical briefs across runs."""
    trends = _trend_list(ctx.run_id, _trend("t1", "AI cooking hacks"), _trend("t2", "Retro gaming"))
    strategist = ContentStrategist()

    first = await strategist.run(ctx, trends)
    second = await strategist.run(ctx, trends)

    assert first == second
    assert first.briefs == second.briefs


async def test_required_fields_populated_within_valid_ranges(ctx: RunContext) -> None:
    """Each brief has valid platform/format/length and non-empty copy."""
    trends = _trend_list(ctx.run_id, _trend("t1", "AI cooking hacks"))
    strategist = ContentStrategist()

    result = await strategist.run(ctx, trends)

    valid_platforms = set(PLATFORM_FORMATS)
    valid_ratios = {"9:16", "1:1", "16:9"}
    for brief in result.briefs:
        assert brief.platform in valid_platforms
        assert brief.aspect_ratio in valid_ratios
        assert 1 <= brief.target_length_seconds <= 180
        assert brief.caption
        assert brief.cta
        assert brief.hashtags
        assert brief.footage_kind in {"sourced", "ai"}
        # Format matches the per-platform defaults.
        expected_ratio, expected_len = PLATFORM_FORMATS[brief.platform]
        assert brief.aspect_ratio == expected_ratio
        assert brief.target_length_seconds == expected_len


async def test_one_brief_per_configured_platform_per_trend() -> None:
    """With a generous cap, each trend yields one brief per configured platform."""
    config = RunConfig(target_clip_count=100)
    ctx = RunContext.new(run_id="fanout-run", config=config)
    trends = _trend_list(ctx.run_id, _trend("t1", "AI cooking hacks"))
    strategist = ContentStrategist()

    result = await strategist.run(ctx, trends)

    assert len(result) == len(DEFAULT_STRATEGY.platforms)
    assert {b.platform for b in result.briefs} == set(DEFAULT_STRATEGY.platforms)


async def test_custom_strategy_targets_single_platform() -> None:
    """Injected strategy config controls which platforms are planned."""
    config = RunConfig(target_clip_count=100)
    ctx = RunContext.new(run_id="single-run", config=config)
    trends = _trend_list(ctx.run_id, _trend("t1", "AI cooking hacks"))
    strategist = ContentStrategist(StrategyConfig(platforms=("tiktok",), footage_kind="ai"))

    result = await strategist.run(ctx, trends)

    assert len(result) == 1
    assert result.briefs[0].platform == "tiktok"
    assert result.briefs[0].footage_kind == "ai"
