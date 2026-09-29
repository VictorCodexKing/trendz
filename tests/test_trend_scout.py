"""Offline tests for the Trend Scout agent and the Orchestrator stub."""

from __future__ import annotations

from trendz.agents.trend_scout import TrendScout, score_trend
from trendz.contracts import RunConfig, RunContext, Trend, TrendList
from trendz.orchestrator import Orchestrator
from trendz.sources.mock_source import MockTrendSource


async def test_ranked_descending_by_score(ctx: RunContext) -> None:
    """Trend Scout returns a TrendList ranked descending by score."""
    scout = TrendScout([MockTrendSource()])

    result = await scout.run(ctx, None)

    assert isinstance(result, TrendList)
    assert result.run_id == ctx.run_id
    assert len(result) == 5
    scores = [t.score for t in result.trends]
    assert scores == sorted(scores, reverse=True)


async def test_scoring_is_deterministic() -> None:
    """The scoring function is pure: same input yields the same score."""
    trend = Trend(
        id="x",
        title="t",
        source="mock",
        velocity=0.8,
        relevance=0.6,
        saturation=0.2,
        shelf_life=0.5,
    )
    assert score_trend(trend) == score_trend(trend)


async def test_higher_velocity_and_relevance_scores_higher() -> None:
    """Relative scoring: stronger positive signals outrank weaker ones."""
    strong = Trend(
        id="strong",
        title="strong",
        source="mock",
        velocity=0.9,
        relevance=0.9,
        saturation=0.2,
        shelf_life=0.5,
    )
    weak = Trend(
        id="weak",
        title="weak",
        source="mock",
        velocity=0.3,
        relevance=0.3,
        saturation=0.2,
        shelf_life=0.5,
    )
    assert score_trend(strong) > score_trend(weak)


async def test_saturation_lowers_score() -> None:
    """A more saturated trend scores lower than an otherwise identical one."""
    fresh = Trend(id="fresh", title="fresh", source="mock", velocity=0.7, saturation=0.1)
    saturated = Trend(id="sat", title="sat", source="mock", velocity=0.7, saturation=0.9)
    assert score_trend(fresh) > score_trend(saturated)


async def test_ordering_matches_relative_signals(ctx: RunContext) -> None:
    """The top-ranked seeded trend is the one with the strongest signals."""
    scout = TrendScout([MockTrendSource()])

    result = await scout.run(ctx, None)

    # t4 (velocity 0.7, relevance 0.9, low saturation, high shelf life) and
    # t1 (velocity 0.9, relevance 0.8) are the two strongest; the crowded,
    # short-lived t3 must not lead.
    top_ids = [t.id for t in result.top(2)]
    assert result.trends[0].id in {"t1", "t4"}
    assert "t3" not in top_ids


async def test_niche_filter_excludes_non_matching() -> None:
    """Niche filters drop trends whose title/source does not match."""
    config = RunConfig(niche_filters=("gaming",))
    ctx = RunContext.new(run_id="niche-run", config=config)
    scout = TrendScout([MockTrendSource()])

    result = await scout.run(ctx, None)

    assert len(result) > 0
    for trend in result.trends:
        assert "gaming" in trend.title.lower()
    # The AI cooking and sustainable fashion trends must be filtered out.
    assert {"t1", "t4"}.isdisjoint({t.id for t in result.trends})


async def test_empty_source_yields_empty_trend_list(ctx: RunContext) -> None:
    """A source that returns nothing yields an empty TrendList, no error."""
    scout = TrendScout([MockTrendSource(trends=())])

    result = await scout.run(ctx, None)

    assert isinstance(result, TrendList)
    assert len(result) == 0


async def test_no_sources_yields_empty_trend_list(ctx: RunContext) -> None:
    """No sources at all yields an empty TrendList without error."""
    scout = TrendScout([])

    result = await scout.run(ctx, None)

    assert isinstance(result, TrendList)
    assert len(result) == 0


async def test_orchestrator_runs_trend_scout_end_to_end() -> None:
    """The Orchestrator stub run() executes the Trend Scout stage."""
    config = RunConfig()
    orchestrator = Orchestrator(config=config, sources=[MockTrendSource()])

    result = await orchestrator.run()

    assert isinstance(result, TrendList)
    assert len(result) == 5
    scores = [t.score for t in result.trends]
    assert scores == sorted(scores, reverse=True)
