"""Offline tests for the Learning / Memory Store agent (stage 8).

Behavioural, fully-offline tests in the style of ``test_performance_analyst``:
they build :class:`PerformanceReports` inputs from small factory helpers, run the
agent against the deterministic in-memory store, and assert the reinforcement
update, seeded bandit allocation, aggregation, feedback-path shape, empty-input
no-op, overall determinism, and default-store wiring.
"""

from __future__ import annotations

from datetime import datetime, timezone

from trendz.agents.content_strategist import DEFAULT_STRATEGY
from trendz.agents.learning_store import (
    LearningStore,
    _ema,
    _normalize_reward,
    _seed_from_run_id,
)
from trendz.agents.trend_scout import DEFAULT_WEIGHTS, ScoringWeights, score_trend
from trendz.contracts import (
    LearningState,
    PerformanceReports,
    Platform,
    PostMetrics,
    PostPerformance,
    RunContext,
    Trend,
)
from trendz.store.in_memory import (
    DEFAULT_ENGAGEMENT_THRESHOLD,
    InMemoryMemoryStore,
    default_learning_state,
)

_BASE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

# A fixed metrics blob: the tests drive behaviour off performance_score, not the
# raw metrics, so a single constant snapshot keeps the factories terse.
_METRICS = PostMetrics(
    views=1000, watch_time_seconds=7000.0, likes=100, shares=20, comments=10, follows=5
)


def _report(
    idx: int,
    trend_id: str,
    platform: Platform,
    score: float,
    variant: int = 0,
) -> PostPerformance:
    """Build a deterministic PostPerformance with a chosen performance_score."""
    return PostPerformance(
        clip_id=f"clip-{idx}",
        brief_id=f"b{idx}",
        trend_id=trend_id,
        platform=platform,
        variant=variant,
        post_id=f"post-{idx}",
        metrics=_METRICS,
        performance_score=score,
        collected_at=_BASE,
    )


def _reports(run_id: str, reports: tuple[PostPerformance, ...]) -> PerformanceReports:
    """Build a PerformanceReports from performance records."""
    return PerformanceReports(run_id=run_id, reports=reports)


def _sample(run_id: str) -> PerformanceReports:
    """A small mixed-platform, two-trend PerformanceReports used across tests."""
    return _reports(
        run_id,
        (
            _report(0, "t0", "tiktok", 3000.0),
            _report(1, "t1", "instagram", 6000.0),
            _report(2, "t0", "tiktok", 3000.0),
        ),
    )


async def test_deterministic_prior_updates_from_known_input(ctx: RunContext) -> None:
    """(a) The reinforcement update produces the exact EMA values given fixed lr."""
    payload = _sample(ctx.run_id)
    store = InMemoryMemoryStore()
    lr = ctx.config.learning_rate  # default 0.2

    state = await LearningStore(store).run(ctx, payload)

    # Overall mean score across the three posts, then its normalized reward.
    overall_mean = (3000.0 + 6000.0 + 3000.0) / 3
    overall_reward = _normalize_reward(overall_mean)

    # Scoring weights: the three positive weights nudge toward the reward, the
    # saturation penalty nudges toward the complement.
    v = DEFAULT_WEIGHTS.velocity
    r = DEFAULT_WEIGHTS.relevance
    s = DEFAULT_WEIGHTS.shelf_life
    sat = DEFAULT_WEIGHTS.saturation
    assert state.scoring_weights[0] == _ema(v, overall_reward, lr)
    assert state.scoring_weights[1] == _ema(r, overall_reward, lr)
    assert state.scoring_weights[2] == _ema(s, overall_reward, lr)
    assert state.scoring_weights[3] == _ema(sat, 1.0 - overall_reward, lr)

    # Format priors: per-platform mean reward, EMA'd from the 0.0 default.
    tiktok_reward = _normalize_reward((3000.0 + 3000.0) / 2)
    instagram_reward = _normalize_reward(6000.0)
    assert state.format_priors["tiktok"] == _ema(0.0, tiktok_reward, lr)
    assert state.format_priors["instagram"] == _ema(0.0, instagram_reward, lr)
    # Unobserved platforms keep their prior (0.0 default).
    assert state.format_priors["facebook"] == 0.0
    assert state.format_priors["youtube_shorts"] == 0.0

    # Engagement threshold EMA'd from the default toward the overall reward.
    assert state.engagement_threshold == _ema(DEFAULT_ENGAGEMENT_THRESHOLD, overall_reward, lr)

    assert state.version == 1
    assert state.updates_applied == 1


async def test_incremental_learning_moves_priors_further(ctx: RunContext) -> None:
    """(b) Two applications against the same store move a prior strictly further."""
    payload = _sample(ctx.run_id)
    store = InMemoryMemoryStore()
    agent = LearningStore(store)

    first = await agent.run(ctx, payload)
    second = await agent.run(ctx, payload)

    # instagram's reward target it is converging toward.
    target = _normalize_reward(6000.0)
    after_one = first.format_priors["instagram"]
    after_two = second.format_priors["instagram"]

    # Both below the target (starting from 0.0), and the second is strictly
    # closer to the target than the first: it 'improves over time'.
    assert 0.0 < after_one < after_two < target
    assert abs(target - after_two) < abs(target - after_one)

    # The version / update counter advances with each application.
    assert first.version == 1
    assert second.version == 2
    assert second.updates_applied == 2


async def test_bandit_allocation_is_deterministic_under_seed(ctx: RunContext) -> None:
    """(c) Two runs with the same seed produce identical arm allocation/pulls."""
    payload = _sample(ctx.run_id)

    first = await LearningStore(InMemoryMemoryStore()).run(ctx, payload)
    second = await LearningStore(InMemoryMemoryStore()).run(ctx, payload)

    assert first.arms == second.arms
    # Exactly one pull is allocated per run (one epsilon-greedy choice).
    assert sum(arm.pulls for arm in first.arms) == 1
    # The seed is a pure function of the run id.
    assert _seed_from_run_id(ctx.run_id) == _seed_from_run_id(ctx.run_id)


async def test_aggregation_matches_by_trend_and_per_platform(ctx: RunContext) -> None:
    """(d) Per-trend and per-platform aggregation match the input counts/totals."""
    payload = _sample(ctx.run_id)

    state = await LearningStore(InMemoryMemoryStore()).run(ctx, payload)

    by_trend = {agg.trend_id: agg for agg in state.trend_aggregates}
    assert set(by_trend) == {"t0", "t1"}
    # t0: two tiktok posts of 3000 each.
    assert by_trend["t0"].post_count == 2
    assert by_trend["t0"].total_performance_score == 6000.0
    assert by_trend["t0"].mean_performance_score == 3000.0
    assert by_trend["t0"].platforms == ("tiktok",)
    # t1: one instagram post of 6000.
    assert by_trend["t1"].post_count == 1
    assert by_trend["t1"].total_performance_score == 6000.0
    assert by_trend["t1"].mean_performance_score == 6000.0
    assert by_trend["t1"].platforms == ("instagram",)

    # Aggregates match by_trend() exactly (counts + totals + trend order).
    expected = payload.by_trend()
    assert [agg.trend_id for agg in state.trend_aggregates] == [tid for tid, _ in expected]
    for agg, (_, group) in zip(state.trend_aggregates, expected, strict=True):
        assert agg.post_count == len(group)
        assert agg.total_performance_score == sum(r.performance_score for r in group)


async def test_feedback_path_is_usable_as_scoring_weights(ctx: RunContext) -> None:
    """(e) The learned state rebuilds a usable Trend Scout ScoringWeights + timing."""
    payload = _sample(ctx.run_id)

    state = await LearningStore(InMemoryMemoryStore()).run(ctx, payload)

    weights = state.as_scoring_weights()
    assert isinstance(weights, ScoringWeights)
    assert weights.velocity == state.scoring_weights[0]
    assert weights.saturation == state.scoring_weights[3]
    # The rebuilt weights are usable in the real scoring function.
    trend = Trend(id="x", title="t", source="mock", velocity=0.8, relevance=0.6)
    assert isinstance(score_trend(trend, weights), float)

    # And the timing slots come back in the audience-timing provider's shape.
    timing_slots = state.as_timing_slots()
    for platform in DEFAULT_STRATEGY.platforms:
        assert isinstance(timing_slots[platform], tuple)
        assert all(isinstance(m, int) for m in timing_slots[platform])


async def test_empty_input_is_a_noop(ctx: RunContext) -> None:
    """(f) Empty reports leave the priors unchanged and do not persist a mutation."""
    store = InMemoryMemoryStore()
    agent = LearningStore(store)

    result = await agent.run(ctx, PerformanceReports(run_id=ctx.run_id))

    # The result is exactly the default/initial state (version 0, no updates).
    assert result == default_learning_state(agent._namespace)
    assert result.version == 0
    assert result.updates_applied == 0
    assert result.trend_aggregates == ()
    # Nothing was persisted: a subsequent load still returns the default.
    assert store.load(agent._namespace) == default_learning_state(agent._namespace)


async def test_empty_input_after_learning_does_not_perturb(ctx: RunContext) -> None:
    """(f) An empty run after a real update returns the prior state unchanged."""
    store = InMemoryMemoryStore()
    agent = LearningStore(store)

    learned = await agent.run(ctx, _sample(ctx.run_id))
    after_empty = await agent.run(ctx, PerformanceReports(run_id=ctx.run_id))

    assert after_empty == learned
    assert after_empty.version == learned.version  # not incremented


async def test_full_run_is_deterministic(ctx: RunContext) -> None:
    """(g) Two full runs over equal input + fresh equal stores produce equal state."""
    payload = _sample(ctx.run_id)

    first = await LearningStore(InMemoryMemoryStore()).run(ctx, payload)
    second = await LearningStore(InMemoryMemoryStore()).run(ctx, payload)

    assert first == second
    assert isinstance(first, LearningState)


async def test_default_store_is_in_memory_stub() -> None:
    """(h) The agent defaults to the deterministic in-memory store."""
    agent = LearningStore()
    assert isinstance(agent._store, InMemoryMemoryStore)


async def test_epsilon_zero_is_pure_exploitation(ctx: RunContext) -> None:
    """With epsilon=0 the bandit always pulls the highest-estimate arm."""
    payload = _sample(ctx.run_id)
    agent = LearningStore(InMemoryMemoryStore(), exploration_epsilon=0.0)

    state = await agent.run(ctx, payload)

    pulled = [arm for arm in state.arms if arm.pulls > 0]
    assert len(pulled) == 1
    best = max(state.arms, key=lambda a: a.estimated_reward)
    # instagram has the higher observed reward, so exploitation pulls it.
    assert pulled[0].platform == best.platform
    assert best.platform == "instagram"
