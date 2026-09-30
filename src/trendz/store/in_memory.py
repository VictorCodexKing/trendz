"""A deterministic, in-process Learning / Memory Store for offline development and tests.

:class:`InMemoryMemoryStore` implements :class:`~trendz.store.base.MemoryStore`
by holding each namespace's :class:`~trendz.contracts.LearningState` in an
instance dict, with no clock, network, API keys, or randomness - mirroring
:class:`~trendz.analytics.stub_provider.StubMetricsProvider` in style and intent.

Because the state lives on the instance, applying reports across MULTIPLE agent
runs against the SAME store instance accumulates the learned priors: this is
what makes the pipeline's 'improves over time' property demonstrable offline.
:meth:`load` on an unseen namespace returns the INITIAL/default state built from
the existing agents' tunable defaults (:func:`default_learning_state`), so the
agent always has a well-defined prior to reinforce. Swap it for a real
Postgres/Redis/vector-store adapter in production.
"""

from __future__ import annotations

from trendz.agents.content_strategist import DEFAULT_STRATEGY
from trendz.agents.trend_scout import DEFAULT_WEIGHTS
from trendz.contracts import BanditArm, LearningState, Platform, RunConfig, RunContext
from trendz.store.base import MemoryStore
from trendz.timing.stub_provider import StubAudienceTimingProvider

# The default Quality Gate engagement threshold the learned state starts from.
# Kept here (not invented on RunConfig) as the stage-8 seed value the
# reinforcement update nudges from measured engagement; a real Quality Gate can
# read the tuned value back off the LearningState.
DEFAULT_ENGAGEMENT_THRESHOLD = 0.5


def default_learning_state(run_id: str) -> LearningState:
    """Build the initial/default learning state from the existing agent defaults.

    The default state reuses the real tunable shapes so the learned priors feed
    straight back: Trend Scout's :data:`~trendz.agents.trend_scout.DEFAULT_WEIGHTS`
    (as the four-float vector), the Content Strategist's configured platforms as
    zeroed per-platform format priors, the default engagement threshold, the
    audience-timing provider's default per-platform slots, and one zeroed bandit
    arm per platform. Version and update counter both start at 0.
    """
    platforms: tuple[Platform, ...] = DEFAULT_STRATEGY.platforms
    timing = StubAudienceTimingProvider()
    # Reuse the timing provider's default slots so the learned timing hints share
    # the Scheduler's tuple[int, ...] slot shape from the outset.
    ctx = RunContext.new(run_id=run_id, config=RunConfig())
    timing_slots: dict[Platform, tuple[int, ...]] = {
        platform: timing.optimal_minutes_of_day(ctx, platform) for platform in platforms
    }
    return LearningState(
        run_id=run_id,
        version=0,
        updates_applied=0,
        scoring_weights=(
            DEFAULT_WEIGHTS.velocity,
            DEFAULT_WEIGHTS.relevance,
            DEFAULT_WEIGHTS.shelf_life,
            DEFAULT_WEIGHTS.saturation,
        ),
        format_priors={platform: 0.0 for platform in platforms},
        engagement_threshold=DEFAULT_ENGAGEMENT_THRESHOLD,
        timing_slots=timing_slots,
        arms=tuple(
            BanditArm(platform=platform, pulls=0, estimated_reward=0.0) for platform in platforms
        ),
        trend_aggregates=(),
    )


class InMemoryMemoryStore(MemoryStore):
    """A Learning / Memory Store that persists state in-process, deterministically."""

    def __init__(self) -> None:
        """Create an empty store (no namespaces persisted yet)."""
        self._states: dict[str, LearningState] = {}

    @property
    def name(self) -> str:
        return "in_memory_memory_store"

    def load(self, namespace: str) -> LearningState:
        """Return the persisted state for ``namespace`` or the default when unseen.

        A pure lookup: no clock, network, or randomness. An unseen namespace
        yields :func:`default_learning_state` (using ``namespace`` as the state's
        run id) so the agent always has a well-defined prior to reinforce.
        """
        state = self._states.get(namespace)
        if state is None:
            return default_learning_state(namespace)
        return state

    def save(self, namespace: str, state: LearningState) -> None:
        """Persist ``state`` for ``namespace`` in-process.

        The stored state is what a subsequent :meth:`load` of the same namespace
        returns, so applying reports repeatedly against the same store instance
        accumulates the learned priors across runs.
        """
        self._states[namespace] = state
