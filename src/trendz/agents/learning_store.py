"""Learning / Memory Store: the eighth and final agent in the pipeline (stage 8).

The Learning / Memory Store is DESIGN.md stage 8 - the self-improvement loop that
closes the pipeline. It receives the :class:`~trendz.contracts.PerformanceReports`
the Performance Analyst produced, attributes the measured outcomes back to the
trends and platforms that drove them, and folds them into the persisted
:class:`~trendz.contracts.LearningState`: the UPDATED tunable priors shaped to
feed straight back into the earlier agents (Trend Scout scoring weights, Content
Strategist per-platform format priors, the Quality Gate engagement threshold, and
the Scheduler's per-platform timing slots), plus the multi-armed-bandit
allocation state.

Everything is DETERMINISTIC and OFFLINE, exactly like the rest of the pipeline:

    - The reinforcement update is a fixed-learning-rate step,
      ``new = old + learning_rate * (normalized_reward - old)`` (an
      exponential-moving-average nudge), a pure function of the prior state, the
      reports, and ``learning_rate``. Rewards are squashed into ``[0, 1]`` by a
      fixed, monotone transform (see :func:`_normalize_reward`), so no
      wall-clock, network, or unseeded randomness is involved.
    - The explore/exploit allocation is an epsilon-greedy multi-armed bandit over
      the platforms, driven by a SEEDED :class:`random.Random` whose seed is
      derived deterministically from ``ctx.run_id``, so the same reports + seed
      always yield the same allocation.

Persistence is delegated to an injected :class:`~trendz.store.base.MemoryStore`
(defaulting to the deterministic
:class:`~trendz.store.in_memory.InMemoryMemoryStore`) so a real Postgres/Redis/
vector store can be swapped in without touching this logic. Because the store
persists across runs, applying reports repeatedly against the SAME store nudges
the priors incrementally - the pipeline's 'improves over time' property.

Empty input (no reports) is a NO-OP: the prior state is returned unchanged (same
weights/priors/threshold/slots, version NOT incremented) and nothing is
persisted, so a run with nothing to learn from never perturbs the memory.
"""

from __future__ import annotations

import hashlib
import random

from trendz.agents.base import BaseAgent
from trendz.contracts import (
    BanditArm,
    LearningState,
    PerformanceReports,
    Platform,
    RunContext,
    TrendAggregate,
)
from trendz.store.base import MemoryStore
from trendz.store.in_memory import InMemoryMemoryStore

# The stable namespace the store keys learned state under. Deliberately NOT the
# per-run id: learning must accumulate across runs, so every run reads and
# writes the same namespace against a shared store instance.
DEFAULT_NAMESPACE = "trendz:learning"

# Fixed half-saturation constant for the reward squash. A per-post performance
# score of this value maps to a normalized reward of 0.5; the transform is a
# pure, monotone map into [0, 1) so higher raw scores always yield higher
# normalized rewards without any clipping discontinuity. Chosen to sit in the
# stub metrics-provider's plausible score range so rewards spread across [0, 1).
_REWARD_HALF_SATURATION = 5000.0

# Fixed learning rate for the bandit arm-estimate EMA. Deliberately SEPARATE
# from ``ctx.config.learning_rate`` (which tunes the priors/threshold): the arm
# estimate is the bandit's own value estimate, and keeping its update rate a
# named module constant makes the bandit's convergence rate explicit and
# independent of the prior-tuning knob rather than a hidden literal. 0.5 weights
# each run's observation and the running estimate equally. Change this constant
# (not the config knob) to retune how fast the bandit's arm values move.
_ARM_ESTIMATE_LEARNING_RATE = 0.5


def _normalize_reward(mean_score: float) -> float:
    """Squash a mean performance score into ``[0, 1)`` deterministically.

    Uses the saturating transform ``score / (score + K)`` with a fixed constant
    ``K`` (:data:`_REWARD_HALF_SATURATION`). It is pure, monotone increasing, and
    bounded: 0 maps to 0.0, ``K`` maps to 0.5, and large scores approach (but
    never reach) 1.0. No randomness or clock, so the same score always maps to
    the same reward.
    """
    if mean_score <= 0.0:
        return 0.0
    return mean_score / (mean_score + _REWARD_HALF_SATURATION)


def _ema(old: float, reward: float, learning_rate: float) -> float:
    """Apply one fixed-step reinforcement update: ``old + lr*(reward - old)``.

    The exponential-moving-average nudge the whole stage uses. A pure function of
    its inputs, so updates are deterministic and reproducible. With ``lr`` in
    ``[0, 1]`` and both ``old`` and ``reward`` in ``[0, 1]`` the result stays in
    ``[0, 1]``.
    """
    return old + learning_rate * (reward - old)


def _seed_from_run_id(run_id: str) -> int:
    """Derive a stable non-negative integer bandit seed from the run id.

    A SHA-256 hash of the run id, so the epsilon-greedy allocation is a pure
    function of the run id (no entropy from the OS or clock): the same run id
    always seeds the same :class:`random.Random`, making the explore/exploit
    choice reproducible.
    """
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return int(digest, 16)


class LearningStore(BaseAgent[PerformanceReports, LearningState]):
    """Folds performance reports into the persisted, fed-back tunable priors."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        learning_rate: float | None = None,
        exploration_epsilon: float | None = None,
    ) -> None:
        """Create the agent.

        Args:
            store: The persistence backend (dependency-injected). Defaults to the
                deterministic offline
                :class:`~trendz.store.in_memory.InMemoryMemoryStore`.
            namespace: The key the learned state is persisted under. A stable
                namespace (not the per-run id) so learning accumulates across
                runs.
            learning_rate: Optional override for the reinforcement step. When
                ``None`` the value is read from ``ctx.config.learning_rate`` in
                :meth:`run`.
            exploration_epsilon: Optional override for the bandit exploration
                probability. When ``None`` the value is read from
                ``ctx.config.exploration_epsilon`` in :meth:`run`.
        """
        super().__init__()
        self._store = store if store is not None else InMemoryMemoryStore()
        self._namespace = namespace
        self._learning_rate = learning_rate
        self._exploration_epsilon = exploration_epsilon

    @property
    def name(self) -> str:
        return "learning_store"

    @staticmethod
    def _aggregate_trends(payload: PerformanceReports) -> tuple[TrendAggregate, ...]:
        """Roll the reports up per originating trend via ``by_trend``.

        For each ``(trend_id, reports)`` group (in sorted trend order) counts the
        posts and sums/means their performance scores, recording the distinct
        platforms observed. A pure function of the reports.
        """
        aggregates: list[TrendAggregate] = []
        for trend_id, reports in payload.by_trend():
            total = sum(report.performance_score for report in reports)
            count = len(reports)
            mean = total / count if count else 0.0
            platforms = tuple(sorted({report.platform for report in reports}))
            aggregates.append(
                TrendAggregate(
                    trend_id=trend_id,
                    post_count=count,
                    total_performance_score=total,
                    mean_performance_score=mean,
                    platforms=platforms,
                )
            )
        return tuple(aggregates)

    @staticmethod
    def _platform_mean_scores(payload: PerformanceReports) -> dict[Platform, float]:
        """Return the mean performance score per platform, in sorted platform order.

        The per-platform reward signal the format priors, bandit arms, and timing
        hints are reinforced from. A pure function of the reports.
        """
        means: dict[Platform, float] = {}
        for platform in payload.platforms:
            reports = payload.for_platform(platform)
            total = sum(report.performance_score for report in reports)
            means[platform] = total / len(reports) if reports else 0.0
        return means

    def _update_format_priors(
        self,
        prior: dict[Platform, float],
        platform_rewards: dict[Platform, float],
        learning_rate: float,
    ) -> dict[Platform, float]:
        """EMA-update the per-platform format priors toward the observed rewards.

        Platforms with observed reports are nudged toward their normalized
        reward; platforms with no observation this run keep their prior value.
        Keys are the union of the prior's platforms and the observed platforms,
        emitted in sorted order for determinism.
        """
        platforms = sorted(set(prior) | set(platform_rewards))
        updated: dict[Platform, float] = {}
        for platform in platforms:
            old = prior.get(platform, 0.0)
            if platform in platform_rewards:
                reward = _normalize_reward(platform_rewards[platform])
                updated[platform] = _ema(old, reward, learning_rate)
            else:
                updated[platform] = old
        return updated

    def _update_timing_slots(
        self,
        prior: dict[Platform, tuple[int, ...]],
        platform_rewards: dict[Platform, float],
        learning_rate: float,
    ) -> dict[Platform, tuple[int, ...]]:
        """Deterministically re-rank each platform's timing slots from its reward.

        The audience-timing provider treats a platform's slot tuple as its
        posting windows in PREFERENCE order (the first slot is the single best
        window). We genuinely learn "best post times" (DESIGN.md stage 8) by
        REORDERING each observed platform's slots so that a stronger measured
        reward promotes a later-listed window to the front - the higher the
        normalized reward, the further into the tuple the preferred slot is
        drawn, then rotated to lead. This is a pure, deterministic function of
        the observed per-platform reward (no clock, network, or randomness): the
        same reward always yields the same ordering, and a different reward
        yields a different lead slot, so the hints genuinely respond to
        performance rather than being a static pass-through.

        The ``learning_rate`` gates responsiveness: a rate of 0 leaves the slots
        untouched (priors never move), matching the reinforcement-update
        semantics used elsewhere. Platforms with no observation this run, and
        slot tuples too short to reorder (fewer than two slots), keep their prior
        ordering unchanged. Keys are the union of the prior's platforms and the
        observed platforms, emitted in sorted order for determinism.
        """
        platforms = sorted(set(prior) | set(platform_rewards))
        updated: dict[Platform, tuple[int, ...]] = {}
        for platform in platforms:
            slots = prior.get(platform, ())
            if learning_rate <= 0.0 or platform not in platform_rewards or len(slots) < 2:
                updated[platform] = slots
                continue
            reward = _normalize_reward(platform_rewards[platform])
            # Map the normalized reward in [0, 1) to a slot index in
            # [0, len(slots)); higher reward promotes a later window to lead.
            # int() floors, and reward < 1.0 keeps the index in range.
            lead = int(reward * len(slots))
            updated[platform] = slots[lead:] + slots[:lead]
        return updated

    def _update_scoring_weights(
        self,
        prior: tuple[float, float, float, float],
        overall_reward: float,
        learning_rate: float,
    ) -> tuple[float, float, float, float]:
        """Nudge the four scoring weights toward the observed overall reward.

        The velocity/relevance/shelf_life weights (the positive contributions)
        are EMA-nudged toward the normalized overall reward, rewarding the
        signals that drove measured performance; the saturation weight (a
        penalty) is nudged toward the complementary ``1 - reward`` so a
        low-reward run tightens the saturation penalty. A pure function of the
        inputs, so the update is deterministic.
        """
        velocity, relevance, shelf_life, saturation = prior
        return (
            _ema(velocity, overall_reward, learning_rate),
            _ema(relevance, overall_reward, learning_rate),
            _ema(shelf_life, overall_reward, learning_rate),
            _ema(saturation, 1.0 - overall_reward, learning_rate),
        )

    def _run_bandit(
        self,
        prior_arms: tuple[BanditArm, ...],
        platform_rewards: dict[Platform, float],
        seed: int,
        epsilon: float,
    ) -> tuple[BanditArm, ...]:
        """Run one epsilon-greedy allocation step and update the arms.

        The arm set is the union of the prior arms and the observed platforms
        (sorted for determinism). Each arm's ``estimated_reward`` is first
        EMA-updated toward its normalized observed reward (arms with no
        observation this run keep their estimate). A SEEDED
        :class:`random.Random` then either explores (with probability
        ``epsilon``, picking a uniformly random arm) or exploits (picks the
        highest-estimate arm, ties broken by sorted platform), and the chosen
        arm's ``pulls`` is incremented. Reproducible given the same seed and
        rewards.
        """
        platforms: list[Platform] = sorted(
            {arm.platform for arm in prior_arms} | set(platform_rewards)
        )
        prior_by_platform = {arm.platform: arm for arm in prior_arms}

        # First fold this run's observed reward into each arm's estimate.
        estimates: dict[Platform, float] = {}
        pulls: dict[Platform, int] = {}
        for platform in platforms:
            existing = prior_by_platform.get(platform)
            old_estimate = existing.estimated_reward if existing else 0.0
            pulls[platform] = existing.pulls if existing else 0
            if platform in platform_rewards:
                reward = _normalize_reward(platform_rewards[platform])
                # A fixed, named estimate learning rate keeps the arm update
                # deterministic and explicitly independent of the prior config
                # knob (see _ARM_ESTIMATE_LEARNING_RATE).
                estimates[platform] = _ema(old_estimate, reward, _ARM_ESTIMATE_LEARNING_RATE)
            else:
                estimates[platform] = old_estimate

        # Epsilon-greedy choice via a seeded RNG (only when there is an arm to
        # pull); explore with probability epsilon, else exploit the best arm.
        if platforms:
            rng = random.Random(seed)
            if rng.random() < epsilon:
                chosen = rng.choice(platforms)
            else:
                chosen = max(platforms, key=lambda p: (estimates[p], -platforms.index(p)))
            pulls[chosen] += 1

        return tuple(
            BanditArm(
                platform=platform,
                pulls=pulls[platform],
                estimated_reward=estimates[platform],
            )
            for platform in platforms
        )

    async def run(self, ctx: RunContext, payload: PerformanceReports) -> LearningState:
        """Fold the reports into the persisted learning state and return it.

        Loads the prior state from the store, aggregates the reports per trend
        and per platform, applies the deterministic fixed-learning-rate
        reinforcement update to the scoring weights, format priors, and engagement
        threshold, deterministically re-ranks each platform's timing slots from
        its observed reward, runs the seeded epsilon-greedy bandit
        allocation over the platform arms, then persists and returns the new
        state with an incremented version.

        Empty input (no reports) is a NO-OP: the prior state is returned
        unchanged and nothing is persisted.
        """
        prior = self._store.load(self._namespace)

        if len(payload) == 0:
            self.log.info(
                "learning_noop",
                reports=0,
                version=prior.version,
                store=self._store.name,
            )
            return prior

        learning_rate = (
            self._learning_rate if self._learning_rate is not None else ctx.config.learning_rate
        )
        epsilon = (
            self._exploration_epsilon
            if self._exploration_epsilon is not None
            else ctx.config.exploration_epsilon
        )

        trend_aggregates = self._aggregate_trends(payload)
        platform_rewards = self._platform_mean_scores(payload)

        overall_mean = (
            sum(report.performance_score for report in payload.reports) / len(payload)
            if len(payload)
            else 0.0
        )
        overall_reward = _normalize_reward(overall_mean)

        scoring_weights = self._update_scoring_weights(
            prior.scoring_weights, overall_reward, learning_rate
        )
        format_priors = self._update_format_priors(
            dict(prior.format_priors), platform_rewards, learning_rate
        )
        engagement_threshold = _ema(prior.engagement_threshold, overall_reward, learning_rate)
        timing_slots = self._update_timing_slots(
            dict(prior.timing_slots), platform_rewards, learning_rate
        )
        arms = self._run_bandit(
            prior.arms, platform_rewards, _seed_from_run_id(ctx.run_id), epsilon
        )

        state = LearningState(
            run_id=ctx.run_id,
            version=prior.version + 1,
            updates_applied=prior.updates_applied + 1,
            scoring_weights=scoring_weights,
            format_priors=format_priors,
            engagement_threshold=engagement_threshold,
            timing_slots=timing_slots,
            arms=arms,
            trend_aggregates=trend_aggregates,
        )
        self._store.save(self._namespace, state)

        self.log.info(
            "learning_done",
            reports=len(payload),
            trends=len(trend_aggregates),
            platforms=len(platform_rewards),
            version=state.version,
            store=self._store.name,
        )
        return state
