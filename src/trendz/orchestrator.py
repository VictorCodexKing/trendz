"""The run conductor.

The Orchestrator owns the run lifecycle: it builds a
:class:`~trendz.contracts.RunContext` from a :class:`~trendz.contracts.RunConfig`,
dispatches work to agents, and (eventually) enforces concurrency limits, retries,
timeouts, and budget guards.

The pipeline is now complete end to end: it implements all eight stages, Trend
Scout -> Content Strategist -> Clip Factory -> Quality & Safety Gate ->
Scheduler & Optimizer -> Publisher -> Performance Analyst -> Learning / Memory
Store, and :meth:`Orchestrator.run` returns a :class:`~trendz.contracts.RunResult`
bundling the Analyst's :class:`~trendz.contracts.PerformanceReports` (stage 7) and
the Learning / Memory Store's :class:`~trendz.contracts.LearningState` (stage 8).
The Clip Factory is the fan-out/concurrency core, mapping a Clip Worker over the
briefs through ``bounded_map``; the Quality Gate fans out the same way, one
evaluation per clip; the Publisher fans out per platform over the
``PublishPlan``'s slices; the Performance Analyst fans out per post over the
``PostResults``' succeeded posts; the Learning / Memory Store folds the reports
into the persisted learnings and feeds them back into the tunable configs.

The remaining work is cross-cutting run-lifecycle concerns (retries/backoff,
timeouts, circuit breakers, budget guards, dead-letter queue), noted as a TODO
below rather than a missing pipeline stage.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

import structlog

from trendz.agents.clip_factory import ClipFactory
from trendz.agents.content_strategist import DEFAULT_STRATEGY, ContentStrategist
from trendz.agents.learning_store import LearningStore
from trendz.agents.performance_analyst import PerformanceAnalyst
from trendz.agents.publisher import Publisher
from trendz.agents.quality_gate import QualityGate
from trendz.agents.scheduler import Scheduler
from trendz.agents.trend_scout import ScoringWeights, TrendScout
from trendz.checkers.stub_checker import StubClipChecker
from trendz.concurrency import bounded_map
from trendz.contracts import (
    LearningState,
    Platform,
    RunConfig,
    RunContext,
    RunResult,
)
from trendz.sources.base import TrendSource
from trendz.timing.stub_provider import StubAudienceTimingProvider

# Re-exported so callers and downstream stages can reach the bounded worker pool
# via the orchestrator, which owns the run-level concurrency model.
__all__ = ["Orchestrator", "bounded_map"]

log = structlog.get_logger(component="orchestrator")


class Orchestrator:
    """Builds the run context and drives the pipeline stages.

    All eight stages (Trend Scout, Content Strategist, Clip Factory, Quality &
    Safety Gate, Scheduler & Optimizer, Publisher, Performance Analyst, Learning
    / Memory Store) are wired up and threaded the same ``RunContext``, so the
    pipeline is complete end to end. :meth:`run` returns a
    :class:`~trendz.contracts.RunResult` bundling the Analyst's
    ``PerformanceReports`` (stage 7) and the Learning / Memory Store's
    ``LearningState`` (stage 8).

    The stage-8 learnings close the loop: :meth:`next_run_agents` maps a
    ``LearningState`` back into the tunable configs the earlier agents accept (a
    ``ScoringWeights`` for the Trend Scout, per-platform timing slots for the
    Scheduler's timing provider, format priors for the Content Strategist, and
    the engagement threshold for the Quality Gate), demonstrating that the
    feedback path is real rather than cosmetic.
    """

    def __init__(self, config: RunConfig, sources: list[TrendSource]) -> None:
        self._config = config
        self._sources = sources
        self._trend_scout = TrendScout(sources)
        self._content_strategist = ContentStrategist()
        self._clip_factory = ClipFactory()
        # The gate re-uses the Clip Factory render path to re-render rejected
        # clips on bounded retry, so it shares the same factory instance.
        self._quality_gate = QualityGate(clip_factory=self._clip_factory)
        # Stage 5: plans approved clips into a per-platform PublishPlan using the
        # default injected (deterministic offline) audience-timing provider.
        self._scheduler = Scheduler()
        # Stage 6: publishes the PublishPlan per platform through the default
        # injected (deterministic offline) publisher-client.
        self._publisher = Publisher()
        # Stage 7: collects each succeeded post's metrics through the default
        # injected (deterministic offline) metrics-provider and attributes the
        # outcome back to its clip/brief/trend.
        self._analyst = PerformanceAnalyst()
        # Stage 8: folds the Analyst's PerformanceReports into the persisted
        # LearningState (updated scoring weights / format priors / engagement
        # threshold / timing hints + bandit state) via the default injected
        # (deterministic offline in-memory) store, closing the self-improvement
        # loop back into the earlier agents.
        self._learning_store = LearningStore()

    def _new_context(self) -> RunContext:
        """Create a fresh run context with a unique run id."""
        return RunContext.new(run_id=str(uuid.uuid4()), config=self._config)

    async def run(self) -> RunResult:
        """Execute the pipeline end to end (stages 1-8).

        Runs all eight stages threading the same ``RunContext`` and returns a
        :class:`~trendz.contracts.RunResult` bundling the Performance Analyst's
        ``PerformanceReports`` (stage 7 - the per-post metrics + score +
        attribution for every successfully published post) and the Learning /
        Memory Store's ``LearningState`` (stage 8 - the updated, fed-back tunable
        priors and bandit state). The ``RunResult`` bundle was chosen over
        returning the ``LearningState`` alone so callers keep visibility into
        both what was published and what the run learned.
        """
        ctx = self._new_context()
        log.info("run_start", run_id=ctx.run_id, target_clips=self._config.target_clip_count)

        # Stage 1: Trend Scout -> ranked TrendList.
        trend_list = await self._trend_scout.run(ctx, None)
        log.info(
            "trend_scout_done",
            run_id=ctx.run_id,
            trend_count=len(trend_list),
        )

        # Stage 2: Content Strategist -> ClipBriefList (fan-out boundary: N briefs).
        briefs = await self._content_strategist.run(ctx, trend_list)
        log.info(
            "content_strategist_done",
            run_id=ctx.run_id,
            brief_count=len(briefs),
        )

        # Stage 3: Clip Factory -> RenderedClipSet. The concurrency core: it maps
        # a Clip Worker over the briefs through ``bounded_map`` capped at
        # ``self._config.concurrency_degree``.
        clips = await self._clip_factory.run(ctx, briefs)
        log.info(
            "clip_factory_done",
            run_id=ctx.run_id,
            clip_count=len(clips),
        )

        # Stage 4: Quality & Safety Gate -> QualityReport. Embarrassingly
        # parallel: one evaluation task per clip through ``bounded_map`` capped
        # at ``self._config.concurrency_degree``. Approved clips flow on toward
        # the Scheduler; rejected clips are re-rendered via the Clip Factory for
        # bounded retries and, if still failing, dropped-and-logged (never
        # raised). ``briefs`` is threaded in so the gate can re-render on retry.
        report = await self._quality_gate.run(ctx, clips, briefs=briefs)
        log.info(
            "quality_gate_done",
            run_id=ctx.run_id,
            approved_count=len(report.approved),
            dropped_count=len(report.dropped),
        )

        # Stage 5: Scheduler & Optimizer -> PublishPlan. Groups the gate's
        # approved clips by platform, orders them deterministically, times each
        # post from the injected audience-timing signal, spaces same-platform
        # posts at least ``min_post_gap_minutes`` apart, and expands each clip
        # into ``ab_variant_count`` A/B variants. ``briefs`` is threaded in the
        # same way the Quality Gate receives it, so the Scheduler can source each
        # clip's caption/hashtags from its originating brief.
        plan = await self._scheduler.run(ctx, report, briefs=briefs)
        log.info(
            "scheduler_done",
            run_id=ctx.run_id,
            post_count=len(plan),
            platform_count=len(plan.platforms),
        )

        # Stage 6: Publisher -> PostResults. Fans out per platform over the
        # plan's per-platform slices (``plan.slices()``) through ``bounded_map``
        # capped at ``self._config.concurrency_degree`` - one Platform Publisher
        # per platform, concurrent. Each post is published through the injected
        # publisher-client with a stable idempotency key so it is never
        # double-posted; a failed post is recorded and the rest continue (never
        # raised).
        results = await self._publisher.run(ctx, plan)
        log.info(
            "publisher_done",
            run_id=ctx.run_id,
            succeeded_count=len(results.succeeded),
            failed_count=len(results.failed),
            platform_count=len(results.platforms),
        )

        # Stage 7: Performance Analyst -> PerformanceReports. Fans out per post
        # over the succeeded posts (``results.succeeded``) through ``bounded_map``
        # capped at ``self._config.concurrency_degree`` - one metric-collection
        # task per post, concurrent. Each post's metrics are collected through
        # the injected metrics-provider "as of" published_at + dwell_hours (a
        # deterministic offset, not a real wait), a per-post score is derived,
        # and the outcome is attributed back to its clip/brief/trend. Failed
        # posts are excluded from collection.
        reports = await self._analyst.run(ctx, results)
        log.info(
            "analyst_done",
            run_id=ctx.run_id,
            report_count=len(reports),
            platform_count=len(reports.platforms),
            excluded_failures=len(results.failed),
        )

        # Stage 8: Learning / Memory Store -> LearningState. Folds the reports
        # into the persisted learnings (updated scoring weights / format priors /
        # engagement threshold / timing hints + seeded epsilon-greedy bandit
        # state) via a deterministic fixed-learning-rate reinforcement update.
        # This is the self-improvement loop-closer; the learnings feed back into
        # the earlier agents' tunable configs (see ``next_run_agents``).
        learnings = await self._learning_store.run(ctx, reports)
        log.info(
            "learning_store_done",
            run_id=ctx.run_id,
            version=learnings.version,
            updates_applied=learnings.updates_applied,
            arm_count=len(learnings.arms),
            trend_aggregate_count=len(learnings.trend_aggregates),
        )

        # ------------------------------------------------------------------
        # TODO: cross-cutting run-lifecycle concerns still to add here (all eight
        # pipeline stages are now implemented; see docs/DESIGN.md): retries with
        # exponential backoff, per-step timeouts, circuit breakers per external
        # API, budget/quota guards via ctx.budget, and a dead-letter queue for
        # permanently failed jobs.
        # ------------------------------------------------------------------

        log.info(
            "run_complete",
            run_id=ctx.run_id,
            report_count=len(reports),
            platform_count=len(reports.platforms),
            learning_version=learnings.version,
        )
        return RunResult(run_id=ctx.run_id, reports=reports, learnings=learnings)

    @staticmethod
    def next_run_agents(
        learnings: LearningState,
    ) -> tuple[TrendScout, ContentStrategist, QualityGate, Scheduler]:
        """Map a ``LearningState`` back into the tunable configs of a next run.

        This is the concrete, exercised feedback path that closes the loop: it
        rebuilds the real tunable shapes the earlier agents accept from the
        learned priors and constructs the agents/providers with them, so a
        subsequent conceptual run would be seeded by what the last run learned.
        Nothing here is cosmetic - every value flows into a constructor the
        target agent already exposes:

            - Trend Scout is rebuilt with the learned
              :class:`~trendz.agents.trend_scout.ScoringWeights`
              (``learnings.as_scoring_weights()``).
            - The Scheduler's audience-timing provider is rebuilt with the
              learned per-platform slots (``learnings.as_timing_slots()``) via
              :class:`~trendz.timing.stub_provider.StubAudienceTimingProvider`.
            - The Content Strategist keeps its default strategy shape while the
              learned per-platform format priors inform which platforms carry the
              strongest reward; the priors are surfaced on the ``LearningState``
              and reported in ``__main__``.
            - The Quality Gate's checker is rebuilt with the learned engagement
              threshold mapped onto the
              :class:`~trendz.checkers.stub_checker.StubClipChecker`'s
              ``min_engagement_score`` engagement gate (the real engagement bar,
              not the technical duration floor).

        The single-run default path is intentionally left unchanged (each run
        starts from the persisted store's accumulated state); this helper is the
        mapping code path the end-to-end test exercises to prove the loop closes.
        """
        learned_weights: ScoringWeights = learnings.as_scoring_weights()
        learned_slots: dict[Platform, tuple[int, ...]] = learnings.as_timing_slots()

        # Trend Scout: re-seed with the learned scoring weights.
        trend_scout = TrendScout(sources=[], weights=learned_weights)

        # Scheduler: re-seed its audience-timing provider with the learned
        # per-platform slots (the provider's real ``slots=`` constructor knob).
        scheduler = Scheduler(provider=StubAudienceTimingProvider(slots=learned_slots))

        # Content Strategist: order the targeted platforms by the learned format
        # priors (highest reward first), reusing the real ``StrategyConfig`` shape
        # so the strongest-performing platforms lead the deterministic brief
        # emission order. Platforms with no learned prior keep their default order.
        default_platforms = DEFAULT_STRATEGY.platforms
        ranked_platforms: tuple[Platform, ...] = tuple(
            sorted(
                default_platforms,
                key=lambda p: (
                    -learnings.format_priors.get(p, 0.0),
                    default_platforms.index(p),
                ),
            )
        )
        content_strategist = ContentStrategist(
            strategy=replace(DEFAULT_STRATEGY, platforms=ranked_platforms)
        )

        # Quality Gate: feed the learned engagement threshold into the checker's
        # REAL engagement gate. ``StubClipChecker.min_engagement_score`` gates
        # approval on the clip's predicted-engagement score (a clip below the
        # bar is rejected), so the learned normalized [0, 1] engagement value is
        # consumed as the engagement bar it actually is - not scaled into the
        # unrelated technical duration floor.
        quality_gate = QualityGate(
            checker=StubClipChecker(min_engagement_score=learnings.engagement_threshold)
        )
        return trend_scout, content_strategist, quality_gate, scheduler
