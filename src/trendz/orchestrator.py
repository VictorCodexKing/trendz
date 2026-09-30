"""The run conductor.

The Orchestrator owns the run lifecycle: it builds a
:class:`~trendz.contracts.RunContext` from a :class:`~trendz.contracts.RunConfig`,
dispatches work to agents, and (eventually) enforces concurrency limits, retries,
timeouts, and budget guards.

Today it implements the first seven stages: Trend Scout -> Content Strategist ->
Clip Factory -> Quality & Safety Gate -> Scheduler & Optimizer -> Publisher ->
Performance Analyst, returning the Analyst's
:class:`~trendz.contracts.PerformanceReports`. The Clip Factory is the
fan-out/concurrency core, mapping a Clip Worker over the briefs through
``bounded_map``; the Quality Gate fans out the same way, one evaluation per clip;
the Publisher fans out per platform over the ``PublishPlan``'s slices; the
Performance Analyst fans out per post over the ``PostResults``' succeeded posts.
The remaining stages (8+) are present as clearly-marked TODO scaffolding so the
module boundaries and the fan-out/concurrency model from the design are visible
and ready to fill in.
"""

from __future__ import annotations

import uuid

import structlog

from trendz.agents.clip_factory import ClipFactory
from trendz.agents.content_strategist import ContentStrategist
from trendz.agents.performance_analyst import PerformanceAnalyst
from trendz.agents.publisher import Publisher
from trendz.agents.quality_gate import QualityGate
from trendz.agents.scheduler import Scheduler
from trendz.agents.trend_scout import TrendScout
from trendz.concurrency import bounded_map
from trendz.contracts import PerformanceReports, RunConfig, RunContext
from trendz.sources.base import TrendSource

# Re-exported so callers and downstream stages can reach the bounded worker pool
# via the orchestrator, which owns the run-level concurrency model.
__all__ = ["Orchestrator", "bounded_map"]

log = structlog.get_logger(component="orchestrator")


class Orchestrator:
    """Builds the run context and drives the pipeline stages.

    Stages 1-7 (Trend Scout, Content Strategist, Clip Factory, Quality & Safety
    Gate, Scheduler & Optimizer, Publisher, Performance Analyst) are wired up and
    threaded the same ``RunContext``, and :meth:`run` returns the Analyst's
    ``PerformanceReports``. Downstream stages are stubbed below with the intended
    data flow and fan-out points documented.
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

    def _new_context(self) -> RunContext:
        """Create a fresh run context with a unique run id."""
        return RunContext.new(run_id=str(uuid.uuid4()), config=self._config)

    async def run(self) -> PerformanceReports:
        """Execute the pipeline.

        Today this runs stages 1-7 and returns the Performance Analyst's
        ``PerformanceReports`` (the per-post metrics + score + attribution for
        every successfully published post). As downstream agents land, each stage
        below is unstubbed and chained on, threading the same ``RunContext``
        through.
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

        # ------------------------------------------------------------------
        # TODO: remaining pipeline stages (see docs/DESIGN.md). Each stage is an
        # agent subclassing BaseAgent, threaded the same RunContext. The fan-out
        # stages use ``bounded_map`` (from trendz.concurrency) to cap concurrency
        # at ``self._config.concurrency_degree``.
        #
        #   await LearningStore(...).run(ctx, reports)  # updates weights/priors
        #
        # TODO: run lifecycle concerns to add here as well: retries with
        # exponential backoff, per-step timeouts, circuit breakers per external
        # API, budget/quota guards via ctx.budget, and a dead-letter queue for
        # permanently failed jobs.
        # ------------------------------------------------------------------

        log.info(
            "run_complete",
            run_id=ctx.run_id,
            report_count=len(reports),
            platform_count=len(reports.platforms),
        )
        return reports
