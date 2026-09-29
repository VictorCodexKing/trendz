"""The run conductor.

The Orchestrator owns the run lifecycle: it builds a
:class:`~trendz.contracts.RunContext` from a :class:`~trendz.contracts.RunConfig`,
dispatches work to agents, and (eventually) enforces concurrency limits, retries,
timeouts, and budget guards.

Today it implements the first four stages: Trend Scout -> Content Strategist ->
Clip Factory -> Quality & Safety Gate. The Clip Factory is the fan-out/
concurrency core, mapping a Clip Worker over the briefs through ``bounded_map``;
the Quality Gate fans out the same way, one evaluation per clip. The remaining
stages (5+) are present as clearly-marked TODO scaffolding so the module
boundaries and the fan-out/concurrency model from the design are visible and
ready to fill in.
"""

from __future__ import annotations

import uuid

import structlog

from trendz.agents.clip_factory import ClipFactory
from trendz.agents.content_strategist import ContentStrategist
from trendz.agents.quality_gate import QualityGate
from trendz.agents.trend_scout import TrendScout
from trendz.concurrency import bounded_map
from trendz.contracts import QualityReport, RunConfig, RunContext
from trendz.sources.base import TrendSource

# Re-exported so callers and downstream stages can reach the bounded worker pool
# via the orchestrator, which owns the run-level concurrency model.
__all__ = ["Orchestrator", "bounded_map"]

log = structlog.get_logger(component="orchestrator")


class Orchestrator:
    """Builds the run context and drives the pipeline stages.

    Stages 1-4 (Trend Scout, Content Strategist, Clip Factory, Quality & Safety
    Gate) are wired up and threaded the same ``RunContext``. Downstream stages
    are stubbed below with the intended data flow and fan-out points documented.
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

    def _new_context(self) -> RunContext:
        """Create a fresh run context with a unique run id."""
        return RunContext.new(run_id=str(uuid.uuid4()), config=self._config)

    async def run(self) -> QualityReport:
        """Execute the pipeline.

        Today this runs stages 1-4 and returns the Quality & Safety Gate's
        ``QualityReport`` (approved clips plus dropped verdicts). As downstream
        agents land, each stage below is unstubbed and chained on, threading the
        same ``RunContext`` through.
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

        # ------------------------------------------------------------------
        # TODO: remaining pipeline stages (see docs/DESIGN.md). Each stage is an
        # agent subclassing BaseAgent, threaded the same RunContext. The fan-out
        # stages use ``bounded_map`` (from trendz.concurrency) to cap concurrency
        # at ``self._config.concurrency_degree``. The Scheduler consumes the
        # gate's approved output (``report.approved``).
        #
        #   plan     = await Scheduler(...).run(ctx, report.approved)  # -> PublishPlan
        #   results  = await bounded_map(
        #                  lambda slice_: Publisher(...).run(ctx, slice_),
        #                  plan.platform_slices, self._config.concurrency_degree,
        #              )                       # fan-out boundary: M platforms -> PostResults
        #   reports  = await Analyst(...).run(ctx, results)      # -> PerformanceReports
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
            approved_count=len(report.approved),
            dropped_count=len(report.dropped),
        )
        return report
