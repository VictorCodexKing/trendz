"""Quality & Safety Gate: the per-clip gatekeeper (DESIGN.md stage 4).

The Quality & Safety Gate runs after the Clip Factory and before the Scheduler.
It is embarrassingly parallel: one evaluation task per rendered clip, fanned out
through :func:`~trendz.concurrency.bounded_map` capped at
``ctx.config.concurrency_degree`` (exactly like the Clip Factory). For each
:class:`~trendz.contracts.RenderedClip` it asks the injected
:class:`~trendz.checkers.base.ClipChecker` for a
:class:`~trendz.contracts.ClipVerdict` covering technical checks, content-safety
checks, and a predicted-engagement score.

Routing follows the design:
    - APPROVED clips are kept and flow on toward the Scheduler.
    - REJECTED clips go BACK to the Clip Factory with their reason for BOUNDED
      retries (up to ``ctx.config.max_quality_retries`` re-renders), re-using the
      existing Clip Factory / Clip Worker render path rather than re-implementing
      rendering.
    - Clips still failing after retries are exhausted (or that error during
      evaluation) are DROPPED and logged - never raised. One bad clip must never
      block the run.

Re-rendering needs each clip's originating :class:`~trendz.contracts.ClipBrief`.
The orchestrator already has the ``ClipBriefList`` in scope before it calls the
Clip Factory, so it threads it into :meth:`run` via the keyword-only ``briefs``
argument; the gate builds a ``brief_id -> ClipBrief`` lookup from it. If a
brief cannot be found (or no briefs were supplied), a rejected clip simply
cannot be retried and is dropped with a reason, which keeps the gate robust.

Results are assembled into a :class:`~trendz.contracts.QualityReport` carrying
the approved clips (in input order, since ``bounded_map`` preserves it) and the
dropped verdicts with their reasons (observability signal the design says feeds
the future Performance Analyst / Learning Store).
"""

from __future__ import annotations

from trendz.agents.base import BaseAgent
from trendz.agents.clip_factory import ClipFactory
from trendz.checkers.base import ClipChecker
from trendz.checkers.stub_checker import StubClipChecker
from trendz.concurrency import bounded_map
from trendz.contracts import (
    ClipBrief,
    ClipBriefList,
    ClipVerdict,
    QualityReport,
    RenderedClip,
    RenderedClipSet,
    RunContext,
)


class QualityGate(BaseAgent[RenderedClipSet, QualityReport]):
    """Evaluates each clip and routes it: approve, bounded-retry, or drop."""

    def __init__(
        self,
        checker: ClipChecker | None = None,
        clip_factory: ClipFactory | None = None,
    ) -> None:
        """Create the agent.

        Args:
            checker: The Clip Checker used to evaluate each clip
                (dependency-injected). Defaults to the deterministic offline
                :class:`~trendz.checkers.stub_checker.StubClipChecker`.
            clip_factory: The Clip Factory used to re-render rejected clips on
                retry (dependency-injected so the gate re-uses the existing
                render path rather than re-implementing rendering). Defaults to a
                fresh :class:`~trendz.agents.clip_factory.ClipFactory`.
        """
        super().__init__()
        self._checker = checker if checker is not None else StubClipChecker()
        self._clip_factory = clip_factory if clip_factory is not None else ClipFactory()

    @property
    def name(self) -> str:
        return "quality_gate"

    async def _rerender(self, ctx: RunContext, brief: ClipBrief) -> RenderedClip:
        """Re-render a single brief by reusing the Clip Factory render path."""
        brief_list = ClipBriefList(run_id=ctx.run_id, briefs=(brief,))
        rendered = await self._clip_factory.run(ctx, brief_list)
        return rendered.clips[0]

    async def _gate_clip(
        self,
        ctx: RunContext,
        clip: RenderedClip,
        briefs: dict[str, ClipBrief],
    ) -> tuple[RenderedClip | None, ClipVerdict]:
        """Evaluate one clip, retrying via re-render, until approved or dropped.

        Returns ``(clip, verdict)`` when approved and ``(None, verdict)`` when
        the clip is dropped (retries exhausted, no brief to retry with, or an
        error during evaluation). Never raises: a per-clip failure becomes a
        drop-with-reason so one bad clip cannot block the run.
        """
        current = clip
        try:
            verdict = await self._checker.evaluate(ctx, current)
            retries = 0
            max_retries = ctx.config.max_quality_retries
            while not verdict.approved and retries < max_retries:
                brief = briefs.get(current.brief_id)
                if brief is None:
                    # Cannot re-render without the originating brief; stop retrying.
                    verdict = verdict.model_copy(
                        update={
                            "reasons": verdict.reasons
                            + ("dropped: no originating brief available to re-render",)
                        }
                    )
                    break
                self.log.info(
                    "clip_rejected_retrying",
                    clip_id=current.id,
                    brief_id=current.brief_id,
                    attempt=retries + 1,
                    reasons=list(verdict.reasons),
                )
                current = await self._rerender(ctx, brief)
                verdict = await self._checker.evaluate(ctx, current)
                retries += 1
            if verdict.approved:
                return current, verdict
            self.log.info(
                "clip_dropped",
                clip_id=current.id,
                brief_id=current.brief_id,
                retries_used=retries,
                reasons=list(verdict.reasons),
            )
            return None, verdict
        except Exception as exc:  # noqa: BLE001 - one bad clip must never block the run.
            reason = f"error: {type(exc).__name__}: {exc}"
            self.log.info(
                "clip_dropped",
                clip_id=current.id,
                brief_id=current.brief_id,
                reasons=[reason],
            )
            verdict = ClipVerdict(
                clip_id=current.id,
                brief_id=current.brief_id,
                trend_id=current.trend_id,
                approved=False,
                reasons=(reason,),
                engagement_score=0.0,
                technical_ok=False,
                safety_ok=False,
            )
            return None, verdict

    async def run(
        self,
        ctx: RunContext,
        payload: RenderedClipSet,
        *,
        briefs: ClipBriefList | None = None,
    ) -> QualityReport:
        """Gate every clip concurrently into a ``QualityReport``.

        The per-clip work is routed through
        :func:`~trendz.concurrency.bounded_map` capped at
        ``ctx.config.concurrency_degree``, so no more than that many evaluations
        run at once. Approved-clip order matches input order (``bounded_map``
        preserves it). An empty ``RenderedClipSet`` yields an empty
        ``QualityReport`` without dispatching any work.

        Args:
            ctx: The shared run context.
            payload: The rendered clips to evaluate.
            briefs: The originating briefs, threaded from the orchestrator, used
                to re-render rejected clips on retry. When absent, rejected clips
                cannot be retried and are dropped.
        """
        brief_lookup: dict[str, ClipBrief] = (
            {brief.id: brief for brief in briefs.briefs} if briefs is not None else {}
        )

        outcomes: list[tuple[RenderedClip | None, ClipVerdict]] = await bounded_map(
            lambda clip: self._gate_clip(ctx, clip, brief_lookup),
            payload.clips,
            ctx.config.concurrency_degree,
        )

        approved: list[RenderedClip] = [clip for clip, _ in outcomes if clip is not None]
        dropped: list[ClipVerdict] = [verdict for clip, verdict in outcomes if clip is None]

        report = QualityReport(
            run_id=ctx.run_id,
            approved=tuple(approved),
            dropped=tuple(dropped),
        )
        self.log.info(
            "quality_gate_done",
            clips_in=len(payload),
            approved=len(approved),
            dropped=len(dropped),
            checker=self._checker.name,
            max_quality_retries=ctx.config.max_quality_retries,
            concurrency=ctx.config.concurrency_degree,
        )
        return report
