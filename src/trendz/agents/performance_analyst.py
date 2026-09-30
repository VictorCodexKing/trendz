"""Performance Analyst: the seventh agent in the pipeline (the per-post fan-out).

The Performance Analyst is DESIGN.md stage 7. It receives the
:class:`~trendz.contracts.PostResults` produced by the Publisher and collects the
engagement metrics for every SUCCESSFULLY published post, deriving a per-post
performance score and attributing the outcome back to its originating clip,
brief, and trend. It emits a :class:`~trendz.contracts.PerformanceReports` the
Learning Store (stage 8) keys off.

Only the succeeded posts (``payload.succeeded`` - those that carry a
``post_id``) get metrics; FAILED posts are excluded from collection (their count
is logged as the Learning Store's failure signal). Per-post metric collection is
not hand-rolled: it is routed through :func:`~trendz.concurrency.bounded_map`
with ``concurrency = ctx.config.concurrency_degree``, exactly like
:class:`~trendz.agents.publisher.Publisher`, so no more than that many
collections run at once and the reports come back in the upstream order
``bounded_map`` preserves.

Two design choices match the rest of the pipeline:
    - DETERMINISTIC, OFFLINE dwell: metrics are collected "as of"
      ``published_at + ctx.config.dwell_hours`` (see
      :attr:`~trendz.contracts.RunConfig.dwell_hours`). This is a modelled
      collection offset, never a real sleep or wall-clock wait.
    - RECORD-AND-CONTINUE is unnecessary here because collection is delegated to
      the injected provider, but the fan-out mirrors the Publisher's shape so a
      real provider can be swapped in without touching this logic.

The actual metric collection is delegated to the injected
:class:`~trendz.analytics.base.MetricsProvider` (defaulting to the deterministic
:class:`~trendz.analytics.stub_provider.StubMetricsProvider`) so a real
analytics API can be swapped in later without touching the fan-out logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trendz.agents.base import BaseAgent
from trendz.analytics.base import MetricsProvider
from trendz.analytics.stub_provider import StubMetricsProvider
from trendz.concurrency import bounded_map
from trendz.contracts import (
    PerformanceReports,
    PostMetrics,
    PostPerformance,
    PostResult,
    PostResults,
    RunContext,
)

# Fixed weights for the composite per-post performance score. Views are the
# base reach signal; watch time is weighted low per-second (it is a large raw
# number); likes/shares/comments/follows are progressively more valuable
# engagement signals (a follow is worth far more than a like). The score is a
# deterministic linear combination of the raw metrics, so identical metrics
# always yield an identical score.
_WEIGHT_VIEWS = 1.0
_WEIGHT_WATCH_TIME = 0.01
_WEIGHT_LIKES = 2.0
_WEIGHT_SHARES = 5.0
_WEIGHT_COMMENTS = 4.0
_WEIGHT_FOLLOWS = 10.0


def _performance_score(metrics: PostMetrics) -> float:
    """Derive a deterministic composite performance score from raw metrics.

    A fixed-weight linear combination of the raw signals::

        score = 1.0*views + 0.01*watch_time_seconds + 2.0*likes
                + 5.0*shares + 4.0*comments + 10.0*follows

    Being a pure function of :class:`~trendz.contracts.PostMetrics`, the same
    metrics always yield the same score, and a post with strictly more
    engagement scores strictly higher. The result is always non-negative because
    every metric and weight is non-negative.
    """
    return (
        _WEIGHT_VIEWS * metrics.views
        + _WEIGHT_WATCH_TIME * metrics.watch_time_seconds
        + _WEIGHT_LIKES * metrics.likes
        + _WEIGHT_SHARES * metrics.shares
        + _WEIGHT_COMMENTS * metrics.comments
        + _WEIGHT_FOLLOWS * metrics.follows
    )


def _collected_at(result: PostResult, dwell_hours: int) -> datetime:
    """Return the deterministic dwell-based collection time for a post.

    Modelled as ``published_at + dwell_hours`` (see
    :attr:`~trendz.contracts.RunConfig.dwell_hours`) - a collection offset, not a
    real wait. ``published_at`` is always set on a succeeded post; if it is
    somehow missing, fall back to the context creation time so the field stays
    an aware-UTC datetime.
    """
    base = result.published_at if result.published_at is not None else datetime.now(timezone.utc)
    return base + timedelta(hours=dwell_hours)


class PerformanceAnalyst(BaseAgent[PostResults, PerformanceReports]):
    """Fans a metrics-provider out per post into a ``PerformanceReports``."""

    def __init__(self, provider: MetricsProvider | None = None) -> None:
        """Create the agent.

        Args:
            provider: The metrics-provider used to collect each post's metrics
                (dependency-injected). Defaults to the deterministic offline
                :class:`~trendz.analytics.stub_provider.StubMetricsProvider`.
        """
        super().__init__()
        self._provider = provider if provider is not None else StubMetricsProvider()

    @property
    def name(self) -> str:
        return "performance_analyst"

    async def _collect_post(self, ctx: RunContext, result: PostResult) -> PostPerformance:
        """Collect metrics for one succeeded post and build its performance record.

        The metrics come from the injected provider; the score is derived
        deterministically from them and ``collected_at`` is the deterministic
        dwell-based collection time. The full attribution linkage
        (clip/brief/trend + platform + variant + post_id) is carried through.
        """
        metrics = await self._provider.collect(ctx, result)
        return PostPerformance(
            clip_id=result.clip_id,
            brief_id=result.brief_id,
            trend_id=result.trend_id,
            platform=result.platform,
            variant=result.variant,
            post_id=result.post_id or "",
            metrics=metrics,
            performance_score=_performance_score(metrics),
            collected_at=_collected_at(result, ctx.config.dwell_hours),
        )

    async def run(self, ctx: RunContext, payload: PostResults) -> PerformanceReports:
        """Collect metrics for every succeeded post, fanning out per post.

        Only the succeeded posts (``payload.succeeded``) are collected; failed
        posts are excluded (their count is logged as the Learning Store's failure
        signal). The per-post collection is routed through
        :func:`~trendz.concurrency.bounded_map` capped at
        ``ctx.config.concurrency_degree``, so no more than that many collections
        run at once. Reports are reassembled in the deterministic order
        ``bounded_map`` preserves (the upstream ``PostResults`` succeeded order).
        An empty or all-failed ``PostResults`` yields an empty
        ``PerformanceReports`` without dispatching any work.
        """
        succeeded = payload.succeeded
        reports: tuple[PostPerformance, ...]
        if not succeeded:
            reports = ()
        else:
            collected = await bounded_map(
                lambda result: self._collect_post(ctx, result),
                succeeded,
                ctx.config.concurrency_degree,
            )
            reports = tuple(collected)

        report = PerformanceReports(run_id=ctx.run_id, reports=reports)
        self.log.info(
            "analyst_done",
            reports=len(report),
            platforms=len(report.platforms),
            excluded_failures=len(payload.failed),
            provider=self._provider.name,
            concurrency=ctx.config.concurrency_degree,
        )
        return report
