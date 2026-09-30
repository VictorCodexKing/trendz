"""Offline entrypoint for the Trendz pipeline.

Builds a default :class:`~trendz.contracts.RunConfig`, wires the deterministic
:class:`~trendz.sources.mock_source.MockTrendSource`, runs the
:class:`~trendz.orchestrator.Orchestrator` (Trend Scout -> Content Strategist ->
Clip Factory -> Quality & Safety Gate -> Scheduler & Optimizer -> Publisher ->
Performance Analyst -> Learning / Memory Store), and prints the resulting
:class:`~trendz.contracts.RunResult`: the ``PerformanceReports`` grouped per
platform (per-post metrics + score) plus an attribution summary by trend, and the
stage-8 ``LearningState`` (updated scoring weights, per-platform format/hook
priors, engagement threshold, timing hints, a per-trend performance summary, and
where each learning feeds back). The pipeline now runs all eight stages fully
offline (no network or API keys). Invoke with ``python -m trendz`` or the
``trendz`` console script.
"""

from __future__ import annotations

import asyncio

from trendz.contracts import LearningState, PerformanceReports, RunConfig, RunResult
from trendz.orchestrator import Orchestrator
from trendz.sources.mock_source import MockTrendSource


def _print_reports(reports: PerformanceReports) -> None:
    """Pretty-print PerformanceReports grouped per platform + by-trend attribution."""
    print(
        f"PerformanceReports for run {reports.run_id} "
        f"({len(reports)} posts across {len(reports.platforms)} platforms):"
    )
    if not len(reports):
        print("  (no performance reports)")
    for platform, platform_reports in reports.slices():
        print(f"  {platform} ({len(platform_reports)} posts):")
        for report in platform_reports:
            metrics = report.metrics
            print(
                f"    clip={report.clip_id}  v{report.variant}  "
                f"id={report.post_id}  score={report.performance_score:.1f}"
            )
            print(
                f"        views={metrics.views}  watch_s={metrics.watch_time_seconds:.0f}  "
                f"likes={metrics.likes}  shares={metrics.shares}  "
                f"comments={metrics.comments}  follows={metrics.follows}  "
                f"collected_at={report.collected_at.isoformat()}"
            )
    print("  Attribution by trend:")
    if not len(reports):
        print("    (none)")
    for trend_id, trend_reports in reports.by_trend():
        total = sum(r.performance_score for r in trend_reports)
        print(f"    trend={trend_id}  posts={len(trend_reports)}  total_score={total:.1f}")


def _print_learnings(learnings: LearningState) -> None:
    """Pretty-print the stage-8 LearningState and where each learning feeds back."""
    print(
        f"LearningState for run {learnings.run_id} "
        f"(version {learnings.version}, {learnings.updates_applied} updates applied):"
    )

    velocity, relevance, shelf_life, saturation = learnings.scoring_weights
    print("  Updated Trend Scout scoring weights:")
    print(
        f"    velocity={velocity:.4f}  relevance={relevance:.4f}  "
        f"shelf_life={shelf_life:.4f}  saturation={saturation:.4f}"
    )

    print("  Content Strategist per-platform format/hook priors:")
    if not learnings.format_priors:
        print("    (none)")
    for platform in sorted(learnings.format_priors):
        print(f"    {platform}: prior={learnings.format_priors[platform]:.4f}")

    print(f"  Quality Gate engagement threshold: {learnings.engagement_threshold:.4f}")

    print("  Scheduler per-platform timing hints (minutes-of-day):")
    if not learnings.timing_slots:
        print("    (none)")
    for platform in sorted(learnings.timing_slots):
        slots = ", ".join(str(slot) for slot in learnings.timing_slots[platform])
        print(f"    {platform}: [{slots}]")

    print("  Bandit arm state (platform / pulls / estimated reward):")
    if not learnings.arms:
        print("    (none)")
    for arm in learnings.arms:
        print(f"    {arm.platform}: pulls={arm.pulls}  reward={arm.estimated_reward:.4f}")

    print("  Per-trend performance summary:")
    if not learnings.trend_aggregates:
        print("    (none)")
    for aggregate in learnings.trend_aggregates:
        platforms = ", ".join(aggregate.platforms)
        print(
            f"    trend={aggregate.trend_id}  posts={aggregate.post_count}  "
            f"total_score={aggregate.total_performance_score:.1f}  "
            f"mean_score={aggregate.mean_performance_score:.1f}  platforms=[{platforms}]"
        )

    print("  Feedback loop (where each learning feeds back):")
    print("    - scoring weights   -> Trend Scout (trend ranking)")
    print("    - format priors     -> Content Strategist (platform emphasis)")
    print("    - engagement threshold -> Quality Gate (clip approval bar)")
    print("    - timing hints      -> Scheduler & Optimizer (post scheduling)")


def _print_result(result: RunResult) -> None:
    """Print the full end-to-end RunResult: stage-7 reports + stage-8 learnings."""
    _print_reports(result.reports)
    print()
    _print_learnings(result.learnings)


async def _run() -> RunResult:
    config = RunConfig()
    orchestrator = Orchestrator(config=config, sources=[MockTrendSource()])
    return await orchestrator.run()


def main() -> None:
    """Console-script / module entrypoint."""
    result = asyncio.run(_run())
    _print_result(result)


if __name__ == "__main__":
    main()
