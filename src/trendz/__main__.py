"""Offline entrypoint for the Trendz pipeline.

Builds a default :class:`~trendz.contracts.RunConfig`, wires the deterministic
:class:`~trendz.sources.mock_source.MockTrendSource`, runs the
:class:`~trendz.orchestrator.Orchestrator` (Trend Scout -> Content Strategist ->
Clip Factory -> Quality & Safety Gate -> Scheduler & Optimizer -> Publisher ->
Performance Analyst), and prints the resulting ``PerformanceReports`` grouped per
platform (per-post metrics + score) plus an attribution summary by trend. The
pipeline now runs stages 1-7 and runs fully offline (no network or API keys).
Invoke with ``python -m trendz`` or the ``trendz`` console script.
"""

from __future__ import annotations

import asyncio

from trendz.contracts import PerformanceReports, RunConfig
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


async def _run() -> PerformanceReports:
    config = RunConfig()
    orchestrator = Orchestrator(config=config, sources=[MockTrendSource()])
    return await orchestrator.run()


def main() -> None:
    """Console-script / module entrypoint."""
    reports = asyncio.run(_run())
    _print_reports(reports)


if __name__ == "__main__":
    main()
