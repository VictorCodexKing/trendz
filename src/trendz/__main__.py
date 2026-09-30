"""Offline entrypoint for the Trendz pipeline.

Builds a default :class:`~trendz.contracts.RunConfig`, wires the deterministic
:class:`~trendz.sources.mock_source.MockTrendSource`, runs the
:class:`~trendz.orchestrator.Orchestrator` (Trend Scout -> Content Strategist ->
Clip Factory -> Quality & Safety Gate), and prints the resulting
``QualityReport``. Runs fully offline (no network or API keys). Invoke with
``python -m trendz`` or the ``trendz`` console script.
"""

from __future__ import annotations

import asyncio

from trendz.contracts import QualityReport, RunConfig
from trendz.orchestrator import Orchestrator
from trendz.sources.mock_source import MockTrendSource


def _print_report(report: QualityReport) -> None:
    """Pretty-print a QualityReport to stdout."""
    print(
        f"QualityReport for run {report.run_id} "
        f"({len(report)} approved, {len(report.dropped)} dropped):"
    )
    if not report.approved:
        print("  (no approved clips)")
    for rank, clip in enumerate(report.approved, start=1):
        print(
            f"  {rank}. {clip.id}  "
            f"[{clip.platform} {clip.aspect_ratio} {clip.duration_seconds}s]  "
            f"trend={clip.trend_id}  {clip.video_uri}"
        )
    for verdict in report.dropped:
        reasons = ", ".join(verdict.reasons)
        print(f"  DROPPED {verdict.clip_id} (brief={verdict.brief_id}): {reasons}")


async def _run() -> QualityReport:
    config = RunConfig()
    orchestrator = Orchestrator(config=config, sources=[MockTrendSource()])
    return await orchestrator.run()


def main() -> None:
    """Console-script / module entrypoint."""
    report = asyncio.run(_run())
    _print_report(report)


if __name__ == "__main__":
    main()
