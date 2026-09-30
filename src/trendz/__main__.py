"""Offline entrypoint for the Trendz pipeline.

Builds a default :class:`~trendz.contracts.RunConfig`, wires the deterministic
:class:`~trendz.sources.mock_source.MockTrendSource`, runs the
:class:`~trendz.orchestrator.Orchestrator` (Trend Scout -> Content Strategist ->
Clip Factory -> Quality & Safety Gate -> Scheduler & Optimizer), and prints the
resulting ``PublishPlan`` grouped per platform. The pipeline now runs stages 1-5
and runs fully offline (no network or API keys). Invoke with ``python -m trendz``
or the ``trendz`` console script.
"""

from __future__ import annotations

import asyncio

from trendz.contracts import PublishPlan, RunConfig
from trendz.orchestrator import Orchestrator
from trendz.sources.mock_source import MockTrendSource


def _print_plan(plan: PublishPlan) -> None:
    """Pretty-print a PublishPlan grouped per platform to stdout."""
    print(
        f"PublishPlan for run {plan.run_id} "
        f"({len(plan)} posts across {len(plan.platforms)} platforms):"
    )
    if not len(plan):
        print("  (no scheduled posts)")
    for platform, posts in plan.slices():
        print(f"  {platform} ({len(posts)} posts):")
        for post in posts:
            when = post.scheduled_at.strftime("%Y-%m-%d %H:%M UTC")
            hashtags = " ".join(f"#{tag}" for tag in post.hashtags) or "(no hashtags)"
            print(
                f"    {when}  clip={post.clip_id}  v{post.variant}  "
                f'"{post.caption}"  {hashtags}'
            )


async def _run() -> PublishPlan:
    config = RunConfig()
    orchestrator = Orchestrator(config=config, sources=[MockTrendSource()])
    return await orchestrator.run()


def main() -> None:
    """Console-script / module entrypoint."""
    plan = asyncio.run(_run())
    _print_plan(plan)


if __name__ == "__main__":
    main()
