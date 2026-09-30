"""Offline entrypoint for the Trendz pipeline.

Builds a default :class:`~trendz.contracts.RunConfig`, wires the deterministic
:class:`~trendz.sources.mock_source.MockTrendSource`, runs the
:class:`~trendz.orchestrator.Orchestrator` (Trend Scout -> Content Strategist ->
Clip Factory -> Quality & Safety Gate -> Scheduler & Optimizer -> Publisher), and
prints the resulting ``PostResults`` grouped per platform. The pipeline now runs
stages 1-6 and runs fully offline (no network or API keys). Invoke with
``python -m trendz`` or the ``trendz`` console script.
"""

from __future__ import annotations

import asyncio

from trendz.contracts import PostResults, RunConfig
from trendz.orchestrator import Orchestrator
from trendz.sources.mock_source import MockTrendSource


def _print_results(results: PostResults) -> None:
    """Pretty-print PostResults grouped per platform to stdout."""
    print(
        f"PostResults for run {results.run_id} "
        f"({len(results)} posts across {len(results.platforms)} platforms):"
    )
    if not len(results):
        print("  (no published posts)")
    for platform, platform_results in results.slices():
        print(f"  {platform} ({len(platform_results)} posts):")
        for result in platform_results:
            if result.succeeded:
                print(
                    f"    OK    clip={result.clip_id}  v{result.variant}  "
                    f"id={result.post_id}  {result.post_url}"
                )
            else:
                print(
                    f"    FAIL  clip={result.clip_id}  v{result.variant}  " f"error={result.error}"
                )
    print(f"  Summary: {len(results.succeeded)} succeeded, {len(results.failed)} failed.")


async def _run() -> PostResults:
    config = RunConfig()
    orchestrator = Orchestrator(config=config, sources=[MockTrendSource()])
    return await orchestrator.run()


def main() -> None:
    """Console-script / module entrypoint."""
    results = asyncio.run(_run())
    _print_results(results)


if __name__ == "__main__":
    main()
