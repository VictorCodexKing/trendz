"""Offline entrypoint for the Trendz pipeline.

Builds a default :class:`~trendz.contracts.RunConfig`, wires the deterministic
:class:`~trendz.sources.mock_source.MockTrendSource`, runs the
:class:`~trendz.orchestrator.Orchestrator`, and prints the ranked ``TrendList``.
Runs fully offline (no network or API keys). Invoke with ``python -m trendz`` or
the ``trendz`` console script.
"""

from __future__ import annotations

import asyncio

from trendz.contracts import RunConfig, TrendList
from trendz.orchestrator import Orchestrator
from trendz.sources.mock_source import MockTrendSource


def _print_trend_list(trend_list: TrendList) -> None:
    """Pretty-print a ranked TrendList to stdout."""
    print(f"Ranked TrendList for run {trend_list.run_id} ({len(trend_list)} trends):")
    if not trend_list.trends:
        print("  (no trends)")
        return
    for rank, trend in enumerate(trend_list.trends, start=1):
        print(f"  {rank}. [{trend.score:+.4f}] {trend.title}  (source={trend.source})")


async def _run() -> TrendList:
    config = RunConfig()
    orchestrator = Orchestrator(config=config, sources=[MockTrendSource()])
    return await orchestrator.run()


def main() -> None:
    """Console-script / module entrypoint."""
    trend_list = asyncio.run(_run())
    _print_trend_list(trend_list)


if __name__ == "__main__":
    main()
