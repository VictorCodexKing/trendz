"""Offline entrypoint for the Trendz pipeline.

Builds a default :class:`~trendz.contracts.RunConfig`, wires the deterministic
:class:`~trendz.sources.mock_source.MockTrendSource`, runs the
:class:`~trendz.orchestrator.Orchestrator` (Trend Scout -> Content Strategist ->
Clip Factory), and prints the resulting ``RenderedClipSet``. Runs fully offline
(no network or API keys). Invoke with ``python -m trendz`` or the ``trendz``
console script.
"""

from __future__ import annotations

import asyncio

from trendz.contracts import RenderedClipSet, RunConfig
from trendz.orchestrator import Orchestrator
from trendz.sources.mock_source import MockTrendSource


def _print_clip_set(clip_set: RenderedClipSet) -> None:
    """Pretty-print a RenderedClipSet to stdout."""
    print(f"RenderedClipSet for run {clip_set.run_id} ({len(clip_set)} clips):")
    if not clip_set.clips:
        print("  (no clips)")
        return
    for rank, clip in enumerate(clip_set.clips, start=1):
        print(
            f"  {rank}. {clip.id}  "
            f"[{clip.platform} {clip.aspect_ratio} {clip.duration_seconds}s]  "
            f"trend={clip.trend_id}  {clip.video_uri}"
        )


async def _run() -> RenderedClipSet:
    config = RunConfig()
    orchestrator = Orchestrator(config=config, sources=[MockTrendSource()])
    return await orchestrator.run()


def main() -> None:
    """Console-script / module entrypoint."""
    clip_set = asyncio.run(_run())
    _print_clip_set(clip_set)


if __name__ == "__main__":
    main()
