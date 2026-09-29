"""Offline tests for the Clip Factory agent and its bounded fan-out."""

from __future__ import annotations

import asyncio

from trendz.agents.clip_factory import ClipFactory
from trendz.contracts import (
    ClipBrief,
    ClipBriefList,
    RenderedClip,
    RenderedClipSet,
    RunConfig,
    RunContext,
)
from trendz.workers.base import ClipWorker
from trendz.workers.stub_worker import StubClipWorker


def _brief(idx: int, trend_id: str | None = None) -> ClipBrief:
    """Build a deterministic clip brief for tests."""
    return ClipBrief(
        id=f"b{idx}",
        trend_id=trend_id if trend_id is not None else f"t{idx}",
        title=f"Trend {idx}",
        angle="angle",
        hook="hook",
        platform="tiktok",
        aspect_ratio="9:16",
        target_length_seconds=30,
        caption="caption",
        cta="cta",
        hashtags=("fyp",),
        footage_kind="sourced",
    )


def _brief_list(run_id: str, count: int) -> ClipBriefList:
    """Build an ordered ClipBriefList with ``count`` briefs."""
    return ClipBriefList(run_id=run_id, briefs=tuple(_brief(i) for i in range(count)))


async def test_one_clip_per_brief_with_linkage_and_order(ctx: RunContext) -> None:
    """Each brief yields one clip, linked by brief_id/trend_id, in input order."""
    briefs = _brief_list(ctx.run_id, 4)
    factory = ClipFactory(StubClipWorker())

    result = await factory.run(ctx, briefs)

    assert isinstance(result, RenderedClipSet)
    assert result.run_id == ctx.run_id
    assert len(result) == len(briefs)
    for brief, clip in zip(briefs.briefs, result.clips, strict=True):
        assert clip.brief_id == brief.id
        assert clip.trend_id == brief.trend_id
        assert clip.platform == brief.platform
        assert clip.aspect_ratio == brief.aspect_ratio
        assert clip.duration_seconds == brief.target_length_seconds
    # Order preserved: the ids line up with the input briefs.
    assert [c.brief_id for c in result.clips] == [b.id for b in briefs.briefs]


async def test_empty_brief_list_yields_empty_clip_set(ctx: RunContext) -> None:
    """An empty ClipBriefList yields an empty RenderedClipSet without error."""
    factory = ClipFactory(StubClipWorker())

    result = await factory.run(ctx, ClipBriefList(run_id=ctx.run_id))

    assert isinstance(result, RenderedClipSet)
    assert len(result) == 0
    assert result.run_id == ctx.run_id


async def test_rendering_is_deterministic(ctx: RunContext) -> None:
    """The same briefs always render to byte-identical clip sets."""
    briefs = _brief_list(ctx.run_id, 3)
    factory = ClipFactory(StubClipWorker())

    first = await factory.run(ctx, briefs)
    second = await factory.run(ctx, briefs)

    assert first == second


async def test_default_worker_is_stub() -> None:
    """The Clip Factory defaults to the deterministic offline stub worker."""
    factory = ClipFactory()
    assert isinstance(factory._worker, StubClipWorker)


class _ConcurrencyProbeWorker(ClipWorker):
    """A worker that tracks the high-water mark of concurrent renders."""

    def __init__(self) -> None:
        self.live = 0
        self.high_water = 0

    @property
    def name(self) -> str:
        return "concurrency_probe"

    async def render(self, ctx: RunContext, brief: ClipBrief) -> RenderedClip:
        self.live += 1
        self.high_water = max(self.high_water, self.live)
        # Hold the slot long enough for the pool to saturate.
        await asyncio.sleep(0.02)
        self.live -= 1
        return RenderedClip(
            id=f"clip-{brief.id}",
            brief_id=brief.id,
            trend_id=brief.trend_id,
            platform=brief.platform,
            aspect_ratio=brief.aspect_ratio,
            duration_seconds=brief.target_length_seconds,
            video_uri=f"stub://clips/{brief.id}.mp4",
        )


async def test_fan_out_reaches_but_never_exceeds_concurrency_cap() -> None:
    """The fan-out runs concurrently up to ctx.config.concurrency_degree, no more.

    Modelled after test_bounded_map_caps_concurrency: with more briefs than the
    cap and a worker that holds each slot, the high-water mark must reach the cap
    (proving real concurrency) but never exceed it (proving bounded_map governs
    the fan-out).
    """
    concurrency = 3
    config = RunConfig(concurrency_degree=concurrency, target_clip_count=100)
    ctx = RunContext.new(run_id="concurrency-run", config=config)
    briefs = _brief_list(ctx.run_id, 12)
    probe = _ConcurrencyProbeWorker()
    factory = ClipFactory(probe)

    result = await factory.run(ctx, briefs)

    assert len(result) == 12
    assert probe.high_water <= concurrency
    # With 12 briefs and a cap of 3, the pool should actually reach the cap.
    assert probe.high_water == concurrency
    # Order is still preserved despite concurrent completion.
    assert [c.brief_id for c in result.clips] == [b.id for b in briefs.briefs]
