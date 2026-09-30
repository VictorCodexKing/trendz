"""Offline tests for the Quality & Safety Gate agent and its bounded fan-out."""

from __future__ import annotations

import asyncio

from trendz.agents.clip_factory import ClipFactory
from trendz.agents.quality_gate import QualityGate
from trendz.checkers.base import ClipChecker
from trendz.checkers.stub_checker import StubClipChecker
from trendz.contracts import (
    ClipBrief,
    ClipBriefList,
    ClipVerdict,
    QualityReport,
    RenderedClip,
    RenderedClipSet,
    RunConfig,
    RunContext,
)


def _brief(idx: int) -> ClipBrief:
    """Build a deterministic clip brief for tests."""
    return ClipBrief(
        id=f"b{idx}",
        trend_id=f"t{idx}",
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


def _clip(idx: int, **overrides: object) -> RenderedClip:
    """Build a deterministic rendered clip for tests (mirrors the stub worker)."""
    fields: dict[str, object] = {
        "id": f"clip-b{idx}",
        "brief_id": f"b{idx}",
        "trend_id": f"t{idx}",
        "platform": "tiktok",
        "aspect_ratio": "9:16",
        "duration_seconds": 30,
        "video_uri": f"stub://clips/b{idx}.mp4",
        "has_audio": True,
        "has_captions": True,
    }
    fields.update(overrides)
    return RenderedClip(**fields)  # type: ignore[arg-type]


def _clip_set(run_id: str, count: int) -> RenderedClipSet:
    """Build an ordered RenderedClipSet with ``count`` clips."""
    return RenderedClipSet(run_id=run_id, clips=tuple(_clip(i) for i in range(count)))


def _brief_list(run_id: str, count: int) -> ClipBriefList:
    """Build a ClipBriefList matching ``_clip_set`` linkage."""
    return ClipBriefList(run_id=run_id, briefs=tuple(_brief(i) for i in range(count)))


async def test_all_approved_preserves_input_order(ctx: RunContext) -> None:
    """All-approving checker yields a report whose approved set equals the input."""
    clips = _clip_set(ctx.run_id, 4)
    gate = QualityGate(checker=StubClipChecker())

    report = await gate.run(ctx, clips)

    assert isinstance(report, QualityReport)
    assert report.run_id == ctx.run_id
    assert len(report) == len(clips)
    assert report.dropped == ()
    assert [c.id for c in report.approved] == [c.id for c in clips.clips]


async def test_rejected_clip_with_no_retries_is_dropped() -> None:
    """A clip the stub rejects with retries=0 is dropped, absent from approved."""
    config = RunConfig(max_quality_retries=0)
    ctx = RunContext.new(run_id="drop-run", config=config)
    clips = _clip_set(ctx.run_id, 3)
    checker = StubClipChecker(reject_brief_ids=frozenset({"b1"}))
    gate = QualityGate(checker=checker)

    report = await gate.run(ctx, clips, briefs=_brief_list(ctx.run_id, 3))

    assert [c.brief_id for c in report.approved] == ["b0", "b2"]
    assert len(report.dropped) == 1
    dropped = report.dropped[0]
    assert dropped.brief_id == "b1"
    assert dropped.approved is False
    assert any("safety" in reason for reason in dropped.reasons)


async def test_reject_then_approve_on_retry_uses_rerender_path() -> None:
    """A clip rejected once but passing on re-render ends up approved."""
    config = RunConfig(max_quality_retries=2)
    ctx = RunContext.new(run_id="retry-run", config=config)
    clips = _clip_set(ctx.run_id, 2)
    # b1 is rejected exactly once; the first re-render then passes.
    checker = StubClipChecker(reject_budget={"b1": 1})
    gate = QualityGate(checker=checker, clip_factory=ClipFactory())

    report = await gate.run(ctx, clips, briefs=_brief_list(ctx.run_id, 2))

    assert len(report) == 2
    assert report.dropped == ()
    assert {c.brief_id for c in report.approved} == {"b0", "b1"}


async def test_reject_budget_exceeding_retries_is_dropped() -> None:
    """A clip whose rejection budget outlasts the retry bound is dropped."""
    config = RunConfig(max_quality_retries=1)
    ctx = RunContext.new(run_id="exhaust-run", config=config)
    clips = _clip_set(ctx.run_id, 1)
    # Rejected 3 times but only 1 retry is allowed -> dropped.
    checker = StubClipChecker(reject_budget={"b0": 3})
    gate = QualityGate(checker=checker)

    report = await gate.run(ctx, clips, briefs=_brief_list(ctx.run_id, 1))

    assert len(report) == 0
    assert len(report.dropped) == 1
    assert report.dropped[0].brief_id == "b0"


async def test_empty_clip_set_yields_empty_report(ctx: RunContext) -> None:
    """An empty RenderedClipSet yields an empty QualityReport without work."""
    gate = QualityGate(checker=StubClipChecker())

    report = await gate.run(ctx, RenderedClipSet(run_id=ctx.run_id))

    assert isinstance(report, QualityReport)
    assert len(report) == 0
    assert report.dropped == ()
    assert report.run_id == ctx.run_id


async def test_evaluation_is_deterministic(ctx: RunContext) -> None:
    """The same input always yields an identical report."""
    clips = _clip_set(ctx.run_id, 3)
    briefs = _brief_list(ctx.run_id, 3)

    first = await QualityGate(checker=StubClipChecker(reject_budget={"b1": 1})).run(
        ctx, clips, briefs=briefs
    )
    second = await QualityGate(checker=StubClipChecker(reject_budget={"b1": 1})).run(
        ctx, clips, briefs=briefs
    )

    assert first == second


class _ExplodingChecker(ClipChecker):
    """A checker that raises for one target brief and approves the rest."""

    def __init__(self, boom_brief_id: str) -> None:
        self._boom_brief_id = boom_brief_id

    @property
    def name(self) -> str:
        return "exploding_checker"

    async def evaluate(self, ctx: RunContext, clip: RenderedClip) -> ClipVerdict:
        if clip.brief_id == self._boom_brief_id:
            raise RuntimeError("checker exploded")
        return ClipVerdict(
            clip_id=clip.id,
            brief_id=clip.brief_id,
            trend_id=clip.trend_id,
            approved=True,
        )


async def test_checker_exception_becomes_drop_not_a_failed_run(ctx: RunContext) -> None:
    """A per-clip exception is turned into a drop-with-reason, run still completes."""
    clips = _clip_set(ctx.run_id, 3)
    gate = QualityGate(checker=_ExplodingChecker("b1"))

    report = await gate.run(ctx, clips, briefs=_brief_list(ctx.run_id, 3))

    assert [c.brief_id for c in report.approved] == ["b0", "b2"]
    assert len(report.dropped) == 1
    assert report.dropped[0].brief_id == "b1"
    assert any("error" in reason for reason in report.dropped[0].reasons)


async def test_default_checker_and_factory_are_stubs() -> None:
    """The gate defaults to the deterministic offline stub checker and a factory."""
    gate = QualityGate()
    assert isinstance(gate._checker, StubClipChecker)
    assert isinstance(gate._clip_factory, ClipFactory)


class _ConcurrencyProbeChecker(ClipChecker):
    """A checker that tracks the high-water mark of concurrent evaluations."""

    def __init__(self) -> None:
        self.live = 0
        self.high_water = 0

    @property
    def name(self) -> str:
        return "concurrency_probe"

    async def evaluate(self, ctx: RunContext, clip: RenderedClip) -> ClipVerdict:
        self.live += 1
        self.high_water = max(self.high_water, self.live)
        # Hold the slot long enough for the pool to saturate.
        await asyncio.sleep(0.02)
        self.live -= 1
        return ClipVerdict(
            clip_id=clip.id,
            brief_id=clip.brief_id,
            trend_id=clip.trend_id,
            approved=True,
        )


async def test_fan_out_reaches_but_never_exceeds_concurrency_cap() -> None:
    """The fan-out runs concurrently up to ctx.config.concurrency_degree, no more.

    Modelled after test_fan_out_reaches_but_never_exceeds_concurrency_cap in
    test_clip_factory.py: with more clips than the cap and a checker that holds
    each slot, the high-water mark must reach the cap (proving real concurrency)
    but never exceed it (proving bounded_map governs the fan-out).
    """
    concurrency = 3
    config = RunConfig(concurrency_degree=concurrency, target_clip_count=100)
    ctx = RunContext.new(run_id="concurrency-run", config=config)
    clips = _clip_set(ctx.run_id, 12)
    probe = _ConcurrencyProbeChecker()
    gate = QualityGate(checker=probe)

    report = await gate.run(ctx, clips)

    assert len(report) == 12
    assert probe.high_water <= concurrency
    # With 12 clips and a cap of 3, the pool should actually reach the cap.
    assert probe.high_water == concurrency
    # Order is still preserved despite concurrent completion.
    assert [c.id for c in report.approved] == [c.id for c in clips.clips]
