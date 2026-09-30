"""Offline tests for the Scheduler & Optimizer agent and its PublishPlan."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trendz.agents.scheduler import Scheduler
from trendz.contracts import (
    ClipBrief,
    ClipBriefList,
    Platform,
    PublishPlan,
    QualityReport,
    RenderedClip,
    RunConfig,
    RunContext,
)
from trendz.timing.stub_provider import StubAudienceTimingProvider

BASE_TIME = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _brief(idx: int, platform: Platform = "tiktok") -> ClipBrief:
    """Build a deterministic clip brief for tests."""
    return ClipBrief(
        id=f"b{idx}",
        trend_id=f"t{idx}",
        title=f"Trend {idx}",
        angle="angle",
        hook="hook",
        platform=platform,
        aspect_ratio="9:16",
        target_length_seconds=30,
        caption=f"caption-{idx}",
        cta="cta",
        hashtags=("fyp", "trend", "viral"),
        footage_kind="sourced",
    )


def _clip(idx: int, platform: Platform = "tiktok", **overrides: object) -> RenderedClip:
    """Build a deterministic rendered clip for tests (mirrors the stub worker)."""
    fields: dict[str, object] = {
        "id": f"clip-b{idx}",
        "brief_id": f"b{idx}",
        "trend_id": f"t{idx}",
        "platform": platform,
        "aspect_ratio": "9:16",
        "duration_seconds": 30,
        "video_uri": f"stub://clips/b{idx}.mp4",
        "has_audio": True,
        "has_captions": True,
    }
    fields.update(overrides)
    return RenderedClip(**fields)  # type: ignore[arg-type]


def _report(run_id: str, clips: tuple[RenderedClip, ...]) -> QualityReport:
    """Build a QualityReport whose approved set is ``clips``."""
    return QualityReport(run_id=run_id, approved=clips)


def _brief_list(run_id: str, briefs: tuple[ClipBrief, ...]) -> ClipBriefList:
    """Build a ClipBriefList from explicit briefs."""
    return ClipBriefList(run_id=run_id, briefs=briefs)


async def test_grouping_and_slices_route_posts_by_platform(ctx: RunContext) -> None:
    """Posts are grouped by platform; for_platform/slices return the right posts."""
    clips = (
        _clip(0, platform="tiktok"),
        _clip(1, platform="instagram"),
        _clip(2, platform="tiktok"),
        _clip(3, platform="youtube_shorts"),
    )
    briefs = _brief_list(
        ctx.run_id,
        (
            _brief(0, platform="tiktok"),
            _brief(1, platform="instagram"),
            _brief(2, platform="tiktok"),
            _brief(3, platform="youtube_shorts"),
        ),
    )
    scheduler = Scheduler(base_time=BASE_TIME)

    plan = await scheduler.run(ctx, _report(ctx.run_id, clips), briefs=briefs)

    assert isinstance(plan, PublishPlan)
    assert plan.run_id == ctx.run_id
    assert len(plan) == 4
    # platforms property is sorted and distinct.
    assert plan.platforms == ("instagram", "tiktok", "youtube_shorts")
    assert {p.clip_id for p in plan.for_platform("tiktok")} == {"clip-b0", "clip-b2"}
    assert {p.clip_id for p in plan.for_platform("instagram")} == {"clip-b1"}
    assert {p.clip_id for p in plan.for_platform("youtube_shorts")} == {"clip-b3"}
    # slices() yields one (platform, posts) pair per platform in sorted order.
    slice_platforms = [platform for platform, _ in plan.slices()]
    assert slice_platforms == ["instagram", "tiktok", "youtube_shorts"]
    for platform, platform_posts in plan.slices():
        assert all(p.platform == platform for p in platform_posts)


async def test_ordering_is_deterministic_by_clip_id(ctx: RunContext) -> None:
    """Within a platform, posts are ordered deterministically by clip id."""
    # Supplied out of clip-id order; scheduling must sort them.
    clips = (_clip(2), _clip(0), _clip(1))
    briefs = _brief_list(ctx.run_id, (_brief(0), _brief(1), _brief(2)))
    scheduler = Scheduler(base_time=BASE_TIME)

    plan = await scheduler.run(ctx, _report(ctx.run_id, clips), briefs=briefs)

    tiktok = plan.for_platform("tiktok")
    assert [p.clip_id for p in tiktok] == ["clip-b0", "clip-b1", "clip-b2"]


async def test_scheduled_at_is_base_time_plus_platform_slot(ctx: RunContext) -> None:
    """The first post is anchored to base_time + the platform's optimal slot."""
    clips = (_clip(0, platform="tiktok"),)
    briefs = _brief_list(ctx.run_id, (_brief(0, platform="tiktok"),))
    scheduler = Scheduler(base_time=BASE_TIME)

    plan = await scheduler.run(ctx, _report(ctx.run_id, clips), briefs=briefs)

    # StubAudienceTimingProvider puts tiktok's best slot at 18:00 UTC.
    expected = BASE_TIME + timedelta(minutes=18 * 60)
    assert plan.for_platform("tiktok")[0].scheduled_at == expected


async def test_spacing_enforced_between_same_platform_posts(ctx: RunContext) -> None:
    """Two posts on the same platform are at least min_post_gap_minutes apart."""
    config = RunConfig(min_post_gap_minutes=45)
    ctx = RunContext.new(run_id="gap-run", config=config)
    clips = (_clip(0), _clip(1), _clip(2))
    briefs = _brief_list(ctx.run_id, (_brief(0), _brief(1), _brief(2)))
    scheduler = Scheduler(base_time=BASE_TIME)

    plan = await scheduler.run(ctx, _report(ctx.run_id, clips), briefs=briefs)

    posts = plan.for_platform("tiktok")
    assert len(posts) == 3
    for earlier, later in zip(posts, posts[1:], strict=False):
        delta = later.scheduled_at - earlier.scheduled_at
        assert delta >= timedelta(minutes=45)


async def test_close_provider_hint_is_pushed_forward(ctx: RunContext) -> None:
    """Even with the same slot for every post, spacing pushes later posts forward."""
    config = RunConfig(min_post_gap_minutes=30)
    ctx = RunContext.new(run_id="push-run", config=config)
    # Force one slot so every post would want the identical time absent spacing.
    provider = StubAudienceTimingProvider(slots={"tiktok": (10 * 60,)})
    clips = (_clip(0), _clip(1))
    briefs = _brief_list(ctx.run_id, (_brief(0), _brief(1)))
    scheduler = Scheduler(provider=provider, base_time=BASE_TIME)

    plan = await scheduler.run(ctx, _report(ctx.run_id, clips), briefs=briefs)

    posts = plan.for_platform("tiktok")
    assert posts[0].scheduled_at == BASE_TIME + timedelta(minutes=10 * 60)
    assert posts[1].scheduled_at == posts[0].scheduled_at + timedelta(minutes=30)


async def test_ab_variants_produce_multiple_distinct_posts() -> None:
    """ab_variant_count=2 yields 2 posts per clip with variants 0/1 and distinct captions."""
    config = RunConfig(ab_variant_count=2, min_post_gap_minutes=15)
    ctx = RunContext.new(run_id="ab-run", config=config)
    clips = (_clip(0),)
    briefs = _brief_list(ctx.run_id, (_brief(0),))
    scheduler = Scheduler(base_time=BASE_TIME)

    plan = await scheduler.run(ctx, _report(ctx.run_id, clips), briefs=briefs)

    posts = plan.for_platform("tiktok")
    assert len(posts) == 2
    assert [p.variant for p in posts] == [0, 1]
    assert posts[0].caption != posts[1].caption
    assert posts[0].caption == "caption-0"
    # Variants are separate posts, so spacing applies across them too.
    assert posts[1].scheduled_at - posts[0].scheduled_at >= timedelta(minutes=15)


async def test_single_variant_yields_one_post_per_clip(ctx: RunContext) -> None:
    """ab_variant_count=1 (the default) yields exactly one variant-0 post per clip."""
    clips = (_clip(0), _clip(1))
    briefs = _brief_list(ctx.run_id, (_brief(0), _brief(1)))
    scheduler = Scheduler(base_time=BASE_TIME)

    plan = await scheduler.run(ctx, _report(ctx.run_id, clips), briefs=briefs)

    posts = plan.for_platform("tiktok")
    assert len(posts) == 2
    assert all(p.variant == 0 for p in posts)


async def test_empty_report_yields_empty_plan(ctx: RunContext) -> None:
    """An empty QualityReport.approved yields an empty PublishPlan."""
    scheduler = Scheduler(base_time=BASE_TIME)

    plan = await scheduler.run(ctx, QualityReport(run_id=ctx.run_id))

    assert isinstance(plan, PublishPlan)
    assert len(plan) == 0
    assert plan.platforms == ()
    assert plan.slices() == ()
    assert plan.run_id == ctx.run_id


async def test_missing_brief_falls_back_to_placeholder_caption(ctx: RunContext) -> None:
    """A clip with no originating brief still schedules with a deterministic caption."""
    clips = (_clip(0),)
    scheduler = Scheduler(base_time=BASE_TIME)

    plan = await scheduler.run(ctx, _report(ctx.run_id, clips))  # no briefs threaded

    post = plan.for_platform("tiktok")[0]
    assert post.caption == "Clip clip-b0"
    assert post.hashtags == ()


async def test_scheduling_is_deterministic(ctx: RunContext) -> None:
    """Two runs with the same inputs produce equal PublishPlans."""
    clips = (_clip(0, platform="instagram"), _clip(1, platform="tiktok"), _clip(2))
    briefs = _brief_list(
        ctx.run_id,
        (_brief(0, platform="instagram"), _brief(1, platform="tiktok"), _brief(2)),
    )

    first = await Scheduler(base_time=BASE_TIME).run(ctx, _report(ctx.run_id, clips), briefs=briefs)
    second = await Scheduler(base_time=BASE_TIME).run(
        ctx, _report(ctx.run_id, clips), briefs=briefs
    )

    assert first == second


async def test_default_provider_is_the_stub() -> None:
    """The scheduler defaults its provider to the deterministic offline stub."""
    scheduler = Scheduler()
    assert isinstance(scheduler._provider, StubAudienceTimingProvider)
