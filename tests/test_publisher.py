"""Offline tests for the Publisher agent and its per-platform bounded fan-out."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from trendz.agents.publisher import Publisher, _idempotency_key
from trendz.contracts import (
    Platform,
    PostResults,
    PublishPlan,
    RunConfig,
    RunContext,
    ScheduledPost,
)
from trendz.publishers.base import PublisherClient, PublishOutcome
from trendz.publishers.stub_client import StubPublisherClient

_BASE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _post(
    idx: int,
    platform: Platform = "tiktok",
    variant: int = 0,
    minute_offset: int = 0,
) -> ScheduledPost:
    """Build a deterministic scheduled post for tests."""
    return ScheduledPost(
        clip_id=f"clip-b{idx}",
        brief_id=f"b{idx}",
        trend_id=f"t{idx}",
        platform=platform,
        scheduled_at=_BASE.replace(minute=minute_offset % 60),
        caption=f"caption {idx}",
        hashtags=("fyp",),
        variant=variant,
    )


def _plan(run_id: str, posts: tuple[ScheduledPost, ...]) -> PublishPlan:
    """Build a PublishPlan from posts."""
    return PublishPlan(run_id=run_id, posts=posts)


async def test_happy_path_every_post_succeeds_with_attribution(ctx: RunContext) -> None:
    """Every scheduled post yields a succeeded PostResult with correct linkage."""
    posts = (
        _post(0, "tiktok", minute_offset=0),
        _post(1, "instagram", minute_offset=5),
        _post(2, "youtube_shorts", minute_offset=10),
        _post(3, "facebook", minute_offset=15),
    )
    plan = _plan(ctx.run_id, posts)
    publisher = Publisher(StubPublisherClient())

    results = await publisher.run(ctx, plan)

    assert isinstance(results, PostResults)
    assert results.run_id == ctx.run_id
    assert len(results) == len(posts)
    assert len(results.succeeded) == len(posts)
    assert len(results.failed) == 0
    by_clip = {result.clip_id: result for result in results.results}
    for post in posts:
        result = by_clip[post.clip_id]
        assert result.succeeded
        assert result.brief_id == post.brief_id
        assert result.trend_id == post.trend_id
        assert result.platform == post.platform
        assert result.variant == post.variant
        assert result.post_id is not None
        assert result.post_url is not None
        assert result.published_at == post.scheduled_at
        assert result.error is None
        assert result.idempotency_key == _idempotency_key(ctx.run_id, post)


async def test_results_grouped_per_platform(ctx: RunContext) -> None:
    """for_platform / slices partition results and cover results.platforms."""
    posts = (
        _post(0, "tiktok"),
        _post(1, "tiktok", minute_offset=5),
        _post(2, "instagram"),
        _post(3, "facebook"),
    )
    plan = _plan(ctx.run_id, posts)

    results = await Publisher(StubPublisherClient()).run(ctx, plan)

    assert set(results.platforms) == {"tiktok", "instagram", "facebook"}
    assert list(results.platforms) == sorted(set(results.platforms))
    assert [platform for platform, _ in results.slices()] == list(results.platforms)
    total = 0
    for platform, platform_results in results.slices():
        assert platform_results == results.for_platform(platform)
        assert all(r.platform == platform for r in platform_results)
        total += len(platform_results)
    assert total == len(results)


class _ConcurrencyProbeClient(PublisherClient):
    """A client that tracks the high-water mark of concurrent platform publishes."""

    def __init__(self) -> None:
        self.live = 0
        self.high_water = 0

    @property
    def name(self) -> str:
        return "concurrency_probe"

    async def publish(
        self, ctx: RunContext, post: ScheduledPost, idempotency_key: str
    ) -> PublishOutcome:
        self.live += 1
        self.high_water = max(self.high_water, self.live)
        # Hold the slot long enough for the per-platform pool to saturate.
        await asyncio.sleep(0.02)
        self.live -= 1
        return PublishOutcome(
            post_id=f"post-{idempotency_key}",
            post_url=f"stub://{post.platform}/posts/{idempotency_key}",
            published_at=post.scheduled_at,
        )


async def test_fan_out_reaches_but_never_exceeds_concurrency_cap() -> None:
    """The per-platform fan-out runs concurrently up to the cap, never beyond.

    A plan spanning all four platforms with a cap of 2: the high-water mark must
    reach the cap (proving real concurrency across platform publishers) but
    never exceed it (proving bounded_map governs the per-platform fan-out).
    """
    concurrency = 2
    config = RunConfig(concurrency_degree=concurrency)
    ctx = RunContext.new(run_id="concurrency-run", config=config)
    posts = (
        _post(0, "tiktok"),
        _post(1, "instagram"),
        _post(2, "youtube_shorts"),
        _post(3, "facebook"),
    )
    plan = _plan(ctx.run_id, posts)
    probe = _ConcurrencyProbeClient()

    results = await Publisher(probe).run(ctx, plan)

    assert len(results) == 4
    assert probe.high_water <= concurrency
    # Four platforms with a cap of 2: the pool should actually reach the cap.
    assert probe.high_water == concurrency


async def test_idempotency_no_double_post_on_repeat(ctx: RunContext) -> None:
    """Publishing the same plan twice does not re-publish; prior result returned."""
    posts = (_post(0, "tiktok"), _post(1, "instagram"))
    plan = _plan(ctx.run_id, posts)
    client = StubPublisherClient()
    publisher = Publisher(client)

    first = await publisher.run(ctx, plan)
    assert client.publish_count == 2

    second = await publisher.run(ctx, plan)
    # No real re-publish happened on the duplicate run.
    assert client.publish_count == 2
    # The prior post ids are returned unchanged.
    first_ids = {r.clip_id: r.post_id for r in first.results}
    second_ids = {r.clip_id: r.post_id for r in second.results}
    assert first_ids == second_ids


async def test_idempotency_duplicate_post_in_plan_not_double_dispatched(ctx: RunContext) -> None:
    """A duplicated post in the plan is dispatched only once (belt-and-suspenders)."""
    dup = _post(0, "tiktok")
    plan = _plan(ctx.run_id, (dup, dup))
    client = StubPublisherClient()

    results = await Publisher(client).run(ctx, plan)

    # Only one real publish and one result despite the duplicate in the plan.
    assert client.publish_count == 1
    assert len(results) == 1


async def test_one_failed_post_does_not_block_the_rest(ctx: RunContext) -> None:
    """A forced platform failure is recorded while every other post succeeds."""
    posts = (
        _post(0, "tiktok"),
        _post(1, "instagram"),
        _post(2, "facebook"),
    )
    plan = _plan(ctx.run_id, posts)
    client = StubPublisherClient(fail_platforms=frozenset({"instagram"}))

    results = await Publisher(client).run(ctx, plan)

    assert len(results) == 3
    failed = results.failed
    assert len(failed) == 1
    assert failed[0].platform == "instagram"
    assert failed[0].succeeded is False
    assert failed[0].error is not None
    assert failed[0].post_id is None
    # The other two platforms still succeeded.
    assert {r.platform for r in results.succeeded} == {"tiktok", "facebook"}


async def test_empty_plan_yields_empty_results_without_dispatch(ctx: RunContext) -> None:
    """An empty PublishPlan yields an empty PostResults without publishing."""
    client = StubPublisherClient()

    results = await Publisher(client).run(ctx, PublishPlan(run_id=ctx.run_id))

    assert isinstance(results, PostResults)
    assert len(results) == 0
    assert results.run_id == ctx.run_id
    assert client.publish_count == 0


async def test_publishing_is_deterministic(ctx: RunContext) -> None:
    """Two runs over the same plan produce equal PostResults."""
    posts = (
        _post(0, "tiktok"),
        _post(1, "instagram"),
        _post(2, "youtube_shorts"),
        _post(3, "facebook"),
    )
    plan = _plan(ctx.run_id, posts)

    first = await Publisher(StubPublisherClient()).run(ctx, plan)
    second = await Publisher(StubPublisherClient()).run(ctx, plan)

    assert first == second


async def test_default_client_is_stub() -> None:
    """The Publisher defaults to the deterministic offline stub client."""
    publisher = Publisher()
    assert isinstance(publisher._client, StubPublisherClient)
