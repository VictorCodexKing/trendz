"""Publisher: the sixth agent in the pipeline (the per-platform fan-out).

The Publisher is DESIGN.md stage 6. It receives the
:class:`~trendz.contracts.PublishPlan` planned by the Scheduler & Optimizer and
fans out PER PLATFORM: it spawns one "Platform Publisher" coroutine per platform
and runs them concurrently up to the run's configured limit. Each Platform
Publisher publishes its platform's whole slice (in scheduled order) through the
dependency-injected :class:`~trendz.publishers.base.PublisherClient`, and the
agent assembles every outcome into a :class:`~trendz.contracts.PostResults`.

Concurrency is not hand-rolled here: the per-platform work is routed through
:func:`~trendz.concurrency.bounded_map` with
``concurrency = ctx.config.concurrency_degree``, so no more than that many
platforms publish at once. The fan-out iterable is ``payload.slices()`` (one
``(platform, posts)`` pair per platform), exactly the boundary
:class:`~trendz.contracts.PublishPlan` exposes for this stage.

Two invariants match the design:
    - IDEMPOTENCY: each post gets a stable :attr:`idempotency_key` derived from
      ``run_id + clip_id + platform + variant`` (see :func:`_idempotency_key`).
      The Publisher also dedupes within the run by tracking keys it has already
      dispatched, so a duplicated post in the plan is not double-dispatched
      (belt-and-suspenders with the client's own idempotency guarantee).
    - RECORD-AND-CONTINUE: one failed post must never block the rest. Each
      single-post publish is wrapped so a failure becomes a failed
      :class:`~trendz.contracts.PostResult` with the reason and the remaining
      posts still publish (mirrors the Quality Gate's never-raise contract).

The actual publishing is delegated to the injected client (defaulting to the
deterministic :class:`~trendz.publishers.stub_client.StubPublisherClient`) so a
real platform API can be swapped in later without touching the fan-out logic.
"""

from __future__ import annotations

from trendz.agents.base import BaseAgent
from trendz.concurrency import bounded_map
from trendz.contracts import (
    Platform,
    PostResult,
    PostResults,
    PublishPlan,
    RunContext,
    ScheduledPost,
)
from trendz.publishers.base import PublisherClient
from trendz.publishers.stub_client import StubPublisherClient


def _idempotency_key(run_id: str, post: ScheduledPost) -> str:
    """Derive a stable idempotency key for one scheduled post.

    The key is ``run_id:clip_id:platform:variant`` - the tuple that uniquely
    identifies a post within a run (a clip may be published to several platforms
    and in several A/B variants, but each ``(clip, platform, variant)`` is
    published exactly once). Being a pure function of those fields, the same post
    always yields the same key, so a retried or duplicated post dedupes.
    """
    return f"{run_id}:{post.clip_id}:{post.platform}:{post.variant}"


class Publisher(BaseAgent[PublishPlan, PostResults]):
    """Fans a publisher-client out per platform into a ``PostResults``."""

    def __init__(self, client: PublisherClient | None = None) -> None:
        """Create the agent.

        Args:
            client: The publisher-client used to publish each post
                (dependency-injected). Defaults to the deterministic offline
                :class:`~trendz.publishers.stub_client.StubPublisherClient`.
        """
        super().__init__()
        self._client = client if client is not None else StubPublisherClient()

    @property
    def name(self) -> str:
        return "publisher"

    async def _publish_post(
        self, ctx: RunContext, post: ScheduledPost, idempotency_key: str
    ) -> PostResult:
        """Publish one post, recording success or failure without ever raising.

        A publish failure is caught and turned into a failed ``PostResult`` with
        the reason, so the remaining posts in the slice still publish (mirrors
        ``QualityGate._gate_clip``'s record-and-continue contract).
        """
        try:
            outcome = await self._client.publish(ctx, post, idempotency_key)
            return PostResult(
                clip_id=post.clip_id,
                brief_id=post.brief_id,
                trend_id=post.trend_id,
                platform=post.platform,
                variant=post.variant,
                idempotency_key=idempotency_key,
                succeeded=True,
                post_id=outcome.post_id,
                post_url=outcome.post_url,
                error=None,
                published_at=outcome.published_at,
            )
        except Exception as exc:  # noqa: BLE001 - one bad post must never block the run.
            reason = f"error: {type(exc).__name__}: {exc}"
            self.log.info(
                "post_failed",
                clip_id=post.clip_id,
                platform=post.platform,
                variant=post.variant,
                reasons=[reason],
            )
            return PostResult(
                clip_id=post.clip_id,
                brief_id=post.brief_id,
                trend_id=post.trend_id,
                platform=post.platform,
                variant=post.variant,
                idempotency_key=idempotency_key,
                succeeded=False,
                post_id=None,
                post_url=None,
                error=reason,
                published_at=None,
            )

    async def _publish_platform(
        self, ctx: RunContext, platform: Platform, posts: tuple[ScheduledPost, ...]
    ) -> list[PostResult]:
        """Publish one platform's whole slice in scheduled order.

        This is a single "Platform Publisher" coroutine. It dedupes within the
        run by tracking the idempotency keys it has already dispatched, so a
        duplicated post in the plan is published only once (belt-and-suspenders
        with the client's own idempotency guarantee).
        """
        results: list[PostResult] = []
        dispatched: set[str] = set()
        for post in posts:
            key = _idempotency_key(ctx.run_id, post)
            if key in dispatched:
                # Duplicate post within this slice: do not dispatch again.
                self.log.info(
                    "post_deduped",
                    clip_id=post.clip_id,
                    platform=post.platform,
                    variant=post.variant,
                    idempotency_key=key,
                )
                continue
            dispatched.add(key)
            results.append(await self._publish_post(ctx, post, key))
        return results

    async def run(self, ctx: RunContext, payload: PublishPlan) -> PostResults:
        """Publish every scheduled post, fanning out per platform.

        The per-platform work is routed through
        :func:`~trendz.concurrency.bounded_map` capped at
        ``ctx.config.concurrency_degree``, so no more than that many platforms
        publish at once. Results are reassembled in a deterministic order
        (grouped by sorted platform - the order ``payload.slices()`` yields -
        then scheduled order within a platform). An empty ``PublishPlan`` yields
        an empty ``PostResults`` without dispatching any work.
        """
        per_platform: list[list[PostResult]] = await bounded_map(
            lambda slice_: self._publish_platform(ctx, slice_[0], slice_[1]),
            payload.slices(),
            ctx.config.concurrency_degree,
        )
        results = tuple(result for group in per_platform for result in group)

        report = PostResults(run_id=ctx.run_id, results=results)
        self.log.info(
            "publish_done",
            published=len(report),
            succeeded=len(report.succeeded),
            failed=len(report.failed),
            platforms=len(report.platforms),
            posts_in=len(payload),
            client=self._client.name,
            concurrency=ctx.config.concurrency_degree,
        )
        return report
