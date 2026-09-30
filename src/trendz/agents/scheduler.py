"""Scheduler & Optimizer: the fifth agent in the pipeline (DESIGN.md stage 5).

The Scheduler & Optimizer runs after the Quality & Safety Gate. It consumes the
gate's :class:`~trendz.contracts.QualityReport` (its ``approved`` clips) and
produces a :class:`~trendz.contracts.PublishPlan` organized by platform, the
stage-6 Publisher fan-out boundary.

For each platform it:
    1. groups the approved clips by :attr:`~trendz.contracts.RenderedClip.platform`;
    2. orders them deterministically (by ``clip.id``) so the plan is stable;
    3. asks the injected
       :class:`~trendz.timing.base.AudienceTimingProvider` for that platform's
       optimal minute-of-day slot and pins each post's ``scheduled_at`` to the
       run's deterministic base time on that slot;
    4. enforces spacing: consecutive posts on the SAME platform are pushed
       forward so they are never closer than ``ctx.config.min_post_gap_minutes``
       apart (avoiding platform spam flags);
    5. expands each clip into ``ctx.config.ab_variant_count`` deterministic A/B
       caption/hashtag variants, each its own :class:`~trendz.contracts.ScheduledPost`
       (variants are separate posts, so spacing applies across them too).

Everything is deterministic and offline: there is no live clock, no network, and
no API keys. Scheduled times derive from an injectable ``base_time`` (defaulting
deterministically) plus the provider's per-platform hint plus the spacing gap, so
two runs with the same inputs produce equal plans.

:class:`~trendz.contracts.RenderedClip` does not carry caption/hashtags (those
live on the originating :class:`~trendz.contracts.ClipBrief`), so
:meth:`Scheduler.run` accepts the originating briefs threaded from the
orchestrator via the keyword-only ``briefs`` argument, exactly as the Quality &
Safety Gate accepts ``briefs``. It builds a ``brief_id -> ClipBrief`` lookup to
source each clip's caption, CTA, and hashtags; when a brief is missing the
Scheduler falls back to a deterministic placeholder caption derived from the clip
id so it stays robust.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trendz.agents.base import BaseAgent
from trendz.contracts import (
    ClipBrief,
    ClipBriefList,
    Platform,
    PublishPlan,
    QualityReport,
    RenderedClip,
    RunContext,
    ScheduledPost,
)
from trendz.timing.base import AudienceTimingProvider
from trendz.timing.stub_provider import StubAudienceTimingProvider

# Deterministic anchor used when no base_time is injected: a fixed, aware-UTC
# midnight so scheduled times are reproducible offline rather than dependent on a
# live wall clock. The provider's minute-of-day slots are added on top of this.
_DEFAULT_BASE_TIME = datetime(2000, 1, 1, tzinfo=timezone.utc)


class Scheduler(BaseAgent[QualityReport, PublishPlan]):
    """Plans approved clips into a per-platform, deterministically-timed ``PublishPlan``."""

    def __init__(
        self,
        provider: AudienceTimingProvider | None = None,
        *,
        base_time: datetime | None = None,
    ) -> None:
        """Create the agent.

        Args:
            provider: The audience-timing provider supplying each platform's
                optimal-post-time signal (dependency-injected). Defaults to the
                deterministic offline
                :class:`~trendz.timing.stub_provider.StubAudienceTimingProvider`.
            base_time: An optional aware-UTC base time the plan is anchored to,
                so scheduling is reproducible in tests. When ``None``, a fixed
                deterministic anchor is used rather than a live wall clock, which
                keeps runs and tests from being flaky.
        """
        super().__init__()
        self._provider = provider if provider is not None else StubAudienceTimingProvider()
        self._base_time = base_time if base_time is not None else _DEFAULT_BASE_TIME

    @property
    def name(self) -> str:
        return "scheduler"

    def _variant_caption(self, base_caption: str, variant: int) -> str:
        """Return a deterministic caption for an A/B ``variant`` (0 is the base)."""
        if variant == 0:
            return base_caption
        # A stable, human-readable suffix so variants differ deterministically
        # without any randomness. Variant 1 is "... (v2)", variant 2 "... (v3)".
        return f"{base_caption} (v{variant + 1})"

    def _variant_hashtags(self, base_hashtags: tuple[str, ...], variant: int) -> tuple[str, ...]:
        """Return deterministic hashtags for an A/B ``variant`` (0 is the base).

        Non-base variants rotate the hashtag order by the variant index, a
        stable permutation that produces a distinct-but-deterministic tag set
        without inventing new tags.
        """
        if variant == 0 or not base_hashtags:
            return base_hashtags
        shift = variant % len(base_hashtags)
        return base_hashtags[shift:] + base_hashtags[:shift]

    def _base_caption_for(self, clip: RenderedClip, brief: ClipBrief | None) -> str:
        """Source a clip's base caption from its brief, or a deterministic fallback."""
        if brief is not None:
            return brief.caption
        # Robust fallback when no originating brief is available: a stable
        # placeholder derived from the clip id (no network, no randomness).
        return f"Clip {clip.id}"

    async def run(
        self,
        ctx: RunContext,
        payload: QualityReport,
        *,
        briefs: ClipBriefList | None = None,
    ) -> PublishPlan:
        """Plan the approved clips into a per-platform ``PublishPlan``.

        Approved clips are grouped by platform, ordered deterministically by
        ``clip.id``, timed from the run's base time plus the injected provider's
        per-platform optimal slot, spaced at least ``min_post_gap_minutes`` apart
        on each platform, and expanded into ``ab_variant_count`` deterministic
        A/B variants. An empty ``QualityReport.approved`` yields an empty
        ``PublishPlan`` without consulting the provider.

        Args:
            ctx: The shared run context.
            payload: The Quality Gate's report; its ``approved`` clips are planned.
            briefs: The originating briefs, threaded from the orchestrator, used
                to source each clip's caption/hashtags (a
                ``RenderedClip`` does not carry them). When a brief is absent a
                deterministic placeholder caption is used instead.
        """
        brief_lookup: dict[str, ClipBrief] = (
            {brief.id: brief for brief in briefs.briefs} if briefs is not None else {}
        )

        variant_count = ctx.config.ab_variant_count
        gap = timedelta(minutes=ctx.config.min_post_gap_minutes)

        # Group approved clips by platform for per-platform timing and spacing.
        by_platform: dict[Platform, list[RenderedClip]] = {}
        for clip in payload.approved:
            by_platform.setdefault(clip.platform, []).append(clip)

        posts: list[ScheduledPost] = []
        for platform in sorted(by_platform):
            clips = sorted(by_platform[platform], key=lambda c: c.id)
            slots = self._provider.optimal_minutes_of_day(ctx, platform)
            # The first slot is the platform's single best window; anchor every
            # post on this platform to base_time + that minute-of-day.
            slot_minutes = slots[0] if slots else 0
            slot_start = self._base_time + timedelta(minutes=slot_minutes)

            # Spacing runs across every post on the platform, including A/B
            # variants (they are separate posts). ``next_at`` is the earliest an
            # upcoming post may be scheduled; we never place two closer than gap.
            next_at = slot_start
            for clip in clips:
                brief = brief_lookup.get(clip.brief_id)
                base_caption = self._base_caption_for(clip, brief)
                base_hashtags = brief.hashtags if brief is not None else ()
                for variant in range(variant_count):
                    scheduled_at = max(slot_start, next_at)
                    posts.append(
                        ScheduledPost(
                            clip_id=clip.id,
                            brief_id=clip.brief_id,
                            trend_id=clip.trend_id,
                            platform=platform,
                            scheduled_at=scheduled_at,
                            caption=self._variant_caption(base_caption, variant),
                            hashtags=self._variant_hashtags(base_hashtags, variant),
                            variant=variant,
                        )
                    )
                    next_at = scheduled_at + gap

        plan = PublishPlan(run_id=ctx.run_id, posts=tuple(posts))
        self.log.info(
            "publish_plan_ready",
            posts=len(plan),
            platforms=len(plan.platforms),
            clips_in=len(payload.approved),
            ab_variant_count=variant_count,
            min_post_gap_minutes=ctx.config.min_post_gap_minutes,
            provider=self._provider.name,
        )
        return plan
