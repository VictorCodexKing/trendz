"""Shared typed data contracts for the Trendz pipeline.

These Pydantic v2 models are the vocabulary every agent speaks. The
Orchestrator builds a :class:`RunContext` from a :class:`RunConfig` and threads
it through the pipeline; Trend Scout emits a :class:`TrendList`; downstream
agents will add their own contracts (ClipBrief, RenderedClip, PublishPlan, ...)
following the same style.

Design choices:
    - Models that represent immutable facts or configuration are ``frozen`` so
      they can be shared safely across concurrent tasks without accidental
      mutation. :class:`BudgetLedger` is intentionally mutable because it tracks
      running spend over the lifetime of a run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    """Return an aware UTC timestamp (default factory for created_at fields)."""
    return datetime.now(timezone.utc)


class RunConfig(BaseModel):
    """Global configuration for a single pipeline run.

    Supplied by whoever triggers a run (a schedule, a trend spike, or a manual
    invocation). The Orchestrator reads it to decide how much work to do and how
    aggressively to parallelise.
    """

    model_config = ConfigDict(frozen=True)

    niche_filters: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Case-insensitive keywords a trend's title must match to be kept. "
            "Empty means no filtering."
        ),
    )
    target_clip_count: int = Field(
        default=5,
        ge=0,
        description="How many clips the run should aim to produce.",
    )
    concurrency_degree: int = Field(
        default=4,
        ge=1,
        description="Maximum number of concurrent workers in fan-out stages.",
    )
    budget_limit: float = Field(
        default=100.0,
        ge=0.0,
        description="Total spend cap (currency units) for the run.",
    )
    quota_limit: int = Field(
        default=1000,
        ge=0,
        description="Total external API call quota for the run.",
    )
    max_quality_retries: int = Field(
        default=2,
        ge=0,
        description=(
            "How many times the Quality & Safety Gate may re-render a rejected "
            "clip before dropping it. 0 means no retries: a rejected clip is "
            "dropped-and-logged immediately."
        ),
    )
    min_post_gap_minutes: int = Field(
        default=30,
        ge=0,
        description=(
            "Minimum spacing, in minutes, between two posts scheduled on the "
            "SAME platform. The Scheduler pushes later posts forward so no two "
            "same-platform posts land closer than this, avoiding spam flags. "
            "0 means no spacing is enforced."
        ),
    )
    ab_variant_count: int = Field(
        default=1,
        ge=1,
        description=(
            "How many caption/hashtag A/B variants the Scheduler produces per "
            "approved clip. 1 means no A/B testing (a single post per clip); "
            "N>1 emits N posts per clip with deterministic caption/hashtag "
            "variations and incrementing variant indices."
        ),
    )


class BudgetLedger(BaseModel):
    """Tracks spend and quota consumption over the lifetime of a run.

    Mutable by design: the Orchestrator debits it as work is performed and
    checks the ``can_*`` helpers before dispatching more work so the run
    throttles or aborts before overrunning its limits.
    """

    budget_limit: float = Field(default=100.0, ge=0.0)
    quota_limit: int = Field(default=1000, ge=0)
    spent: float = Field(default=0.0, ge=0.0)
    quota_used: int = Field(default=0, ge=0)

    @property
    def budget_remaining(self) -> float:
        """Currency units still available before hitting the budget cap."""
        return max(0.0, self.budget_limit - self.spent)

    @property
    def quota_remaining(self) -> int:
        """External API calls still available before hitting the quota cap."""
        return max(0, self.quota_limit - self.quota_used)

    def can_spend(self, amount: float) -> bool:
        """Return True if ``amount`` fits within the remaining budget."""
        return self.spent + amount <= self.budget_limit

    def can_use_quota(self, calls: int = 1) -> bool:
        """Return True if ``calls`` fit within the remaining quota."""
        return self.quota_used + calls <= self.quota_limit

    def charge(self, amount: float = 0.0, calls: int = 0) -> None:
        """Debit the ledger by ``amount`` spend and ``calls`` quota."""
        if amount < 0 or calls < 0:
            raise ValueError("charge amounts must be non-negative")
        self.spent += amount
        self.quota_used += calls

    @classmethod
    def from_config(cls, config: RunConfig) -> BudgetLedger:
        """Build a fresh ledger seeded from a run's configured limits."""
        return cls(budget_limit=config.budget_limit, quota_limit=config.quota_limit)


class RunContext(BaseModel):
    """Immutable context passed to every agent for a given run.

    Carries the run identity, the run configuration, and the (mutable) budget
    ledger. The ledger is a nested model so it can be debited in place while the
    surrounding context stays frozen.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    config: RunConfig
    budget: BudgetLedger
    created_at: datetime = Field(default_factory=_utcnow)

    @classmethod
    def new(cls, run_id: str, config: RunConfig) -> RunContext:
        """Construct a context with a budget ledger derived from ``config``."""
        return cls(run_id=run_id, config=config, budget=BudgetLedger.from_config(config))


class Trend(BaseModel):
    """A single discovered trend with its raw scoring signals.

    Signals are normalised to the ``[0, 1]`` range so the scoring function can
    combine them with tunable weights. ``score`` is assigned by Trend Scout;
    it defaults to 0.0 for a freshly fetched, not-yet-scored trend.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    source: str
    velocity: float = Field(default=0.0, ge=0.0, le=1.0, description="How fast it is rising.")
    relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Fit to the niche.")
    saturation: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How crowded/overdone it already is."
    )
    shelf_life: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How long it will stay relevant."
    )
    score: float = Field(default=0.0, description="Composite score assigned by Trend Scout.")

    def with_score(self, score: float) -> Trend:
        """Return a copy of this trend with ``score`` set (models are frozen)."""
        return self.model_copy(update={"score": score})


class TrendList(BaseModel):
    """An ordered, ranked collection of trends for a run.

    Trend Scout returns this. Trends are expected to be ordered descending by
    ``score``; :meth:`top` is a convenience for the downstream Content
    Strategist, which only needs the strongest N.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    trends: tuple[Trend, ...] = Field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.trends)

    def top(self, n: int) -> tuple[Trend, ...]:
        """Return the first ``n`` trends (the highest scored when ranked)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        return self.trends[:n]


# Short platform identifiers the Content Strategist targets. Each maps to a
# canonical short-video surface (YouTube Shorts, Instagram, TikTok, Facebook);
# the strategist chooses aspect ratio and length per platform. Kept as a Literal
# (not a free string) so a brief can only name a platform the pipeline knows how
# to render and publish.
Platform = Literal["youtube_shorts", "instagram", "tiktok", "facebook"]

# Canonical short-video aspect ratios the Assembly step supports (per DESIGN.md
# stage 3): vertical, square, and landscape.
AspectRatio = Literal["9:16", "1:1", "16:9"]

# Whether a clip's footage is sourced (stock/licensed B-roll) or AI-generated.
# Modelled as a Literal enum rather than a bare bool so the design's "sourced vs
# AI footage" decision reads explicitly at every call site and can grow a third
# option (e.g. "hybrid") without a signature change.
FootageKind = Literal["sourced", "ai"]


class ClipBrief(BaseModel):
    """A concrete, per-clip production brief emitted by the Content Strategist.

    A brief is the fan-out unit into the Clip Factory: one brief becomes one
    rendered clip. It captures the stage-2 creative decisions from DESIGN.md
    (angle/hook, platform, format, target length, caption/CTA, hashtags, sourced
    vs AI footage) and carries ``trend_id`` back to the originating
    :class:`Trend` so the Performance Analyst can attribute outcomes to a trend.

    Frozen because a brief is an immutable fact once planned: the Clip Factory
    consumes it concurrently and must never mutate it.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    trend_id: str = Field(description="ID of the originating Trend, for attribution.")
    title: str = Field(description="Human-readable title, derived from the trend.")
    angle: str = Field(description="The creative angle/framing for the clip.")
    hook: str = Field(description="The opening hook that stops the scroll.")
    platform: Platform = Field(description="Target short-video platform.")
    aspect_ratio: AspectRatio = Field(description="Frame aspect ratio for the platform.")
    target_length_seconds: int = Field(
        default=30,
        ge=1,
        le=180,
        description="Intended clip length in seconds.",
    )
    caption: str = Field(description="Post caption/description copy.")
    cta: str = Field(description="Call to action for the viewer.")
    hashtags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Hashtags to publish with the clip (without leading '#').",
    )
    footage_kind: FootageKind = Field(
        default="sourced",
        description="Whether footage is sourced (stock) or AI-generated.",
    )


class ClipBriefList(BaseModel):
    """An ordered collection of clip briefs for a run.

    The Content Strategist returns this; it is the fan-out boundary the Clip
    Factory maps a Clip Worker over. Mirrors :class:`TrendList`: frozen, with
    ``__len__`` and a :meth:`top` helper for consumers that only need the first
    N briefs.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    briefs: tuple[ClipBrief, ...] = Field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.briefs)

    def top(self, n: int) -> tuple[ClipBrief, ...]:
        """Return the first ``n`` briefs in order."""
        if n < 0:
            raise ValueError("n must be non-negative")
        return self.briefs[:n]


class RenderedClip(BaseModel):
    """A single assembled clip produced by a Clip Worker in the Clip Factory.

    One :class:`ClipBrief` becomes exactly one ``RenderedClip``. It carries the
    stage-3 assembly output from DESIGN.md (stitched audio + video + captions,
    formatted per platform aspect ratio) as stub media fields: the real Clip
    Worker will populate these with paths to FFmpeg/TTS output, while the
    deterministic offline worker fills them with mocked URIs. ``brief_id`` links
    back to the originating brief and ``trend_id`` carries attribution through so
    the Performance Analyst can trace an outcome all the way back to its trend.

    Frozen because a rendered clip is an immutable fact once assembled: the
    Quality Gate and downstream stages consume it concurrently and must never
    mutate it.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    brief_id: str = Field(description="ID of the originating ClipBrief.")
    trend_id: str = Field(description="ID of the originating Trend, for attribution.")
    platform: Platform = Field(description="Target short-video platform.")
    aspect_ratio: AspectRatio = Field(description="Frame aspect ratio for the platform.")
    duration_seconds: int = Field(
        ge=1,
        le=180,
        description="Actual assembled clip duration in seconds.",
    )
    video_uri: str = Field(description="URI/path to the assembled video (stubbed offline).")
    has_audio: bool = Field(
        default=True,
        description="Whether a voice/audio track was assembled (script + TTS marker).",
    )
    has_captions: bool = Field(
        default=True,
        description="Whether burned-in captions/subtitles were assembled.",
    )


class RenderedClipSet(BaseModel):
    """An ordered collection of rendered clips for a run.

    The Clip Factory returns this after fanning a Clip Worker out over the
    briefs. Mirrors :class:`ClipBriefList`/:class:`TrendList`: frozen, with
    ``__len__`` and a :meth:`top` helper for consumers that only need the first
    N clips. Order mirrors the input ``ClipBriefList`` because the fan-out routes
    through :func:`~trendz.concurrency.bounded_map`, which preserves input order.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    clips: tuple[RenderedClip, ...] = Field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.clips)

    def top(self, n: int) -> tuple[RenderedClip, ...]:
        """Return the first ``n`` clips in order."""
        if n < 0:
            raise ValueError("n must be non-negative")
        return self.clips[:n]


class ClipVerdict(BaseModel):
    """The Quality & Safety Gate's evaluation of a single rendered clip.

    Produced by a :class:`~trendz.checkers.base.ClipChecker` for one
    :class:`RenderedClip`, this captures DESIGN.md stage 4's per-clip decision:
    the ``technical_ok`` (resolution/audio/no black frames/length),
    ``safety_ok`` (brand safety, copyright/music licensing, platform policy),
    and ``engagement_score`` (the predicted-engagement signal) sub-checks, rolled
    up into a single ``approved`` flag. ``reasons`` records the human-readable
    rejection/safety reasons (empty when approved); the design notes these feed
    the future Performance Analyst / Learning Store as failure-learning signal.

    ``clip_id``/``brief_id``/``trend_id`` carry the same linkage the evaluated
    clip does so a verdict can be traced back to its clip, brief, and trend.

    Frozen because a verdict is an immutable fact once decided: the gate collects
    verdicts across concurrent tasks and must never mutate them.
    """

    model_config = ConfigDict(frozen=True)

    clip_id: str = Field(description="ID of the evaluated RenderedClip.")
    brief_id: str = Field(description="ID of the originating ClipBrief.")
    trend_id: str = Field(description="ID of the originating Trend, for attribution.")
    approved: bool = Field(description="Whether the clip passed the gate.")
    reasons: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Rejection/safety reasons; empty when approved.",
    )
    engagement_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Predicted-engagement score in [0, 1].",
    )
    technical_ok: bool = Field(
        default=True,
        description="Technical checks passed (resolution, audio, no black frames, length).",
    )
    safety_ok: bool = Field(
        default=True,
        description="Content-safety checks passed (brand safety, licensing, platform policy).",
    )


class QualityReport(BaseModel):
    """The Quality & Safety Gate's output: approved clips plus rejection signal.

    Returned by the gate after evaluating every :class:`RenderedClip` (fanning a
    :class:`~trendz.checkers.base.ClipChecker` out through
    :func:`~trendz.concurrency.bounded_map`). ``approved`` holds the clips that
    passed and flow on toward the Scheduler; ``dropped`` holds the verdicts of
    clips that failed the gate even after their bounded re-render retries were
    exhausted (or errored). The dropped verdicts carry their reasons for
    observability - DESIGN.md notes these become failure-learning signal for the
    future Performance Analyst / Learning Store.

    Mirrors :class:`RenderedClipSet`/:class:`ClipBriefList`: frozen, with
    ``__len__`` (over the approved clips) and a :meth:`top` helper. Approved
    order mirrors the input clip order because the fan-out routes through
    :func:`~trendz.concurrency.bounded_map`, which preserves input order.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    approved: tuple[RenderedClip, ...] = Field(default_factory=tuple)
    dropped: tuple[ClipVerdict, ...] = Field(
        default_factory=tuple,
        description="Verdicts of clips dropped after exhausting retries or erroring.",
    )

    def __len__(self) -> int:
        return len(self.approved)

    def top(self, n: int) -> tuple[RenderedClip, ...]:
        """Return the first ``n`` approved clips in order."""
        if n < 0:
            raise ValueError("n must be non-negative")
        return self.approved[:n]


class ScheduledPost(BaseModel):
    """A single planned post produced by the Scheduler & Optimizer (stage 5).

    One approved :class:`RenderedClip` becomes one or more scheduled posts: one
    per A/B caption variant (see :attr:`variant`). Each post pins the platform,
    the aware-UTC :attr:`scheduled_at` time the Scheduler chose (derived from a
    deterministic base time plus the platform's optimal-time hint and the
    configured minimum spacing gap), and the caption/hashtags to publish with
    it. Because :class:`RenderedClip` does not carry caption/hashtags (those live
    on the originating :class:`ClipBrief`), the Scheduler sources them from the
    brief and records the resolved copy here so stage 6 (the Publisher) needs no
    further lookup.

    ``clip_id``/``brief_id``/``trend_id`` carry the same attribution linkage the
    :class:`RenderedClip` and :class:`ClipVerdict` carry, so a published post can
    be traced back to its clip, brief, and originating trend by the future
    Performance Analyst / Learning Store.

    Frozen because a scheduled post is an immutable fact once planned: the plan
    is assembled and fanned out downstream and must never be mutated in place.
    """

    model_config = ConfigDict(frozen=True)

    clip_id: str = Field(description="ID of the scheduled RenderedClip.")
    brief_id: str = Field(description="ID of the originating ClipBrief.")
    trend_id: str = Field(description="ID of the originating Trend, for attribution.")
    platform: Platform = Field(description="Target short-video platform.")
    scheduled_at: datetime = Field(description="Aware UTC time the post is scheduled for.")
    caption: str = Field(description="Post caption/description copy for this variant.")
    hashtags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Hashtags to publish with this variant (without leading '#').",
    )
    variant: int = Field(
        default=0,
        ge=0,
        description=(
            "Index of the A/B caption/hashtag variant for this clip. 0 is the "
            "base caption; 1..N are deterministic derivations."
        ),
    )


class PublishPlan(BaseModel):
    """The Scheduler & Optimizer's output: scheduled posts organized by platform.

    Returned by the Scheduler after it groups the Quality Gate's approved clips
    by platform, orders them deterministically, decides each post's
    :attr:`~ScheduledPost.scheduled_at` from the injected audience-timing signal
    plus the configured minimum spacing gap, and expands each clip into its A/B
    caption variants.

    This is the stage-6 Publisher fan-out boundary, exactly as
    :class:`ClipBriefList` was the Clip Factory boundary: stage 6 will publish
    each platform's posts through its own platform adapter, so the per-platform
    slice must be trivially extractable. :meth:`for_platform` returns one
    platform's posts in scheduled order, and :meth:`slices` returns one
    ``(platform, posts)`` pair per platform in sorted platform order (the
    fan-out iterable).

    ``posts`` holds every scheduled post in a single deterministic global order.
    Mirrors :class:`RenderedClipSet`/:class:`QualityReport`: frozen, with
    ``__len__`` over the posts.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    posts: tuple[ScheduledPost, ...] = Field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.posts)

    @property
    def platforms(self) -> tuple[Platform, ...]:
        """Return the sorted tuple of distinct platforms present (deterministic)."""
        return tuple(sorted({post.platform for post in self.posts}))

    def for_platform(self, platform: Platform) -> tuple[ScheduledPost, ...]:
        """Return ``platform``'s posts in scheduled order.

        This is the per-platform slice stage 6 parallelizes across. Posts are
        returned sorted by :attr:`~ScheduledPost.scheduled_at` (ties broken by
        ``clip_id`` then ``variant``) so the order is stable and deterministic.
        """
        selected = [post for post in self.posts if post.platform == platform]
        selected.sort(key=lambda post: (post.scheduled_at, post.clip_id, post.variant))
        return tuple(selected)

    def slices(self) -> tuple[tuple[Platform, tuple[ScheduledPost, ...]], ...]:
        """Return one ``(platform, posts)`` pair per platform in sorted order.

        The fan-out iterable for stage 6: each pair carries a platform and its
        posts in scheduled order, with platforms in the same deterministic order
        as :attr:`platforms`.
        """
        return tuple((platform, self.for_platform(platform)) for platform in self.platforms)
