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
from typing import TYPE_CHECKING, Literal

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
    dwell_hours: int = Field(
        default=48,
        ge=0,
        description=(
            "Dwell period, in hours, after a post is published before the "
            "Performance Analyst collects its metrics. Applied deterministically "
            "as 'metrics collected as of published_at + dwell_hours' - it is a "
            "modelled collection offset, NOT a real sleep or wall-clock wait. "
            "0 means metrics are collected as of the publish time itself."
        ),
    )
    learning_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Fixed reinforcement-update step the Learning / Memory Store applies "
            "when it retunes priors from measured performance: "
            "new = old + learning_rate * (normalized_reward - old). A FIXED step "
            "(not an adaptive schedule) so updates are deterministic and "
            "reproducible. 0 means priors never move; 1 means each update jumps "
            "straight to the observed target."
        ),
    )
    exploration_epsilon: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Exploration probability for the Learning / Memory Store's "
            "multi-armed-bandit format/timing allocation (epsilon-greedy). "
            "Applied via a SEEDED, deterministic policy (a random.Random seeded "
            "from the run id) so the explore/exploit allocation is reproducible. "
            "0 means pure exploitation (always the best-known arm); higher values "
            "explore more."
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


class PostResult(BaseModel):
    """The outcome of publishing a single :class:`ScheduledPost` (stage 6).

    Produced by the Publisher for one scheduled post. It mirrors
    :class:`ScheduledPost`'s attribution linkage -
    ``clip_id``/``brief_id``/``trend_id`` plus ``platform`` and ``variant`` - so
    a published (or failed) post can be traced all the way back to its clip,
    brief, and originating trend by the future Performance Analyst / Learning
    Store. It also carries the :attr:`idempotency_key` the Publisher derived for
    this post, which the publisher-client uses to guarantee a post is not
    published twice within a run.

    On success :attr:`succeeded` is ``True`` and :attr:`post_id`,
    :attr:`post_url`, and :attr:`published_at` (aware UTC) are set while
    :attr:`error` is ``None``. On failure :attr:`succeeded` is ``False``,
    :attr:`error` records the reason, and the success fields are ``None`` - the
    Publisher records the failure and continues (one bad post never blocks the
    run).

    Frozen because a result is an immutable fact once recorded: results are
    collected across concurrent per-platform publishers and must never be
    mutated in place.
    """

    model_config = ConfigDict(frozen=True)

    clip_id: str = Field(description="ID of the published RenderedClip.")
    brief_id: str = Field(description="ID of the originating ClipBrief.")
    trend_id: str = Field(description="ID of the originating Trend, for attribution.")
    platform: Platform = Field(description="Target short-video platform.")
    variant: int = Field(
        default=0,
        ge=0,
        description="Index of the A/B caption/hashtag variant (mirrors ScheduledPost.variant).",
    )
    idempotency_key: str = Field(
        description="Stable key the Publisher derived for this post to prevent double-posting."
    )
    succeeded: bool = Field(description="Whether the post was published successfully.")
    post_id: str | None = Field(
        default=None,
        description="Platform post id assigned on success; None on failure.",
    )
    post_url: str | None = Field(
        default=None,
        description="Platform post URL assigned on success; None on failure.",
    )
    error: str | None = Field(
        default=None,
        description="Failure reason; None on success.",
    )
    published_at: datetime | None = Field(
        default=None,
        description="Aware UTC time the post was published; None on failure.",
    )


class PostResults(BaseModel):
    """The Publisher's output: the outcome of every scheduled post (stage 6).

    Returned by the Publisher after it fans the run's :class:`PublishPlan` out
    per platform (one Platform Publisher per platform, concurrent) and publishes
    each platform's slice through the injected publisher-client. ``results``
    holds every :class:`PostResult` in a single deterministic global order
    (grouped by sorted platform, then scheduled order within a platform), so a
    run is reproducible.

    Mirrors :class:`PublishPlan`/:class:`QualityReport`: frozen, with ``__len__``
    over the results and the same per-platform helpers (:attr:`platforms`,
    :meth:`for_platform`, :meth:`slices`) so per-platform success/failure is
    trivially inspectable. :attr:`succeeded`/:attr:`failed` split the results by
    outcome, and :meth:`top` mirrors the other containers.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    results: tuple[PostResult, ...] = Field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def platforms(self) -> tuple[Platform, ...]:
        """Return the sorted tuple of distinct platforms present (deterministic)."""
        return tuple(sorted({result.platform for result in self.results}))

    def for_platform(self, platform: Platform) -> tuple[PostResult, ...]:
        """Return ``platform``'s results in the deterministic global order.

        Results are already stored grouped by sorted platform then scheduled
        order, so filtering preserves that stable order.
        """
        return tuple(result for result in self.results if result.platform == platform)

    def slices(self) -> tuple[tuple[Platform, tuple[PostResult, ...]], ...]:
        """Return one ``(platform, results)`` pair per platform in sorted order.

        Mirrors :meth:`PublishPlan.slices`: each pair carries a platform and its
        results in the same deterministic order as :attr:`platforms`.
        """
        return tuple((platform, self.for_platform(platform)) for platform in self.platforms)

    @property
    def succeeded(self) -> tuple[PostResult, ...]:
        """Return the results that published successfully, in global order."""
        return tuple(result for result in self.results if result.succeeded)

    @property
    def failed(self) -> tuple[PostResult, ...]:
        """Return the results that failed to publish, in global order."""
        return tuple(result for result in self.results if not result.succeeded)

    def top(self, n: int) -> tuple[PostResult, ...]:
        """Return the first ``n`` results in the deterministic global order."""
        if n < 0:
            raise ValueError("n must be non-negative")
        return self.results[:n]


class PostMetrics(BaseModel):
    """The raw engagement stats a real analytics API returns for one post (stage 7).

    This is the raw payload the Performance Analyst collects for a successfully
    published post, mirroring what a real per-platform analytics API would report
    (views, watch time, and the reaction/share/comment/follow signals). The
    Analyst derives a :attr:`PostPerformance.performance_score` from these.

    Frozen because a metrics snapshot is an immutable fact once collected: it is
    a point-in-time reading (as of the deterministic dwell-based collection time)
    and must never be mutated in place.
    """

    model_config = ConfigDict(frozen=True)

    views: int = Field(ge=0, description="Total views the post accumulated.")
    watch_time_seconds: float = Field(
        ge=0.0,
        description="Total watch time across all views, in seconds.",
    )
    likes: int = Field(ge=0, description="Number of likes/reactions.")
    shares: int = Field(ge=0, description="Number of shares/reposts.")
    comments: int = Field(ge=0, description="Number of comments.")
    follows: int = Field(ge=0, description="New follows attributed to the post.")


class PostPerformance(BaseModel):
    """The Performance Analyst's per-post record: metrics + score + attribution.

    Produced by the Analyst for one successfully published :class:`PostResult`,
    this pairs the collected :class:`PostMetrics` with a derived
    :attr:`performance_score` and carries the full attribution linkage -
    ``clip_id``/``brief_id``/``trend_id`` plus ``platform`` and ``variant`` -
    that flows through the whole pipeline, so an outcome can be traced back to
    its clip, brief, and originating trend by the Learning Store. ``post_id`` is
    the platform id of the published post the metrics were collected for.

    :attr:`collected_at` records the aware-UTC time the metrics were collected,
    modelled deterministically as ``published_at + dwell_hours`` (see
    :attr:`RunConfig.dwell_hours`) - a collection offset, not a real wait.

    Frozen because a performance record is an immutable fact once derived: the
    Analyst collects records across concurrent tasks and must never mutate them.
    """

    model_config = ConfigDict(frozen=True)

    clip_id: str = Field(description="ID of the published RenderedClip.")
    brief_id: str = Field(description="ID of the originating ClipBrief.")
    trend_id: str = Field(description="ID of the originating Trend, for attribution.")
    platform: Platform = Field(description="Target short-video platform.")
    variant: int = Field(
        default=0,
        ge=0,
        description="Index of the A/B caption/hashtag variant (mirrors PostResult.variant).",
    )
    post_id: str = Field(description="Platform post id the metrics were collected for.")
    metrics: PostMetrics = Field(description="Raw engagement stats collected for the post.")
    performance_score: float = Field(
        ge=0.0,
        description="Derived composite performance score for the post.",
    )
    collected_at: datetime = Field(
        description=(
            "Aware UTC time the metrics were collected, modelled deterministically "
            "as published_at + RunConfig.dwell_hours (a collection offset, not a wait)."
        ),
    )


class PerformanceReports(BaseModel):
    """The Performance Analyst's output: a performance record per published post (stage 7).

    Returned by the Analyst after it collects metrics for every successfully
    published :class:`PostResult` (fanning the injected
    :class:`~trendz.analytics.base.MetricsProvider` out through
    :func:`~trendz.concurrency.bounded_map`) and derives a per-post score.
    ``reports`` holds every :class:`PostPerformance` in a single deterministic
    global order (mirroring the upstream :class:`PostResults` order), so a run is
    reproducible.

    Mirrors :class:`PostResults`: frozen, with ``__len__`` over the reports and
    the same per-platform helpers (:attr:`platforms`, :meth:`for_platform`,
    :meth:`slices`) plus :meth:`top`. It adds :meth:`by_trend`, the attribution
    helper the Learning Store keys off, grouping reports by originating trend.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    reports: tuple[PostPerformance, ...] = Field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.reports)

    @property
    def platforms(self) -> tuple[Platform, ...]:
        """Return the sorted tuple of distinct platforms present (deterministic)."""
        return tuple(sorted({report.platform for report in self.reports}))

    def for_platform(self, platform: Platform) -> tuple[PostPerformance, ...]:
        """Return ``platform``'s reports in the deterministic global order.

        Reports are stored in the upstream global order, so filtering preserves
        that stable order.
        """
        return tuple(report for report in self.reports if report.platform == platform)

    def slices(self) -> tuple[tuple[Platform, tuple[PostPerformance, ...]], ...]:
        """Return one ``(platform, reports)`` pair per platform in sorted order.

        Mirrors :meth:`PostResults.slices`: each pair carries a platform and its
        reports in the same deterministic order as :attr:`platforms`.
        """
        return tuple((platform, self.for_platform(platform)) for platform in self.platforms)

    def by_trend(self) -> tuple[tuple[str, tuple[PostPerformance, ...]], ...]:
        """Return one ``(trend_id, reports)`` pair per trend in sorted trend order.

        The attribution helper the Learning Store keys off: reports are grouped
        by their originating ``trend_id`` (trends in sorted order), with each
        group's reports left in the deterministic global order. This makes it
        trivial to attribute per-post outcomes back to the trend that spawned
        them.
        """
        trend_ids = tuple(sorted({report.trend_id for report in self.reports}))
        return tuple(
            (trend_id, tuple(r for r in self.reports if r.trend_id == trend_id))
            for trend_id in trend_ids
        )

    def top(self, n: int) -> tuple[PostPerformance, ...]:
        """Return the ``n`` highest-scoring reports, ranked by ``performance_score``.

        Unlike :meth:`PostResults.top` (which has no intrinsic ranking), reports
        carry a :attr:`PostPerformance.performance_score`, so ``top`` ranks by
        it descending - the same "top means best" meaning :meth:`TrendResults.top`
        carries. Ties break on the deterministic global order (Python's sort is
        stable), so two runs over the same input yield the same ordering.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        ranked = sorted(self.reports, key=lambda report: report.performance_score, reverse=True)
        return tuple(ranked[:n])


class TrendAggregate(BaseModel):
    """A per-trend rollup of the observed performance for one originating trend (stage 8).

    The Learning / Memory Store derives one of these per trend from
    :meth:`PerformanceReports.by_trend`, folding every post attributed to a trend
    into a single record: how many posts it produced, their total and mean
    performance score, and the distinct platforms it was published to. These are
    the attribution-keyed aggregates the reinforcement update reads to decide how
    to nudge the tunable priors.

    Frozen because an aggregate is an immutable fact once derived from a fixed
    :class:`PerformanceReports`: the same reports always yield the same rollup.
    """

    model_config = ConfigDict(frozen=True)

    trend_id: str = Field(description="ID of the originating Trend the posts share.")
    post_count: int = Field(ge=0, description="Number of posts attributed to this trend.")
    total_performance_score: float = Field(
        ge=0.0,
        description="Sum of the per-post performance scores across the trend's posts.",
    )
    mean_performance_score: float = Field(
        ge=0.0,
        description="Mean per-post performance score (0.0 when the trend has no posts).",
    )
    platforms: tuple[Platform, ...] = Field(
        default_factory=tuple,
        description="Distinct platforms the trend was published to, in sorted order.",
    )


class BanditArm(BaseModel):
    """A single multi-armed-bandit arm: one platform's observed reward + pull state (stage 8).

    Each arm corresponds to a per-platform allocation choice the Learning /
    Memory Store's epsilon-greedy policy can pull. It carries the arm's
    incrementally-updated ``pulls`` count and ``estimated_reward`` (a running,
    normalized estimate of that platform's performance). The bandit explores
    (picks a random arm with probability ``exploration_epsilon``) or exploits
    (picks the highest ``estimated_reward`` arm) using a SEEDED
    :class:`random.Random`, so the allocation is reproducible.

    Frozen because an arm is a snapshot of the persisted state at a given
    version; a new update produces a new arm rather than mutating this one.
    """

    model_config = ConfigDict(frozen=True)

    platform: Platform = Field(description="The platform this arm allocates to.")
    pulls: int = Field(ge=0, description="How many times this arm has been pulled/allocated.")
    estimated_reward: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Running normalized reward estimate for this arm, in [0, 1].",
    )


class LearningState(BaseModel):
    """The Learning / Memory Store's output: the updated, fed-back tunable priors (stage 8).

    This is the loop-closer of the pipeline (DESIGN.md stage 8). It carries the
    UPDATED tunable priors, shaped so they can be fed straight back into the
    existing agents rather than living in a parallel vocabulary:

        - :attr:`scoring_weights` mirrors Trend Scout's ``ScoringWeights`` (the
          four floats velocity/relevance/shelf_life/saturation). Use
          :meth:`as_scoring_weights` to rebuild a ``ScoringWeights`` for the
          Trend Scout.
        - :attr:`format_priors` are the Content Strategist's per-platform prior
          weights, keyed by :data:`Platform` (its format/hook allocation signal).
        - :attr:`engagement_threshold` is the Quality Gate's updated
          minimum-engagement threshold.
        - :attr:`timing_slots` mirrors the audience-timing provider's per-platform
          ``tuple[int, ...]`` minute-of-day slot shape. Use
          :meth:`as_timing_slots` to rebuild the provider's slot mapping.

    It also carries the bandit allocation state (:attr:`arms`) and the
    aggregation records it derived (:attr:`trend_aggregates`), plus ``run_id`` and
    a monotonically-incrementing :attr:`version` / :attr:`updates_applied` counter
    so 'improves over time' (incremental updates across runs against the same
    store) is observable.

    Frozen because a learning snapshot is an immutable fact once produced: a
    subsequent update produces a NEW state with an incremented version rather
    than mutating this one.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    version: int = Field(
        default=0,
        ge=0,
        description=(
            "Monotonic version of the learned state. 0 is the initial/default "
            "state; each applied (non-empty) update increments it."
        ),
    )
    updates_applied: int = Field(
        default=0,
        ge=0,
        description="Total number of non-empty updates folded into this state.",
    )
    scoring_weights: tuple[float, float, float, float] = Field(
        default=(0.35, 0.35, 0.15, 0.15),
        description=(
            "Updated Trend Scout scoring weights as "
            "(velocity, relevance, shelf_life, saturation), mirroring "
            "ScoringWeights. Defaults match DEFAULT_WEIGHTS."
        ),
    )
    format_priors: dict[Platform, float] = Field(
        default_factory=dict,
        description=(
            "Updated Content Strategist per-platform format/hook prior weights, "
            "keyed by Platform. Each is a normalized reward estimate in [0, 1]."
        ),
    )
    engagement_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Updated Quality Gate minimum-engagement threshold, in [0, 1].",
    )
    timing_slots: dict[Platform, tuple[int, ...]] = Field(
        default_factory=dict,
        description=(
            "Updated Scheduler per-platform best posting slots (minutes-of-day), "
            "mirroring the audience-timing provider's tuple[int, ...] slot shape."
        ),
    )
    arms: tuple[BanditArm, ...] = Field(
        default_factory=tuple,
        description="The multi-armed-bandit arm state (per-platform pulls + reward).",
    )
    trend_aggregates: tuple[TrendAggregate, ...] = Field(
        default_factory=tuple,
        description="The per-trend aggregation records this state was derived from.",
    )

    def as_scoring_weights(self) -> ScoringWeights:
        """Rebuild a Trend Scout ``ScoringWeights`` from the learned weights.

        Proves the feedback path is real: the learned four-float weight vector is
        exactly the shape Trend Scout consumes, so a downstream run can be
        re-seeded with the tuned weights. Imported lazily to keep
        :mod:`trendz.contracts` free of an agent-module import cycle.
        """
        from trendz.agents.trend_scout import ScoringWeights

        velocity, relevance, shelf_life, saturation = self.scoring_weights
        return ScoringWeights(
            velocity=velocity,
            relevance=relevance,
            shelf_life=shelf_life,
            saturation=saturation,
        )

    def as_timing_slots(self) -> dict[Platform, tuple[int, ...]]:
        """Return the per-platform timing slots as the provider's slot mapping.

        A copy of :attr:`timing_slots` in exactly the
        :class:`~trendz.timing.stub_provider.StubAudienceTimingProvider` ``slots``
        shape, so the tuned timings can be injected back into the Scheduler's
        timing provider.
        """
        return {platform: tuple(slots) for platform, slots in self.timing_slots.items()}

    def arm_for(self, platform: Platform) -> BanditArm | None:
        """Return the bandit arm for ``platform`` if present, else ``None``."""
        for arm in self.arms:
            if arm.platform == platform:
                return arm
        return None


class RunResult(BaseModel):
    """The Orchestrator's end-to-end run output bundle (stages 1-8).

    A single frozen bundle returned by :meth:`~trendz.orchestrator.Orchestrator.run`
    now that the pipeline is complete end to end. It carries:

        - :attr:`run_id`: the run's unique id (the same one threaded through the
          ``RunContext`` and stamped on every contract the run produced).
        - :attr:`reports`: the Performance Analyst's :class:`PerformanceReports`
          (stage 7) - the per-post metrics + score + attribution for every
          successfully published post.
        - :attr:`learnings`: the Learning / Memory Store's :class:`LearningState`
          (stage 8) - the updated, fed-back tunable priors and bandit state.

    Bundling both keeps the ``run()`` return type honest about the whole
    pipeline: callers can inspect what was published *and* what the run learned,
    and feed the learnings back into the next run's tunable configs.

    Frozen because a run result is an immutable fact once the run completes.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    reports: PerformanceReports = Field(
        description="The Performance Analyst's per-post reports (stage 7)."
    )
    learnings: LearningState = Field(
        description="The Learning / Memory Store's updated tunable priors (stage 8)."
    )


if TYPE_CHECKING:  # pragma: no cover - typing-only import to avoid a cycle
    from trendz.agents.trend_scout import ScoringWeights
