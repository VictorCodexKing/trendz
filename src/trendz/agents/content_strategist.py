"""Content Strategist: the second agent in the pipeline.

The Content Strategist is the first consumer of the Trend Scout's output. It
receives a ranked :class:`~trendz.contracts.TrendList` and turns the strongest
trends into concrete :class:`~trendz.contracts.ClipBrief` objects, one per
targeted platform per trend. This is the fan-out boundary described in
DESIGN.md stage 2: each brief becomes one job for the downstream Clip Factory.

Brief generation is pure and deterministic (no network, no LLM): a real LLM
client can be dependency-injected later following the same pattern the sources
use. All creative fields are derived mechanically from the trend and the
strategy config, so the same input always yields byte-identical briefs.

The strategy configuration lives in one place (:class:`StrategyConfig` and
:data:`DEFAULT_STRATEGY`) precisely so a future Learning / Memory Store can tune
the targeted platforms, briefs-per-trend, footage choice, and default hashtags
from measured performance without touching the planning logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from trendz.agents.base import BaseAgent
from trendz.contracts import (
    AspectRatio,
    ClipBrief,
    ClipBriefList,
    FootageKind,
    Platform,
    RunContext,
    Trend,
    TrendList,
)

# Per-platform format defaults: the canonical aspect ratio and target clip
# length (seconds) for each surface. Kept beside the strategy config so a
# Learning Store can retune lengths from performance without touching the
# planning loop.
PLATFORM_FORMATS: dict[Platform, tuple[AspectRatio, int]] = {
    "tiktok": ("9:16", 30),
    "reels": ("9:16", 30),
    "shorts": ("9:16", 45),
}


@dataclass(frozen=True)
class StrategyConfig:
    """Tunable strategy configuration for the Content Strategist.

    Everything the strategist uses to shape briefs lives here so a future
    Learning Store can produce tuned instances from measured performance.

    Attributes:
        platforms: Which platforms to plan a brief for, per selected trend. The
            order is significant: it fixes the deterministic order in which
            briefs are emitted (and therefore which briefs survive the
            ``target_clip_count`` cap).
        default_hashtags: Hashtags appended to every brief in addition to the
            ones derived from the trend title.
        footage_kind: Whether briefs default to sourced (stock) or AI footage.
        max_derived_hashtags: Upper bound on hashtags derived from the trend
            title, keeping captions tidy and output deterministic.
    """

    platforms: tuple[Platform, ...] = ("tiktok", "reels", "shorts")
    default_hashtags: tuple[str, ...] = ("fyp", "trending")
    footage_kind: FootageKind = "sourced"
    max_derived_hashtags: int = 3

    def __post_init__(self) -> None:
        if self.max_derived_hashtags < 0:
            raise ValueError("max_derived_hashtags must be non-negative")


DEFAULT_STRATEGY = StrategyConfig()


def _derive_hashtags(trend: Trend, strategy: StrategyConfig) -> tuple[str, ...]:
    """Derive hashtags for a trend deterministically from its title.

    Words in the title are lowercased, stripped of non-alphanumeric characters,
    and short stop-words are dropped; the first ``max_derived_hashtags`` distinct
    survivors become tags, followed by the strategy's default hashtags. Order is
    preserved and duplicates removed so the same trend always yields the same
    tags.
    """
    tags: list[str] = []
    seen: set[str] = set()
    for word in trend.title.lower().split():
        cleaned = "".join(ch for ch in word if ch.isalnum())
        if len(cleaned) <= 2 or cleaned in seen:
            continue
        seen.add(cleaned)
        tags.append(cleaned)
        if len(tags) >= strategy.max_derived_hashtags:
            break
    for tag in strategy.default_hashtags:
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tuple(tags)


class ContentStrategist(BaseAgent[TrendList, ClipBriefList]):
    """Turns a ranked ``TrendList`` into a bounded set of ``ClipBrief`` objects.

    Brief generation is deterministic. For each trend, in ranked order, the
    strategist plans one brief per configured platform::

        for trend in trends:            # already ranked by Trend Scout
            for platform in strategy.platforms:
                emit ClipBrief(...)      # format/length from PLATFORM_FORMATS

    Trends are admitted on **whole-trend boundaries**: a trend contributes its
    full platform set or nothing at all. Planning stops before the first trend
    whose complete platform set would overflow ``ctx.config.target_clip_count``,
    so the total never exceeds the run's clip budget and no trend is ever left
    with a partial platform set. (This is the cleaner Option A from the review:
    an all-or-nothing per-trend cap, chosen over documenting a mid-trend
    truncation skew, because downstream attribution and per-trend fairness are
    easier to reason about when a trend's platform coverage is never partial.)

    Every brief carries the originating ``trend_id`` for downstream attribution.
    Creative fields (angle, hook, caption, CTA, hashtags) are derived
    mechanically from the trend title and the strategy config; no LLM or network
    is involved.
    """

    def __init__(self, strategy: StrategyConfig = DEFAULT_STRATEGY) -> None:
        """Create the agent.

        Args:
            strategy: Tunable strategy configuration; defaults to
                :data:`DEFAULT_STRATEGY`.
        """
        super().__init__()
        self._strategy = strategy

    @property
    def name(self) -> str:
        return "content_strategist"

    async def run(self, ctx: RunContext, payload: TrendList) -> ClipBriefList:
        """Plan clip briefs from the incoming ``TrendList``.

        Produces at most ``ctx.config.target_clip_count`` briefs, one per
        configured platform per trend in ranked order. Trends are admitted on
        whole-trend boundaries, so a trend never receives a partial platform
        set: planning stops before the first trend whose full platform set would
        exceed the cap. An empty input (or a zero cap) yields an empty
        ``ClipBriefList``.

        Args:
            ctx: The run context; ``ctx.config.target_clip_count`` bounds output.
            payload: The ranked trends to plan from. Trend ids **must be unique**
                because each brief id is minted as ``f"{trend.id}-{platform}"``
                and ``brief_id``/``trend_id`` are the attribution keys used by
                the downstream Quality Gate and Performance Analyst. Rather than
                silently producing colliding ids, this precondition is enforced.

        Raises:
            ValueError: If ``payload`` contains duplicate trend ids.
        """
        self._require_unique_trend_ids(payload)

        cap = ctx.config.target_clip_count
        per_trend = len(self._strategy.platforms)
        briefs: list[ClipBrief] = []

        for trend in payload.trends:
            # Admit trends on whole-trend boundaries: only start a trend if its
            # full platform set still fits under the cap (Option A).
            if len(briefs) + per_trend > cap:
                break
            for platform in self._strategy.platforms:
                briefs.append(self._plan_brief(trend, platform))

        result = ClipBriefList(run_id=ctx.run_id, briefs=tuple(briefs))
        self.log.info(
            "briefs_planned",
            planned=len(result),
            trends_in=len(payload),
            cap=cap,
            platforms=list(self._strategy.platforms),
        )
        return result

    @staticmethod
    def _require_unique_trend_ids(payload: TrendList) -> None:
        """Enforce the trend-id uniqueness precondition of the fan-out.

        Brief ids are minted as ``f"{trend.id}-{platform}"`` and downstream
        attribution keys off ``brief_id``/``trend_id``; duplicate trend ids would
        mint colliding brief ids and corrupt that attribution. We fail fast with
        a clear error rather than silently de-duplicating, so the upstream Trend
        Scout is the single source of truth for trend identity.
        """
        seen: set[str] = set()
        duplicates: list[str] = []
        for trend in payload.trends:
            if trend.id in seen:
                duplicates.append(trend.id)
            else:
                seen.add(trend.id)
        if duplicates:
            raise ValueError(
                "TrendList contains duplicate trend ids, which would mint "
                f"colliding brief ids: {sorted(set(duplicates))}"
            )

    def _plan_brief(self, trend: Trend, platform: Platform) -> ClipBrief:
        """Build one deterministic brief for a trend on a platform.

        Format and length come from :data:`PLATFORM_FORMATS`; the id encodes the
        trend id and platform so it is stable and unique per (trend, platform).
        """
        aspect_ratio, length = PLATFORM_FORMATS[platform]
        hashtags = _derive_hashtags(trend, self._strategy)
        return ClipBrief(
            id=f"{trend.id}-{platform}",
            trend_id=trend.id,
            title=trend.title,
            angle=f"A fresh take on {trend.title} for {platform}",
            hook=f"Here's why {trend.title} is blowing up right now",
            platform=platform,
            aspect_ratio=aspect_ratio,
            target_length_seconds=length,
            caption=f"{trend.title} - you won't want to miss this.",
            cta="Follow for more and drop a comment!",
            hashtags=hashtags,
            footage_kind=self._strategy.footage_kind,
        )
