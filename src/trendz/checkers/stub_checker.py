"""A deterministic, in-memory Clip Checker for offline development and tests.

:class:`StubClipChecker` evaluates a :class:`~trendz.contracts.RenderedClip`
into a :class:`~trendz.contracts.ClipVerdict` as a pure function of the clip and
the checker's own configuration, with no real media analysis, network, or API
keys. It models DESIGN.md stage 4's three check families deterministically:

    - Technical: a clip must have audio, captions, and a length within
      ``[min_duration_seconds, max_duration_seconds]`` (a stand-in for the real
      resolution / no-black-frames / length checks).
    - Content safety: a clip is unsafe iff its ``brief_id`` is listed in
      ``reject_brief_ids`` (a stand-in for brand-safety / licensing / policy).
    - Predicted engagement: a deterministic score derived from the clip's
      duration, hashed into ``[0, 1]``. When a ``min_engagement_score`` is
      configured, a clip scoring below it FAILS the engagement check and is
      rejected - this is the real engagement bar the Learning / Memory Store's
      learned engagement threshold feeds (a threshold of 0.0 disables gating).

To let tests force approve, force reject, and (crucially) reject-then-approve on
re-render within the retry bound, the safety check consults an optional
``reject_budget`` mapping keyed by ``brief_id``. Each time a brief is evaluated
its remaining budget is decremented; while the budget is positive the clip is
rejected, and once it reaches zero the same brief's re-render passes. Because the
gate re-renders through the Clip Factory (which yields a clip with the same
``brief_id``), this makes a bounded retry deterministically succeed.
"""

from __future__ import annotations

from trendz.checkers.base import ClipChecker
from trendz.contracts import ClipVerdict, RenderedClip, RunContext


class StubClipChecker(ClipChecker):
    """A Clip Checker that produces a deterministic verdict with no real media."""

    def __init__(
        self,
        *,
        reject_brief_ids: frozenset[str] | None = None,
        reject_budget: dict[str, int] | None = None,
        min_duration_seconds: int = 1,
        max_duration_seconds: int = 180,
        min_engagement_score: float = 0.0,
    ) -> None:
        """Create the checker.

        Args:
            reject_brief_ids: Brief ids that always fail the content-safety
                check (force reject, never recoverable by a re-render).
            reject_budget: Optional per-brief rejection budget. A brief with a
                positive remaining budget fails the content-safety check and has
                its budget decremented; once it reaches zero the brief passes.
                Lets a re-render within the retry bound deterministically pass.
            min_duration_seconds: Minimum acceptable clip duration (technical).
            max_duration_seconds: Maximum acceptable clip duration (technical).
            min_engagement_score: Minimum predicted-engagement score in [0, 1] a
                clip must meet to pass the ENGAGEMENT check. This is the real
                engagement bar (distinct from the technical duration floor): a
                clip whose ``engagement_score`` is below it is rejected. Defaults
                to 0.0 (no engagement gating), so existing behaviour is
                unchanged unless a threshold is supplied - this is the knob the
                Learning / Memory Store's learned engagement threshold feeds.
        """
        self._reject_brief_ids = reject_brief_ids or frozenset()
        # Copied so the caller's mapping is not mutated as budgets are consumed.
        self._reject_budget = dict(reject_budget) if reject_budget else {}
        self._min_duration_seconds = min_duration_seconds
        self._max_duration_seconds = max_duration_seconds
        self._min_engagement_score = min_engagement_score

    @property
    def name(self) -> str:
        return "stub_clip_checker"

    def _engagement_score(self, clip: RenderedClip) -> float:
        """Return a deterministic predicted-engagement score in [0, 1]."""
        # A pure function of the clip: stable across runs and re-renders (a
        # re-render of the same brief keeps the same duration, so the score is
        # unchanged). Hashed into a bounded range without any randomness.
        return ((clip.duration_seconds * 37 + 11) % 100) / 100.0

    async def evaluate(self, ctx: RunContext, clip: RenderedClip) -> ClipVerdict:
        """Evaluate one rendered clip into a deterministic verdict.

        The verdict is a pure function of the clip and the checker's configured
        state, so (given the same budget state) the same clip always yields the
        same verdict. No network, media decoding, or API is involved.
        """
        reasons: list[str] = []

        technical_ok = True
        if not clip.has_audio:
            technical_ok = False
            reasons.append("technical: missing audio track")
        if not clip.has_captions:
            technical_ok = False
            reasons.append("technical: missing captions")
        if not (self._min_duration_seconds <= clip.duration_seconds <= self._max_duration_seconds):
            technical_ok = False
            reasons.append(
                "technical: duration "
                f"{clip.duration_seconds}s outside "
                f"[{self._min_duration_seconds}, {self._max_duration_seconds}]s"
            )

        safety_ok = True
        if clip.brief_id in self._reject_brief_ids:
            safety_ok = False
            reasons.append("safety: brief on reject list (brand/licensing/policy)")
        remaining = self._reject_budget.get(clip.brief_id, 0)
        if remaining > 0:
            self._reject_budget[clip.brief_id] = remaining - 1
            safety_ok = False
            reasons.append("safety: pending re-render (rejection budget not yet exhausted)")

        # Predicted-engagement gate: a clip whose engagement score falls below
        # the configured minimum fails the ENGAGEMENT check (distinct from the
        # technical and safety checks). With the default threshold of 0.0 this
        # never rejects; a learned/raised threshold makes the engagement bar
        # actually gate approval.
        engagement_score = self._engagement_score(clip)
        engagement_ok = engagement_score >= self._min_engagement_score
        if not engagement_ok:
            reasons.append(
                "engagement: predicted engagement "
                f"{engagement_score:.2f} below minimum {self._min_engagement_score:.2f}"
            )

        return ClipVerdict(
            clip_id=clip.id,
            brief_id=clip.brief_id,
            trend_id=clip.trend_id,
            approved=technical_ok and safety_ok and engagement_ok,
            reasons=tuple(reasons),
            engagement_score=engagement_score,
            technical_ok=technical_ok,
            safety_ok=safety_ok,
        )
