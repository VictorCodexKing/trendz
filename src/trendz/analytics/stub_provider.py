"""A deterministic, in-memory metrics-provider for offline development and tests.

:class:`StubMetricsProvider` simulates collecting a post's engagement metrics
with no real network, API keys, or wall clock: the returned
:class:`~trendz.contracts.PostMetrics` is a PURE function of the post's stable
attribution fields (``post_id``/``clip_id``/``platform``/``variant``/
``trend_id``), so the same post always yields the same metrics and the whole
pipeline (and its tests) runs fully offline and deterministically. Swap it for a
real :class:`~trendz.analytics.base.MetricsProvider` adapter (each platform's
analytics API) in production.

The values are derived from a stable hash of those fields and kept plausibly
correlated (watch time follows from views, and the reaction/share/comment/follow
counts are progressively rarer fractions of views), mirroring
:class:`~trendz.publishers.stub_client.StubPublisherClient` and
:class:`~trendz.timing.stub_provider.StubAudienceTimingProvider` in style and
intent.
"""

from __future__ import annotations

import hashlib

from trendz.analytics.base import MetricsProvider
from trendz.contracts import PostMetrics, PostResult, RunContext


def _seed(result: PostResult) -> int:
    """Return a stable non-negative integer seed for a post's stable fields.

    Uses a hash of the post's attribution triple plus platform/variant/post_id
    so the seed is a deterministic pure function of the post (no clock, no
    randomness) and identical inputs always produce the identical seed.
    """
    key = "|".join(
        (
            result.post_id or "",
            result.clip_id,
            result.platform,
            str(result.variant),
            result.trend_id,
        )
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest, 16)


class StubMetricsProvider(MetricsProvider):
    """A metrics-provider that simulates analytics collection deterministically, offline."""

    @property
    def name(self) -> str:
        return "stub_metrics_provider"

    async def collect(self, ctx: RunContext, result: PostResult) -> PostMetrics:
        """Return deterministic, plausibly-correlated metrics for one post.

        The metrics are a pure function of the post's stable fields via
        :func:`_seed`, so repeated collection of the same post yields identical
        metrics. Views drive watch time; likes/shares/comments/follows are
        progressively rarer fractions of views. No network, API keys, or sleep.
        """
        seed = _seed(result)
        # Views land in a plausible, bounded range (1000..50999) so downstream
        # scoring has meaningful spread while staying deterministic.
        views = 1000 + (seed % 50000)
        # Average watch of ~7-21 seconds per view, derived deterministically.
        avg_watch = 7 + (seed % 15)
        watch_time_seconds = float(views * avg_watch)
        # Reactions are progressively rarer fractions of views.
        likes = views // 10
        shares = views // 50
        comments = views // 100
        follows = views // 200
        return PostMetrics(
            views=views,
            watch_time_seconds=watch_time_seconds,
            likes=likes,
            shares=shares,
            comments=comments,
            follows=follows,
        )
