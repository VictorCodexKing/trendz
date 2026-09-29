"""The pluggable trend source abstraction.

A :class:`TrendSource` knows how to fetch raw trends from one place. Trend Scout
is dependency-injected with one or more of them, so the same agent works against
a deterministic in-memory source in tests and against real network adapters in
production.

Real adapters plug in by subclassing :class:`TrendSource` and implementing
:meth:`fetch`, for example (added in later features):

    - ``PlatformTrendingSource`` -> TikTok / Instagram / YouTube trending APIs
    - ``RedditSource`` / ``XSource`` -> subreddit / hashtag firehoses
    - ``GoogleTrendsSource`` -> Google Trends interest-over-time
    - ``RssSource`` -> curated RSS / competitor feeds

Each adapter is responsible for its own auth, rate limiting, and mapping the
provider's payload onto normalised :class:`~trendz.contracts.Trend` signals in
the ``[0, 1]`` range. The mock source in :mod:`trendz.sources.mock_source`
implements this interface with no network access for offline tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trendz.contracts import RunContext, Trend


class TrendSource(ABC):
    """Abstract base for a single trend discovery adapter."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this source (used as ``Trend.source``)."""
        raise NotImplementedError

    @abstractmethod
    async def fetch(self, ctx: RunContext) -> list[Trend]:
        """Fetch raw, unscored trends from this source.

        Args:
            ctx: The run context (for run-scoped config and budget checks).

        Returns:
            A list of trends with their raw signals populated. Scoring and
            ranking are the responsibility of Trend Scout, not the source.
        """
        raise NotImplementedError
