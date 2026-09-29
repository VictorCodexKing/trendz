"""A deterministic, in-memory trend source for offline development and tests.

:class:`MockTrendSource` returns a fixed set of sample trends with hand-picked
signal values. Because it needs no network, no API keys, and returns the same
data every time, it lets the whole pipeline (and its tests) run fully offline and
deterministically. Swap it for a real :class:`~trendz.sources.base.TrendSource`
adapter in production.
"""

from __future__ import annotations

from trendz.contracts import RunContext, Trend
from trendz.sources.base import TrendSource

# Deterministic sample trends spanning a range of signal profiles so tests can
# assert relative ordering (e.g. high velocity + relevance should outrank a
# saturated, short-lived trend).
_SAMPLE_TRENDS: tuple[Trend, ...] = (
    Trend(
        id="t1",
        title="AI cooking hacks",
        source="mock",
        velocity=0.9,
        relevance=0.8,
        saturation=0.2,
        shelf_life=0.7,
    ),
    Trend(
        id="t2",
        title="Retro gaming speedruns",
        source="mock",
        velocity=0.6,
        relevance=0.5,
        saturation=0.5,
        shelf_life=0.6,
    ),
    Trend(
        id="t3",
        title="Overdone dance challenge",
        source="mock",
        velocity=0.4,
        relevance=0.3,
        saturation=0.9,
        shelf_life=0.2,
    ),
    Trend(
        id="t4",
        title="Sustainable fashion tips",
        source="mock",
        velocity=0.7,
        relevance=0.9,
        saturation=0.3,
        shelf_life=0.8,
    ),
    Trend(
        id="t5",
        title="Budget travel gaming setups",
        source="mock",
        velocity=0.5,
        relevance=0.6,
        saturation=0.4,
        shelf_life=0.5,
    ),
)


class MockTrendSource(TrendSource):
    """A trend source seeded with deterministic in-memory sample data."""

    def __init__(self, trends: tuple[Trend, ...] | None = None) -> None:
        """Create the source.

        Args:
            trends: Optional override of the seeded sample trends. Defaults to a
                built-in deterministic set. Pass an empty tuple to simulate a
                source that returns nothing.
        """
        self._trends: tuple[Trend, ...] = _SAMPLE_TRENDS if trends is None else trends

    @property
    def name(self) -> str:
        return "mock"

    async def fetch(self, ctx: RunContext) -> list[Trend]:
        """Return a fresh copy of the seeded sample trends."""
        return list(self._trends)
