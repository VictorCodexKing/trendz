"""A deterministic, offline audience-timing provider for development and tests.

:class:`StubAudienceTimingProvider` returns each
:class:`~trendz.contracts.Platform`'s best posting slots as a pure function of
the platform, with no clock, network, randomness, or API keys, mirroring
:class:`~trendz.checkers.stub_checker.StubClipChecker` and
:class:`~trendz.workers.stub_worker.StubClipWorker`.

Each platform is given a fixed, distinct set of minute-of-day slots (stand-ins
for the real "when is this audience most active" signal). The mapping can be
overridden via the constructor so tests can force specific timings, exactly as
the stub checker exposes configurable knobs. Because the result depends only on
the platform, the Scheduler's timing decisions are stable across runs.
"""

from __future__ import annotations

from trendz.contracts import Platform, RunContext
from trendz.timing.base import AudienceTimingProvider

# Fixed, distinct per-platform slots (minutes since 00:00 UTC), in preference
# order. Chosen to be recognisable and well-spaced across platforms so the
# Scheduler's grouping and ordering are easy to reason about offline. The first
# slot in each tuple is that platform's single best posting window.
_DEFAULT_SLOTS: dict[Platform, tuple[int, ...]] = {
    # 18:00 and 12:00 UTC.
    "tiktok": (18 * 60, 12 * 60),
    # 11:00 and 19:00 UTC.
    "reels": (11 * 60, 19 * 60),
    # 15:00 and 20:00 UTC.
    "shorts": (15 * 60, 20 * 60),
}

# Fallback slot used when a platform has no configured hint (defensive: the
# Platform Literal is closed, so this is only reachable via an override that
# omits a platform). Noon UTC.
_FALLBACK_SLOTS: tuple[int, ...] = (12 * 60,)


class StubAudienceTimingProvider(AudienceTimingProvider):
    """An audience-timing provider that is a pure function of the platform."""

    def __init__(self, *, slots: dict[Platform, tuple[int, ...]] | None = None) -> None:
        """Create the provider.

        Args:
            slots: Optional per-platform override of the minute-of-day slots.
                When omitted, a fixed, distinct default per platform is used.
                Copied so the caller's mapping is not shared or mutated.
        """
        self._slots: dict[Platform, tuple[int, ...]] = (
            dict(slots) if slots is not None else dict(_DEFAULT_SLOTS)
        )

    @property
    def name(self) -> str:
        return "stub_audience_timing"

    def optimal_minutes_of_day(self, ctx: RunContext, platform: Platform) -> tuple[int, ...]:
        """Return the platform's fixed best posting slots (minutes-of-day).

        A pure function of the platform: no clock, network, or randomness, so
        the same platform always yields the same slots.
        """
        return self._slots.get(platform, _FALLBACK_SLOTS)
