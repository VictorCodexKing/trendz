"""The pluggable audience-timing abstraction for the Scheduler & Optimizer.

An :class:`AudienceTimingProvider` knows, for a given short-video
:class:`~trendz.contracts.Platform`, the best minute-of-day slots to post into.
The Scheduler & Optimizer is dependency-injected with a provider and consults it
per platform, so the same agent works against a deterministic in-memory stub in
tests and against a real audience-analytics backend in production.

Per DESIGN.md stage 5, the Scheduler decides an optimal post time per platform.
The timing signal is expressed as an ordered tuple of minute-of-day slots (each
in ``[0, 1440)``, minutes since 00:00 UTC) describing that platform's best
posting windows in preference order. The Scheduler combines a run's deterministic
base time with the first such slot, then applies the configured minimum spacing
gap; exposing several slots lets a provider express secondary windows a future
Scheduler can spread posts across.

The stub provider in :mod:`trendz.timing.stub_provider` implements this
interface with no clock, network, or API keys for offline tests. The future
Learning Store will supply a provider that learns these slots from observed
performance rather than hard-coding them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trendz.contracts import Platform, RunContext


class AudienceTimingProvider(ABC):
    """Abstract base for a single audience-timing provider adapter."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this provider (used in logs)."""
        raise NotImplementedError

    @abstractmethod
    def optimal_minutes_of_day(self, ctx: RunContext, platform: Platform) -> tuple[int, ...]:
        """Return the platform's best posting slots as minutes-of-day.

        Args:
            ctx: The run context, for run-scoped config. The provider must not
                depend on a live wall clock: the signal is a function of the
                platform (and any run-scoped configuration) so scheduling stays
                deterministic and reproducible.
            platform: The short-video platform to return timing signal for.

        Returns:
            A non-empty, deterministic tuple of minute-of-day slots, each in
            ``[0, 1440)`` (minutes since 00:00 UTC), in preference order (the
            first slot is the single best window). Returning several slots lets
            the provider express secondary windows.
        """
        raise NotImplementedError
