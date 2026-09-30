"""The pluggable metrics-provider abstraction for the Performance Analyst (stage 7).

A :class:`MetricsProvider` knows how to collect the engagement metrics for one
successfully published post. The Performance Analyst is dependency-injected with
a provider and fans it out per post, so the same agent works against a
deterministic in-memory stub in tests and against a real analytics API in
production.

Per DESIGN.md stage 7, a real provider calls each platform's analytics API,
keyed by the post's ``post_id``, to read back its accumulated
views/watch-time/likes/shares/comments/follows once the post has had time to
gather engagement. The Analyst models that maturation window as the deterministic
:attr:`~trendz.contracts.RunConfig.dwell_hours` offset ('metrics as of
published_at + dwell'), never a real sleep or wall-clock wait.

:meth:`collect` returns a :class:`~trendz.contracts.PostMetrics` for the given
:class:`~trendz.contracts.PostResult`; the Analyst derives a per-post score and
attribution record from it. The stub provider in
:mod:`trendz.analytics.stub_provider` implements this interface as a pure
deterministic function with no network, API keys, or wall-clock for offline
tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trendz.contracts import PostMetrics, PostResult, RunContext


class MetricsProvider(ABC):
    """Abstract base for a single analytics metrics-provider adapter."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this provider (used in logs)."""
        raise NotImplementedError

    @abstractmethod
    async def collect(self, ctx: RunContext, result: PostResult) -> PostMetrics:
        """Collect the engagement metrics for one successfully published post.

        Args:
            ctx: The run context, for run-scoped config. The caller (Performance
                Analyst) is responsible for any budget/quota guard around this
                call via ``ctx.budget``. A provider must not depend on a live
                wall clock: the dwell maturation window is modelled by the
                Analyst as the deterministic ``ctx.config.dwell_hours`` offset.
            result: The succeeded :class:`~trendz.contracts.PostResult` to
                collect metrics for. Must not be mutated: results are frozen and
                shared across concurrent collections. A real provider keys the
                analytics lookup off ``result.post_id``.

        Returns:
            A :class:`~trendz.contracts.PostMetrics` snapshot of the post's
            engagement stats. The stub returns the same metrics for the same
            input so the pipeline stays deterministic and reproducible.

        Raises:
            Exception: Any failure to collect. The Performance Analyst catches it
                and continues (one bad collection never blocks the run).
        """
        raise NotImplementedError
