"""The pluggable Clip Checker abstraction for the Quality & Safety Gate.

A :class:`ClipChecker` knows how to evaluate one
:class:`~trendz.contracts.RenderedClip` into one
:class:`~trendz.contracts.ClipVerdict`. The Quality & Safety Gate is
dependency-injected with a checker and fans it out over the clips, so the same
agent works against a deterministic in-memory stub in tests and against a real
evaluation backend in production.

Per DESIGN.md stage 4, a real checker runs three families of checks: technical
(resolution, audio, no black frames, length), content safety (brand safety,
copyright/music licensing, platform policy), and a predicted-engagement score.
The stub checker in :mod:`trendz.checkers.stub_checker` implements this
interface with no real media or network access for offline tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trendz.contracts import ClipVerdict, RenderedClip, RunContext


class ClipChecker(ABC):
    """Abstract base for a single Clip Checker adapter."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this checker (used in logs)."""
        raise NotImplementedError

    @abstractmethod
    async def evaluate(self, ctx: RunContext, clip: RenderedClip) -> ClipVerdict:
        """Evaluate one rendered clip into a verdict.

        Args:
            ctx: The run context, for run-scoped config. The caller (Quality &
                Safety Gate) is responsible for any budget/quota guard around
                this call via ``ctx.budget``.
            clip: The rendered clip to evaluate. Must not be mutated: clips are
                frozen and shared across concurrent tasks.

        Returns:
            A :class:`~trendz.contracts.ClipVerdict` linked back to the clip (via
            ``clip_id``), its brief (via ``brief_id``), and its originating trend
            (via ``trend_id``), with ``approved`` set and ``reasons`` populated
            when rejected.
        """
        raise NotImplementedError
