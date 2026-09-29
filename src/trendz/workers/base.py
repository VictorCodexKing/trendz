"""The pluggable Clip Worker abstraction.

A :class:`ClipWorker` knows how to turn one :class:`~trendz.contracts.ClipBrief`
into one :class:`~trendz.contracts.RenderedClip`. The Clip Factory is
dependency-injected with a worker and fans it out over the briefs, so the same
agent works against a deterministic in-memory stub in tests and against a real
media pipeline in production.

Per DESIGN.md stage 3, a real worker coordinates parallel sub-steps -
Script/Voice (script + TTS), Visual (stock/AI B-roll), Caption/Subtitle, and
Assembly (stitch audio + video + captions, format per platform aspect ratio) -
and is responsible for its own media tooling (FFmpeg, a TTS provider, a
text-to-video/stock provider). The stub worker in
:mod:`trendz.workers.stub_worker` implements this interface with no real media
or network access for offline tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trendz.contracts import ClipBrief, RenderedClip, RunContext


class ClipWorker(ABC):
    """Abstract base for a single Clip Worker adapter."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this worker (used in logs)."""
        raise NotImplementedError

    @abstractmethod
    async def render(self, ctx: RunContext, brief: ClipBrief) -> RenderedClip:
        """Render one clip brief into a rendered clip.

        Args:
            ctx: The run context, for run-scoped config. The caller (Clip
                Factory) is responsible for any budget/quota guard around this
                call via ``ctx.budget``.
            brief: The clip brief to render. Must not be mutated: briefs are
                frozen and shared across concurrent workers.

        Returns:
            A :class:`~trendz.contracts.RenderedClip` linked back to the brief
            (via ``brief_id``) and its originating trend (via ``trend_id``).
        """
        raise NotImplementedError
