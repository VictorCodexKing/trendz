"""Clip Factory: the third agent in the pipeline (the concurrency core).

The Clip Factory is the fan-out stage of DESIGN.md stage 3. It receives the
:class:`~trendz.contracts.ClipBriefList` planned by the Content Strategist and
spawns one Clip Worker per brief, running them concurrently up to the run's
configured limit. Each worker turns one :class:`~trendz.contracts.ClipBrief`
into one :class:`~trendz.contracts.RenderedClip`; the factory assembles the
results into a :class:`~trendz.contracts.RenderedClipSet`.

Concurrency is not hand-rolled here: the per-brief work is routed through
:func:`~trendz.concurrency.bounded_map` with
``concurrency = ctx.config.concurrency_degree``, so the factory respects the
run's render/GPU/API budget and the bounded worker pool the design mandates.
Because ``bounded_map`` preserves input order, the output clips line up with the
input briefs.

The actual media work is delegated to a dependency-injected
:class:`~trendz.workers.base.ClipWorker` (defaulting to the deterministic
:class:`~trendz.workers.stub_worker.StubClipWorker`) so a real FFmpeg/TTS
pipeline can be swapped in later without touching the fan-out logic.
"""

from __future__ import annotations

from trendz.agents.base import BaseAgent
from trendz.concurrency import bounded_map
from trendz.contracts import ClipBriefList, RenderedClip, RenderedClipSet, RunContext
from trendz.workers.base import ClipWorker
from trendz.workers.stub_worker import StubClipWorker


class ClipFactory(BaseAgent[ClipBriefList, RenderedClipSet]):
    """Fans a Clip Worker out over the briefs into a ``RenderedClipSet``."""

    def __init__(self, worker: ClipWorker | None = None) -> None:
        """Create the agent.

        Args:
            worker: The Clip Worker to render each brief (dependency-injected).
                Defaults to the deterministic offline
                :class:`~trendz.workers.stub_worker.StubClipWorker`.
        """
        super().__init__()
        self._worker = worker if worker is not None else StubClipWorker()

    @property
    def name(self) -> str:
        return "clip_factory"

    async def run(self, ctx: RunContext, payload: ClipBriefList) -> RenderedClipSet:
        """Render every brief concurrently into a ``RenderedClipSet``.

        The per-brief work is routed through
        :func:`~trendz.concurrency.bounded_map` capped at
        ``ctx.config.concurrency_degree``, so no more than that many workers run
        at once. Output order matches input order (``bounded_map`` preserves
        it). An empty ``ClipBriefList`` yields an empty ``RenderedClipSet``
        without dispatching any work.
        """
        clips: list[RenderedClip] = await bounded_map(
            lambda brief: self._worker.render(ctx, brief),
            payload.briefs,
            ctx.config.concurrency_degree,
        )
        result = RenderedClipSet(run_id=ctx.run_id, clips=tuple(clips))
        self.log.info(
            "clips_rendered",
            rendered=len(result),
            briefs_in=len(payload),
            worker=self._worker.name,
            concurrency=ctx.config.concurrency_degree,
        )
        return result
