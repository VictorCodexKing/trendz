"""A deterministic, in-memory Clip Worker for offline development and tests.

:class:`StubClipWorker` builds a :class:`~trendz.contracts.RenderedClip` from a
:class:`~trendz.contracts.ClipBrief` with no real FFmpeg/TTS/API calls: media
fields are mocked URIs derived from the brief, so the whole pipeline (and its
tests) runs fully offline and deterministically. Swap it for a real
:class:`~trendz.workers.base.ClipWorker` adapter in production.
"""

from __future__ import annotations

from trendz.contracts import ClipBrief, RenderedClip, RunContext
from trendz.workers.base import ClipWorker


class StubClipWorker(ClipWorker):
    """A Clip Worker that assembles a deterministic clip with no real media."""

    @property
    def name(self) -> str:
        return "stub_clip_worker"

    async def render(self, ctx: RunContext, brief: ClipBrief) -> RenderedClip:
        """Assemble a deterministic rendered clip from a brief.

        The output is a pure function of the brief: the id and video URI are
        derived from the brief id, and the format/duration are carried through
        from the brief. No network, FFmpeg, or TTS is involved, so the same
        brief always yields a byte-identical rendered clip.
        """
        return RenderedClip(
            id=f"clip-{brief.id}",
            brief_id=brief.id,
            trend_id=brief.trend_id,
            platform=brief.platform,
            aspect_ratio=brief.aspect_ratio,
            duration_seconds=brief.target_length_seconds,
            video_uri=f"stub://clips/{brief.id}.mp4",
            has_audio=True,
            has_captions=True,
        )
