"""Clip Worker adapters for the Trendz Clip Factory.

This package holds the pluggable Clip Worker abstraction the Clip Factory fans
out over: a :class:`~trendz.workers.base.ClipWorker` interface plus a
deterministic in-memory :class:`~trendz.workers.stub_worker.StubClipWorker` so
the pipeline (and its tests) run fully offline. A real worker (FFmpeg assembly,
a TTS provider, a text-to-video/stock provider) plugs in by subclassing
``ClipWorker`` without touching the Clip Factory.
"""
