"""Pluggable clip-checker adapters for the Quality & Safety Gate.

A :class:`~trendz.checkers.base.ClipChecker` evaluates one
:class:`~trendz.contracts.RenderedClip` into a
:class:`~trendz.contracts.ClipVerdict` (technical + content-safety checks plus a
predicted-engagement score). The Quality & Safety Gate is dependency-injected
with a checker and fans it out over the clips, so the same agent works against a
deterministic in-memory stub in tests and against a real evaluation backend in
production.

Implemented so far:
    - :class:`~trendz.checkers.stub_checker.StubClipChecker`: a deterministic,
      offline checker for development and tests (no network, no API keys).
"""
