"""Pluggable audience-timing providers for the Scheduler & Optimizer.

An :class:`~trendz.timing.base.AudienceTimingProvider` supplies the per-platform
optimal-post-time signal the Scheduler uses to decide when each clip should go
out (DESIGN.md stage 5). The Scheduler is dependency-injected with a provider,
so the same agent works against a deterministic in-memory stub in tests and
against a real audience-analytics backend in production. This is the swappable,
offline-testable source of timing signal the future Learning Store will replace
once it can learn best posting slots from observed performance.

Implemented so far:
    - :class:`~trendz.timing.stub_provider.StubAudienceTimingProvider`: a
      deterministic, offline provider for development and tests (no clock, no
      network, no API keys).
"""
