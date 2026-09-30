"""Publisher-client adapters for the Trendz Publisher (stage 6).

This package holds the pluggable publisher-client abstraction the Publisher fans
out over per platform: a :class:`~trendz.publishers.base.PublisherClient`
interface plus a deterministic in-memory
:class:`~trendz.publishers.stub_client.StubPublisherClient` so the pipeline (and
its tests) run fully offline. A real client (each platform's publishing API)
plugs in by subclassing ``PublisherClient`` without touching the Publisher
agent's fan-out logic, exactly like the Clip Factory's Clip Worker, the Quality
Gate's Clip Checker, and the Scheduler's audience-timing provider.
"""
