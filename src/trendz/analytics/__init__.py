"""Metrics-provider adapters for the Trendz Performance Analyst (stage 7).

This package holds the pluggable metrics-provider abstraction the Performance
Analyst fans out over per post: a :class:`~trendz.analytics.base.MetricsProvider`
interface plus a deterministic in-memory
:class:`~trendz.analytics.stub_provider.StubMetricsProvider` so the pipeline (and
its tests) run fully offline. A real provider (each platform's analytics API,
keyed by ``post_id``) plugs in by subclassing ``MetricsProvider`` without
touching the Analyst's fan-out logic, exactly like the Clip Factory's Clip
Worker, the Quality Gate's Clip Checker, the Scheduler's audience-timing
provider, and the Publisher's publisher-client.
"""
