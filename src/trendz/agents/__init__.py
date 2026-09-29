"""Pipeline agents for the Trendz system.

Each agent follows the same pattern: a ``BaseAgent`` subclass with an async
``run`` method that takes a typed input contract and returns a typed output
contract. The Orchestrator dispatches work between them and owns concurrency
and fan-out.

Implemented so far:
    - Trend Scout (first agent, head of the pipeline) -- see ``trend_scout``.
    - Content Strategist: turns trends into concrete clip briefs (fan-out
      boundary) -- see ``content_strategist``.
    - Clip Factory: one Clip Worker per brief; script/voice, visual, caption,
      assembly (concurrency core) -- see ``clip_factory``.

TODO: remaining agents to be added as their own modules under this package:
    - Quality & Safety Gate: automated per-clip technical/safety/engagement gate.
    - Scheduler & Optimizer: decides when and where each clip posts.
    - Publisher: one Platform Publisher per platform (fan-out per platform).
    - Performance Analyst: collects post metrics and attributes outcomes.
    - Learning / Memory Store: self-improvement feedback loop and shared memory.
"""
