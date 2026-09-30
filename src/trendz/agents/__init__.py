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
    - Quality & Safety Gate: per-clip technical/safety/engagement gate; approves,
      re-renders rejected clips for bounded retries, or drops-and-logs
      (embarrassingly parallel) -- see ``quality_gate``.
    - Scheduler & Optimizer: consumes the Quality Gate's approved clips and
      emits a per-platform PublishPlan (deterministic timing, spacing, and A/B
      variants), the stage-6 Publisher fan-out boundary -- see ``scheduler``.
    - Publisher: one Platform Publisher per platform (fan-out per platform);
      publishes each platform's slice through an injected publisher-client with
      a stable idempotency key (no double-post) and records failures without
      raising (record-and-continue) into a PostResults -- see ``publisher``.
    - Performance Analyst: collects each succeeded post's metrics via an injected
      offline metrics-provider (fan-out per post), derives a per-post
      performance score, attributes outcomes back to trend/brief/clip, and emits
      a PerformanceReports -- see ``performance_analyst``.
    - Learning / Memory Store: the self-improvement feedback loop and shared
      memory; folds the Analyst's PerformanceReports into the persisted
      LearningState via a deterministic fixed-learning-rate reinforcement update
      plus a seeded epsilon-greedy bandit allocation, producing updated scoring
      weights/format priors/engagement threshold/timing hints shaped to feed back
      into the earlier agents -- see ``learning_store``.
"""
