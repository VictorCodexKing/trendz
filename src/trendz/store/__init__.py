"""Persistence abstraction for the Learning / Memory Store (stage 8).

This DI subpackage mirrors :mod:`trendz.analytics`: an ABC (:mod:`base`) plus a
deterministic offline stub (:mod:`in_memory`). The Learning / Memory Store agent
is injected with a :class:`~trendz.store.base.MemoryStore` so it loads the prior
:class:`~trendz.contracts.LearningState`, applies its reinforcement update, and
persists the new state - with the real implementation swappable for Postgres /
Redis / a vector store per DESIGN.md's tech mapping without touching the agent.
"""
