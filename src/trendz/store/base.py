"""The pluggable persistence abstraction for the Learning / Memory Store (stage 8).

A :class:`MemoryStore` is where the Learning / Memory Store agent keeps the
learned :class:`~trendz.contracts.LearningState` between runs. The agent is
dependency-injected with a store and, on each run, (1) :meth:`load`s the prior
state for its namespace and (2) :meth:`save`s the state it produced, so the same
agent works against a deterministic in-memory stub in tests and against a real
backing store in production.

Per DESIGN.md's stage-8 tech mapping, a real implementation would persist to
Postgres (structured priors + version history), Redis (hot bandit-arm state), or
a vector store (embedded trend/format memory). Naming note: this ABC is
``MemoryStore`` deliberately, so it does NOT collide with the ``LearningStore``
AGENT that owns it - the agent is the behaviour, the store is the persistence.

:meth:`load` must always return a well-defined state: when nothing has been
persisted for a namespace yet it returns the INITIAL/default state (Trend Scout's
:data:`~trendz.agents.trend_scout.DEFAULT_WEIGHTS`, the Content Strategist's
default per-platform priors, the default engagement threshold, the default
per-platform timing slots, and zeroed bandit arms) at ``version == 0``. The
offline stub in :mod:`trendz.store.in_memory` implements this interface purely
in-process with no clock, network, API keys, or randomness for offline tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trendz.contracts import LearningState


class MemoryStore(ABC):
    """Abstract base for a Learning / Memory Store persistence backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this store (used in logs)."""
        raise NotImplementedError

    @abstractmethod
    def load(self, namespace: str) -> LearningState:
        """Load the current persisted learning state for ``namespace``.

        Args:
            namespace: The key the state is stored under. The Learning Store uses
                a stable namespace (not the per-run id) so learning accumulates
                across runs; a fresh namespace starts from the default state.

        Returns:
            The persisted :class:`~trendz.contracts.LearningState`, or a
            well-defined INITIAL/default state (default scoring weights, default
            per-platform format priors, default engagement threshold, default
            per-platform timing slots, zeroed bandit arms, ``version == 0``) when
            nothing has been persisted for the namespace yet. Never ``None``, so
            the agent's update path always has a prior to reinforce.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, namespace: str, state: LearningState) -> None:
        """Persist ``state`` as the current learning state for ``namespace``.

        Args:
            namespace: The key to store the state under (see :meth:`load`).
            state: The updated :class:`~trendz.contracts.LearningState` to
                persist. A subsequent :meth:`load` of the same namespace must
                return this state, so learning is cumulative across runs against
                the same store.
        """
        raise NotImplementedError
