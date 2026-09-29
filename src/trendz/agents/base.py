"""The reusable agent abstraction every Trendz pipeline agent subclasses.

``BaseAgent`` defines the single contract shared by all agents: an async
:meth:`run` that takes the shared :class:`~trendz.contracts.RunContext` plus a
typed input payload and returns a typed output. It is generic over the input and
output types so subclasses declare their own contracts while sharing the logging
and lifecycle boilerplate.

The abstraction is deliberately framework-agnostic: it has no hard dependency on
any LLM or agent framework. Reasoning agents that need an LLM will inject a
client of their choosing; infrastructure agents (like Trend Scout) may need none
at all.

Downstream agents subclass it identically, for example::

    class ContentStrategist(BaseAgent[TrendList, list[ClipBrief]]):
        @property
        def name(self) -> str:
            return "content_strategist"

        async def run(self, ctx: RunContext, payload: TrendList) -> list[ClipBrief]:
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import structlog

from trendz.contracts import RunContext

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class BaseAgent(ABC, Generic[TInput, TOutput]):
    """Abstract base for every pipeline agent.

    Subclasses implement :meth:`run` and :attr:`name`. Each instance gets a
    structlog logger bound to its agent name so all log lines are attributable to
    the emitting agent (part of the design's observability goal).
    """

    def __init__(self) -> None:
        # Bound lazily on first access so the name property (which subclasses
        # define) is available before we build the logger.
        self._log: structlog.stdlib.BoundLogger | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable, human-readable identifier for this agent (used in logs)."""
        raise NotImplementedError

    @property
    def log(self) -> structlog.stdlib.BoundLogger:
        """A structlog logger bound with this agent's name."""
        if self._log is None:
            self._log = structlog.get_logger(agent=self.name)
        return self._log

    @abstractmethod
    async def run(self, ctx: RunContext, payload: TInput) -> TOutput:
        """Execute the agent's work for a run.

        Args:
            ctx: The shared run context (identity, config, budget ledger).
            payload: The typed input contract for this agent. Head-of-pipeline
                agents that have no upstream input accept a trivial payload
                (for example ``None``).

        Returns:
            The typed output contract produced by this agent.
        """
        raise NotImplementedError
