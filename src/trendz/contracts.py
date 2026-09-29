"""Shared typed data contracts for the Trendz pipeline.

These Pydantic v2 models are the vocabulary every agent speaks. The
Orchestrator builds a :class:`RunContext` from a :class:`RunConfig` and threads
it through the pipeline; Trend Scout emits a :class:`TrendList`; downstream
agents will add their own contracts (ClipBrief, RenderedClip, PublishPlan, ...)
following the same style.

Design choices:
    - Models that represent immutable facts or configuration are ``frozen`` so
      they can be shared safely across concurrent tasks without accidental
      mutation. :class:`BudgetLedger` is intentionally mutable because it tracks
      running spend over the lifetime of a run.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    """Return an aware UTC timestamp (default factory for created_at fields)."""
    return datetime.now(timezone.utc)


class RunConfig(BaseModel):
    """Global configuration for a single pipeline run.

    Supplied by whoever triggers a run (a schedule, a trend spike, or a manual
    invocation). The Orchestrator reads it to decide how much work to do and how
    aggressively to parallelise.
    """

    model_config = ConfigDict(frozen=True)

    niche_filters: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Case-insensitive keywords a trend's title must match to be kept. "
            "Empty means no filtering."
        ),
    )
    target_clip_count: int = Field(
        default=5,
        ge=0,
        description="How many clips the run should aim to produce.",
    )
    concurrency_degree: int = Field(
        default=4,
        ge=1,
        description="Maximum number of concurrent workers in fan-out stages.",
    )
    budget_limit: float = Field(
        default=100.0,
        ge=0.0,
        description="Total spend cap (currency units) for the run.",
    )
    quota_limit: int = Field(
        default=1000,
        ge=0,
        description="Total external API call quota for the run.",
    )


class BudgetLedger(BaseModel):
    """Tracks spend and quota consumption over the lifetime of a run.

    Mutable by design: the Orchestrator debits it as work is performed and
    checks the ``can_*`` helpers before dispatching more work so the run
    throttles or aborts before overrunning its limits.
    """

    budget_limit: float = Field(default=100.0, ge=0.0)
    quota_limit: int = Field(default=1000, ge=0)
    spent: float = Field(default=0.0, ge=0.0)
    quota_used: int = Field(default=0, ge=0)

    @property
    def budget_remaining(self) -> float:
        """Currency units still available before hitting the budget cap."""
        return max(0.0, self.budget_limit - self.spent)

    @property
    def quota_remaining(self) -> int:
        """External API calls still available before hitting the quota cap."""
        return max(0, self.quota_limit - self.quota_used)

    def can_spend(self, amount: float) -> bool:
        """Return True if ``amount`` fits within the remaining budget."""
        return self.spent + amount <= self.budget_limit

    def can_use_quota(self, calls: int = 1) -> bool:
        """Return True if ``calls`` fit within the remaining quota."""
        return self.quota_used + calls <= self.quota_limit

    def charge(self, amount: float = 0.0, calls: int = 0) -> None:
        """Debit the ledger by ``amount`` spend and ``calls`` quota."""
        if amount < 0 or calls < 0:
            raise ValueError("charge amounts must be non-negative")
        self.spent += amount
        self.quota_used += calls

    @classmethod
    def from_config(cls, config: RunConfig) -> BudgetLedger:
        """Build a fresh ledger seeded from a run's configured limits."""
        return cls(budget_limit=config.budget_limit, quota_limit=config.quota_limit)


class RunContext(BaseModel):
    """Immutable context passed to every agent for a given run.

    Carries the run identity, the run configuration, and the (mutable) budget
    ledger. The ledger is a nested model so it can be debited in place while the
    surrounding context stays frozen.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    config: RunConfig
    budget: BudgetLedger
    created_at: datetime = Field(default_factory=_utcnow)

    @classmethod
    def new(cls, run_id: str, config: RunConfig) -> RunContext:
        """Construct a context with a budget ledger derived from ``config``."""
        return cls(run_id=run_id, config=config, budget=BudgetLedger.from_config(config))


class Trend(BaseModel):
    """A single discovered trend with its raw scoring signals.

    Signals are normalised to the ``[0, 1]`` range so the scoring function can
    combine them with tunable weights. ``score`` is assigned by Trend Scout;
    it defaults to 0.0 for a freshly fetched, not-yet-scored trend.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    source: str
    velocity: float = Field(default=0.0, ge=0.0, le=1.0, description="How fast it is rising.")
    relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Fit to the niche.")
    saturation: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How crowded/overdone it already is."
    )
    shelf_life: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How long it will stay relevant."
    )
    score: float = Field(default=0.0, description="Composite score assigned by Trend Scout.")

    def with_score(self, score: float) -> Trend:
        """Return a copy of this trend with ``score`` set (models are frozen)."""
        return self.model_copy(update={"score": score})


class TrendList(BaseModel):
    """An ordered, ranked collection of trends for a run.

    Trend Scout returns this. Trends are expected to be ordered descending by
    ``score``; :meth:`top` is a convenience for the downstream Content
    Strategist, which only needs the strongest N.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    trends: tuple[Trend, ...] = Field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.trends)

    def top(self, n: int) -> tuple[Trend, ...]:
        """Return the first ``n`` trends (the highest scored when ranked)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        return self.trends[:n]
