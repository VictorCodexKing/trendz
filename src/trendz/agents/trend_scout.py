"""Trend Scout: the first agent in the pipeline.

Trend Scout is the head of the pipeline and has no upstream agent input. It:

    1. fetches raw trends from one or more dependency-injected trend sources,
    2. scores each trend with a documented weighted scoring function,
    3. applies the run's niche filters,
    4. ranks the survivors descending by score, and
    5. returns a :class:`~trendz.contracts.TrendList`.

The scoring weights live in one place (:data:`DEFAULT_WEIGHTS`) precisely so a
future Learning / Memory Store can tune them from performance feedback without
touching the scoring logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from trendz.agents.base import BaseAgent
from trendz.concurrency import bounded_map
from trendz.contracts import RunContext, Trend, TrendList
from trendz.sources.base import TrendSource

# Estimated external-API quota cost of a single source fetch. Real network
# adapters can refine this later; it exists so the budget ledger is exercised on
# the live path and the debit/guard idiom is established for cloned agents.
QUOTA_PER_FETCH = 1


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for the Trend Scout scoring function.

    ``velocity``, ``relevance``, and ``shelf_life`` are positive contributions;
    ``saturation`` is subtracted because an already-crowded trend is worth less.
    A future Learning Store can produce tuned instances of this from measured
    performance.
    """

    velocity: float = 0.35
    relevance: float = 0.35
    shelf_life: float = 0.15
    saturation: float = 0.15


DEFAULT_WEIGHTS = ScoringWeights()


def score_trend(trend: Trend, weights: ScoringWeights = DEFAULT_WEIGHTS) -> float:
    """Compute a composite score for a trend.

    The score is a weighted linear combination of the normalised signals::

        score = w_v * velocity
              + w_r * relevance
              + w_s * shelf_life
              - w_sat * saturation

    Higher velocity, relevance, and shelf life raise the score; higher
    saturation lowers it. All signals are in ``[0, 1]``, so the score is bounded
    and comparable across trends and runs. The function is pure and
    deterministic: the same trend always yields the same score.
    """
    return (
        weights.velocity * trend.velocity
        + weights.relevance * trend.relevance
        + weights.shelf_life * trend.shelf_life
        - weights.saturation * trend.saturation
    )


class TrendScout(BaseAgent[None, TrendList]):
    """Discovers, scores, filters, and ranks trends into a ``TrendList``."""

    def __init__(
        self,
        sources: list[TrendSource],
        weights: ScoringWeights = DEFAULT_WEIGHTS,
    ) -> None:
        """Create the agent.

        Args:
            sources: Trend sources to fetch from (dependency-injected). May be
                empty, in which case the agent returns an empty ``TrendList``.
            weights: Scoring weights; defaults to :data:`DEFAULT_WEIGHTS`.
        """
        super().__init__()
        self._sources = sources
        self._weights = weights

    @property
    def name(self) -> str:
        return "trend_scout"

    async def run(self, ctx: RunContext, payload: None = None) -> TrendList:
        """Produce a ranked ``TrendList`` for the run.

        The ``payload`` is unused: Trend Scout is the head of the pipeline.
        """
        raw = await self._gather(ctx)
        self.log.info("fetched_trends", count=len(raw), sources=len(self._sources))

        scored = [trend.with_score(score_trend(trend, self._weights)) for trend in raw]
        filtered = [t for t in scored if self._matches_niche(t, ctx.config.niche_filters)]

        ranked = sorted(filtered, key=lambda t: t.score, reverse=True)
        result = TrendList(run_id=ctx.run_id, trends=tuple(ranked))

        self.log.info(
            "ranked_trends",
            kept=len(ranked),
            dropped=len(scored) - len(ranked),
            top=[(t.id, round(t.score, 4)) for t in result.top(3)],
        )
        return result

    async def _gather(self, ctx: RunContext) -> list[Trend]:
        """Fetch trends from all sources, capped at the configured concurrency.

        Fetching is routed through :func:`~trendz.concurrency.bounded_map` so the
        head stage respects ``ctx.config.concurrency_degree`` just like the
        downstream fan-out stages, rather than fanning out unbounded. Empty and
        no-source cases yield an empty list without error.
        """
        if not self._sources:
            return []
        results = await bounded_map(
            lambda source: self._fetch_source(ctx, source),
            self._sources,
            ctx.config.concurrency_degree,
        )
        return [trend for batch in results for trend in batch]

    async def _fetch_source(self, ctx: RunContext, source: TrendSource) -> list[Trend]:
        """Fetch one source, checking and debiting the run's quota ledger.

        This establishes the budget-guard idiom the design assigns to the run:
        skip the fetch when the run has no quota left, otherwise debit the
        estimated cost before doing the work. Downstream agents copy this shape
        for their own external calls (see ``ctx.budget`` helpers).
        """
        if not ctx.budget.can_use_quota(QUOTA_PER_FETCH):
            self.log.warning("quota_exhausted", source=source.name)
            return []
        ctx.budget.charge(calls=QUOTA_PER_FETCH)
        return await source.fetch(ctx)

    @staticmethod
    def _matches_niche(trend: Trend, niche_filters: tuple[str, ...]) -> bool:
        """Return True if the trend matches the niche filters.

        Empty filters match everything. Otherwise a trend is kept when any
        filter keyword appears (case-insensitively) in its title. Matching is
        title-only on purpose: the ``source`` string is an infrastructure
        identifier (for example "reddit" or "mock"), not topic content, so
        keying niche selection off it would silently widen the filter.
        """
        if not niche_filters:
            return True
        haystack = trend.title.lower()
        return any(keyword.lower() in haystack for keyword in niche_filters)
