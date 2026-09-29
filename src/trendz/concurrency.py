"""Reusable concurrency primitives for the Trendz pipeline.

:func:`bounded_map` is the semaphore-bounded worker pool that every fan-out
stage funnels through (Trend Scout over its sources, Clip Factory over N briefs,
Publisher over M platforms). Keeping it in a dependency-free module lets both the
Orchestrator and the individual agents import it without a circular dependency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def bounded_map(
    func: Callable[[T], Awaitable[R]],
    items: Iterable[T],
    concurrency: int,
) -> list[R]:
    """Run ``func`` over ``items`` concurrently, capped at ``concurrency``.

    This is the reusable bounded worker-pool helper the fan-out stages use
    (Clip Factory over N briefs, Publisher over M platforms). An
    ``asyncio.Semaphore`` limits how many coroutines run at once so the run
    respects API rate limits and render/GPU budget. Results preserve input
    order.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    semaphore = asyncio.Semaphore(concurrency)

    async def _worker(item: T) -> R:
        async with semaphore:
            return await func(item)

    return await asyncio.gather(*(_worker(item) for item in items))
