"""A deterministic, in-memory publisher-client for offline development and tests.

:class:`StubPublisherClient` simulates publishing a
:class:`~trendz.contracts.ScheduledPost` with no real network or API keys: the
``post_id``/``post_url`` are pure functions of the idempotency key and the
``published_at`` is carried through from the post's scheduled time, so the whole
pipeline (and its tests) runs fully offline and deterministically. Swap it for a
real :class:`~trendz.publishers.base.PublisherClient` adapter in production.

It models the two behaviours a real client must guarantee:

    - IDEMPOTENCY: it remembers every idempotency key it has published and, on a
      repeat, returns the SAME prior :class:`~trendz.publishers.base.PublishOutcome`
      instead of publishing again. :attr:`publish_count` counts only real
      (non-deduped) publishes so tests can assert a duplicate did not double-post.
    - CONFIGURABLE FAILURE: a constructor knob (``fail_platforms`` / ``fail_keys``)
      forces deterministic failures for the record-and-continue test path.

Mirrors :class:`~trendz.workers.stub_worker.StubClipWorker` and
:class:`~trendz.checkers.stub_checker.StubClipChecker` in style and intent.
"""

from __future__ import annotations

from trendz.contracts import Platform, RunContext, ScheduledPost
from trendz.publishers.base import PublisherClient, PublishOutcome


class StubPublisherClient(PublisherClient):
    """A publisher-client that simulates posting deterministically, offline."""

    def __init__(
        self,
        *,
        fail_platforms: frozenset[Platform] | None = None,
        fail_keys: frozenset[str] | None = None,
    ) -> None:
        """Create the stub client.

        Args:
            fail_platforms: Platforms whose publishes should deterministically
                fail (raise), for exercising record-and-continue. Copied so the
                caller's set is not shared or mutated.
            fail_keys: Idempotency keys whose publishes should deterministically
                fail (raise). Copied so the caller's set is not shared.
        """
        self._fail_platforms: frozenset[Platform] = (
            frozenset(fail_platforms) if fail_platforms is not None else frozenset()
        )
        self._fail_keys: frozenset[str] = (
            frozenset(fail_keys) if fail_keys is not None else frozenset()
        )
        # idempotency_key -> prior outcome. A repeat key returns the prior
        # outcome without re-publishing (models the platform's dedup contract).
        self._published: dict[str, PublishOutcome] = {}
        # Counts only real (non-deduped) publishes, so tests can assert a
        # duplicate key did not increment it.
        self._publish_count = 0

    @property
    def name(self) -> str:
        return "stub_publisher_client"

    @property
    def publish_count(self) -> int:
        """Number of real (non-deduped) publishes performed so far."""
        return self._publish_count

    @property
    def published_keys(self) -> frozenset[str]:
        """Snapshot view of the idempotency keys published so far."""
        return frozenset(self._published)

    async def publish(
        self, ctx: RunContext, post: ScheduledPost, idempotency_key: str
    ) -> PublishOutcome:
        """Simulate publishing a post, honouring idempotency and forced failures.

        A repeated ``idempotency_key`` returns the prior outcome without
        re-publishing (``publish_count`` unchanged). Otherwise the outcome is a
        pure function of the key (``post_id``/``post_url``) and the post's
        scheduled time (``published_at``), so the same input always yields the
        same result. If the post's platform or key is configured to fail, this
        raises instead so the Publisher can record a failed result.
        """
        prior = self._published.get(idempotency_key)
        if prior is not None:
            return prior

        if post.platform in self._fail_platforms or idempotency_key in self._fail_keys:
            raise RuntimeError(f"stub forced publish failure for {idempotency_key}")

        outcome = PublishOutcome(
            post_id=f"post-{idempotency_key}",
            post_url=f"stub://{post.platform}/posts/{idempotency_key}",
            published_at=post.scheduled_at,
        )
        self._published[idempotency_key] = outcome
        self._publish_count += 1
        return outcome
