"""The pluggable publisher-client abstraction for the Publisher (stage 6).

A :class:`PublisherClient` knows how to publish one
:class:`~trendz.contracts.ScheduledPost` to its target platform. The Publisher
agent is dependency-injected with a client and fans it out per platform, so the
same agent works against a deterministic in-memory stub in tests and against a
real platform API in production.

Per DESIGN.md stage 6, a real client calls each platform's publishing API
(YouTube Shorts, Instagram, TikTok, Facebook), handling auth, upload, and the
platform's own idempotency/dedup semantics. Publishing is modelled around an
:attr:`idempotency_key` the Publisher derives per post: a client MUST treat a
repeated key as the same post and return the prior outcome rather than
publishing again, so a retried or duplicated post is never double-published.

A successful publish returns a :class:`PublishOutcome` carrying the platform
``post_id``/``post_url`` and the aware-UTC ``published_at`` time; a failure is
signalled by raising, which the Publisher records as a failed
:class:`~trendz.contracts.PostResult` and continues (one bad post never blocks
the run). The stub client in :mod:`trendz.publishers.stub_client` implements this
interface with no network or API keys for offline tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from trendz.contracts import RunContext, ScheduledPost


class PublishOutcome(BaseModel):
    """The successful result of publishing a single post to a platform.

    Frozen: an outcome is an immutable fact once a post lands. The Publisher uses
    it to build a :class:`~trendz.contracts.PostResult`, and the client returns
    the same outcome for a repeated idempotency key (dedup) rather than
    publishing again.
    """

    model_config = ConfigDict(frozen=True)

    post_id: str = Field(description="Platform-assigned post id.")
    post_url: str = Field(description="Platform URL the post is reachable at.")
    published_at: datetime = Field(description="Aware UTC time the post was published.")


class PublisherClient(ABC):
    """Abstract base for a single platform publisher-client adapter."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this client (used in logs)."""
        raise NotImplementedError

    @abstractmethod
    async def publish(
        self, ctx: RunContext, post: ScheduledPost, idempotency_key: str
    ) -> PublishOutcome:
        """Publish one scheduled post to its platform.

        Args:
            ctx: The run context, for run-scoped config. The caller (Publisher)
                is responsible for any budget/quota guard around this call via
                ``ctx.budget``.
            post: The scheduled post to publish. Must not be mutated: posts are
                frozen and shared across concurrent publishers.
            idempotency_key: A stable key identifying this post within the run.
                The client MUST return the prior outcome for a repeated key
                instead of publishing again, guaranteeing no double-post.

        Returns:
            A :class:`PublishOutcome` with the platform post id/url and the aware
            UTC publish time.

        Raises:
            Exception: Any failure to publish. The Publisher catches it and
                records a failed :class:`~trendz.contracts.PostResult`, so one
                bad post never blocks the run.
        """
        raise NotImplementedError
