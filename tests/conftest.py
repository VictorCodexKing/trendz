"""Shared pytest fixtures for the Trendz test suite."""

from __future__ import annotations

import pytest

from trendz.contracts import RunConfig, RunContext


@pytest.fixture
def config() -> RunConfig:
    """A default run configuration with no niche filters."""
    return RunConfig()


@pytest.fixture
def ctx(config: RunConfig) -> RunContext:
    """A run context built from the default config."""
    return RunContext.new(run_id="test-run", config=config)
