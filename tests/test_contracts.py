"""Offline tests for the shared data contracts, focused on the budget ledger."""

from __future__ import annotations

import pytest

from trendz.contracts import BudgetLedger, RunConfig


def test_from_config_seeds_limits_from_run_config() -> None:
    """A ledger built from config inherits the run's budget and quota caps."""
    config = RunConfig(budget_limit=50.0, quota_limit=200)
    ledger = BudgetLedger.from_config(config)

    assert ledger.budget_limit == 50.0
    assert ledger.quota_limit == 200
    assert ledger.spent == 0.0
    assert ledger.quota_used == 0


def test_charge_debits_spend_and_quota() -> None:
    """``charge`` debits both spend and quota in place."""
    ledger = BudgetLedger(budget_limit=100.0, quota_limit=10)

    ledger.charge(amount=25.0, calls=3)

    assert ledger.spent == 25.0
    assert ledger.quota_used == 3


def test_remaining_properties_track_debits_and_floor_at_zero() -> None:
    """``budget_remaining`` / ``quota_remaining`` follow debits and never go negative."""
    ledger = BudgetLedger(budget_limit=30.0, quota_limit=5)

    ledger.charge(amount=10.0, calls=2)
    assert ledger.budget_remaining == 20.0
    assert ledger.quota_remaining == 3

    # Overshooting the caps clamps the remaining values at zero.
    ledger.charge(amount=100.0, calls=100)
    assert ledger.budget_remaining == 0.0
    assert ledger.quota_remaining == 0


def test_can_spend_respects_the_budget_cap() -> None:
    """``can_spend`` is true up to the cap and false beyond it."""
    ledger = BudgetLedger(budget_limit=20.0, quota_limit=10)
    ledger.charge(amount=15.0)

    assert ledger.can_spend(5.0) is True
    assert ledger.can_spend(5.01) is False


def test_can_use_quota_respects_the_quota_cap() -> None:
    """``can_use_quota`` is true up to the cap and false beyond it."""
    ledger = BudgetLedger(budget_limit=20.0, quota_limit=4)
    ledger.charge(calls=3)

    assert ledger.can_use_quota(1) is True
    assert ledger.can_use_quota(2) is False
    # Defaults to a single call.
    assert ledger.can_use_quota() is True


def test_charge_rejects_negative_amounts() -> None:
    """Negative debits are a programming error and are rejected."""
    ledger = BudgetLedger(budget_limit=20.0, quota_limit=10)

    with pytest.raises(ValueError):
        ledger.charge(amount=-1.0)
    with pytest.raises(ValueError):
        ledger.charge(calls=-1)
