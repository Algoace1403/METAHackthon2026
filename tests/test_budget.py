"""Tests for the intervention budget system.

Verifies that budget tracking, action costs, and budget visibility
in observations work correctly.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataclean_env.server.environment import ACTION_COSTS, DataCleanEnvironment
from dataclean_env.models import DataCleanAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_action(action_type: str, **params: Any) -> DataCleanAction:
    return DataCleanAction(action_type=action_type, params=params)


def _reset_env(task_id: str = "easy_contacts", seed: int = 42) -> tuple:
    env = DataCleanEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    return env, obs


# ---------------------------------------------------------------------------
# Budget initialisation
# ---------------------------------------------------------------------------


def test_budget_initialized():
    """After reset the budget_remaining should be positive."""
    env, obs = _reset_env("easy_contacts")
    assert obs.budget_remaining > 0
    assert obs.budget_spent == 0.0


# ---------------------------------------------------------------------------
# Budget decreases on action
# ---------------------------------------------------------------------------


def test_budget_decreases_on_action():
    """fix_value costs 1.0 -- budget should decrease accordingly."""
    env, obs = _reset_env("easy_contacts")
    initial_remaining = obs.budget_remaining

    first_row_id = obs.rows[0][0]
    col = obs.columns[1]
    obs2 = env.step(_make_action("fix_value", row_id=first_row_id, column=col, new_value="X"))

    assert obs2.budget_spent == pytest.approx(ACTION_COSTS["fix_value"])
    assert obs2.budget_remaining == pytest.approx(initial_remaining - ACTION_COSTS["fix_value"])


# ---------------------------------------------------------------------------
# Expensive action costs more
# ---------------------------------------------------------------------------


def test_expensive_action_costs_more():
    """delete_row costs 6.0 -- much more than fix_value's 1.0."""
    env, obs = _reset_env("easy_contacts")
    initial_remaining = obs.budget_remaining

    first_row_id = obs.rows[0][0]
    obs2 = env.step(_make_action("delete_row", row_id=first_row_id))

    assert obs2.budget_spent == pytest.approx(ACTION_COSTS["delete_row"])
    assert obs2.budget_remaining == pytest.approx(initial_remaining - ACTION_COSTS["delete_row"])
    # delete_row should cost more than fix_value
    assert ACTION_COSTS["delete_row"] > ACTION_COSTS["fix_value"]


# ---------------------------------------------------------------------------
# Budget visible in observation
# ---------------------------------------------------------------------------


def test_budget_in_observation():
    """Observation should expose budget_spent and budget_remaining."""
    env, obs = _reset_env("easy_contacts")
    assert hasattr(obs, "budget_spent")
    assert hasattr(obs, "budget_remaining")
    assert isinstance(obs.budget_spent, (int, float))
    assert isinstance(obs.budget_remaining, (int, float))


def test_action_costs_in_observation():
    """Observation should expose the action_costs dict."""
    env, obs = _reset_env("easy_contacts")
    assert hasattr(obs, "action_costs")
    assert isinstance(obs.action_costs, dict)
    assert "fix_value" in obs.action_costs
    assert "delete_row" in obs.action_costs
    assert obs.action_costs["fix_value"] == pytest.approx(1.0)
    assert obs.action_costs["delete_row"] == pytest.approx(6.0)
