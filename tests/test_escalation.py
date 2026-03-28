"""Tests for the escalate_to_human action.

Verifies that escalation succeeds, errors on invalid rows,
records state, and consumes the correct budget cost.
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
# Escalation succeeds
# ---------------------------------------------------------------------------


def test_escalate_action_succeeds():
    """escalate_to_human on a valid row/column returns success."""
    env, obs = _reset_env("easy_contacts")
    first_row_id = obs.rows[0][0]
    col = obs.columns[1]

    obs2 = env.step(_make_action(
        "escalate_to_human", row_id=first_row_id, column=col,
        confidence=0.3, reason="unsure",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "success"


# ---------------------------------------------------------------------------
# Escalation with invalid row
# ---------------------------------------------------------------------------


def test_escalate_invalid_row():
    """escalate_to_human with a non-existent row_id returns error."""
    env, obs = _reset_env("easy_contacts")
    obs2 = env.step(_make_action(
        "escalate_to_human", row_id=99999, column="first_name",
        confidence=0.5, reason="bad row",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "error"
    assert "not found" in obs2.last_action_result.message


# ---------------------------------------------------------------------------
# Escalation recorded in state
# ---------------------------------------------------------------------------


def test_escalate_recorded_in_state():
    """Each successful escalation appends to escalated_cells in state."""
    env, obs = _reset_env("easy_contacts")
    first_row_id = obs.rows[0][0]
    col = obs.columns[1]

    assert len(env.state.escalated_cells) == 0

    env.step(_make_action(
        "escalate_to_human", row_id=first_row_id, column=col,
        confidence=0.4, reason="ambiguous",
    ))
    assert len(env.state.escalated_cells) == 1
    assert env.state.escalated_cells[0]["row_id"] == first_row_id
    assert env.state.escalated_cells[0]["column"] == col

    # Escalate a second cell
    second_row_id = obs.rows[1][0]
    env.step(_make_action(
        "escalate_to_human", row_id=second_row_id, column=col,
        confidence=0.2, reason="also ambiguous",
    ))
    assert len(env.state.escalated_cells) == 2


# ---------------------------------------------------------------------------
# Escalation cost
# ---------------------------------------------------------------------------


def test_escalate_cost():
    """escalate_to_human costs 0.5 budget."""
    env, obs = _reset_env("easy_contacts")
    first_row_id = obs.rows[0][0]
    col = obs.columns[1]

    env.step(_make_action(
        "escalate_to_human", row_id=first_row_id, column=col,
        confidence=0.5, reason="test",
    ))
    assert env.state.budget_spent == pytest.approx(ACTION_COSTS["escalate_to_human"])
    assert ACTION_COSTS["escalate_to_human"] == pytest.approx(0.5)
