"""Grader invariance and determinism tests.

Verifies that the same seed produces the same data and scores,
different seeds produce different data, and grading is deterministic.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from dataclean_env.server.environment import DataCleanEnvironment
from dataclean_env.server.grader import DataCleanGrader
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


def _run_fixed_actions_and_complete(env: "DataCleanEnvironment", obs: Any) -> float:
    """Apply a deterministic sequence of actions and return the final score."""
    first_row_id = obs.rows[0][0]
    col = obs.columns[1]
    env.step(_make_action("fix_value", row_id=first_row_id, column=col, new_value="TestDet"))
    obs_final = env.step(_make_action("mark_complete"))
    assert obs_final.done is True
    return obs_final.reward


# ---------------------------------------------------------------------------
# Same seed produces same score
# ---------------------------------------------------------------------------


def test_same_seed_same_score():
    """Resetting with seed=42, performing identical actions, yields identical final scores."""
    env1, obs1 = _reset_env("easy_contacts", seed=42)
    score1 = _run_fixed_actions_and_complete(env1, obs1)

    env2, obs2 = _reset_env("easy_contacts", seed=42)
    score2 = _run_fixed_actions_and_complete(env2, obs2)

    assert score1 == pytest.approx(score2), (
        f"Same seed produced different scores: {score1} vs {score2}"
    )


# ---------------------------------------------------------------------------
# Different seeds produce different dirty data
# ---------------------------------------------------------------------------


def test_different_seed_different_data():
    """seed=42 vs seed=99 should produce different dirty data rows."""
    env1, obs1 = _reset_env("easy_contacts", seed=42)
    env2, obs2 = _reset_env("easy_contacts", seed=99)

    # Both should have the same columns (schema is fixed)
    assert obs1.columns == obs2.columns

    # But the row values should differ (different corruption noise)
    rows1 = [tuple(row) for row in obs1.rows]
    rows2 = [tuple(row) for row in obs2.rows]
    assert rows1 != rows2, "Different seeds produced identical dirty data"


# ---------------------------------------------------------------------------
# Grading is deterministic
# ---------------------------------------------------------------------------


def test_score_deterministic():
    """Grading the same data twice returns identical results."""
    env, obs = _reset_env("easy_contacts", seed=42)
    grader = DataCleanGrader()

    state = env.state
    kwargs = dict(
        final_data=copy.deepcopy(state.current_data),
        ground_truth=copy.deepcopy(state.ground_truth),
        original_data=copy.deepcopy(state.original_dirty),
        action_history=[],
        schema=state.schema_def,
        flagged_cells=[],
        budget_spent=0.0,
        action_budget=state.action_budget,
        escalated_cells=[],
        ambiguous_cells=[],
        utility_probes=[],
    )

    result1 = grader.grade(**kwargs)
    result2 = grader.grade(**kwargs)

    assert result1.score == pytest.approx(result2.score)
    assert result1.accuracy == pytest.approx(result2.accuracy)
    assert result1.completeness == pytest.approx(result2.completeness)
    assert result1.efficiency == pytest.approx(result2.efficiency)
    assert result1.penalties == pytest.approx(result2.penalties)
    assert result1.bonuses == pytest.approx(result2.bonuses)
