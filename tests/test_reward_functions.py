"""Tests for GRPO reward functions in scripts/reward_functions.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add scripts/ to path so we can import reward_functions
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from reward_functions import (
    reward_valid_action,
    reward_episode_score,
    reward_efficiency,
    reward_no_destruction,
)


# -----------------------------------------------------------------------
# reward_valid_action
# -----------------------------------------------------------------------


class TestRewardValidAction:
    def test_valid_action_gets_1(self):
        completions = ['{"action_type": "fix_value", "params": {"row_id": 0}}']
        assert reward_valid_action(completions) == [1.0]

    def test_invalid_json_gets_0(self):
        completions = ["this is not json at all"]
        assert reward_valid_action(completions) == [0.0]

    def test_unknown_action_gets_partial(self):
        completions = ['{"action_type": "unknown_action"}']
        assert reward_valid_action(completions) == [0.3]

    def test_mark_complete_valid(self):
        completions = ['{"action_type": "mark_complete", "params": {}}']
        assert reward_valid_action(completions) == [1.0]

    def test_markdown_fenced_json(self):
        completions = ['```json\n{"action_type": "fill_missing", "params": {}}\n```']
        assert reward_valid_action(completions) == [1.0]

    def test_batch(self):
        completions = [
            '{"action_type": "fix_value"}',
            "garbage",
            '{"action_type": "delete_row"}',
        ]
        result = reward_valid_action(completions)
        assert result == [1.0, 0.0, 1.0]


# -----------------------------------------------------------------------
# reward_episode_score
# -----------------------------------------------------------------------


class TestRewardEpisodeScore:
    def test_reads_env_reward(self):
        env = MagicMock()
        env.reward = 0.85
        result = reward_episode_score(["completion"], environments=[env])
        assert result == [0.85]

    def test_no_environments_returns_zero(self):
        result = reward_episode_score(["completion"])
        assert result == [0.0]

    def test_none_reward_returns_zero(self):
        env = MagicMock()
        env.reward = None
        result = reward_episode_score(["completion"], environments=[env])
        assert result == [0.0]


# -----------------------------------------------------------------------
# reward_efficiency
# -----------------------------------------------------------------------


class TestRewardEfficiency:
    def test_no_budget_spent(self):
        env = MagicMock()
        env._state.action_budget = 100.0
        env._state.budget_spent = 0.0
        result = reward_efficiency(["completion"], environments=[env])
        assert result == [1.0]

    def test_half_budget_spent(self):
        env = MagicMock()
        env._state.action_budget = 100.0
        env._state.budget_spent = 50.0
        result = reward_efficiency(["completion"], environments=[env])
        assert result == [0.5]

    def test_no_environments_returns_default(self):
        result = reward_efficiency(["completion"])
        assert result == [0.5]


# -----------------------------------------------------------------------
# reward_no_destruction
# -----------------------------------------------------------------------


class TestRewardNoDestruction:
    def test_no_destructive_actions(self):
        env = MagicMock()
        env._state.action_log = [
            {"action": "fix_value", "status": "success"},
            {"action": "fill_missing", "status": "success"},
        ]
        result = reward_no_destruction(["completion"], environments=[env])
        assert result == [1.0]

    def test_delete_row_penalized(self):
        env = MagicMock()
        env._state.action_log = [
            {"action": "delete_row", "status": "success"},
        ]
        result = reward_no_destruction(["completion"], environments=[env])
        assert result == [0.85]  # 1.0 - 1 * 0.15

    def test_multiple_destructions_capped(self):
        env = MagicMock()
        env._state.action_log = [
            {"action": "delete_row", "status": "success"},
            {"action": "delete_row", "status": "success"},
            {"action": "delete_row", "status": "success"},
            {"action": "delete_row", "status": "success"},
            {"action": "delete_row", "status": "success"},
            {"action": "delete_row", "status": "success"},
            {"action": "delete_row", "status": "success"},
        ]
        result = reward_no_destruction(["completion"], environments=[env])
        assert result == [0.0]  # capped at 0

    def test_failed_delete_not_penalized(self):
        env = MagicMock()
        env._state.action_log = [
            {"action": "delete_row", "status": "error"},
        ]
        result = reward_no_destruction(["completion"], environments=[env])
        assert result == [1.0]
