"""Decomposed reward functions for GRPO training on DataClean-Env.

Each function follows TRL's reward function signature:
    def reward_fn(completions, **kwargs) -> list[float]

When used with `environment_factory`, kwargs includes `environments` —
a list of environment instances (one per completion). The reward functions
read accumulated state from these instances.
"""

from __future__ import annotations

import json
import re
from typing import Any


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

VALID_ACTIONS = {
    "fix_value", "delete_row", "fill_missing", "standardize_format",
    "merge_duplicates", "flag_anomaly", "split_column", "rename_column",
    "cast_type", "escalate_to_human", "mark_complete",
}


def _parse_action(text: str) -> dict:
    """Extract action JSON from model completion text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action_type" in data:
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\"action_type\".*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# -----------------------------------------------------------------------
# Reward 1: Action Validity (self-contained, no environment needed)
# -----------------------------------------------------------------------

def reward_valid_action(completions: list[str], **kwargs: Any) -> list[float]:
    """1.0 if the completion is a parseable, valid action; 0.0 otherwise."""
    rewards = []
    for text in completions:
        parsed = _parse_action(text)
        action_type = parsed.get("action_type", "")
        if action_type in VALID_ACTIONS:
            rewards.append(1.0)
        elif parsed:
            rewards.append(0.3)
        else:
            rewards.append(0.0)
    return rewards


# -----------------------------------------------------------------------
# Reward 2: Episode Score (reads from environment instance)
# -----------------------------------------------------------------------

def reward_episode_score(completions: list[str], **kwargs: Any) -> list[float]:
    """Final composite score from the grader at episode end.

    Reads `env.reward` from the environment instance provided via
    `environment_factory`. Falls back to 0.0 if not available.
    """
    environments = kwargs.get("environments", [])
    rewards = []
    for i, _ in enumerate(completions):
        if i < len(environments):
            env = environments[i]
            score = getattr(env, "reward", 0.0) or 0.0
            rewards.append(float(score))
        else:
            rewards.append(0.0)
    return rewards


# -----------------------------------------------------------------------
# Reward 3: Efficiency (reads budget from environment)
# -----------------------------------------------------------------------

def reward_efficiency(completions: list[str], **kwargs: Any) -> list[float]:
    """Reward frugal budget usage. Reads from environment state."""
    environments = kwargs.get("environments", [])
    rewards = []
    for i, _ in enumerate(completions):
        if i < len(environments):
            env = environments[i]
            state = getattr(env, "_state", None)
            if state is not None:
                budget_total = getattr(state, "action_budget", 100.0)
                budget_spent = getattr(state, "budget_spent", 0.0)
                if budget_total > 0:
                    rewards.append(max(0.0, 1.0 - (budget_spent / budget_total)))
                else:
                    rewards.append(0.5)
            else:
                rewards.append(0.5)
        else:
            rewards.append(0.5)
    return rewards


# -----------------------------------------------------------------------
# Reward 4: No Destruction (reads action log from environment)
# -----------------------------------------------------------------------

def reward_no_destruction(completions: list[str], **kwargs: Any) -> list[float]:
    """Penalize destructive actions (delete_row, wrong merges)."""
    environments = kwargs.get("environments", [])
    rewards = []
    for i, _ in enumerate(completions):
        if i < len(environments):
            env = environments[i]
            state = getattr(env, "_state", None)
            action_log = getattr(state, "action_log", []) if state else []
            destructions = 0.0
            for entry in action_log:
                action = entry.get("action", "")
                status = entry.get("status", "")
                if action == "delete_row" and status == "success":
                    destructions += 1.0
                if action == "merge_duplicates" and status == "success":
                    destructions += 0.5
            penalty = min(destructions * 0.15, 1.0)
            rewards.append(max(0.0, 1.0 - penalty))
        else:
            rewards.append(1.0)
    return rewards
