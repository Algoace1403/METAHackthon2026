"""Tests for the inference script's helper functions and system prompt contract."""

from __future__ import annotations

import pytest

from dataclean_env.inference import _normalize_params, SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Param normalization
# ---------------------------------------------------------------------------


def test_normalize_params_fix_value():
    result = _normalize_params("fix_value", {"row_id": 0, "column": "name", "value": "Alice"})
    assert "new_value" in result
    assert result["new_value"] == "Alice"
    assert "value" not in result


def test_normalize_params_fix_value_preserves_new_value():
    """When new_value is already present, value should NOT overwrite it."""
    result = _normalize_params("fix_value", {"row_id": 0, "column": "name", "new_value": "Bob", "value": "Alice"})
    assert result["new_value"] == "Bob"


def test_normalize_params_merge():
    result = _normalize_params("merge_duplicates", {"row_id_1": 1, "row_id_2": 2, "strategy": "keep_first"})
    assert "row_id1" in result
    assert result["row_id1"] == 1
    assert "row_id2" in result
    assert result["row_id2"] == 2
    assert "row_id_1" not in result
    assert "row_id_2" not in result


def test_normalize_params_merge_preserves_canonical():
    """When row_id1/row_id2 already present, row_id_1/row_id_2 should NOT overwrite."""
    result = _normalize_params("merge_duplicates", {"row_id1": 10, "row_id2": 20, "row_id_1": 99, "row_id_2": 88})
    assert result["row_id1"] == 10
    assert result["row_id2"] == 20


def test_normalize_params_noop_for_other_actions():
    """Actions that need no normalization should pass through unchanged."""
    original = {"row_id": 5, "column": "email", "reason": "suspicious"}
    result = _normalize_params("flag_anomaly", original)
    assert result == original


# ---------------------------------------------------------------------------
# System prompt contract
# ---------------------------------------------------------------------------


_ALL_ACTIONS = [
    "fix_value",
    "delete_row",
    "fill_missing",
    "standardize_format",
    "merge_duplicates",
    "flag_anomaly",
    "split_column",
    "rename_column",
    "cast_type",
    "mark_complete",
]


def test_system_prompt_has_all_actions():
    for action in _ALL_ACTIONS:
        assert action in SYSTEM_PROMPT, f"System prompt missing action type: {action}"


def test_system_prompt_mentions_row_id():
    assert "row_id" in SYSTEM_PROMPT


def test_system_prompt_mentions_json():
    assert "JSON" in SYSTEM_PROMPT
