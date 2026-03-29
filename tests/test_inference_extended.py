"""Extended tests for inference.py: _parse_action edge cases and task validation."""

from __future__ import annotations

import json

import pytest

from dataclean_env.inference import _parse_action
from dataclean_env.server.tasks import get_task


# ---------------------------------------------------------------------------
# _parse_action tests
# ---------------------------------------------------------------------------


def test_parse_action_clean_json():
    """Clean JSON with action_type should parse correctly."""
    text = json.dumps({"action_type": "fix_value", "params": {"row_id": 0, "column": "name", "new_value": "Alice"}})
    action = _parse_action(text)
    assert action.action_type == "fix_value"
    assert action.params["new_value"] == "Alice"


def test_parse_action_strips_markdown():
    """JSON wrapped in ```json fences should still parse."""
    text = '```json\n{"action_type": "delete_row", "params": {"row_id": 5}}\n```'
    action = _parse_action(text)
    assert action.action_type == "delete_row"
    assert action.params["row_id"] == 5


def test_parse_action_fallback_on_garbage():
    """Completely unparseable text should fall back to mark_complete."""
    action = _parse_action("This is not JSON at all, just random text.")
    assert action.action_type == "mark_complete"


def test_parse_action_with_surrounding_text():
    """JSON embedded in surrounding explanation text should be extracted."""
    text = 'I think we should do this: {"action_type": "flag_anomaly", "params": {"row_id": 3, "column": "email", "reason": "suspicious"}} and that should fix it.'
    action = _parse_action(text)
    assert action.action_type == "flag_anomaly"
    assert action.params["row_id"] == 3


def test_parse_action_nested_params():
    """Params containing nested structures like new_names list should parse."""
    text = json.dumps({
        "action_type": "split_column",
        "params": {"column": "full_name", "delimiter": " ", "new_names": ["first", "last"]},
    })
    action = _parse_action(text)
    assert action.action_type == "split_column"
    assert action.params["new_names"] == ["first", "last"]


# ---------------------------------------------------------------------------
# Task validation tests
# ---------------------------------------------------------------------------


def test_get_task_invalid_key_raises():
    """Requesting a non-existent task should raise KeyError."""
    with pytest.raises(KeyError, match="not found"):
        get_task("nonexistent_task_xyz")


def test_medium_task_has_corruptions():
    """The medium task should define at least one corruption."""
    task = get_task("medium_employees")
    assert len(task.corruptions) > 0
    assert task.difficulty == "medium"
    assert task.max_steps > 0


def test_hard_task_has_utility_probes():
    """The hard task should include utility probes for downstream validation."""
    task = get_task("hard_patients")
    assert len(task.utility_probes) > 0
    # Each probe should have the required fields
    for probe in task.utility_probes:
        assert probe.name
        assert probe.query_fn
        assert probe.expected_result is not None
