"""Tests for DataCleanGrader: scoring, cell matching, penalties, format checks."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest

from dataclean_env.server.grader import DataCleanGrader


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _simple_gt() -> List[Dict[str, Any]]:
    """Small ground truth dataset for grader tests."""
    return [
        {"_entity_id": "E1", "name": "Alice", "email": "alice@x.com", "phone": "(555) 123-4567", "date": "2023-01-15"},
        {"_entity_id": "E2", "name": "Bob", "email": "bob@x.com", "phone": "(555) 234-5678", "date": "2023-06-20"},
        {"_entity_id": "E3", "name": "Carol", "email": "carol@x.com", "phone": "(555) 345-6789", "date": "2023-09-01"},
    ]


def _simple_schema() -> Dict[str, Any]:
    return {
        "primary_key": None,
        "expected_types": {
            "name": "name",
            "email": "email",
            "phone": "phone",
            "date": "date",
        },
        "constraints": {},
    }


def _grade(final_data, ground_truth, original_data=None, action_history=None, schema=None):
    grader = DataCleanGrader()
    if original_data is None:
        original_data = copy.deepcopy(ground_truth)
    if action_history is None:
        action_history = []
    if schema is None:
        schema = _simple_schema()
    return grader.grade(
        final_data=final_data,
        ground_truth=ground_truth,
        original_data=original_data,
        action_history=action_history,
        schema=schema,
        flagged_cells=[],
    )


# ---------------------------------------------------------------------------
# Score tests
# ---------------------------------------------------------------------------


def test_perfect_score():
    gt = _simple_gt()
    result = _grade(copy.deepcopy(gt), gt)
    assert result.score == 1.0


def test_dirty_score_less_than_perfect():
    gt = _simple_gt()
    dirty = copy.deepcopy(gt)
    dirty[0]["name"] = "Alcie"  # typo
    dirty[1]["email"] = None  # null
    result = _grade(dirty, gt, original_data=dirty)
    assert result.score < 1.0


def test_partial_fix_improves_score():
    gt = _simple_gt()
    dirty = copy.deepcopy(gt)
    dirty[0]["name"] = "Alcie"
    dirty[1]["email"] = None

    score_dirty = _grade(dirty, gt, original_data=dirty).score

    partial = copy.deepcopy(dirty)
    partial[0]["name"] = "Alice"  # fix one cell

    score_partial = _grade(partial, gt, original_data=dirty).score
    assert score_partial > score_dirty


def test_scores_monotonic():
    gt = _simple_gt()
    dirty = copy.deepcopy(gt)
    dirty[0]["name"] = "Alcie"
    dirty[1]["email"] = None
    dirty[2]["phone"] = "broken"

    scores = []
    current = copy.deepcopy(dirty)
    scores.append(_grade(current, gt, original_data=dirty).score)

    current[0]["name"] = "Alice"
    scores.append(_grade(current, gt, original_data=dirty).score)

    current[1]["email"] = "bob@x.com"
    scores.append(_grade(current, gt, original_data=dirty).score)

    current[2]["phone"] = "(555) 345-6789"
    scores.append(_grade(current, gt, original_data=dirty).score)

    # Each fix should yield >= previous score
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i - 1], f"Score dropped: {scores[i-1]} -> {scores[i]}"


# ---------------------------------------------------------------------------
# Penalty tests
# ---------------------------------------------------------------------------


def test_delete_valid_row_penalty():
    gt = _simple_gt()
    full = copy.deepcopy(gt)
    # Simulate: agent deleted row with entity E2
    deleted_data = copy.deepcopy(gt)
    del deleted_data[1]  # remove E2

    action_history = [
        {
            "action": "delete_row",
            "status": "success",
            "deleted_data": {"_entity_id": "E2", "name": "Bob"},
        }
    ]
    result = _grade(deleted_data, gt, original_data=full, action_history=action_history)
    assert result.penalties > 0


def test_merge_different_entities_penalty():
    gt = _simple_gt()
    action_history = [
        {
            "action": "merge_duplicates",
            "status": "success",
            "entity_id1": "E1",
            "entity_id2": "E2",
        }
    ]
    merged = copy.deepcopy(gt)
    del merged[1]  # E2 removed by merge
    result = _grade(merged, gt, original_data=copy.deepcopy(gt), action_history=action_history)
    assert result.penalties >= 0.10


# ---------------------------------------------------------------------------
# Cell matching tests
# ---------------------------------------------------------------------------


def test_case_sensitive_str():
    grader = DataCleanGrader()
    assert grader._cell_match("co", "CO", "str") is False


def test_case_insensitive_name():
    grader = DataCleanGrader()
    assert grader._cell_match("john", "John", "name") is True


def test_date_matching():
    grader = DataCleanGrader()
    assert grader._cell_match("2023-01-15", "01/15/2023", "date") is True


def test_phone_matching():
    grader = DataCleanGrader()
    assert grader._cell_match("(555) 123-4567", "5551234567", "phone") is True


# ---------------------------------------------------------------------------
# Format matching tests
# ---------------------------------------------------------------------------


def test_format_matching_named():
    assert DataCleanGrader._matches_format("2023-01-15", "YYYY-MM-DD") is True
    assert DataCleanGrader._matches_format("01/15/2023", "YYYY-MM-DD") is False


def test_format_matching_regex():
    assert DataCleanGrader._matches_format("12345", r"^\d{5}$") is True
    assert DataCleanGrader._matches_format("1234", r"^\d{5}$") is False


# ---------------------------------------------------------------------------
# Entity ID alignment
# ---------------------------------------------------------------------------


def test_entity_id_alignment():
    grader = DataCleanGrader()
    gt = _simple_gt()
    # Reverse the order -- alignment should still map correctly by _entity_id
    shuffled = list(reversed(copy.deepcopy(gt)))
    schema = _simple_schema()
    alignment = grader._align_rows(shuffled, gt, schema)
    # alignment maps gt_index -> final_data_index
    for gt_i, fd_i in alignment.items():
        assert gt[gt_i]["_entity_id"] == shuffled[fd_i]["_entity_id"]
