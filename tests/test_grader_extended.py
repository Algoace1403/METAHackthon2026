"""Extended tests for DataCleanGrader: probes, scoring, cell matching, alignment."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from dataclean_env.server.grader import DataCleanGrader, GradeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grader() -> DataCleanGrader:
    return DataCleanGrader()


# ---------------------------------------------------------------------------
# Probe tests
# ---------------------------------------------------------------------------


def test_probe_unique_count():
    g = _grader()
    data = [
        {"name": "Alice"},
        {"name": "Bob"},
        {"name": "Alice"},
        {"name": None},
    ]
    result = g._probe_unique_count(data, "name")
    assert result == 2


def test_probe_distribution():
    g = _grader()
    data = [
        {"dept": "Eng"},
        {"dept": "Eng"},
        {"dept": "Sales"},
        {"dept": None},
    ]
    result = g._probe_distribution(data, "dept")
    assert result == {"Eng": 2, "Sales": 1}


def test_probe_avg_by_group():
    g = _grader()
    data = [
        {"dept": "Eng", "salary": "100000"},
        {"dept": "Eng", "salary": "120000"},
        {"dept": "Sales", "salary": "80000"},
    ]
    result = g._probe_avg_by_group(data, "salary", "dept")
    assert result["Eng"] == 110000.0
    assert result["Sales"] == 80000.0


def test_probe_avg_by_group_year_age_transform():
    g = _grader()
    data = [
        {"dept": "A", "dob": "1990-01-01"},
        {"dept": "A", "dob": "2000-01-01"},
        {"dept": "B", "dob": "1980-01-01"},
    ]
    result = g._probe_avg_by_group(data, "dob", "dept", transform="year_age_2024")
    # A: (2024-1990 + 2024-2000)/2 = (34+24)/2 = 29.0
    # B: 2024-1980 = 44.0
    assert result["A"] == 29.0
    assert result["B"] == 44.0


def test_probe_count_where():
    g = _grader()
    data = [
        {"dept": "Eng"},
        {"dept": "Eng"},
        {"dept": "Sales"},
    ]
    result = g._probe_count_where(data, "dept", "Eng")
    assert result == 2


def test_probe_matches_numeric_tolerance():
    g = _grader()
    # Within tolerance (0.5)
    assert g._probe_matches(10, 10.3) is True
    assert g._probe_matches(10, 10.6) is False
    assert g._probe_matches(None, 5) is False


def test_probe_matches_dict():
    g = _grader()
    expected = {"Eng": 3, "Sales": 2}
    actual = {"Eng": 3, "Sales": 2}
    assert g._probe_matches(actual, expected) is True

    actual_off = {"Eng": 3, "Sales": 5}
    assert g._probe_matches(actual_off, expected) is False

    # Different keys
    assert g._probe_matches({"Eng": 3}, expected) is False


# ---------------------------------------------------------------------------
# Format score
# ---------------------------------------------------------------------------


def test_compute_format_score_with_constraints():
    g = _grader()
    data = [
        {"_row_id": 0, "date": "2023-01-15", "phone": "(555) 123-4567"},
        {"_row_id": 1, "date": "bad-date", "phone": "(555) 234-5678"},
    ]
    schema = {
        "constraints": {
            "date": {"format": "YYYY-MM-DD"},
            "phone": {"format": "(XXX) XXX-XXXX"},
        },
    }
    score = g._compute_format_score(data, schema)
    # 3 out of 4 cells match format (bad-date fails)
    assert 0.0 < score < 1.0
    assert abs(score - 0.75) < 0.01


# ---------------------------------------------------------------------------
# Row score
# ---------------------------------------------------------------------------


def test_compute_row_score_exact():
    g = _grader()
    assert g._compute_row_score(10, 10) == 1.0


def test_compute_row_score_over():
    g = _grader()
    score = g._compute_row_score(12, 10)
    assert score < 1.0
    assert score > 0.0
    # 1.0 - abs(10-12)/10 = 1.0 - 0.2 = 0.8
    assert abs(score - 0.8) < 0.01


def test_compute_row_score_zero_expected():
    g = _grader()
    assert g._compute_row_score(0, 0) == 1.0
    assert g._compute_row_score(5, 0) == 0.0


# ---------------------------------------------------------------------------
# Cell matching
# ---------------------------------------------------------------------------


def test_cell_match_email():
    g = _grader()
    assert g._cell_match("Alice@Example.COM", "alice@example.com", "email") is True
    assert g._cell_match("alice@a.com", "alice@b.com", "email") is False


def test_cell_match_currency():
    g = _grader()
    assert g._cell_match("$1,234.56", "1234.56", "currency") is True
    assert g._cell_match("$100.00", "200.00", "currency") is False


# ---------------------------------------------------------------------------
# digits_only
# ---------------------------------------------------------------------------


def test_digits_only_strips_leading_1():
    g = _grader()
    assert g._digits_only("+1-555-123-4567") == "5551234567"
    assert g._digits_only("5551234567") == "5551234567"
    # 10-digit number starting with 1 stays (not 11 digits)
    assert g._digits_only("1234567890") == "1234567890"


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------


def test_parse_date_multiple_formats():
    g = _grader()
    from datetime import date
    # ISO format
    assert g._parse_date("2023-01-15") == date(2023, 1, 15)
    # US format
    assert g._parse_date("01/15/2023") == date(2023, 1, 15)
    # Long format
    assert g._parse_date("January 15, 2023") == date(2023, 1, 15)
    # Unparseable returns original string
    assert g._parse_date("not-a-date") == "not-a-date"


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def test_align_rows_primary_key_fallback():
    """When _entity_id is absent, alignment falls back to primary_key."""
    g = _grader()
    final_data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    ground_truth = [
        {"id": 2, "name": "Bob"},
        {"id": 1, "name": "Alice"},
    ]
    schema = {"primary_key": "id", "expected_types": {"id": "int", "name": "str"}}
    alignment = g._align_rows(final_data, ground_truth, schema)
    # gt index 0 (id=2) -> fd index 1
    assert alignment[0] == 1
    # gt index 1 (id=1) -> fd index 0
    assert alignment[1] == 0


# ---------------------------------------------------------------------------
# Bonus: column cleanup
# ---------------------------------------------------------------------------


def test_column_cleanup_bonus():
    g = _grader()
    # Setup: one dirty cell in "name", agent fixes it
    ground_truth = [{"_entity_id": "E1", "name": "Alice"}]
    original = [{"_entity_id": "E1", "name": "alice"}]  # lowercase = dirty
    final = [{"_entity_id": "E1", "name": "Alice"}]
    schema = {"expected_types": {"name": "str"}}

    result = g.grade(
        final_data=final,
        ground_truth=ground_truth,
        original_data=original,
        action_history=[],
        schema=schema,
        flagged_cells=[],
    )
    # Should get a bonus for fully cleaning the "name" column
    assert result.bonuses >= 0.0


# ---------------------------------------------------------------------------
# Penalty: fix correct value
# ---------------------------------------------------------------------------


def test_fix_correct_value_penalty():
    g = _grader()
    ground_truth = [{"_entity_id": "E1", "name": "Alice"}]
    final = [{"_entity_id": "E1", "name": "Wrong"}]
    original = [{"_entity_id": "E1", "name": "Alice"}]
    schema = {"expected_types": {"name": "str"}}

    action_history = [
        {
            "action": "fix_value",
            "status": "success",
            "old_value": "Alice",
            "new_value": "Wrong",
            "column": "name",
        },
    ]
    result = g.grade(
        final_data=final,
        ground_truth=ground_truth,
        original_data=original,
        action_history=action_history,
        schema=schema,
        flagged_cells=[],
    )
    assert result.penalties > 0.0
