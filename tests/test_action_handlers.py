"""Tests for action handlers and edge cases in DataCleanEnvironment."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from dataclean_env.server.environment import DataCleanEnvironment
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
# split_column
# ---------------------------------------------------------------------------


def test_split_column_success():
    env, obs = _reset_env("easy_contacts")
    # Combine first_name and last_name into a "full_name" column first, then split.
    # Instead, let's directly test split on a column that exists with a delimiter.
    # We'll manufacture a scenario: set a cell to contain a delimiter-value then split.
    first_rid = obs.rows[0][0]
    col = "first_name"
    # Put a value with a space so we can split on space
    env.step(_make_action("fix_value", row_id=first_rid, column=col, new_value="Alice Marie"))
    obs2 = env.step(_make_action(
        "split_column",
        column="first_name",
        delimiter=" ",
        new_names=["first", "middle"],
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "success"
    assert "first_name" not in obs2.columns
    assert "first" in obs2.columns


def test_split_column_no_column_no_effect():
    env, obs = _reset_env("easy_contacts")
    obs2 = env.step(_make_action(
        "split_column",
        column="nonexistent_col",
        delimiter=",",
        new_names=["a", "b"],
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "no_effect"


# ---------------------------------------------------------------------------
# rename_column
# ---------------------------------------------------------------------------


def test_rename_column_success():
    env, obs = _reset_env("easy_contacts")
    obs2 = env.step(_make_action(
        "rename_column", old_name="first_name", new_name="given_name",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "success"
    assert "given_name" in obs2.columns


def test_rename_column_not_found_error():
    env, obs = _reset_env("easy_contacts")
    obs2 = env.step(_make_action(
        "rename_column", old_name="nonexistent", new_name="whatever",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "error"
    assert "not found" in obs2.last_action_result.message


def test_rename_column_target_exists_error():
    env, obs = _reset_env("easy_contacts")
    obs2 = env.step(_make_action(
        "rename_column", old_name="first_name", new_name="last_name",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "error"
    assert "already exists" in obs2.last_action_result.message


# ---------------------------------------------------------------------------
# cast_type
# ---------------------------------------------------------------------------


def test_cast_type_to_str_success():
    env, obs = _reset_env("easy_contacts")
    obs2 = env.step(_make_action("cast_type", column="state", target_type="str"))
    assert obs2.last_action_result is not None
    # All values are already strings, but the cast still processes them
    assert obs2.last_action_result.status in ("success", "no_effect")


def test_cast_type_to_int_success():
    env, obs = _reset_env("medium_employees")
    obs2 = env.step(_make_action("cast_type", column="salary", target_type="int"))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "success"


def test_cast_type_all_fail_returns_error():
    """When every cast fails, status should be 'error' not 'success'."""
    env, obs = _reset_env("easy_contacts")
    # Set all values in 'first_name' column -- they are strings that can't be cast to int
    # But the cast actually does int(float(val)) which would fail on "Alice" etc.
    # First, let's make sure column has only non-numeric values
    obs2 = env.step(_make_action("cast_type", column="first_name", target_type="int"))
    assert obs2.last_action_result is not None
    # All casts fail => nullified > 0, modified == 0 => status == "error"
    assert obs2.last_action_result.status == "error"


# ---------------------------------------------------------------------------
# fill_missing
# ---------------------------------------------------------------------------


def test_fill_missing_on_non_null_cell_error():
    env, obs = _reset_env("easy_contacts")
    first_rid = obs.rows[0][0]
    obs2 = env.step(_make_action(
        "fill_missing", row_id=first_rid, column="last_name", value="Test",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "error"
    assert "not empty" in obs2.last_action_result.message


def test_fill_missing_column_not_found_error():
    env, obs = _reset_env("easy_contacts")
    first_rid = obs.rows[0][0]
    obs2 = env.step(_make_action(
        "fill_missing", row_id=first_rid, column="nonexistent_col", value="X",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "error"


# ---------------------------------------------------------------------------
# fix_value
# ---------------------------------------------------------------------------


def test_fix_value_column_not_found_error():
    env, obs = _reset_env("easy_contacts")
    first_rid = obs.rows[0][0]
    obs2 = env.step(_make_action(
        "fix_value", row_id=first_rid, column="zzz_missing", new_value="X",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "error"
    assert "not found" in obs2.last_action_result.message


# ---------------------------------------------------------------------------
# merge_duplicates
# ---------------------------------------------------------------------------


def test_merge_same_row_id_error():
    env, obs = _reset_env("medium_employees")
    rid = obs.rows[0][0]
    obs2 = env.step(_make_action(
        "merge_duplicates", row_id1=rid, row_id2=rid, strategy="keep_first",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "error"
    assert "itself" in obs2.last_action_result.message


def test_merge_keep_first_strategy():
    env, obs = _reset_env("medium_employees")
    rid1 = obs.rows[0][0]
    rid2 = obs.rows[1][0]
    obs2 = env.step(_make_action(
        "merge_duplicates", row_id1=rid1, row_id2=rid2, strategy="keep_first",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "success"
    # rid2 is gone
    remaining_ids = [row[0] for row in obs2.rows]
    assert rid1 in remaining_ids
    assert rid2 not in remaining_ids


def test_merge_keep_second_strategy():
    env, obs = _reset_env("medium_employees")
    rid1 = obs.rows[0][0]
    rid2 = obs.rows[1][0]
    obs2 = env.step(_make_action(
        "merge_duplicates", row_id1=rid1, row_id2=rid2, strategy="keep_second",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "success"
    # Merged row keeps rid1 (per environment logic), but data is from second
    remaining_ids = [row[0] for row in obs2.rows]
    assert rid1 in remaining_ids
    assert rid2 not in remaining_ids


def test_merge_prefer_row1_strategy():
    env, obs = _reset_env("medium_employees")
    rid1 = obs.rows[0][0]
    rid2 = obs.rows[1][0]
    obs2 = env.step(_make_action(
        "merge_duplicates", row_id1=rid1, row_id2=rid2, strategy="merge_prefer_row1",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "success"


def test_merge_prefer_row2_strategy():
    env, obs = _reset_env("medium_employees")
    rid1 = obs.rows[0][0]
    rid2 = obs.rows[1][0]
    obs2 = env.step(_make_action(
        "merge_duplicates", row_id1=rid1, row_id2=rid2, strategy="merge_prefer_row2",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "success"


# ---------------------------------------------------------------------------
# flag_anomaly
# ---------------------------------------------------------------------------


def test_flag_anomaly_invalid_row_error():
    env, obs = _reset_env("easy_contacts")
    obs2 = env.step(_make_action(
        "flag_anomaly", row_id=99999, column="email", reason="test",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "error"
    assert "not found" in obs2.last_action_result.message


# ---------------------------------------------------------------------------
# standardize_format (various format types)
# ---------------------------------------------------------------------------


def test_standardize_format_email_lowercase():
    env, obs = _reset_env("easy_contacts")
    # First, set an email to uppercase to ensure the format changes something
    first_rid = obs.rows[0][0]
    env.step(_make_action("fix_value", row_id=first_rid, column="email", new_value="ALICE@EXAMPLE.COM"))
    obs2 = env.step(_make_action(
        "standardize_format", column="email", format_type="email:lowercase",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "success"


def test_standardize_format_name_title_case():
    env, obs = _reset_env("easy_contacts")
    obs2 = env.step(_make_action(
        "standardize_format", column="first_name", format_type="name:title_case",
    ))
    assert obs2.last_action_result is not None
    # Some names may have been case-corrupted, so this should produce at least some changes
    assert obs2.last_action_result.status in ("success", "no_effect")


def test_standardize_format_zip_5digit():
    env, obs = _reset_env("medium_employees")
    obs2 = env.step(_make_action(
        "standardize_format", column="office_zip", format_type="zip:5digit",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status in ("success", "no_effect")


def test_standardize_format_currency_float():
    env, obs = _reset_env("medium_employees")
    # Set a salary to a string with $ and comma
    first_rid = obs.rows[0][0]
    env.step(_make_action("fix_value", row_id=first_rid, column="salary", new_value="$145,000.00"))
    obs2 = env.step(_make_action(
        "standardize_format", column="salary", format_type="currency:float",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "success"


def test_standardize_format_state_abbreviation():
    env, obs = _reset_env("easy_contacts")
    # Force a state to full name so abbreviation conversion works
    first_rid = obs.rows[0][0]
    env.step(_make_action("fix_value", row_id=first_rid, column="state", new_value="California"))
    obs2 = env.step(_make_action(
        "standardize_format", column="state", format_type="state:abbreviation",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status == "success"


def test_standardize_format_phone_e164():
    env, obs = _reset_env("easy_contacts")
    obs2 = env.step(_make_action(
        "standardize_format", column="phone", format_type="phone:E164",
    ))
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.status in ("success", "no_effect")
