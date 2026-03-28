"""Tests for the task registry and task definitions."""

from __future__ import annotations

import pytest

from dataclean_env.server.tasks import get_task, list_tasks, _TASK_REGISTRY


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_all_tasks_registered():
    assert len(_TASK_REGISTRY) == 3


def test_easy_task_loads():
    task = get_task("easy_contacts")
    assert task.task_id == "easy_contacts"
    assert task.name == "Customer Contact Cleanup"
    assert task.difficulty == "easy"
    assert len(task.ground_truth) > 0
    assert len(task.corruptions) > 0
    assert task.max_steps > 0


def test_medium_task_loads():
    task = get_task("medium_employees")
    assert task.task_id == "medium_employees"
    assert task.difficulty == "medium"
    assert len(task.ground_truth) > 0


def test_hard_task_loads():
    task = get_task("hard_patients")
    assert task.task_id == "hard_patients"
    assert task.difficulty == "hard"
    assert len(task.ground_truth) > 0


# ---------------------------------------------------------------------------
# Corruption characteristics
# ---------------------------------------------------------------------------


def test_easy_no_duplicates():
    task = get_task("easy_contacts")
    dup_types = {"duplicate_with_noise", "duplicate_cluster"}
    for c in task.corruptions:
        assert c["type"] not in dup_types, f"Easy task should have no duplicates, found {c['type']}"


def test_hard_has_cross_field():
    task = get_task("hard_patients")
    assert "cross_field_rules" in task.schema, "Hard schema should have cross_field_rules"


# ---------------------------------------------------------------------------
# Entity ID presence
# ---------------------------------------------------------------------------


def test_all_rows_have_entity_id():
    for task_id in ["easy_contacts", "medium_employees", "hard_patients"]:
        task = get_task(task_id)
        for i, row in enumerate(task.ground_truth):
            assert "_entity_id" in row, f"Row {i} in {task_id} missing _entity_id"
            assert row["_entity_id"], f"Row {i} in {task_id} has empty _entity_id"


# ---------------------------------------------------------------------------
# False positives in hard task
# ---------------------------------------------------------------------------


def test_false_positives_in_ground_truth():
    task = get_task("hard_patients")
    eids = {row["_entity_id"] for row in task.ground_truth}
    # PAT007 and PAT008 are separate entities (two different Michael Davis)
    assert "PAT007" in eids
    assert "PAT008" in eids
    assert "PAT007" != "PAT008"
    # PAT009 and PAT010 are separate entities (two different Maria Garcia)
    assert "PAT009" in eids
    assert "PAT010" in eids
    assert "PAT009" != "PAT010"


# ---------------------------------------------------------------------------
# list_tasks metadata
# ---------------------------------------------------------------------------


def test_list_tasks_returns_metadata():
    tasks = list_tasks()
    assert isinstance(tasks, list)
    assert len(tasks) == 3
    for t in tasks:
        assert "task_id" in t
        assert "name" in t
        assert "difficulty" in t
        assert isinstance(t["task_id"], str)
        assert isinstance(t["name"], str)
        assert isinstance(t["difficulty"], str)
