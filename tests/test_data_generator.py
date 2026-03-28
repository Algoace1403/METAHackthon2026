"""Tests for the deterministic data corruption pipeline."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest

from dataclean_env.server.data_generator import generate_dirty_data, DataCorruptor


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def _sample_clean() -> List[Dict[str, Any]]:
    return [
        {"_entity_id": "A", "name": "Alice", "email": "alice@x.com", "phone": "(555) 111-2222"},
        {"_entity_id": "B", "name": "Bob", "email": "bob@x.com", "phone": "(555) 333-4444"},
        {"_entity_id": "C", "name": "Carol", "email": "carol@x.com", "phone": "(555) 555-6666"},
    ]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_same_output():
    clean = _sample_clean()
    corruptions = [
        {"type": "char_swap", "columns": ["name"], "probability": 1.0, "mode": "swap"},
    ]
    out1 = generate_dirty_data(clean, corruptions, seed=42)
    out2 = generate_dirty_data(clean, corruptions, seed=42)
    assert out1 == out2


def test_different_seed_different_output():
    clean = _sample_clean()
    corruptions = [
        {"type": "char_swap", "columns": ["name", "email"], "probability": 1.0, "mode": "swap"},
    ]
    out1 = generate_dirty_data(clean, corruptions, seed=42)
    out2 = generate_dirty_data(clean, corruptions, seed=99)
    assert out1 != out2


# ---------------------------------------------------------------------------
# Corruption handlers
# ---------------------------------------------------------------------------


def test_char_swap_corrupts():
    clean = _sample_clean()
    corruptions = [
        {"type": "char_swap", "targets": [{"row_idx": 0, "field": "name"}], "mode": "swap"},
    ]
    dirty = generate_dirty_data(clean, corruptions, seed=42)
    # The name should be different from the original
    assert dirty[0]["name"] != "Alice"


def test_null_inject_creates_null():
    clean = _sample_clean()
    corruptions = [
        {"type": "null_inject", "targets": [{"row_idx": 1, "field": "email"}]},
    ]
    dirty = generate_dirty_data(clean, corruptions, seed=42)
    assert dirty[1]["email"] is None


def test_duplicate_preserves_entity_id():
    clean = _sample_clean()
    corruptions = [
        {"type": "duplicate_with_noise", "source_indices": [0], "noise_fields": ["name"]},
    ]
    dirty = generate_dirty_data(clean, corruptions, seed=42)
    # Should have at least one extra row
    assert len(dirty) > len(clean)
    # The duplicated row should share _entity_id with the source
    dup_row = dirty[-1]  # appended at end
    assert dup_row["_entity_id"] == clean[0]["_entity_id"]


# ---------------------------------------------------------------------------
# Original not mutated
# ---------------------------------------------------------------------------


def test_original_not_mutated():
    clean = _sample_clean()
    original_snapshot = copy.deepcopy(clean)
    corruptions = [
        {"type": "char_swap", "targets": [{"row_idx": 0, "field": "name"}], "mode": "swap"},
        {"type": "null_inject", "targets": [{"row_idx": 2, "field": "phone"}]},
    ]
    _ = generate_dirty_data(clean, corruptions, seed=42)
    assert clean == original_snapshot
