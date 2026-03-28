"""Deterministic grader for the DataClean-Env environment.

Compares the agent's final cleaned dataset against ground truth using:
- Entity-ID based row alignment (primary) with similarity fallback
- Type-aware cell matching (case-insensitive strings, date parsing, phone digits)
- Weighted scoring: accuracy 40%, completeness 20%, format 10%, row count 10%,
  efficiency 10%, utility 10%
- Downstream utility probes: verify aggregate analytics match expected results
- Penalties for destructive actions, bonuses for full column cleanup
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple


# Date formats for flexible parsing
_DATE_FORMATS = [
    "%Y-%m-%d",     # 2023-01-15 (unambiguous)
    "%Y/%m/%d",     # 2023/01/15 (unambiguous)
    "%B %d, %Y",    # January 15, 2023 (unambiguous)
    "%b %d, %Y",    # Jan 15, 2023 (unambiguous)
    "%d %B %Y",     # 15 January 2023 (unambiguous)
    "%B %d %Y",     # January 15 2023 (unambiguous)
    "%d-%b-%Y",     # 15-Jan-2023 (unambiguous)
    "%m/%d/%Y",     # 01/15/2023 (US convention, before d/m/Y)
    "%d/%m/%Y",     # 15/01/2023 (EU convention, after m/d/Y)
    "%m-%d-%Y",     # 01-15-2023 (last resort, ambiguous with d-m-Y)
]


@dataclass
class GradeResult:
    """Result of grading the agent's cleaned dataset."""

    score: float  # 0.0-1.0 final composite score
    accuracy: float = 0.0
    completeness: float = 0.0
    format_consistency: float = 0.0
    row_correctness: float = 0.0
    efficiency: float = 0.0
    utility_score: float = 0.0
    penalties: float = 0.0
    bonuses: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)
    utility_details: List[Dict[str, Any]] = field(default_factory=list)


class DataCleanGrader:
    """Deterministic grader using entity-ID alignment and type-aware matching."""

    WEIGHTS = {
        "accuracy": 0.40,
        "completeness": 0.20,
        "format_consistency": 0.10,
        "row_correctness": 0.10,
        "efficiency": 0.10,
        "utility": 0.10,
    }

    def grade(
        self,
        final_data: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        original_data: List[Dict[str, Any]],
        action_history: List[Dict[str, Any]],
        schema: Dict[str, Any],
        flagged_cells: List[Dict[str, str]],
        budget_spent: float = 0.0,
        action_budget: float = 100.0,
        escalated_cells: Optional[List[Dict[str, Any]]] = None,
        ambiguous_cells: Optional[List[Tuple[str, str]]] = None,
        utility_probes: Optional[List[Any]] = None,
    ) -> GradeResult:
        """Grade the agent's cleaned dataset against ground truth.

        Returns a GradeResult with composite score in [0.0, 1.0].

        Args:
            budget_spent: Total action cost spent during the episode.
            action_budget: Total budget allocated for the episode.
        """
        if not ground_truth:
            return GradeResult(score=1.0)

        # Step 1: Align rows using _entity_id (primary) or similarity (fallback)
        alignment = self._align_rows(final_data, ground_truth, schema)

        # Step 2: Identify which cells were dirty in the original
        dirty_cells = self._identify_dirty_cells(original_data, ground_truth, schema)

        # Step 3: Compute scoring components
        types = schema.get("expected_types", {})
        accuracy = self._compute_accuracy(final_data, ground_truth, alignment, dirty_cells, types)
        completeness = self._compute_completeness(final_data, ground_truth, alignment, types)
        format_score = self._compute_format_score(final_data, schema)
        row_score = self._compute_row_score(len(final_data), len(ground_truth))

        # Efficiency bonus: reward achieving same quality with less cost
        if action_budget > 0:
            efficiency = max(0.0, 1.0 - (budget_spent / action_budget))
        else:
            efficiency = 1.0

        # Step 3b: Downstream utility probes
        utility_score, utility_details = self._compute_utility_score(
            final_data, utility_probes or [],
        )

        # Step 4: Penalties and bonuses
        penalties = self._compute_penalties(
            action_history, ground_truth, schema,
            ambiguous_cells=ambiguous_cells or [],
            final_data=final_data,
            alignment=alignment,
            types=types,
        )
        bonuses = self._compute_bonuses(
            final_data, ground_truth, alignment, dirty_cells, flagged_cells, types,
            escalated_cells=escalated_cells or [],
            ambiguous_cells=ambiguous_cells or [],
        )

        # Step 5: Weighted composite
        base_score = (
            self.WEIGHTS["accuracy"] * accuracy
            + self.WEIGHTS["completeness"] * completeness
            + self.WEIGHTS["format_consistency"] * format_score
            + self.WEIGHTS["row_correctness"] * row_score
            + self.WEIGHTS["efficiency"] * efficiency
            + self.WEIGHTS["utility"] * utility_score
        )
        final_score = max(0.0, min(1.0, base_score - penalties + bonuses))

        return GradeResult(
            score=round(final_score, 4),
            accuracy=round(accuracy, 4),
            completeness=round(completeness, 4),
            format_consistency=round(format_score, 4),
            row_correctness=round(row_score, 4),
            efficiency=round(efficiency, 4),
            utility_score=round(utility_score, 4),
            penalties=round(penalties, 4),
            bonuses=round(bonuses, 4),
            utility_details=utility_details,
        )

    # ------------------------------------------------------------------
    # Row Alignment (entity_id primary, similarity fallback)
    # ------------------------------------------------------------------

    def _align_rows(
        self,
        final_data: List[Dict],
        ground_truth: List[Dict],
        schema: Dict,
    ) -> Dict[int, int]:
        """Align ground_truth rows to final_data rows.

        Returns mapping: {ground_truth_index: final_data_index}.
        Uses _entity_id for alignment when available, otherwise similarity.
        """
        # Strategy 1: Entity ID matching (hidden field from data generator)
        gt_has_eid = all("_entity_id" in row for row in ground_truth)
        fd_has_eid = all("_entity_id" in row for row in final_data)

        if gt_has_eid and fd_has_eid:
            alignment: Dict[int, int] = {}
            fd_by_eid: Dict[str, List[int]] = {}
            for i, row in enumerate(final_data):
                eid = row.get("_entity_id", "")
                fd_by_eid.setdefault(eid, []).append(i)

            used_fd: Set[int] = set()
            for gt_i, gt_row in enumerate(ground_truth):
                gt_eid = gt_row.get("_entity_id", "")
                candidates = fd_by_eid.get(gt_eid, [])
                for fd_i in candidates:
                    if fd_i not in used_fd:
                        alignment[gt_i] = fd_i
                        used_fd.add(fd_i)
                        break
            return alignment

        # Strategy 2: Primary key matching
        pk = schema.get("primary_key")
        if pk:
            alignment = {}
            fd_by_pk: Dict[Any, int] = {}
            for i, row in enumerate(final_data):
                pk_val = row.get(pk)
                if pk_val is not None:
                    fd_by_pk[pk_val] = i
            for gt_i, gt_row in enumerate(ground_truth):
                gt_pk = gt_row.get(pk)
                if gt_pk in fd_by_pk:
                    alignment[gt_i] = fd_by_pk[gt_pk]
            return alignment

        # Strategy 3: Greedy similarity matching
        return self._align_by_similarity(final_data, ground_truth, schema)

    def _align_by_similarity(
        self,
        final_data: List[Dict],
        ground_truth: List[Dict],
        schema: Dict,
    ) -> Dict[int, int]:
        """Greedy best-match alignment using row similarity."""
        types = schema.get("expected_types", {})
        used_fd: Set[int] = set()
        alignment: Dict[int, int] = {}

        for gt_i, gt_row in enumerate(ground_truth):
            best_score = -1.0
            best_fd = -1
            for fd_i, fd_row in enumerate(final_data):
                if fd_i in used_fd:
                    continue
                sim = self._row_similarity(gt_row, fd_row, types)
                if sim > best_score:
                    best_score = sim
                    best_fd = fd_i
            if best_score > 0.3 and best_fd >= 0:
                alignment[gt_i] = best_fd
                used_fd.add(best_fd)
        return alignment

    def _row_similarity(
        self, row_a: Dict, row_b: Dict, types: Dict[str, str],
    ) -> float:
        """Compute fraction of matching cells between two rows."""
        cols = [c for c in set(list(row_a.keys()) + list(row_b.keys()))
                if not c.startswith("_")]
        if not cols:
            return 0.0
        matches = sum(
            1 for c in cols
            if self._cell_match(row_a.get(c), row_b.get(c), types.get(c, "str"))
        )
        return matches / len(cols)

    # ------------------------------------------------------------------
    # Cell Matching (type-aware)
    # ------------------------------------------------------------------

    def _cell_match(self, val_a: Any, val_b: Any, col_type: str) -> bool:
        """Type-aware comparison. Returns True if semantically equal."""
        if val_a is None and val_b is None:
            return True
        if val_a is None or val_b is None:
            return False

        a_str = str(val_a).strip()
        b_str = str(val_b).strip()

        if col_type == "name":
            # Names are case-insensitive (John == john)
            return a_str.lower() == b_str.lower()
        elif col_type == "str":
            # Generic strings are CASE-SENSITIVE (so case corruptions are detected)
            return a_str == b_str
        elif col_type in ("int", "float", "currency"):
            try:
                a_num = float(a_str.replace(",", "").replace("$", ""))
                b_num = float(b_str.replace(",", "").replace("$", ""))
                return abs(a_num - b_num) < 0.01
            except (ValueError, TypeError):
                return a_str.lower() == b_str.lower()
        elif col_type == "date":
            return self._parse_date(a_str) == self._parse_date(b_str)
        elif col_type in ("phone", "tel"):
            return self._digits_only(a_str) == self._digits_only(b_str)
        elif col_type == "email":
            return a_str.lower() == b_str.lower()
        else:
            return a_str.lower() == b_str.lower()

    @staticmethod
    def _digits_only(s: str) -> str:
        d = "".join(c for c in s if c.isdigit())
        if d.startswith("1") and len(d) == 11:
            d = d[1:]
        return d

    @staticmethod
    def _parse_date(s: str) -> Any:
        """Try multiple date formats, return date object or original string."""
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(s.strip(), fmt).date()
            except ValueError:
                continue
        return s

    # ------------------------------------------------------------------
    # Scoring Components
    # ------------------------------------------------------------------

    def _identify_dirty_cells(
        self,
        original: List[Dict],
        ground_truth: List[Dict],
        schema: Dict,
    ) -> Set[Tuple[int, str]]:
        """Find cells that differ between original dirty data and ground truth."""
        dirty: Set[Tuple[int, str]] = set()
        types = schema.get("expected_types", {})

        # Align original to ground truth
        alignment = self._align_rows(original, ground_truth, schema)

        # Invert: for each gt row, find the original row
        gt_to_orig: Dict[int, int] = {}
        for orig_i, gt_candidates in self._invert_alignment(alignment, ground_truth).items():
            for gt_i in gt_candidates:
                gt_to_orig[gt_i] = orig_i

        for gt_i, gt_row in enumerate(ground_truth):
            if gt_i not in gt_to_orig:
                # This ground truth row has no original (e.g., it was split from a merge)
                continue
            orig_i = gt_to_orig[gt_i]
            if orig_i >= len(original):
                continue
            orig_row = original[orig_i]
            for col in gt_row:
                if col.startswith("_"):
                    continue
                col_type = types.get(col, "str")
                if not self._cell_match(orig_row.get(col), gt_row.get(col), col_type):
                    dirty.add((gt_i, col))

        return dirty

    @staticmethod
    def _invert_alignment(
        alignment: Dict[int, int], ground_truth: List[Dict],
    ) -> Dict[int, List[int]]:
        """Invert alignment from {gt->fd} to {fd->[gt]}."""
        inverted: Dict[int, List[int]] = {}
        for gt_i, fd_i in alignment.items():
            inverted.setdefault(fd_i, []).append(gt_i)
        return inverted

    def _compute_accuracy(
        self,
        final_data: List[Dict],
        ground_truth: List[Dict],
        alignment: Dict[int, int],
        dirty_cells: Set[Tuple[int, str]],
        types: Dict[str, str],
    ) -> float:
        """What fraction of dirty cells were fixed correctly?"""
        if not dirty_cells:
            return 1.0
        fixed = 0
        for gt_i, col in dirty_cells:
            if gt_i not in alignment:
                continue
            fd_i = alignment[gt_i]
            if fd_i >= len(final_data):
                continue
            col_type = types.get(col, "str")
            if self._cell_match(
                final_data[fd_i].get(col), ground_truth[gt_i].get(col), col_type,
            ):
                fixed += 1
        return fixed / len(dirty_cells)

    def _compute_completeness(
        self,
        final_data: List[Dict],
        ground_truth: List[Dict],
        alignment: Dict[int, int],
        types: Dict[str, str],
    ) -> float:
        """What fraction of expected non-null cells are correct?"""
        expected = 0
        correct = 0
        for gt_i, gt_row in enumerate(ground_truth):
            for col, val in gt_row.items():
                if col.startswith("_"):
                    continue
                if val is None:
                    continue
                expected += 1
                if gt_i in alignment:
                    fd_i = alignment[gt_i]
                    if fd_i < len(final_data):
                        fd_val = final_data[fd_i].get(col)
                        col_type = types.get(col, "str")
                        if fd_val is not None and self._cell_match(fd_val, val, col_type):
                            correct += 1
        return correct / expected if expected > 0 else 1.0

    def _compute_format_score(
        self, final_data: List[Dict], schema: Dict,
    ) -> float:
        """What fraction of format-constrained cells are correctly formatted?"""
        constraints = schema.get("constraints", {})
        total = 0
        correct = 0
        for row in final_data:
            for col, val in row.items():
                if col.startswith("_") or val is None:
                    continue
                col_constraints = constraints.get(col, {})
                fmt = col_constraints.get("format")
                if fmt:
                    total += 1
                    if self._matches_format(val, fmt):
                        correct += 1
        return correct / total if total > 0 else 1.0

    def _compute_row_score(self, actual_rows: int, expected_rows: int) -> float:
        """Score based on having the correct number of rows."""
        if expected_rows == 0:
            return 1.0 if actual_rows == 0 else 0.0
        return 1.0 - min(abs(expected_rows - actual_rows) / expected_rows, 1.0)

    # ------------------------------------------------------------------
    # Penalties
    # ------------------------------------------------------------------

    def _compute_penalties(
        self,
        action_history: List[Dict],
        ground_truth: List[Dict],
        schema: Dict,
        ambiguous_cells: Optional[List[Tuple[str, str]]] = None,
        final_data: Optional[List[Dict]] = None,
        alignment: Optional[Dict[int, int]] = None,
        types: Optional[Dict[str, str]] = None,
    ) -> float:
        """Compute penalties for destructive or incorrect actions."""
        penalty = 0.0
        schema_types = types or schema.get("expected_types", {})
        ambiguous_set: Set[Tuple[str, str]] = set(ambiguous_cells or [])

        for action in action_history:
            status = action.get("status")
            if status != "success":
                continue

            action_type = action.get("action", "")

            # Penalty: deleted a row that exists in ground truth
            if action_type == "delete_row":
                deleted = action.get("deleted_data", {})
                eid = deleted.get("_entity_id")
                if eid:
                    gt_eids = {r.get("_entity_id") for r in ground_truth}
                    if eid in gt_eids:
                        penalty += 0.10
                else:
                    pk = schema.get("primary_key")
                    if pk:
                        pk_val = deleted.get(pk)
                        gt_pks = {r.get(pk) for r in ground_truth}
                        if pk_val in gt_pks:
                            penalty += 0.10

            # Penalty: changed a correct value to an incorrect one
            if action_type in ("fix_value", "fill_missing"):
                old_val = action.get("old_value")
                new_val = action.get("new_value")
                col = action.get("column")
                if col and old_val is not None:
                    col_type = schema_types.get(col, "str")
                    for gt_row in ground_truth:
                        if self._cell_match(old_val, gt_row.get(col), col_type):
                            if not self._cell_match(new_val, gt_row.get(col), col_type):
                                # Higher penalty for wrong fix on ambiguous cell
                                eid = gt_row.get("_entity_id", "")
                                if (eid, col) in ambiguous_set:
                                    penalty += 0.08
                                else:
                                    penalty += 0.05
                                break

            # Penalty: merged two rows that are distinct entities
            if action_type == "merge_duplicates":
                eid1 = action.get("entity_id1", "")
                eid2 = action.get("entity_id2", "")
                if eid1 and eid2 and eid1 != eid2:
                    # Different entity IDs = merged two distinct people
                    penalty += 0.10

        return min(penalty, 0.50)  # Cap total penalties

    # ------------------------------------------------------------------
    # Bonuses
    # ------------------------------------------------------------------

    def _compute_bonuses(
        self,
        final_data: List[Dict],
        ground_truth: List[Dict],
        alignment: Dict[int, int],
        dirty_cells: Set[Tuple[int, str]],
        flagged_cells: List[Dict[str, str]],
        types: Dict[str, str],
        escalated_cells: Optional[List[Dict[str, Any]]] = None,
        ambiguous_cells: Optional[List[Tuple[str, str]]] = None,
    ) -> float:
        """Compute bonuses for thorough cleaning."""
        bonus = 0.0

        # Bonus: +0.10 for fully cleaning all issues in a column
        cols_with_issues: Dict[str, List[int]] = {}
        for gt_i, col in dirty_cells:
            cols_with_issues.setdefault(col, []).append(gt_i)

        for col, gt_indices in cols_with_issues.items():
            col_type = types.get(col, "str")
            all_fixed = True
            for gt_i in gt_indices:
                if gt_i not in alignment:
                    all_fixed = False
                    break
                fd_i = alignment[gt_i]
                if fd_i >= len(final_data):
                    all_fixed = False
                    break
                if not self._cell_match(
                    final_data[fd_i].get(col), ground_truth[gt_i].get(col), col_type,
                ):
                    all_fixed = False
                    break
            if all_fixed and gt_indices:
                bonus += 0.10

        # Bonus: +0.02 for correctly flagging a dirty cell (exact row+column match)
        dirty_cell_set = {(gt_i, col) for gt_i, col in dirty_cells}
        for flag in flagged_cells:
            flag_col = flag.get("column")
            # Check if any dirty cell in that column matches
            for gt_i, col in dirty_cell_set:
                if col == flag_col and gt_i in alignment:
                    # Verify the flag's row_id maps to this gt row
                    fd_i = alignment[gt_i]
                    if fd_i < len(final_data):
                        flagged_rid = flag.get("row_id", flag.get("row"))
                        actual_rid = final_data[fd_i].get("_row_id")
                        if flagged_rid == actual_rid:
                            bonus += 0.02
                            break

        # Calibrated abstention: escalated_cells scoring
        ambiguous_set: Set[Tuple[str, str]] = set(ambiguous_cells or [])
        for esc in (escalated_cells or []):
            esc_eid = self._resolve_entity_id_for_row_id(
                esc.get("row_id"), final_data,
            )
            esc_col = esc.get("column", "")
            if (esc_eid, esc_col) in ambiguous_set:
                # Correct escalation on genuinely ambiguous cell
                bonus += 0.03
            else:
                # Escalation on a clearly fixable cell wastes human time
                bonus -= 0.02

        return min(bonus, 0.30)  # Cap bonuses

    @staticmethod
    def _resolve_entity_id_for_row_id(
        row_id: Any, data: List[Dict],
    ) -> str:
        """Map a runtime _row_id back to the stable _entity_id."""
        if row_id is None:
            return ""
        for row in data:
            if row.get("_row_id") == row_id:
                return str(row.get("_entity_id", ""))
        return ""

    # ------------------------------------------------------------------
    # Downstream Utility Probes
    # ------------------------------------------------------------------

    def _compute_utility_score(
        self,
        final_data: List[Dict[str, Any]],
        utility_probes: List[Any],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Run downstream utility probes and score correctness.

        Returns (score, details) where score is the fraction of probes passed
        and details is a list of per-probe result dicts.
        """
        if not utility_probes:
            return 1.0, []

        details: List[Dict[str, Any]] = []
        passed = 0
        for probe in utility_probes:
            actual = self._run_probe(final_data, probe)
            match = self._probe_matches(actual, probe.expected_result)
            details.append({
                "probe": probe.name,
                "description": probe.description,
                "expected": probe.expected_result,
                "actual": actual,
                "passed": match,
            })
            if match:
                passed += 1
        return passed / len(utility_probes), details

    def _run_probe(
        self, data: List[Dict[str, Any]], probe: Any,
    ) -> Any:
        """Execute a single utility probe against the dataset."""
        fn_name = probe.query_fn
        params = probe.params

        if fn_name == "unique_count":
            return self._probe_unique_count(data, params["column"])
        elif fn_name == "distribution":
            return self._probe_distribution(data, params["column"])
        elif fn_name == "avg_by_group":
            transform = params.get("transform")
            return self._probe_avg_by_group(
                data, params["value_col"], params["group_col"], transform,
            )
        elif fn_name == "count_where":
            return self._probe_count_where(
                data, params["column"], params["value"],
            )
        return None

    @staticmethod
    def _probe_unique_count(data: List[Dict], column: str) -> int:
        """Count unique non-null values in a column."""
        values = set()
        for row in data:
            val = row.get(column)
            if val is not None:
                values.add(val)
        return len(values)

    @staticmethod
    def _probe_distribution(data: List[Dict], column: str) -> Dict[str, int]:
        """Count occurrences per distinct value in a column."""
        counts: Dict[str, int] = {}
        for row in data:
            val = row.get(column)
            if val is not None:
                key = str(val).strip()
                counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _probe_avg_by_group(
        data: List[Dict],
        value_col: str,
        group_col: str,
        transform: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute average of value_col grouped by group_col.

        If transform == 'year_age_2024', interpret value_col as a date string
        and compute age as (2024 - birth_year).
        """
        groups: Dict[str, List[float]] = {}
        for row in data:
            group_val = row.get(group_col)
            raw_val = row.get(value_col)
            if group_val is None or raw_val is None:
                continue

            group_key = str(group_val).strip()

            if transform == "year_age_2024":
                try:
                    if isinstance(raw_val, str):
                        year = int(raw_val.strip()[:4])
                        numeric_val = float(2024 - year)
                    else:
                        continue
                except (ValueError, IndexError):
                    continue
            else:
                try:
                    numeric_val = float(
                        str(raw_val).replace(",", "").replace("$", "")
                    )
                except (ValueError, TypeError):
                    continue

            groups.setdefault(group_key, []).append(numeric_val)

        return {
            k: round(sum(v) / len(v), 2)
            for k, v in sorted(groups.items())
            if v
        }

    @staticmethod
    def _probe_count_where(
        data: List[Dict], column: str, value: Any,
    ) -> int:
        """Count rows where column equals value (case-sensitive string match)."""
        count = 0
        for row in data:
            row_val = row.get(column)
            if row_val is not None and str(row_val).strip() == str(value):
                count += 1
        return count

    @staticmethod
    def _probe_matches(actual: Any, expected: Any) -> bool:
        """Check if a probe's actual result matches the expected result.

        Supports int, float, str, and dict comparisons.
        For dicts, all keys and values must match (numeric values use tolerance).
        """
        if actual is None:
            return False

        if isinstance(expected, dict) and isinstance(actual, dict):
            if set(expected.keys()) != set(actual.keys()):
                return False
            for key in expected:
                exp_v = expected[key]
                act_v = actual.get(key)
                if act_v is None:
                    return False
                try:
                    if abs(float(exp_v) - float(act_v)) > 0.5:
                        return False
                except (ValueError, TypeError):
                    if str(exp_v) != str(act_v):
                        return False
            return True

        if isinstance(expected, (int, float)):
            try:
                return abs(float(actual) - float(expected)) < 0.5
            except (ValueError, TypeError):
                return False

        return str(actual) == str(expected)

    # ------------------------------------------------------------------
    # Format Matching
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_format(value: Any, format_spec: str) -> bool:
        """Check if a value matches the expected format.

        Supports named keys ('YYYY-MM-DD') and raw regex patterns.
        """
        s = str(value)
        named_patterns: Dict[str, str] = {
            "YYYY-MM-DD": r"^\d{4}-\d{2}-\d{2}$",
            "(XXX) XXX-XXXX": r"^\(\d{3}\) \d{3}-\d{4}$",
            "email": r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
            "5_digit": r"^\d{5}$",
            "+1XXXXXXXXXX": r"^\+1\d{10}$",
        }
        # Try named key first
        pattern = named_patterns.get(format_spec)
        if pattern:
            return bool(re.match(pattern, s))
        # Fallback: treat format_spec as a raw regex
        try:
            return bool(re.match(format_spec, s))
        except re.error:
            return True
