"""DataClean Environment: core logic for the data cleaning RL environment.

Implements reset(), step(), state property following OpenEnv spec.
All 10 action handlers fully implemented. Delta reward system.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server import Environment

from dataclean_env.models import (
    ActionResult,
    DataCleanAction,
    DataCleanObservation,
    DataCleanState,
    DataSummary,
    IssueGroup,
    QualityIssue,
)
from dataclean_env.server.grader import DataCleanGrader
from dataclean_env.server.tasks import get_task, list_tasks


# US state name -> abbreviation mapping
US_STATES: Dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}

# Date parsing formats (most specific first)
DATE_PARSE_FORMATS = [
    "%Y-%m-%d",       # 2023-01-15
    "%m/%d/%Y",       # 01/15/2023
    "%d-%m-%Y",       # 15-01-2023
    "%B %d, %Y",      # January 15, 2023
    "%b %d, %Y",      # Jan 15, 2023
    "%d %B %Y",       # 15 January 2023
    "%d-%b-%Y",       # 15-Jan-2023
    "%m-%d-%Y",       # 01-15-2023
    "%B %d %Y",       # January 15 2023
    "%d/%m/%Y",       # 15/01/2023
    "%Y/%m/%d",       # 2023/01/15
]


# Per-action costs for the intervention budget system
ACTION_COSTS: Dict[str, float] = {
    "fix_value": 1.0,
    "delete_row": 6.0,
    "fill_missing": 1.0,
    "standardize_format": 2.0,
    "merge_duplicates": 4.0,
    "flag_anomaly": 0.5,
    "split_column": 3.0,
    "rename_column": 0.5,
    "cast_type": 2.0,
    "escalate_to_human": 0.5,
    "mark_complete": 0.0,
}

# Budget allocation per difficulty level
DIFFICULTY_BUDGETS: Dict[str, float] = {
    "easy": 50.0,
    "medium": 100.0,
    "hard": 200.0,
}


class DataCleanEnvironment(
    Environment[DataCleanAction, DataCleanObservation, DataCleanState]
):
    """Data Cleaning environment for training AI agents."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        super().__init__()
        self._state = DataCleanState()
        self._grader = DataCleanGrader()
        self._utility_probes: list = []
        self._last_grade_result = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DataCleanObservation:
        """Initialize a new data cleaning episode."""
        task_id = kwargs.get("task_id", "easy_contacts")
        task = get_task(task_id)

        actual_seed = seed if seed is not None else 42

        from dataclean_env.server.data_generator import generate_dirty_data

        dirty_data = generate_dirty_data(
            clean_data=task.ground_truth,
            corruptions=task.corruptions,
            seed=actual_seed,
        )

        # Assign stable row_ids (persist through delete/merge within episode)
        self._next_row_id = 0
        for row in dirty_data:
            row["_row_id"] = self._next_row_id
            self._next_row_id += 1

        # Compute initial score (dirty data vs ground truth)
        initial_score = self._grader.grade(
            final_data=dirty_data,
            ground_truth=task.ground_truth,
            original_data=dirty_data,
            action_history=[],
            schema=task.schema,
            flagged_cells=[],
            escalated_cells=[],
            ambiguous_cells=list(getattr(task, "ambiguous_cells", [])),
            utility_probes=list(getattr(task, "utility_probes", [])),
        ).score

        budget = DIFFICULTY_BUDGETS.get(task.difficulty, 100.0)
        self._state = DataCleanState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            difficulty=task.difficulty,
            current_data=copy.deepcopy(dirty_data),
            ground_truth=copy.deepcopy(task.ground_truth),
            original_dirty=copy.deepcopy(dirty_data),
            schema_def=task.schema,
            action_log=[],
            flagged_cells=[],
            escalated_cells=[],
            max_steps=task.max_steps,
            is_complete=False,
            previous_score=initial_score,
            action_budget=budget,
            budget_spent=0.0,
            budget_remaining=budget,
        )
        self._task_name = task.name
        self._ambiguous_cells: List[tuple[str, str]] = list(
            getattr(task, "ambiguous_cells", [])
        )
        self._utility_probes = list(getattr(task, "utility_probes", []))

        return self._build_observation(reward=None, done=False)

    def step(
        self,
        action: DataCleanAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DataCleanObservation:
        """Process one cleaning action. Returns observation with delta reward."""
        self._state.step_count += 1

        # Execute the action
        result = self._execute_action(action)
        self._state.action_log.append(result)

        # Deduct action cost from budget
        cost = ACTION_COSTS.get(action.action_type, 1.0)
        self._state.budget_spent += cost
        self._state.budget_remaining -= cost

        # Check termination
        is_done = (
            action.action_type == "mark_complete"
            or self._state.step_count >= self._state.max_steps
        )
        self._state.is_complete = is_done

        # Compute reward
        if is_done:
            # Terminal: return absolute final score
            grade_result = self._grader.grade(
                final_data=self._state.current_data,
                ground_truth=self._state.ground_truth,
                original_data=self._state.original_dirty,
                action_history=self._state.action_log,
                schema=self._state.schema_def,
                flagged_cells=self._state.flagged_cells,
                budget_spent=self._state.budget_spent,
                action_budget=self._state.action_budget,
                escalated_cells=self._state.escalated_cells,
                ambiguous_cells=self._ambiguous_cells,
                utility_probes=self._utility_probes,
            )
            reward = grade_result.score
            self._last_grade_result = grade_result
        else:
            # Non-terminal: delta reward
            reward = self._compute_delta_reward(result)

        return self._build_observation(reward=reward, done=is_done)

    @property
    def state(self) -> DataCleanState:
        return self._state

    # ------------------------------------------------------------------
    # Delta Reward System
    # ------------------------------------------------------------------

    def _compute_delta_reward(self, action_result: Dict[str, Any]) -> float:
        """Compute reward = current_score - previous_score - step_cost.

        Penalizes no-ops and errors explicitly.
        """
        step_cost = 0.005

        # Explicit penalties for bad actions
        if action_result.get("status") == "error":
            return -0.02
        if action_result.get("status") == "no_effect":
            return -0.01
        if action_result.get("cells_modified", 0) == 0 and action_result.get("action") not in ("flag_anomaly", "escalate_to_human"):
            return -0.01

        # Compute current score
        current_score = self._grader.grade(
            final_data=self._state.current_data,
            ground_truth=self._state.ground_truth,
            original_data=self._state.original_dirty,
            action_history=self._state.action_log,
            schema=self._state.schema_def,
            flagged_cells=self._state.flagged_cells,
            budget_spent=self._state.budget_spent,
            action_budget=self._state.action_budget,
            escalated_cells=self._state.escalated_cells,
            ambiguous_cells=self._ambiguous_cells,
            utility_probes=self._utility_probes,
        ).score

        delta = current_score - self._state.previous_score - step_cost
        self._state.previous_score = current_score
        return round(delta, 4)

    # ------------------------------------------------------------------
    # Action Dispatch
    # ------------------------------------------------------------------

    def _execute_action(self, action: DataCleanAction) -> Dict[str, Any]:
        """Dispatch action to the appropriate handler."""
        handler = getattr(self, f"_action_{action.action_type}", None)
        if handler is None:
            return {
                "action": action.action_type,
                "status": "error",
                "message": f"Unknown action type: {action.action_type}",
                "cells_modified": 0,
            }
        try:
            return handler(action.params)
        except (KeyError, TypeError, IndexError) as exc:
            return {
                "action": action.action_type,
                "status": "error",
                "message": f"Invalid params: {exc}",
                "cells_modified": 0,
            }
        except Exception as exc:
            return {
                "action": action.action_type,
                "status": "error",
                "message": str(exc),
                "cells_modified": 0,
            }

    # ------------------------------------------------------------------
    # Row Lookup by Stable row_id
    # ------------------------------------------------------------------

    def _find_row_by_id(self, row_id: int) -> tuple[int, Dict[str, Any] | None]:
        """Find the list index and row dict for a given stable row_id.

        Returns (index, row_dict) or (-1, None) if not found.
        """
        for i, row in enumerate(self._state.current_data):
            if row.get("_row_id") == row_id:
                return i, row
        return -1, None

    # ------------------------------------------------------------------
    # Action Handlers (10 total) — all use stable row_id
    # ------------------------------------------------------------------

    def _action_fix_value(self, params: Dict[str, Any]) -> Dict[str, Any]:
        row_id = int(params["row_id"])
        column = str(params["column"])
        new_value = params["new_value"]

        idx, row = self._find_row_by_id(row_id)
        if row is None:
            return {"action": "fix_value", "status": "error",
                    "message": f"row_id {row_id} not found", "cells_modified": 0}
        if column not in row or column.startswith("_"):
            return {"action": "fix_value", "status": "error",
                    "message": f"Column '{column}' not found", "cells_modified": 0}

        old_value = row[column]
        if str(old_value) == str(new_value):
            return {"action": "fix_value", "status": "no_effect",
                    "message": f"Value unchanged at (row_id={row_id}, '{column}')", "cells_modified": 0}

        row[column] = new_value
        return {"action": "fix_value", "status": "success",
                "message": f"(row_id={row_id}, '{column}'): '{old_value}' -> '{new_value}'",
                "cells_modified": 1, "old_value": old_value, "new_value": new_value,
                "row_id": row_id, "column": column}

    def _action_delete_row(self, params: Dict[str, Any]) -> Dict[str, Any]:
        row_id = int(params["row_id"])
        idx, row = self._find_row_by_id(row_id)

        if row is None:
            return {"action": "delete_row", "status": "error",
                    "message": f"row_id {row_id} not found", "cells_modified": 0}

        deleted = self._state.current_data.pop(idx)
        return {"action": "delete_row", "status": "success",
                "message": f"row_id={row_id} deleted",
                "cells_modified": len(deleted), "deleted_data": deleted,
                "row_id": row_id, "deleted_entity_id": deleted.get("_entity_id")}

    def _action_fill_missing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        row_id = int(params["row_id"])
        column = str(params["column"])
        value = params["value"]

        idx, row = self._find_row_by_id(row_id)
        if row is None:
            return {"action": "fill_missing", "status": "error",
                    "message": f"row_id {row_id} not found", "cells_modified": 0}
        if column not in row or column.startswith("_"):
            return {"action": "fill_missing", "status": "error",
                    "message": f"Column '{column}' not found", "cells_modified": 0}

        current = row.get(column)
        if current is not None and str(current).strip() != "":
            return {"action": "fill_missing", "status": "error",
                    "message": f"Cell (row_id={row_id}, '{column}') is not empty: '{current}'",
                    "cells_modified": 0}

        row[column] = value
        return {"action": "fill_missing", "status": "success",
                "message": f"(row_id={row_id}, '{column}'): NULL -> '{value}'",
                "cells_modified": 1, "row_id": row_id, "column": column, "new_value": value}

    def _action_standardize_format(self, params: Dict[str, Any]) -> Dict[str, Any]:
        column = str(params["column"])
        format_type = str(params["format_type"])
        data = self._state.current_data
        modified = 0
        errors: List[str] = []

        for row in data:
            if column not in row or row[column] is None:
                continue
            try:
                new_val = self._apply_format(row[column], format_type)
                if str(new_val) != str(row[column]):
                    row[column] = new_val
                    modified += 1
            except (ValueError, TypeError) as exc:
                errors.append(f"row_id={row.get('_row_id', '?')}: {exc}")

        if modified == 0 and not errors:
            return {"action": "standardize_format", "status": "no_effect",
                    "message": f"No changes needed in '{column}' for {format_type}",
                    "cells_modified": 0}

        msg = f"Formatted {modified} cell(s) in '{column}' to {format_type}"
        if errors:
            msg += f". {len(errors)} parse failure(s)."
        return {"action": "standardize_format", "status": "success",
                "message": msg, "cells_modified": modified}

    def _action_merge_duplicates(self, params: Dict[str, Any]) -> Dict[str, Any]:
        row_id1 = int(params["row_id1"])
        row_id2 = int(params["row_id2"])
        strategy = str(params.get("strategy", "merge_prefer_nonnull"))

        if row_id1 == row_id2:
            return {"action": "merge_duplicates", "status": "error",
                    "message": "Cannot merge a row with itself", "cells_modified": 0}

        idx1, r1 = self._find_row_by_id(row_id1)
        idx2, r2 = self._find_row_by_id(row_id2)

        if r1 is None or r2 is None:
            missing = row_id1 if r1 is None else row_id2
            return {"action": "merge_duplicates", "status": "error",
                    "message": f"row_id {missing} not found", "cells_modified": 0}

        # Track entity IDs for penalty checking
        eid1 = r1.get("_entity_id", "")
        eid2 = r2.get("_entity_id", "")

        merged = self._merge_rows(r1, r2, strategy)
        # Merged row keeps the first row's entity_id and row_id
        merged["_entity_id"] = eid1
        merged["_row_id"] = r1["_row_id"]

        # Remove both, insert merged at first position
        data = self._state.current_data
        lo_idx = min(idx1, idx2)
        hi_idx = max(idx1, idx2)
        data.pop(hi_idx)
        data.pop(lo_idx)
        data.insert(lo_idx, merged)

        return {"action": "merge_duplicates", "status": "success",
                "message": f"Merged row_id={row_id1} and row_id={row_id2} using '{strategy}'",
                "cells_modified": len(merged),
                "row_id1": row_id1, "row_id2": row_id2,
                "entity_id1": eid1, "entity_id2": eid2,
                "deleted_entity_id": eid2,
                "strategy": strategy}

    def _action_flag_anomaly(self, params: Dict[str, Any]) -> Dict[str, Any]:
        row_id = int(params["row_id"])
        column = str(params["column"])
        reason = str(params.get("reason", ""))

        idx, row = self._find_row_by_id(row_id)
        if row is None:
            return {"action": "flag_anomaly", "status": "error",
                    "message": f"row_id {row_id} not found", "cells_modified": 0}

        self._state.flagged_cells.append(
            {"row_id": row_id, "column": column, "reason": reason}
        )
        return {"action": "flag_anomaly", "status": "success",
                "message": f"Flagged (row_id={row_id}, '{column}'): {reason}",
                "cells_modified": 0}

    def _action_escalate_to_human(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Escalate a cell to human review -- agent signals it is uncertain."""
        row_id = int(params["row_id"])
        column = str(params["column"])
        confidence = float(params.get("confidence", 0.5))
        reason = str(params.get("reason", ""))

        idx, row = self._find_row_by_id(row_id)
        if row is None:
            return {"action": "escalate_to_human", "status": "error",
                    "message": f"row_id {row_id} not found", "cells_modified": 0}

        self._state.escalated_cells.append({
            "row_id": row_id, "column": column,
            "confidence": confidence, "reason": reason,
        })
        return {"action": "escalate_to_human", "status": "success",
                "message": f"Escalated (row_id={row_id}, '{column}'): {reason} (confidence={confidence})",
                "cells_modified": 0}

    def _action_split_column(self, params: Dict[str, Any]) -> Dict[str, Any]:
        column = str(params["column"])
        delimiter = str(params["delimiter"])
        new_names = list(params["new_names"])
        data = self._state.current_data
        modified = 0

        for row in data:
            if column not in row or row[column] is None:
                continue
            parts = str(row[column]).split(delimiter, maxsplit=len(new_names) - 1)
            for i, name in enumerate(new_names):
                row[name] = parts[i].strip() if i < len(parts) else None
            del row[column]
            modified += 1

        if modified == 0:
            return {"action": "split_column", "status": "no_effect",
                    "message": f"Column '{column}' not found or all null", "cells_modified": 0}
        return {"action": "split_column", "status": "success",
                "message": f"Split '{column}' into {new_names} ({modified} rows)",
                "cells_modified": modified}

    def _action_rename_column(self, params: Dict[str, Any]) -> Dict[str, Any]:
        old_name = str(params["old_name"])
        new_name = str(params["new_name"])
        data = self._state.current_data

        if not data or old_name not in data[0]:
            return {"action": "rename_column", "status": "error",
                    "message": f"Column '{old_name}' not found", "cells_modified": 0}
        if data and new_name in data[0]:
            return {"action": "rename_column", "status": "error",
                    "message": f"Column '{new_name}' already exists", "cells_modified": 0}

        for row in data:
            if old_name in row:
                row[new_name] = row.pop(old_name)
        return {"action": "rename_column", "status": "success",
                "message": f"Renamed '{old_name}' -> '{new_name}'", "cells_modified": 0}

    def _action_cast_type(self, params: Dict[str, Any]) -> Dict[str, Any]:
        column = str(params["column"])
        target_type = str(params["target_type"])
        data = self._state.current_data
        modified = 0
        nullified = 0

        for row in data:
            if column not in row or row[column] is None:
                continue
            try:
                row[column] = self._cast_value(row[column], target_type)
                modified += 1
            except (ValueError, TypeError):
                row[column] = None
                nullified += 1

        msg = f"Cast {modified} cell(s) in '{column}' to {target_type}"
        if nullified:
            msg += f" ({nullified} failed -> null)"
        return {"action": "cast_type", "status": "success",
                "message": msg, "cells_modified": modified + nullified}

    def _action_mark_complete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "mark_complete", "status": "success",
                "message": "Agent signaled completion", "cells_modified": 0}

    # ------------------------------------------------------------------
    # Format Standardization (8 types, fully implemented)
    # ------------------------------------------------------------------

    def _apply_format(self, value: Any, format_type: str) -> Any:
        """Apply format transformation to a single value."""
        val_str = str(value).strip()
        if not val_str:
            return value

        if format_type == "date:YYYY-MM-DD":
            return self._format_date_iso(val_str)
        elif format_type == "phone:US":
            return self._format_phone_us(val_str)
        elif format_type == "phone:E164":
            return self._format_phone_e164(val_str)
        elif format_type == "name:title_case":
            return val_str.title()
        elif format_type == "email:lowercase":
            return val_str.lower()
        elif format_type == "zip:5digit":
            return self._format_zip_5digit(val_str)
        elif format_type == "currency:float":
            return self._format_currency_float(val_str)
        elif format_type == "state:abbreviation":
            return self._format_state_abbrev(val_str)
        else:
            raise ValueError(f"Unknown format type: {format_type}")

    def _format_date_iso(self, val: str) -> str:
        """Parse various date formats and return YYYY-MM-DD."""
        for fmt in DATE_PARSE_FORMATS:
            try:
                dt = datetime.strptime(val.strip(), fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: '{val}'")

    def _format_phone_us(self, val: str) -> str:
        """Normalize phone to (XXX) XXX-XXXX format."""
        digits = re.sub(r"\D", "", val)
        if digits.startswith("1") and len(digits) == 11:
            digits = digits[1:]
        if len(digits) != 10:
            raise ValueError(f"Phone must have 10 digits, got {len(digits)}: '{val}'")
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"

    def _format_phone_e164(self, val: str) -> str:
        """Normalize phone to +1XXXXXXXXXX format."""
        digits = re.sub(r"\D", "", val)
        if digits.startswith("1") and len(digits) == 11:
            digits = digits[1:]
        if len(digits) != 10:
            raise ValueError(f"Phone must have 10 digits, got {len(digits)}: '{val}'")
        return f"+1{digits}"

    def _format_zip_5digit(self, val: str) -> str:
        """Normalize ZIP to 5 digits (pad or truncate)."""
        digits = re.sub(r"\D", "", val.split("-")[0])
        if not digits:
            raise ValueError(f"No digits in ZIP: '{val}'")
        return digits[:5].zfill(5)

    def _format_currency_float(self, val: str) -> float:
        """Parse currency string to float. '$1,234.56' -> 1234.56"""
        cleaned = val.replace("$", "").replace(",", "").strip()
        if cleaned.lower().endswith("k"):
            return float(cleaned[:-1]) * 1000
        return float(cleaned)

    def _format_state_abbrev(self, val: str) -> str:
        """Convert full state name to 2-letter abbreviation."""
        if len(val) == 2 and val.upper() in US_STATES.values():
            return val.upper()
        lower = val.strip().lower()
        if lower in US_STATES:
            return US_STATES[lower]
        raise ValueError(f"Unknown state: '{val}'")

    # ------------------------------------------------------------------
    # Row Merging (all strategies)
    # ------------------------------------------------------------------

    def _merge_rows(self, r1: Dict, r2: Dict, strategy: str) -> Dict:
        """Merge two rows according to the given strategy."""
        if strategy == "keep_first":
            return copy.deepcopy(r1)
        elif strategy == "keep_second":
            return copy.deepcopy(r2)
        elif strategy == "merge_prefer_nonnull":
            merged: Dict[str, Any] = {}
            for key in dict.fromkeys(list(r1.keys()) + list(r2.keys())):
                v1 = r1.get(key)
                v2 = r2.get(key)
                if v1 is not None and str(v1).strip():
                    merged[key] = v1
                elif v2 is not None and str(v2).strip():
                    merged[key] = v2
                else:
                    merged[key] = v1
            return merged
        elif strategy == "merge_prefer_row1":
            merged = copy.deepcopy(r2)
            for key, val in r1.items():
                if val is not None and str(val).strip():
                    merged[key] = val
            return merged
        elif strategy == "merge_prefer_row2":
            merged = copy.deepcopy(r1)
            for key, val in r2.items():
                if val is not None and str(val).strip():
                    merged[key] = val
            return merged
        else:
            raise ValueError(f"Unknown merge strategy: '{strategy}'")

    # ------------------------------------------------------------------
    # Type Casting
    # ------------------------------------------------------------------

    def _cast_value(self, value: Any, target_type: str) -> Any:
        """Cast a value to the target type."""
        val_str = str(value).strip()
        if target_type == "int":
            return int(float(val_str.replace(",", "").replace("$", "")))
        elif target_type == "float":
            return float(val_str.replace(",", "").replace("$", ""))
        elif target_type == "str":
            return val_str
        elif target_type == "bool":
            return val_str.lower() in ("true", "1", "yes", "y")
        elif target_type == "date":
            return self._format_date_iso(val_str)
        else:
            raise ValueError(f"Unknown target type: '{target_type}'")

    # ------------------------------------------------------------------
    # Observation Builder
    # ------------------------------------------------------------------

    def _build_observation(
        self, reward: float | None, done: bool
    ) -> DataCleanObservation:
        """Build issue-first observation from current state."""
        data = self._state.current_data
        columns = list(data[0].keys()) if data else []

        # Filter out internal fields EXCEPT _row_id (renamed to row_id for agent)
        hidden = {"_entity_id"}
        visible_columns = ["row_id"] + [c for c in columns if c not in hidden and c != "_row_id"]
        rows = [
            [row.get("_row_id")] + [row.get(col) for col in columns if col not in hidden and col != "_row_id"]
            for row in data
        ]

        # Quality analysis
        quality_issues = self._analyze_quality()
        issue_groups = self._group_issues(quality_issues)

        # Data summary
        null_count = sum(
            1 for row in data for col in visible_columns
            if row.get(col) is None
        )
        data_summary = DataSummary(
            row_count=len(data),
            column_count=len(visible_columns),
            total_cells=len(visible_columns) * len(data),
            null_count=null_count,
            issue_count=len(quality_issues),
            columns=visible_columns,
            dtypes={
                col: self._state.schema_def.get("expected_types", {}).get(col, "str")
                for col in visible_columns
            },
        )

        # Recent actions (last 5)
        recent = [
            ActionResult(
                action=a.get("action", ""),
                status=a.get("status", ""),
                message=a.get("message", ""),
                cells_modified=a.get("cells_modified", 0),
            )
            for a in self._state.action_log[-5:]
        ]

        # Build metadata with grade breakdown when episode ends
        metadata: Dict[str, Any] = {}
        grade = getattr(self, "_last_grade_result", None)
        if done and grade is not None:
            metadata = {
                "grade_breakdown": {
                    "accuracy": grade.accuracy,
                    "completeness": grade.completeness,
                    "format_consistency": grade.format_consistency,
                    "row_correctness": grade.row_correctness,
                    "efficiency": grade.efficiency,
                    "utility_score": grade.utility_score,
                    "penalties": grade.penalties,
                    "bonuses": grade.bonuses,
                },
                "utility_details": grade.utility_details,
            }

        return DataCleanObservation(
            done=done,
            reward=reward,
            metadata=metadata,
            data_summary=data_summary,
            quality_issues=quality_issues[:20],  # Cap at 20 for readability
            issue_groups=issue_groups,
            issues_remaining=len(quality_issues),
            columns=visible_columns,
            rows=rows,
            row_count=len(data),
            schema_info=self._state.schema_def,
            step_number=self._state.step_count,
            max_steps=self._state.max_steps,
            steps_remaining=self._state.max_steps - self._state.step_count,
            budget_spent=self._state.budget_spent,
            budget_remaining=self._state.budget_remaining,
            action_costs=ACTION_COSTS,
            last_action_result=recent[-1] if recent else None,
            recent_actions=recent,
            task_id=self._state.task_id,
            task_name=getattr(self, "_task_name", self._state.task_id),
            difficulty=self._state.difficulty,
        )

    # ------------------------------------------------------------------
    # Quality Analysis
    # ------------------------------------------------------------------

    def _analyze_quality(self) -> List[QualityIssue]:
        """Analyze current data and return detected quality issues."""
        issues: List[QualityIssue] = []
        schema = self._state.schema_def
        data = self._state.current_data
        constraints = schema.get("constraints", {})

        for row in data:
            rid = row.get("_row_id", 0)
            for col in [c for c in row if not c.startswith("_")]:
                val = row.get(col)
                col_constraints = constraints.get(col, {})

                # Null check
                if val is None and col_constraints.get("not_null"):
                    issues.append(QualityIssue(
                        row_id=rid, column=col, issue_type="null",
                        description="Required field is null",
                    ))

                if val is None:
                    continue

                # Format check
                fmt = col_constraints.get("format")
                if fmt and not self._matches_format(val, fmt):
                    issues.append(QualityIssue(
                        row_id=rid, column=col, issue_type="format",
                        description=f"Does not match format: {fmt}",
                        suggestion=f"Use standardize_format('{col}', '{self._suggest_format_type(fmt)}')",
                    ))

                # Allowed values
                allowed = col_constraints.get("allowed_values")
                if allowed and str(val) not in allowed:
                    issues.append(QualityIssue(
                        row_id=rid, column=col, issue_type="type_violation",
                        description=f"Value '{val}' not in allowed values",
                    ))

        # Duplicate detection
        issues.extend(self._detect_potential_duplicates())

        # Cross-field validation (for hard mode)
        issues.extend(self._detect_cross_field_issues())

        return issues

    def _detect_cross_field_issues(self) -> List[QualityIssue]:
        """Detect cross-field inconsistencies: zip/city, date relationships, insurance ID prefixes."""
        issues: List[QualityIssue] = []
        data = self._state.current_data
        schema = self._state.schema_def
        cross_field_rules = schema.get("cross_field_rules", {})

        # Rule: zip_city_match — zip code should correspond to the city
        zip_city_map = cross_field_rules.get("zip_city_map", {})
        if zip_city_map:
            for row in data:
                rid = row.get("_row_id", 0)
                zip_val = str(row.get("zip", row.get("office_zip", ""))).strip()
                city_val = str(row.get("city", row.get("office_city", ""))).strip().lower()
                if zip_val in zip_city_map:
                    expected_city = zip_city_map[zip_val].lower()
                    if city_val and city_val != expected_city:
                        issues.append(QualityIssue(
                            row_id=rid, column="zip",
                            issue_type="cross_field",
                            description=f"ZIP '{zip_val}' should map to '{zip_city_map[zip_val]}', got '{row.get('city', row.get('office_city', ''))}'",
                            suggestion=f"fix_value(row_id={rid}, column='city', new_value='{zip_city_map[zip_val]}')",
                        ))

        # Rule: date_order — dob must be before last_visit_date
        if "dob" in schema.get("expected_types", {}) and "last_visit_date" in schema.get("expected_types", {}):
            for row in data:
                rid = row.get("_row_id", 0)
                dob = row.get("dob")
                visit = row.get("last_visit_date")
                if dob and visit:
                    try:
                        dob_dt = datetime.strptime(str(dob), "%Y-%m-%d")
                        visit_dt = datetime.strptime(str(visit), "%Y-%m-%d")
                        if dob_dt > visit_dt:
                            issues.append(QualityIssue(
                                row_id=rid, column="dob",
                                issue_type="cross_field",
                                description=f"DOB '{dob}' is after last_visit_date '{visit}'",
                            ))
                        if dob_dt > datetime.now():
                            issues.append(QualityIssue(
                                row_id=rid, column="dob",
                                issue_type="cross_field",
                                description=f"DOB '{dob}' is in the future",
                            ))
                    except ValueError:
                        pass

        # Rule: insurance_prefix — insurance_id prefix must match provider
        prefix_map = cross_field_rules.get("insurance_prefix_map", {})
        if prefix_map:
            for row in data:
                rid = row.get("_row_id", 0)
                provider = str(row.get("insurance_provider", "")).strip()
                ins_id = str(row.get("insurance_id", "")).strip()
                if provider and ins_id and provider in prefix_map:
                    expected_prefix = prefix_map[provider]
                    if not ins_id.startswith(expected_prefix):
                        issues.append(QualityIssue(
                            row_id=rid, column="insurance_id",
                            issue_type="cross_field",
                            description=f"Insurance ID '{ins_id}' should start with '{expected_prefix}' for provider '{provider}'",
                        ))

        return issues

    def _detect_potential_duplicates(self) -> List[QualityIssue]:
        """Detect potential duplicate rows by email, phone, or name similarity."""
        issues: List[QualityIssue] = []
        data = self._state.current_data

        # Check by email
        email_index: Dict[str, List[int]] = {}
        for row in data:
            rid = row.get("_row_id", 0)
            email = row.get("email")
            if email and str(email).strip():
                key = str(email).strip().lower()
                email_index.setdefault(key, []).append(rid)
        for email, row_ids in email_index.items():
            if len(row_ids) > 1:
                issues.append(QualityIssue(
                    row_id=row_ids[0], column="email", issue_type="duplicate",
                    description=f"Rows {row_ids} share email '{email}'",
                    suggestion=f"Consider merge_duplicates(row_id1={row_ids[0]}, row_id2={row_ids[1]}, strategy='merge_prefer_nonnull')",
                ))

        # Check by phone (digit-only comparison)
        phone_index: Dict[str, List[int]] = {}
        for row in data:
            rid = row.get("_row_id", 0)
            phone = row.get("phone")
            if phone and str(phone).strip():
                digits = re.sub(r"\D", "", str(phone))
                if digits.startswith("1") and len(digits) == 11:
                    digits = digits[1:]
                if len(digits) == 10:
                    phone_index.setdefault(digits, []).append(rid)
        for digits, row_ids in phone_index.items():
            if len(row_ids) > 1:
                # Avoid duplicate issues if already flagged by email
                issues.append(QualityIssue(
                    row_id=row_ids[0], column="phone", issue_type="duplicate",
                    description=f"Rows {row_ids} share phone digits '{digits}'",
                ))

        return issues

    def _group_issues(self, issues: List[QualityIssue]) -> List[IssueGroup]:
        """Group issues by type for compact display."""
        type_counter: Dict[str, List[QualityIssue]] = {}
        for issue in issues:
            type_counter.setdefault(issue.issue_type, []).append(issue)

        return [
            IssueGroup(
                issue_type=itype,
                count=len(items),
                examples=items[:3],  # Show max 3 examples per type
            )
            for itype, items in sorted(type_counter.items())
        ]

    def _matches_format(self, value: Any, format_spec: str) -> bool:
        """Check if a value matches the expected format."""
        val_str = str(value)
        format_patterns: Dict[str, str] = {
            "YYYY-MM-DD": r"^\d{4}-\d{2}-\d{2}$",
            "(XXX) XXX-XXXX": r"^\(\d{3}\) \d{3}-\d{4}$",
            "email": r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
            "5_digit": r"^\d{5}$",
            "+1XXXXXXXXXX": r"^\+1\d{10}$",
        }
        pattern = format_patterns.get(format_spec)
        if pattern:
            return bool(re.match(pattern, val_str))
        return True

    def _suggest_format_type(self, format_spec: str) -> str:
        """Suggest the standardize_format type for a given format spec."""
        mapping = {
            "YYYY-MM-DD": "date:YYYY-MM-DD",
            "(XXX) XXX-XXXX": "phone:US",
            "email": "email:lowercase",
            "5_digit": "zip:5digit",
            "+1XXXXXXXXXX": "phone:E164",
        }
        return mapping.get(format_spec, format_spec)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self):  # type: ignore[override]
        from openenv.core.env_server.types import EnvironmentMetadata

        return EnvironmentMetadata(
            name="dataclean_env",
            description=(
                "Data Cleaning environment for training AI agents to clean "
                "messy tabular data. Supports 3 difficulty levels (easy, medium, hard) "
                "with deterministic grading via cell-by-cell comparison against ground truth."
            ),
            version="0.1.0",
        )
