"""Prove grader variance across different agent strategies.

Runs 4 strategies on all 3 tasks and prints a formatted score table.
No server needed -- imports the environment directly.

Usage:
    python3 scripts/run_score_variance.py
"""

from __future__ import annotations

import random
import sys
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Bootstrap: install openenv mock if the real package is absent.
# ---------------------------------------------------------------------------


def _ensure_openenv_mock() -> None:
    """Install a lightweight openenv mock into sys.modules if needed."""
    try:
        import openenv.core.env_server  # noqa: F401
        return
    except ImportError:
        pass

    from types import ModuleType

    class _Base:
        def __init__(self, **kw: object) -> None:
            for k, v in kw.items():
                setattr(self, k, v)

    class _Environment:
        def __init__(self) -> None:
            pass

        def __class_getitem__(cls, item):  # type: ignore[override]
            return cls

    class _EnvClient:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def __class_getitem__(cls, item):  # type: ignore[override]
            return cls

    names = [
        "openenv", "openenv.core", "openenv.core.env_server",
        "openenv.core.env_server.types", "openenv.core.env_client",
        "openenv.core.client_types",
    ]
    mods = {n: ModuleType(n) for n in names}
    for n, m in mods.items():
        sys.modules[n] = m

    mods["openenv"].core = mods["openenv.core"]  # type: ignore[attr-defined]
    mods["openenv.core"].env_server = mods["openenv.core.env_server"]  # type: ignore[attr-defined]
    mods["openenv.core"].env_client = mods["openenv.core.env_client"]  # type: ignore[attr-defined]
    mods["openenv.core"].client_types = mods["openenv.core.client_types"]  # type: ignore[attr-defined]

    for attr in ("Action", "Observation", "State"):
        setattr(mods["openenv.core.env_server"], attr, type(attr, (_Base,), {}))
    setattr(mods["openenv.core.env_server"], "Environment", _Environment)
    setattr(mods["openenv.core.env_server.types"], "EnvironmentMetadata", _Base)
    setattr(mods["openenv.core.env_client"], "EnvClient", _EnvClient)
    setattr(mods["openenv.core.client_types"], "StepResult", _Base)


_ensure_openenv_mock()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from dataclean_env.server.environment import DataCleanEnvironment
    from dataclean_env.models import DataCleanAction
    from dataclean_env.server.tasks import get_task
except ImportError as exc:
    print(f"FAIL: Could not import DataClean-Env modules: {exc}")
    print("Hint: run from the project root with the package installed or on PYTHONPATH.")
    sys.exit(1)


TASK_IDS = ["easy_contacts", "medium_employees", "hard_patients"]

# Format columns and their standardize_format types, per task.
# Derived from each task's schema constraints that have a "format" key.
FORMAT_COLUMNS: Dict[str, List[Dict[str, str]]] = {
    "easy_contacts": [
        {"column": "signup_date", "format_type": "date:YYYY-MM-DD"},
        {"column": "phone", "format_type": "phone:US"},
    ],
    "medium_employees": [
        {"column": "hire_date", "format_type": "date:YYYY-MM-DD"},
        {"column": "phone", "format_type": "phone:US"},
        {"column": "email", "format_type": "email:lowercase"},
        {"column": "office_zip", "format_type": "zip:5digit"},
    ],
    "hard_patients": [
        {"column": "dob", "format_type": "date:YYYY-MM-DD"},
        {"column": "last_visit_date", "format_type": "date:YYYY-MM-DD"},
        {"column": "phone", "format_type": "phone:US"},
        {"column": "email", "format_type": "email:lowercase"},
        {"column": "zip", "format_type": "zip:5digit"},
    ],
}


def _action(action_type: str, **params: Any) -> DataCleanAction:
    return DataCleanAction(action_type=action_type, params=params)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def strategy_noop(task_id: str) -> float:
    """Reset and immediately mark_complete. Baseline: do nothing."""
    env = DataCleanEnvironment()
    env.reset(seed=42, task_id=task_id)
    obs = env.step(_action("mark_complete"))
    return obs.reward


def strategy_random(task_id: str) -> float:
    """Apply 10 random fix_value actions with garbage values, then mark_complete."""
    rng = random.Random(123)
    env = DataCleanEnvironment()
    obs = env.reset(seed=42, task_id=task_id)

    columns = [c for c in obs.columns if not c.startswith("_")]
    row_ids = [row[0] for row in obs.rows] if obs.rows else []

    for _ in range(min(10, obs.steps_remaining - 1)):
        if not row_ids or not columns:
            break
        rid = rng.choice(row_ids)
        col = rng.choice(columns)
        val = rng.choice(["RANDOM", "999", "", "null", "42", "test@x.com"])
        obs = env.step(_action("fix_value", row_id=rid, column=col, new_value=val))
        if obs.done:
            return obs.reward

    obs = env.step(_action("mark_complete"))
    return obs.reward


def strategy_format_only(task_id: str) -> float:
    """Standardize format on all known format columns, then mark_complete."""
    env = DataCleanEnvironment()
    env.reset(seed=42, task_id=task_id)

    fmt_cols = FORMAT_COLUMNS.get(task_id, [])
    for fc in fmt_cols:
        obs = env.step(_action(
            "standardize_format",
            column=fc["column"],
            format_type=fc["format_type"],
        ))
        if obs.done:
            return obs.reward

    obs = env.step(_action("mark_complete"))
    return obs.reward


def strategy_heuristic(task_id: str) -> float:
    """Use quality_issues to apply targeted fixes. Should beat format-only.

    Applies: format standardization, null fills, duplicate merges using
    the suggestions from the observation. Does NOT peek at ground truth.
    """
    env = DataCleanEnvironment()
    obs = env.reset(seed=42, task_id=task_id)

    # Phase 1: Standardize format columns (known per-task)
    fmt_cols = FORMAT_COLUMNS.get(task_id, [])
    for fc in fmt_cols:
        if obs.done:
            return obs.reward
        obs = env.step(_action(
            "standardize_format",
            column=fc["column"],
            format_type=fc["format_type"],
        ))

    # Phase 2: Merge duplicates using suggestions from quality_issues
    merged_pairs: set = set()
    for issue in list(obs.quality_issues):
        if obs.done:
            return obs.reward
        if issue.issue_type == "duplicate" and issue.suggestion:
            import re as _re
            nums = _re.findall(r"row[_\s]*(?:id)?[_\s]*(\d+)", issue.suggestion, _re.IGNORECASE)
            if not nums:
                nums = _re.findall(r"\b(\d+)\b", issue.suggestion)
            if len(nums) >= 2:
                r1, r2 = int(nums[0]), int(nums[1])
                pair = (min(r1, r2), max(r1, r2))
                if pair not in merged_pairs:
                    merged_pairs.add(pair)
                    obs = env.step(_action(
                        "merge_duplicates",
                        row_id1=r1,
                        row_id2=r2,
                        strategy="merge_prefer_nonnull",
                    ))
                    if obs.done:
                        return obs.reward

    # Phase 3: Schema-driven cell fixes using observation data
    # Title-case name columns, lowercase email, fix common department typos
    DEPT_CORRECTIONS: Dict[str, str] = {
        "engneering": "Engineering", "enginering": "Engineering",
        "engg": "Engineering", "eng": "Engineering",
        "mktg": "Marketing", "marketting": "Marketing",
        "hr": "Human Resources", "h.r.": "Human Resources",
        "human resources": "Human Resources",
        "finance": "Finance", "fin": "Finance",
        "sales": "Sales", "sls": "Sales",
        "operations": "Operations", "ops": "Operations",
    }

    for row in obs.rows:
        if obs.done:
            return obs.reward
        row_id = row[0]
        for i, col in enumerate(obs.columns):
            if col.startswith("_") or row[i] is None:
                continue
            val = str(row[i])

            # Title-case name fields
            if col in ("name", "first_name", "last_name", "full_name"):
                titled = val.strip().title()
                if titled != val:
                    obs = env.step(_action(
                        "fix_value", row_id=row_id, column=col, new_value=titled,
                    ))
                    if obs.done:
                        return obs.reward

            # Fix department typos
            if col == "department":
                corrected = DEPT_CORRECTIONS.get(val.lower().strip())
                if corrected and corrected != val:
                    obs = env.step(_action(
                        "fix_value", row_id=row_id, column=col, new_value=corrected,
                    ))
                    if obs.done:
                        return obs.reward

            # Lowercase email
            if col == "email" and val != val.lower():
                obs = env.step(_action(
                    "fix_value", row_id=row_id, column=col, new_value=val.lower(),
                ))
                if obs.done:
                    return obs.reward

    obs = env.step(_action("mark_complete"))
    return obs.reward


def strategy_oracle(task_id: str) -> float:
    """Submit ground truth values for every dirty cell and merge duplicates. Should score near 1.0."""
    env = DataCleanEnvironment()
    obs = env.reset(seed=42, task_id=task_id)

    task = get_task(task_id)
    gt = task.ground_truth

    # Build a map from _entity_id to ground truth row
    gt_by_eid: Dict[str, Dict[str, Any]] = {}
    for row in gt:
        eid = row.get("_entity_id", "")
        if eid:
            gt_by_eid[eid] = row

    # Phase 1: Delete duplicate rows (rows whose _entity_id already has a
    # canonical row in the data). Keep the first occurrence, delete the rest.
    state = env.state
    seen_eids: Dict[str, int] = {}  # eid -> first row_id
    duplicate_row_ids: list = []
    for data_row in state.current_data:
        eid = data_row.get("_entity_id", "")
        row_id = data_row.get("_row_id")
        if eid in seen_eids:
            duplicate_row_ids.append(row_id)
        else:
            seen_eids[eid] = row_id

    for dup_rid in duplicate_row_ids:
        obs = env.step(_action("delete_row", row_id=dup_rid))
        if obs.done:
            return obs.reward

    # Phase 2: Fix every cell to match ground truth
    state = env.state
    for data_row in state.current_data:
        eid = data_row.get("_entity_id", "")
        gt_row = gt_by_eid.get(eid)
        if gt_row is None:
            continue
        row_id = data_row.get("_row_id")
        for col in gt_row:
            if col.startswith("_"):
                continue
            gt_val = gt_row[col]
            cur_val = data_row.get(col)
            # Only fix if values differ (string comparison)
            if str(cur_val).strip() != str(gt_val).strip():
                obs = env.step(_action(
                    "fix_value",
                    row_id=row_id,
                    column=col,
                    new_value=gt_val,
                ))
                if obs.done:
                    return obs.reward

    obs = env.step(_action("mark_complete"))
    return obs.reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STRATEGIES = {
    "no-op": strategy_noop,
    "random": strategy_random,
    "format-only": strategy_format_only,
    "heuristic": strategy_heuristic,
    "oracle": strategy_oracle,
}


def main() -> None:
    results: Dict[str, Dict[str, float]] = {}

    for name, fn in STRATEGIES.items():
        results[name] = {}
        for task_id in TASK_IDS:
            score = fn(task_id)
            results[name][task_id] = score

    # Print formatted table
    col_width = 20
    header = f"{'Strategy':<15}"
    for tid in TASK_IDS:
        header += f"  {tid:>{col_width}}"
    header += f"  {'Mean':>{col_width}}"

    print("\n" + "=" * len(header))
    print("Score Variance Table")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name in STRATEGIES:
        row = f"{name:<15}"
        scores = []
        for tid in TASK_IDS:
            s = results[name][tid]
            scores.append(s)
            row += f"  {s:>{col_width}.4f}"
        mean = sum(scores) / len(scores)
        row += f"  {mean:>{col_width}.4f}"
        print(row)

    print("-" * len(header))
    print("\nExpected ordering: no-op < 0.15, random ~ 0, format-only > no-op, heuristic > format-only, oracle ~ 1.0")
    print("If oracle is not near 1.0, check that ground truth data and grader are aligned.")


if __name__ == "__main__":
    main()
