"""Full grader calibration audit across all 3 tasks.

Tests: no-op, random, format-only, oracle on all tasks.
Plus hard-task specific: false-positive merge, gender trap, dedup-only, cell-fixes-only.

Usage:
    python3 scripts/grader_calibration_audit.py
"""

from __future__ import annotations

import random
import sys
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Bootstrap: install openenv mock if the real package is absent.
# ---------------------------------------------------------------------------


def _ensure_openenv_mock() -> None:
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

from dataclean_env.server.environment import DataCleanEnvironment
from dataclean_env.models import DataCleanAction
from dataclean_env.server.tasks import get_task

TASK_IDS = ["easy_contacts", "medium_employees", "hard_patients"]

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


# =====================================================================
# Standard strategies (all tasks)
# =====================================================================


def strategy_noop(task_id: str) -> float:
    env = DataCleanEnvironment()
    env.reset(seed=42, task_id=task_id)
    obs = env.step(_action("mark_complete"))
    return obs.reward


def strategy_random(task_id: str) -> float:
    rng = random.Random(123)
    env = DataCleanEnvironment()
    obs = env.reset(seed=42, task_id=task_id)
    columns = [c for c in obs.columns if not c.startswith("_")]
    row_ids = [row[0] for row in obs.rows] if obs.rows else []
    for _ in range(min(5, obs.steps_remaining - 1)):
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
    env = DataCleanEnvironment()
    env.reset(seed=42, task_id=task_id)
    for fc in FORMAT_COLUMNS.get(task_id, []):
        obs = env.step(_action(
            "standardize_format",
            column=fc["column"],
            format_type=fc["format_type"],
        ))
        if obs.done:
            return obs.reward
    obs = env.step(_action("mark_complete"))
    return obs.reward


def strategy_oracle(task_id: str) -> float:
    env = DataCleanEnvironment()
    obs = env.reset(seed=42, task_id=task_id)
    task = get_task(task_id)
    gt = task.ground_truth
    gt_by_eid: Dict[str, Dict[str, Any]] = {}
    for row in gt:
        eid = row.get("_entity_id", "")
        if eid:
            gt_by_eid[eid] = row

    state = env.state

    # Phase 1: Merge duplicates (rows with same entity_id)
    eid_to_rows: Dict[str, list] = {}
    for row in state.current_data:
        eid = row.get("_entity_id", "")
        if eid:
            eid_to_rows.setdefault(eid, []).append(row)

    for eid, rows in eid_to_rows.items():
        if len(rows) > 1 and eid in gt_by_eid:
            primary = rows[0]
            for extra in rows[1:]:
                obs = env.step(_action(
                    "merge_duplicates",
                    row_id1=primary["_row_id"],
                    row_id2=extra["_row_id"],
                    strategy="merge_prefer_nonnull",
                ))
                if obs.done:
                    return obs.reward

    # Phase 2: Delete rows not in ground truth
    gt_eids = set(gt_by_eid.keys())
    for row in list(state.current_data):
        eid = row.get("_entity_id", "")
        if eid and eid not in gt_eids:
            obs = env.step(_action("delete_row", row_id=row["_row_id"]))
            if obs.done:
                return obs.reward

    # Phase 3: Fix all cell values to match ground truth
    for row in state.current_data:
        eid = row.get("_entity_id", "")
        gt_row = gt_by_eid.get(eid)
        if gt_row is None:
            continue
        row_id = row.get("_row_id")
        for col in gt_row:
            if col.startswith("_"):
                continue
            gt_val = gt_row[col]
            cur_val = row.get(col)
            if str(cur_val).strip() != str(gt_val).strip():
                obs = env.step(_action(
                    "fix_value",
                    row_id=row_id,
                    column=col,
                    new_value=gt_val,
                ))
                if obs.done:
                    return obs.reward

    # Phase 4: Standardize formats
    for fc in FORMAT_COLUMNS.get(task_id, []):
        obs = env.step(_action(
            "standardize_format",
            column=fc["column"],
            format_type=fc["format_type"],
        ))
        if obs.done:
            return obs.reward

    obs = env.step(_action("mark_complete"))
    return obs.reward


# =====================================================================
# Hard-task specific tests
# =====================================================================


def hard_false_positive_merge() -> float:
    """Merge PAT032 and PAT033 (two different David Kims) -- should penalize."""
    env = DataCleanEnvironment()
    env.reset(seed=42, task_id="hard_patients")
    state = env.state
    rid32 = rid33 = None
    for row in state.current_data:
        if row.get("_entity_id") == "PAT032":
            rid32 = row["_row_id"]
        if row.get("_entity_id") == "PAT033":
            rid33 = row["_row_id"]
    if rid32 is not None and rid33 is not None:
        env.step(_action(
            "merge_duplicates",
            row_id1=rid32,
            row_id2=rid33,
            strategy="merge_prefer_nonnull",
        ))
    obs = env.step(_action("mark_complete"))
    return obs.reward


def hard_gender_trap() -> float:
    """Fix PAT031 Morgan's gender from M to F -- should penalize."""
    env = DataCleanEnvironment()
    env.reset(seed=42, task_id="hard_patients")
    state = env.state
    rid31 = None
    for row in state.current_data:
        if row.get("_entity_id") == "PAT031":
            rid31 = row["_row_id"]
    if rid31 is not None:
        env.step(_action("fix_value", row_id=rid31, column="gender", new_value="F"))
    obs = env.step(_action("mark_complete"))
    return obs.reward


def hard_dedup_only() -> float:
    """Only merge true duplicates and delete extras, fix nothing else."""
    env = DataCleanEnvironment()
    obs = env.reset(seed=42, task_id="hard_patients")
    state = env.state
    task = get_task("hard_patients")
    gt_eids = {r.get("_entity_id") for r in task.ground_truth}

    eid_to_rows: Dict[str, list] = {}
    for row in state.current_data:
        eid = row.get("_entity_id", "")
        if eid:
            eid_to_rows.setdefault(eid, []).append(row)

    merge_count = 0
    for eid, rows in eid_to_rows.items():
        if len(rows) > 1 and eid in gt_eids:
            primary = rows[0]
            for extra in rows[1:]:
                obs = env.step(_action(
                    "merge_duplicates",
                    row_id1=primary["_row_id"],
                    row_id2=extra["_row_id"],
                    strategy="merge_prefer_nonnull",
                ))
                merge_count += 1
                if obs.done:
                    break
            if obs.done:
                break

    del_count = 0
    for row in list(state.current_data):
        if obs.done:
            break
        eid = row.get("_entity_id", "")
        if eid and eid not in gt_eids:
            obs = env.step(_action("delete_row", row_id=row["_row_id"]))
            del_count += 1

    if not obs.done:
        obs = env.step(_action("mark_complete"))
    return obs.reward


def hard_fix_cells_no_dedup() -> float:
    """Fix all cell-level issues but skip deduplication entirely."""
    env = DataCleanEnvironment()
    obs = env.reset(seed=42, task_id="hard_patients")
    state = env.state
    task = get_task("hard_patients")
    gt_by_eid = {r.get("_entity_id", ""): r for r in task.ground_truth}

    fix_count = 0
    for row in state.current_data:
        eid = row.get("_entity_id", "")
        gt_row = gt_by_eid.get(eid)
        if gt_row is None:
            continue
        row_id = row.get("_row_id")
        for col in gt_row:
            if col.startswith("_"):
                continue
            gt_val = gt_row[col]
            cur_val = row.get(col)
            if str(cur_val).strip() != str(gt_val).strip():
                obs = env.step(_action(
                    "fix_value",
                    row_id=row_id,
                    column=col,
                    new_value=gt_val,
                ))
                fix_count += 1
                if obs.done:
                    break
        if obs.done:
            break

    if not obs.done:
        obs = env.step(_action("mark_complete"))

    print(f"    (cell-fixes-only used {fix_count} fix_value actions, "
          f"budget_spent={state.budget_spent:.1f}, "
          f"steps={state.step_count})")
    return obs.reward


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    strategies = {
        "no-op": strategy_noop,
        "random (5 acts)": strategy_random,
        "format-only": strategy_format_only,
        "oracle": strategy_oracle,
    }

    print()
    print("=" * 90)
    print("GRADER CALIBRATION AUDIT -- ALL TASKS")
    print("=" * 90)

    results: Dict[str, Dict[str, Any]] = {}
    for name, fn in strategies.items():
        results[name] = {}
        for tid in TASK_IDS:
            try:
                score = fn(tid)
                results[name][tid] = score
            except Exception as e:
                results[name][tid] = f"ERR: {e}"

    # Print table
    print()
    hdr = f"{'Strategy':<18} {'easy_contacts':>15} {'medium_employees':>18} {'hard_patients':>15} {'Mean':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name in strategies:
        row = f"{name:<18}"
        scores = []
        for tid in TASK_IDS:
            s = results[name][tid]
            if isinstance(s, float):
                row += f"  {s:>13.4f}"
                scores.append(s)
            else:
                row += f"  {str(s):>13}"
        if scores:
            mean = sum(scores) / len(scores)
            row += f"  {mean:>6.4f}"
        print(row)

    # Hard-task specific tests
    print()
    print("=" * 90)
    print("HARD TASK SPECIFIC TESTS")
    print("=" * 90)

    hard_tests = [
        ("false-positive merge (PAT032+PAT033)", hard_false_positive_merge),
        ("gender trap (PAT031 M->F)", hard_gender_trap),
        ("dedup-only (no cell fixes)", hard_dedup_only),
        ("cell-fixes-only (no dedup)", hard_fix_cells_no_dedup),
    ]

    hard_scores: Dict[str, float] = {}
    for name, fn in hard_tests:
        try:
            score = fn()
            hard_scores[name] = score
            print(f"  {name:<45} score = {score:.4f}")
        except Exception as e:
            print(f"  {name:<45} ERR: {e}")

    # Variance ordering check
    print()
    print("=" * 90)
    print("VARIANCE ORDERING CHECK")
    print("=" * 90)
    all_pass = True
    for tid in TASK_IDS:
        noop = results["no-op"].get(tid, -1)
        fmt = results["format-only"].get(tid, -1)
        oracle_s = results["oracle"].get(tid, -1)
        if isinstance(noop, float) and isinstance(fmt, float) and isinstance(oracle_s, float):
            ok1 = noop < fmt
            ok2 = fmt < oracle_s
            status = "PASS" if ok1 and ok2 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {tid:<22} no-op={noop:.4f} < format={fmt:.4f} < oracle={oracle_s:.4f}  [{status}]")
        else:
            print(f"  {tid:<22} ERROR in results")
            all_pass = False

    # No-op generosity check
    print()
    print("NO-OP GENEROSITY CHECK (should be < 0.15):")
    for tid in TASK_IDS:
        s = results["no-op"].get(tid, -1)
        if isinstance(s, float):
            status = "PASS" if s < 0.15 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {tid:<22} no-op={s:.4f}  [{status}]")

    # Hard-specific anomaly checks
    print()
    print("HARD-SPECIFIC ANOMALY CHECKS:")
    noop_hard = results["no-op"].get("hard_patients", -1)
    fp_merge = hard_scores.get("false-positive merge (PAT032+PAT033)", -1)
    gender = hard_scores.get("gender trap (PAT031 M->F)", -1)
    dedup = hard_scores.get("dedup-only (no cell fixes)", -1)
    cells = hard_scores.get("cell-fixes-only (no dedup)", -1)

    if isinstance(fp_merge, float) and isinstance(noop_hard, float):
        # False positive merge should score WORSE than no-op (penalty applied)
        ok = fp_merge < noop_hard
        status = "PASS" if ok else "ANOMALY"
        print(f"  false-pos merge ({fp_merge:.4f}) < no-op ({noop_hard:.4f})?  [{status}]")

    if isinstance(gender, float) and isinstance(noop_hard, float):
        # Gender trap should score WORSE than no-op (penalty for wrong fix)
        ok = gender < noop_hard
        status = "PASS" if ok else "ANOMALY"
        print(f"  gender trap ({gender:.4f}) < no-op ({noop_hard:.4f})?  [{status}]")

    if isinstance(dedup, float) and isinstance(cells, float):
        # Both partial strategies should be between no-op and oracle
        oracle_hard = results["oracle"].get("hard_patients", -1)
        print(f"  dedup-only={dedup:.4f}, cell-fixes-only={cells:.4f}")
        if isinstance(oracle_hard, float):
            print(f"  (for reference: no-op={noop_hard:.4f}, oracle={oracle_hard:.4f})")

    print()
    if all_pass:
        print("ALL CHECKS PASSED.")
    else:
        print("SOME CHECKS FAILED -- see above.")


if __name__ == "__main__":
    main()
