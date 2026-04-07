"""Investigate grader anomalies found in calibration audit.

Anomaly 1: hard_patients format-only == no-op (both 0.0680)
Anomaly 2: hard_patients cell-fixes-only scores 1.0 (same as oracle)

Usage:
    python3 scripts/grader_anomaly_investigation.py
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Bootstrap
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

from dataclean_env.server.environment import DataCleanEnvironment
from dataclean_env.models import DataCleanAction
from dataclean_env.server.grader import DataCleanGrader
from dataclean_env.server.tasks import get_task


def _action(action_type: str, **params: object) -> DataCleanAction:
    return DataCleanAction(action_type=action_type, params=params)


def main() -> None:
    # =====================================================================
    # ANOMALY 1: format-only == no-op for hard_patients
    # =====================================================================
    print("=" * 80)
    print("ANOMALY 1: hard_patients format-only == no-op (both 0.0680)")
    print("=" * 80)

    env = DataCleanEnvironment()
    obs = env.reset(seed=42, task_id="hard_patients")
    print(f"Initial rows: {obs.row_count}")

    for fc in [
        {"column": "dob", "format_type": "date:YYYY-MM-DD"},
        {"column": "last_visit_date", "format_type": "date:YYYY-MM-DD"},
        {"column": "phone", "format_type": "phone:US"},
        {"column": "email", "format_type": "email:lowercase"},
        {"column": "zip", "format_type": "zip:5digit"},
    ]:
        obs = env.step(_action(
            "standardize_format",
            column=fc["column"],
            format_type=fc["format_type"],
        ))
        ar = obs.last_action_result
        status = ar.status if ar else "?"
        cells = ar.cells_modified if ar else "?"
        msg = ar.message if ar else "?"
        print(f"  standardize {fc['column']:<20}: status={status}, cells={cells}, msg={msg}")

    obs = env.step(_action("mark_complete"))
    print(f"Final format-only score: {obs.reward:.4f}")

    # =====================================================================
    # ANOMALY 2: cell-fixes-only == oracle
    # =====================================================================
    print()
    print("=" * 80)
    print("ANOMALY 2: cell-fixes-only scores 1.0 (means dedup has zero impact)")
    print("=" * 80)

    task = get_task("hard_patients")
    gt_count = len(task.ground_truth)
    print(f"Ground truth rows: {gt_count}")

    env2 = DataCleanEnvironment()
    obs2 = env2.reset(seed=42, task_id="hard_patients")
    print(f"Dirty data rows: {obs2.row_count}")

    state2 = env2.state
    eid_counts: dict = {}
    for row in state2.current_data:
        eid = row.get("_entity_id", "")
        eid_counts[eid] = eid_counts.get(eid, 0) + 1

    dups = {k: v for k, v in eid_counts.items() if v > 1}
    print(f"Entity IDs with duplicate rows: {len(dups)}")
    for eid, cnt in sorted(dups.items()):
        print(f"  {eid}: {cnt} rows")

    gt_eids = {r.get("_entity_id") for r in task.ground_truth}
    extra = set(eid_counts.keys()) - gt_eids
    print(f"Extra entity IDs not in GT: {len(extra)}")
    for eid in sorted(extra):
        if eid:
            print(f"  {eid}: {eid_counts[eid]} rows")

    # Do cell-fixes-only and get detailed grade
    env3 = DataCleanEnvironment()
    env3.reset(seed=42, task_id="hard_patients")
    state3 = env3.state
    gt_by_eid = {r.get("_entity_id", ""): r for r in task.ground_truth}

    fix_count = 0
    done_early = False
    for row in state3.current_data:
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
                obs3 = env3.step(_action(
                    "fix_value",
                    row_id=row_id,
                    column=col,
                    new_value=gt_val,
                ))
                fix_count += 1
                if obs3.done:
                    done_early = True
                    break
        if done_early:
            break

    # Detailed grading
    grader = DataCleanGrader()
    result = grader.grade(
        final_data=state3.current_data,
        ground_truth=task.ground_truth,
        original_data=state3.original_dirty,
        action_history=state3.action_log,
        schema=task.schema,
        flagged_cells=state3.flagged_cells,
        budget_spent=state3.budget_spent,
        action_budget=150.0,
        escalated_cells=[],
        ambiguous_cells=list(getattr(task, "ambiguous_cells", [])),
        utility_probes=list(getattr(task, "utility_probes", [])),
    )

    print()
    print("Cell-fixes-only DETAILED grade:")
    print(f"  accuracy       = {result.accuracy:.4f}  (weight 0.40)")
    print(f"  completeness   = {result.completeness:.4f}  (weight 0.20)")
    print(f"  format_consist = {result.format_consistency:.4f}  (weight 0.10)")
    print(f"  row_correctness= {result.row_correctness:.4f}  (weight 0.10)")
    print(f"  efficiency     = {result.efficiency:.4f}  (weight 0.10)")
    print(f"  utility_score  = {result.utility_score:.4f}  (weight 0.10)")
    print(f"  penalties      = {result.penalties:.4f}")
    print(f"  bonuses        = {result.bonuses:.4f}")
    print(f"  FINAL SCORE    = {result.score:.4f}")
    print(f"  Row count: final={len(state3.current_data)}, gt={gt_count}")
    print(f"  fix_value actions used: {fix_count}")
    print(f"  budget_spent: {state3.budget_spent}")

    # Also get detailed grade for no-op on hard
    print()
    print("No-op DETAILED grade for hard_patients:")
    env4 = DataCleanEnvironment()
    env4.reset(seed=42, task_id="hard_patients")
    state4 = env4.state
    r_noop = grader.grade(
        final_data=state4.current_data,
        ground_truth=task.ground_truth,
        original_data=state4.original_dirty,
        action_history=[],
        schema=task.schema,
        flagged_cells=[],
        budget_spent=0.0,
        action_budget=150.0,
        escalated_cells=[],
        ambiguous_cells=list(getattr(task, "ambiguous_cells", [])),
        utility_probes=list(getattr(task, "utility_probes", [])),
    )
    print(f"  accuracy       = {r_noop.accuracy:.4f}")
    print(f"  completeness   = {r_noop.completeness:.4f}")
    print(f"  format_consist = {r_noop.format_consistency:.4f}")
    print(f"  row_correctness= {r_noop.row_correctness:.4f}")
    print(f"  efficiency     = {r_noop.efficiency:.4f}")
    print(f"  utility_score  = {r_noop.utility_score:.4f}")
    print(f"  penalties      = {r_noop.penalties:.4f}")
    print(f"  bonuses        = {r_noop.bonuses:.4f}")
    print(f"  FINAL SCORE    = {r_noop.score:.4f}")


if __name__ == "__main__":
    main()
