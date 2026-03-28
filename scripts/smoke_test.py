"""Quick smoke test for DataClean-Env: reset, step, score each task locally.

Runs without a server or openenv-core installed at the script level.
Imports the environment and grader directly from dataclean_env.server.

Usage:
    python3 scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import time

# ---------------------------------------------------------------------------
# Bootstrap: install openenv mock if the real package is absent.
# This lets the scripts run without openenv-core being installed.
# ---------------------------------------------------------------------------


def _ensure_openenv_mock() -> None:
    """Install a lightweight openenv mock into sys.modules if needed."""
    try:
        import openenv.core.env_server  # noqa: F401
        return  # real package present, nothing to do
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
# Import environment directly (no openenv client needed)
# ---------------------------------------------------------------------------

try:
    from dataclean_env.server.environment import DataCleanEnvironment
    from dataclean_env.models import DataCleanAction
except ImportError as exc:
    print(f"FAIL: Could not import DataClean-Env modules: {exc}")
    print("Hint: run from the project root with the package installed or on PYTHONPATH.")
    sys.exit(1)


TASK_IDS = ["easy_contacts", "medium_employees", "hard_patients"]


def _make_action(action_type: str, **params: object) -> DataCleanAction:
    return DataCleanAction(action_type=action_type, params=params)


def run_smoke_test() -> bool:
    """Run smoke test across all 3 tasks. Returns True on success."""
    all_passed = True
    start = time.monotonic()

    for task_id in TASK_IDS:
        print(f"\n--- Task: {task_id} ---")

        # 1. Reset
        env = DataCleanEnvironment()
        obs = env.reset(seed=42, task_id=task_id)
        assert obs.done is False, f"Expected done=False after reset for {task_id}"
        assert obs.row_count > 0, f"Expected rows > 0 after reset for {task_id}"
        print(f"  Reset OK: {obs.row_count} rows, {len(obs.columns)} columns")

        # 2. One fix_value step (pick first row, first non-internal column)
        first_row_id = obs.rows[0][0] if obs.rows else 0  # _row_id is typically first
        # Use a column from the observation
        target_col = obs.columns[1] if len(obs.columns) > 1 else obs.columns[0]
        obs = env.step(_make_action(
            "fix_value",
            row_id=first_row_id,
            column=target_col,
            new_value="smoke_test_value",
        ))
        assert obs.done is False, f"Expected done=False after fix_value for {task_id}"
        print(f"  fix_value step OK: reward={obs.reward}")

        # 3. mark_complete
        obs = env.step(_make_action("mark_complete"))
        assert obs.done is True, f"Expected done=True after mark_complete for {task_id}"

        score = obs.reward
        print(f"  mark_complete OK: score={score}")

        # 4. Verify score is in valid range
        if not isinstance(score, (int, float)):
            print(f"  FAIL: score is not numeric: {type(score)}")
            all_passed = False
            continue

        if not (0.0 <= score <= 1.0):
            print(f"  FAIL: score {score} out of range [0.0, 1.0]")
            all_passed = False
        else:
            print(f"  PASS: score {score:.4f} in [0.0, 1.0]")

    elapsed = time.monotonic() - start
    print(f"\nCompleted in {elapsed:.2f}s")
    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    if success:
        print("\nAll smoke tests PASSED.")
        sys.exit(0)
    else:
        print("\nSome smoke tests FAILED.")
        sys.exit(1)
