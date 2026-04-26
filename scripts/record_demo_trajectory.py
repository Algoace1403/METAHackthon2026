"""Record a single hard_drift episode under ScriptedDriftAwarePolicy and
serialise every step to JSON for the static demo viewer.

Output: ``docs/assets/demo_trajectory.json``

The JSON shape is:

    {
      "task_id": "hard_drift",
      "seed": 42,
      "policy": "ScriptedDriftAwarePolicy",
      "drift_events": [{"step": 14, "from": "v1.3", "to": "v1.4"}, ...],
      "n_claims": 12,
      "final_score": 0.9996,
      "steps": [
        {
          "step": 0,
          "action": {"tool": "insurance_lookup", "params": {...}},
          "result": {"status": "success", "summary": "..."},
          "active_policy": "v1.3",
          "drift_fired": false,
          "submitted": [],
          "score_so_far": 0.0
        },
        ...
      ]
    }

Run::

    python scripts/record_demo_trajectory.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from medibill.baselines import ScriptedDriftAwarePolicy
from medibill.models import MediBillAction
from medibill.server.environment import MediBillEnvironment
from medibill.server.grader import MediBillGrader

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "demo_trajectory.json"


def _action_summary(action: MediBillAction) -> dict[str, Any]:
    return {
        "tool": action.action_type,
        "params": dict(action.params),
    }


def _result_summary(env: MediBillEnvironment) -> dict[str, Any]:
    last = env.state.tool_log[-1] if env.state.tool_log else {}
    return {
        "status": last.get("status"),
        "message": last.get("message", "")[:160],
        "cost": last.get("cost", 0.0),
    }


def _grade_now(env: MediBillEnvironment) -> float:
    g = MediBillGrader()
    res = g.grade(
        ground_truth=env.state.ground_truth,
        final_claims=env.state.claims,
        submitted_claim_ids=env.state.submitted_claim_ids,
        tool_log=env.state.tool_log,
        lookup_history=env.state.lookup_history,
        drift_events_fired=[d.model_dump() for d in env.state.drift_events_fired],
        provider=env.state.provider,
        ambiguous_cells=env.state.ambiguous_cells,
        budget=env.state.budget,
        budget_spent=env.state.budget_spent,
        active_policy_version=env.state.active_policy_version,
    )
    return float(res.score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="hard_drift")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    env = MediBillEnvironment()
    obs = env.reset(seed=args.seed, task_id=args.task)
    policy = ScriptedDriftAwarePolicy()
    rng = random.Random(args.seed)

    steps: list[dict[str, Any]] = []
    drift_seen: set[int] = set()

    while not env.state.is_complete and env.state.step_count < env.state.max_steps:
        action = policy(obs, env, rng)
        prev_drift_count = len(env.state.drift_events_fired)
        obs = env.step(action)
        new_drift_count = len(env.state.drift_events_fired)
        drift_fired_this_step = new_drift_count > prev_drift_count

        if drift_fired_this_step:
            for d in env.state.drift_events_fired[prev_drift_count:]:
                drift_seen.add(d.step)

        steps.append({
            "step": env.state.step_count,
            "action": _action_summary(action),
            "result": _result_summary(env),
            "active_policy": env.state.active_policy_version,
            "drift_fired_this_step": drift_fired_this_step,
            "submitted_count": len(env.state.submitted_claim_ids),
            "score_so_far": round(_grade_now(env), 4),
            "budget_remaining": round(env.state.budget_remaining, 2),
        })

    final_score = _grade_now(env)
    payload = {
        "task_id": args.task,
        "seed": args.seed,
        "policy": "ScriptedDriftAwarePolicy",
        "drift_events": [
            {"step": d.step, "from": d.from_version, "to": d.to_version}
            for d in env.state.drift_events_fired
        ],
        "n_claims": len(env.state.claims),
        "n_steps": env.state.step_count,
        "final_score": round(final_score, 4),
        "submitted_claim_ids": list(env.state.submitted_claim_ids),
        "steps": steps,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Recorded {env.state.step_count} steps. Final score = {final_score:.4f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
