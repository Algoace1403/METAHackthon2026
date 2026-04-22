"""Trajectory generator for MediBill-Env.

Runs a reference policy across seeds/tasks and dumps JSONL trajectories for
downstream SFT / RFT filtering. Each trajectory record contains per-step
``(observation_text, action, reward, done)`` tuples plus the terminal grade.

Usage:
    python -m medibill.generate_trajectories \\
        --task easy_cashless --seeds 5 --out traces/easy.jsonl

    python -m medibill.generate_trajectories \\
        --task all --seeds 20 --out traces/all.jsonl

Transport: this script calls ``MediBillEnvironment`` directly (no HTTP). That
matches how TRL's ``environment_factory`` runs the env during GRPO training
and avoids the per-request setup overhead of the HTTP/WebSocket path. The
HTTP client (:mod:`medibill.client`) is separately exercised in the client
smoke test; both paths share the same Pydantic models, so the prompt and
action formats generated here are directly usable through either transport.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from medibill.baselines import ScriptedHeuristicPolicy, no_op_agent, random_agent
from medibill.models import MediBillAction, MediBillObservation
from medibill.prompting import PROMPT_VERSION, SYSTEM_PROMPT, format_observation
from medibill.server.environment import MediBillEnvironment
from medibill.server.grader import GradeResult, MediBillGrader


TASKS_ALL = ("easy_cashless", "medium_multi_payer", "hard_drift")

# Policy registry — name -> zero-arg factory returning a callable
# ``agent(obs, env, rng) -> MediBillAction``.
POLICIES: Dict[str, Any] = {
    "scripted_heuristic": ScriptedHeuristicPolicy,
    "no_op":              lambda: no_op_agent,
    "random":             lambda: random_agent,
}


def _serialize_action(action: MediBillAction) -> Dict[str, Any]:
    return {"action_type": action.action_type, "params": action.params}


def _grade(env: MediBillEnvironment) -> GradeResult:
    grader = MediBillGrader()
    return grader.grade(
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


def run_trajectory(
    policy_name: str,
    task_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Run a single episode of *policy_name* on *task_id* @ *seed*.

    Returns a trajectory record ready to serialise as one JSONL line.
    """
    if policy_name not in POLICIES:
        raise ValueError(
            f"Unknown policy '{policy_name}'. Known: {list(POLICIES)}"
        )
    agent = POLICIES[policy_name]()

    env = MediBillEnvironment()
    obs: MediBillObservation = env.reset(seed=seed, task_id=task_id)
    rng = random.Random(seed)

    steps: List[Dict[str, Any]] = []
    total_reward = 0.0
    safety = env.state.max_steps + 5

    while not bool(obs.done) and safety > 0:
        # Capture the observation_step_number BEFORE env.step() mutates state.
        # observation_text and observation_step_number both describe the env
        # at the moment the agent made its decision; result_step_number is
        # the env's step count AFTER the action has been applied.
        observation_step_number = int(obs.step_number)
        observation_text = format_observation(obs)
        action = agent(obs, env, rng)
        next_obs = env.step(action)
        reward = float(next_obs.reward) if next_obs.reward is not None else 0.0
        total_reward += reward
        steps.append({
            "observation_step_number": observation_step_number,
            "result_step_number": int(next_obs.step_number),
            "observation_text": observation_text,
            "action": _serialize_action(action),
            "reward": reward,
            "done": bool(next_obs.done),
        })
        obs = next_obs
        safety -= 1

    grade = _grade(env)
    return {
        "trajectory_id": str(uuid.uuid4()),
        "task_id": task_id,
        "seed": seed,
        "policy": policy_name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Prompt handshake: downstream trainers MUST reject trajectories whose
        # prompt_version does not equal medibill.prompting.PROMPT_VERSION —
        # that indicates the system prompt has changed since generation and
        # the SFT pairs no longer match inference-time conditioning.
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "final_score": grade.score,
        "total_reward": round(total_reward, 4),
        "n_steps": len(steps),
        "done": bool(obs.done),
        "grade": {
            "score": grade.score,
            "penalties": grade.penalties,
            "bonuses": grade.bonuses,
            "axes": [asdict(ax) for ax in grade.axes],
            "penalty_breakdown": grade.penalty_breakdown,
            "bonus_breakdown": grade.bonus_breakdown,
        },
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MediBill-Env trajectories (one per JSONL line)."
    )
    parser.add_argument(
        "--task",
        default="easy_cashless",
        help="Task id, or 'all' to run every task. Default: easy_cashless.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds per task. Default: 3.",
    )
    parser.add_argument(
        "--policy",
        default="scripted_heuristic",
        choices=sorted(POLICIES),
        help="Reference policy to drive rollouts.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Start seed index (seeds are seed_offset..seed_offset+seeds-1).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL path. Parent directory will be created.",
    )
    args = parser.parse_args()

    task_ids = TASKS_ALL if args.task == "all" else (args.task,)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    scores_by_task: Dict[str, List[float]] = {tid: [] for tid in task_ids}
    with args.out.open("w", encoding="utf-8") as fh:
        for tid in task_ids:
            for seed in range(args.seed_offset, args.seed_offset + args.seeds):
                traj = run_trajectory(args.policy, tid, seed)
                fh.write(json.dumps(traj) + "\n")
                scores_by_task[tid].append(traj["final_score"])
                n_written += 1
                print(
                    f"  {tid} seed={seed:>3}  "
                    f"score={traj['final_score']:.3f}  "
                    f"steps={traj['n_steps']:>3}  "
                    f"total_reward={traj['total_reward']:+.3f}"
                )

    print()
    print(f"Wrote {n_written} trajectory(ies) to {args.out}")
    for tid, scores in scores_by_task.items():
        if scores:
            mean = sum(scores) / len(scores)
            lo, hi = min(scores), max(scores)
            print(
                f"  {tid:22s} n={len(scores)}  mean={mean:.3f}  "
                f"range=[{lo:.3f}, {hi:.3f}]"
            )


if __name__ == "__main__":
    main()
