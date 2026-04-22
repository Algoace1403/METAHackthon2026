"""Grader score-separation gate.

Runs three reference policies across multiple seeds and prints the composite
score distribution. Spec v3 §6 requires:

    scripted_heuristic - random >= 0.20  (per task)

If this gate fails the grader is broken and all downstream training is
wasted. Run this BEFORE any SFT/RFT trajectory generation.

Usage:
    python -m medibill.validate_grader --episodes 20

Increase --episodes for the real pre-training check (target 100+).
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Callable

from medibill.baselines import (
    ScriptedHeuristicPolicy,
    no_op_agent,
    random_agent,
    run_episode,
)
from medibill.server.grader import MediBillGrader


@dataclass
class PolicyStats:
    name: str
    task_id: str
    scores: list[float]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.scores) if self.scores else 0.0

    @property
    def stdev(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        return statistics.stdev(self.scores)


def _score_episode(env, task_id: str) -> float:
    grader = MediBillGrader()
    result = grader.grade(
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
    return result.score


def run_policy(
    name: str,
    agent_factory: Callable,
    task_id: str,
    n_episodes: int,
) -> PolicyStats:
    scores: list[float] = []
    for seed in range(n_episodes):
        agent = agent_factory() if callable(agent_factory) else agent_factory
        env, _ = run_episode(agent, task_id, seed=seed)
        scores.append(_score_episode(env, task_id))
    return PolicyStats(name=name, task_id=task_id, scores=scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per (policy, task). 100+ for the real gate.")
    parser.add_argument("--task", default="easy_cashless",
                        choices=["easy_cashless", "medium_multi_payer", "hard_drift", "all"])
    args = parser.parse_args()

    tasks = ["easy_cashless", "medium_multi_payer", "hard_drift"] if args.task == "all" else [args.task]

    policies = [
        ("random",             lambda: random_agent),
        ("no_op",              lambda: no_op_agent),
        ("scripted_heuristic", lambda: ScriptedHeuristicPolicy()),
    ]

    any_fail = False
    for task_id in tasks:
        print(f"\n=== Task: {task_id} (n={args.episodes}) ===")
        stats: dict[str, PolicyStats] = {}
        for name, factory in policies:
            s = run_policy(name, factory, task_id, args.episodes)
            stats[name] = s
            print(f"  {name:22s} mean={s.mean:.3f}  stdev={s.stdev:.3f}  "
                  f"min={min(s.scores):.3f}  max={max(s.scores):.3f}")
        separation = stats["scripted_heuristic"].mean - stats["random"].mean
        gate = "PASS" if separation >= 0.20 else "FAIL"
        print(f"  -> scripted - random = {separation:+.3f}  gate(>=0.20) = {gate}")
        if gate == "FAIL":
            any_fail = True

    print()
    if any_fail:
        print("GRADER SEPARATION GATE FAILED — fix grader or baselines before training.")
        raise SystemExit(1)
    print("GRADER SEPARATION GATE PASSED.")


if __name__ == "__main__":
    main()
