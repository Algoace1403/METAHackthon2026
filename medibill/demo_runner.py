"""Live narratable demo of MediBill-Env on one hard_drift episode.

Runs a tool-faithful scripted policy against ``hard_drift`` and prints every
step in a screen-recording-friendly format: clear boundaries, coloured step
labels, and a deliberately loud annotation the moment the silent drift
fires. Output is designed to be readable at normal speaking speed and to
produce a 60–90 second demo video when recorded live.

Usage::

    python -m medibill.demo_runner                 # seed 44 by default
    python -m medibill.demo_runner --seed 7        # different drift step
    python -m medibill.demo_runner --no-color      # plain text for capture tools that mangle ANSI

The policy is :class:`medibill.baselines.ScriptedHeuristicPolicy` — the
same tool-faithful baseline used by the separation gate. No trained
adapter is required; this script runs on any machine in under 30 seconds.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List

from medibill.baselines import ScriptedHeuristicPolicy
from medibill.models import MediBillAction
from medibill.server.environment import MediBillEnvironment
from medibill.server.grader import MediBillGrader


# ANSI colour helpers — kept minimal so the --no-color path is a clean no-op.
class Ansi:
    def __init__(self, enabled: bool) -> None:
        self._e = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self._e else text

    def bold(self, t: str) -> str:   return self._wrap("1", t)
    def dim(self, t: str) -> str:    return self._wrap("2", t)
    def red(self, t: str) -> str:    return self._wrap("31", t)
    def green(self, t: str) -> str:  return self._wrap("32", t)
    def yellow(self, t: str) -> str: return self._wrap("33", t)
    def blue(self, t: str) -> str:   return self._wrap("34", t)
    def mag(self, t: str) -> str:    return self._wrap("35", t)
    def cyan(self, t: str) -> str:   return self._wrap("36", t)


def _format_action_summary(action: MediBillAction) -> str:
    params = action.params
    t = action.action_type
    if t == "ehr_query":
        return f"{t}(claim_id={params.get('claim_id', '?')[:18]})"
    if t == "insurance_lookup":
        return f"{t}(provider={params.get('provider', '?')})"
    if t == "coding_engine":
        cid = params.get("claim_id", "?")[:18]
        field = params.get("field", "?")
        val = params.get("value", "?")
        val_str = repr(val)
        if len(val_str) > 32:
            val_str = val_str[:29] + "...'"
        return f"{t}(claim={cid}, field={field}, value={val_str})"
    if t == "escalate_to_human":
        return f"{t}(claim={params.get('claim_id', '?')[:18]}, field={params.get('field', '?')})"
    if t == "submit_claim":
        return f"{t}(claim={params.get('claim_id', '?')[:18]})"
    return f"{t}({params})"


def _grade(env: MediBillEnvironment) -> Dict[str, float]:
    result = MediBillGrader().grade(
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
    return {
        "score": result.score,
        "penalties": result.penalties,
        "bonuses": result.bonuses,
        "axes": {ax.name: (ax.raw, ax.effective, ax.contribution) for ax in result.axes},
    }


def run_demo(seed: int, color: bool, max_narrated_steps: int | None) -> int:
    c = Ansi(color)
    env = MediBillEnvironment()
    obs = env.reset(seed=seed, task_id="hard_drift")
    agent = ScriptedHeuristicPolicy()
    rng = random.Random(seed)

    drift_step = env.state.drift_events_pending[0]["step"] if env.state.drift_events_pending else None
    target_version = env.state.drift_events_pending[0]["to_version"] if env.state.drift_events_pending else None

    print()
    print(c.bold("═" * 72))
    print(c.bold(f"  MediBill-Env — live demo of silent policy-drift mechanic"))
    print(c.bold("═" * 72))
    print(f"  Task:        {c.cyan('hard_drift')}  (seed={seed})")
    print(f"  Provider:    {c.cyan(env.state.provider)}")
    print(f"  Initial policy version:  {c.green(env.state.active_policy_version)}")
    print(f"  Claims:      {len(env.state.claims)}    "
          f"Budget:  {env.state.budget}    Max steps: {env.state.max_steps}")
    if drift_step is not None:
        print(f"  {c.yellow(f'Scheduled drift: step {drift_step} → policy will silently change to {target_version}')}")
    print(c.dim("  (drift is not announced by the env. Only the scripted policy's next"))
    print(c.dim("   insurance_lookup reveals it.)"))
    print()

    # Run episode
    drift_announced = False
    last_observed_version = env.state.active_policy_version
    step_idx = 0
    while not bool(obs.done):
        step_idx += 1
        action = agent(obs, env, rng)

        # Fire the step
        obs = env.step(action)
        last_tool = obs.last_tool_result
        status = last_tool.status if last_tool else "?"

        # If drift just fired (but no one's seen it yet), call it out loudly.
        fired = [d for d in env.state.drift_events_fired if d.step == step_idx]
        if fired and not drift_announced:
            print()
            print(c.red(c.bold(
                f"  *** DRIFT FIRED SILENTLY at step {step_idx}: "
                f"{fired[0].from_version} → {fired[0].to_version} ***"
            )))
            print(c.dim(
                "      (no observation field announces this — the agent only learns"
                " via a fresh insurance_lookup call)"))
            print()
            drift_announced = True

        # If the agent just did an insurance_lookup, show what version it got back
        if last_tool and last_tool.tool == "insurance_lookup" and status == "success":
            new_version = last_tool.payload.get("rules", {}).get("policy_version", "?")
            if new_version != last_observed_version and last_observed_version:
                print(c.green(c.bold(
                    f"  ! Agent re-queried policy — version changed "
                    f"{last_observed_version} → {new_version}. Agent has detected drift."
                )))
                print()
            last_observed_version = new_version

        # Compact step line
        tag = (c.green("OK") if status == "success"
               else c.red("ERR") if status == "error"
               else c.yellow(status[:3].upper()))
        action_str = _format_action_summary(action)
        remaining = env.state.max_steps - step_idx
        budget = obs.budget_remaining

        line = (
            f"  step {step_idx:>3}  [{tag}]  "
            f"{action_str:<58}  "
            f"budget={budget:>6.1f}  unsubmitted={obs.claims_remaining}"
        )
        if max_narrated_steps is None or step_idx <= max_narrated_steps:
            print(line)
        elif step_idx == max_narrated_steps + 1:
            print(c.dim(f"  ... (suppressing steps {step_idx}–; --max-narrated-steps to adjust)"))

    # Grade
    grade = _grade(env)
    axes = grade["axes"]

    print()
    print(c.bold("─" * 72))
    print(c.bold("  Episode complete."))
    print(c.bold("─" * 72))
    print(f"  Total steps:        {step_idx}")
    print(f"  Budget spent:       {env.state.budget_spent:.1f}/{env.state.budget}")
    print(f"  Drift events fired: {len(env.state.drift_events_fired)}")
    for d in env.state.drift_events_fired:
        print(f"    · step {d.step}: {d.from_version} → {d.to_version}")
    print(f"  Submitted claims:   {len(env.state.submitted_claim_ids)}/{len(env.state.ground_truth)}")

    print()
    print(c.bold("  Grader breakdown:"))
    for name, (raw, eff, contrib) in axes.items():
        eff_str = f"{eff:.3f}" if eff != raw else f"{raw:.3f}"
        print(f"    {name:<22}  raw={raw:.3f}  effective={eff_str}  contribution={contrib:+.3f}")
    print(f"    penalties              {grade['penalties']:+.3f}")
    print(f"    bonuses                {grade['bonuses']:+.3f}")
    print()
    print(c.bold(f"  FINAL COMPOSITE SCORE:  {c.cyan(f'{grade['score']:.3f}')}"))
    print(c.dim("  (scripted baseline on hard_drift scores ≈ 0.75. That score is the cost"))
    print(c.dim("   of carrying a stale policy model into submit, not a sign of recovery —"))
    print(c.dim("   it is the behavioral gap the training pipeline is designed to target."))
    print(c.dim("   See docs/round2-spec-v3.md §7.6 for the SFT target-coverage scope.)"))
    print()

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Narratable MediBill-Env demo runner for screen recording.",
    )
    parser.add_argument("--seed", type=int, default=44,
                        help="Episode seed. Different seeds produce different drift steps.")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colour (useful if the terminal or capture tool mangles codes).")
    parser.add_argument(
        "--max-narrated-steps", type=int, default=40,
        help="Show detailed per-step lines for the first N steps, elide afterwards. "
             "Default 40 keeps the demo under ~60 seconds at normal speaking speed.",
    )
    args = parser.parse_args()

    color_enabled = not args.no_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    raise SystemExit(run_demo(
        seed=args.seed,
        color=color_enabled,
        max_narrated_steps=args.max_narrated_steps,
    ))


if __name__ == "__main__":
    main()
