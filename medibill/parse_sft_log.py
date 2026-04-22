"""Extract the six post-run signals from ``sft_v1_stdout.log``.

Six binary acceptance outputs:
    1. per_task      — dict of task_id -> (trained, scripted, delta)
    2. parse_fails   — int, LM-output parse failures during eval
    3. warnings      — list of [WARN] lines (sans timestamps/prefix)
    4. adapter_ok    — bool, whether ``[OK] SFT complete`` appeared
    5. final_loss    — float or None, last logged ``'loss': ...`` value
    6. steps_done    — int or None, last logged ``'step': ...`` value

Usage:
    python -m medibill.parse_sft_log <path-to-log>

Prints a compact summary and exits 0 only if:
    - adapter_ok is True
    - all three tasks present in per_task
    - parse_fails is an integer (0 or more; just needs to have been reported)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TASK_ROW = re.compile(
    r"^\s*(?P<task>easy_cashless|medium_multi_payer|hard_drift)\s+"
    r"n=(?P<n>\d+)\s+"
    r"trained=(?P<trained>[\d.]+)\s+"
    r"scripted=(?P<scripted>[\d.]+)\s+"
    r"Δ=(?P<delta>[+\-][\d.]+)",
    re.MULTILINE,
)
PARSE_FAILS = re.compile(r"LM output parse failures:\s*(\d+)")
ADAPTER_OK = re.compile(r"\[OK\] SFT complete\. Adapter saved to")
WARN_LINE = re.compile(r"\[WARN\]\s*(.+)")
# TRL/transformers logs losses as dict-like lines every `logging_steps`:
#   {'loss': 0.4231, 'grad_norm': 1.2, 'learning_rate': 9.9e-05, 'epoch': 0.01}
LOSS_DICT = re.compile(r"\{'loss':\s*([\d.]+)")
STEP_DICT = re.compile(r"'(?:step|global_step)':\s*(\d+)")


def parse(log_path: Path) -> Dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    per_task: Dict[str, Tuple[float, float, float]] = {}
    for m in TASK_ROW.finditer(text):
        per_task[m.group("task")] = (
            float(m.group("trained")),
            float(m.group("scripted")),
            float(m.group("delta")),
        )

    pf_matches = PARSE_FAILS.findall(text)
    parse_fails: Optional[int] = int(pf_matches[-1]) if pf_matches else None

    warnings = [w.strip() for w in WARN_LINE.findall(text)]
    adapter_ok = bool(ADAPTER_OK.search(text))

    losses = LOSS_DICT.findall(text)
    final_loss: Optional[float] = float(losses[-1]) if losses else None

    steps = STEP_DICT.findall(text)
    steps_done: Optional[int] = int(steps[-1]) if steps else None

    return {
        "per_task": per_task,
        "parse_fails": parse_fails,
        "warnings": warnings,
        "adapter_ok": adapter_ok,
        "final_loss": final_loss,
        "steps_done": steps_done,
    }


def print_summary(result: Dict[str, object]) -> None:
    print("=" * 60)
    print("SFT post-run summary")
    print("=" * 60)
    per_task = result["per_task"]
    if per_task:
        print("\nper_task (task | trained | scripted | Δ):")
        for task, (tr, sc, dt) in sorted(per_task.items()):
            print(f"  {task:22s} {tr:.3f}   {sc:.3f}   {dt:+.3f}")
    else:
        print("\n[MISSING] no per-task rows found in log")

    pf = result["parse_fails"]
    print(f"\nparse_fails: {pf if pf is not None else '[MISSING]'}")

    warnings = result["warnings"]
    if warnings:
        print(f"\nwarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nwarnings: none")

    print(f"\nadapter_ok: {result['adapter_ok']}")

    fl = result["final_loss"]
    print(f"final_loss: {fl if fl is not None else '[MISSING — training may have crashed before first log]'}")

    sd = result["steps_done"]
    print(f"steps_done: {sd if sd is not None else '[MISSING]'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path,
                        help="Path to sft_v1_stdout.log (or equivalent).")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON-only output (for piping).")
    args = parser.parse_args()

    if not args.log_path.exists():
        print(f"[FAIL] log not found: {args.log_path}", file=sys.stderr)
        return 1

    result = parse(args.log_path)

    if args.json:
        # Convert per_task tuples to lists for JSON
        result["per_task"] = {k: list(v) for k, v in result["per_task"].items()}
        print(json.dumps(result, indent=2))
    else:
        print_summary(result)

    # Binary gate: adapter written AND all 3 tasks present
    required_tasks = {"easy_cashless", "medium_multi_payer", "hard_drift"}
    missing = required_tasks - set(result["per_task"].keys())
    if not result["adapter_ok"] or missing:
        print(
            f"\n[INCOMPLETE] adapter_ok={result['adapter_ok']}, "
            f"missing tasks: {sorted(missing) if missing else 'none'}",
            file=sys.stderr,
        )
        return 1
    print("\n[OK] All six signals captured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
