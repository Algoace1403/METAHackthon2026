"""Colab setup smoke test — run before burning a full training session.

Checks: package imports, PROMPT_VERSION is set, dataset and eval files exist
and are non-empty, the first dataset row has the expected chat shape with
matching prompt_version, and a CUDA device is (or isn't) available. Optional
plumbing-mode eval exercises the eval harness without training.

Usage:
    python -m medibill.colab_smoke
    python -m medibill.colab_smoke --with-plumbing-eval

Exits 0 on success, 1 on any hard failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATASET = Path("datasets/sft_v1.jsonl")
EVAL = Path("traces/eval.jsonl")


def _count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def main() -> int:
    failures: list[str] = []

    try:
        import medibill  # noqa: WPS433
        from medibill.prompting import PROMPT_VERSION  # noqa: WPS433
        print(
            f"[OK] medibill v{getattr(medibill, '__version__', '?')}; "
            f"PROMPT_VERSION={PROMPT_VERSION}"
        )
    except Exception as exc:
        print(f"[FAIL] medibill import: {exc}")
        return 1

    n_ds = _count_lines(DATASET)
    if n_ds > 0:
        print(f"[OK] dataset {DATASET}: {n_ds} rows")
    else:
        failures.append(f"dataset missing or empty: {DATASET}")

    n_eval = _count_lines(EVAL)
    if n_eval > 0:
        print(f"[OK] eval {EVAL}: {n_eval} trajectories")
    else:
        failures.append(f"eval missing or empty: {EVAL}")

    if n_ds:
        try:
            with DATASET.open("r", encoding="utf-8") as fh:
                row = json.loads(fh.readline())
            msgs = row["messages"]
            assert len(msgs) == 3
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
            assert row["prompt_version"] == PROMPT_VERSION
            json.loads(msgs[-1]["content"])  # assistant is valid JSON
            print("[OK] first dataset row: chat shape valid + prompt_version matches")
        except Exception as exc:
            failures.append(f"first dataset row malformed: {exc}")

    try:
        import torch  # noqa: WPS433
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "cpu-only"
        print(f"[OK] torch {torch.__version__}; cuda={cuda}; device={dev}")
    except ImportError:
        print("[WARN] torch not installed — SFT will fail. pip install '.[train]'")

    if "--with-plumbing-eval" in sys.argv:
        try:
            from medibill.evaluate_sft import (  # noqa: WPS433
                evaluate_against_eval_split,
                make_scripted_agent,
            )
            report = evaluate_against_eval_split(EVAL, make_scripted_agent())
            tasks = sorted(report["per_task"].keys())
            print(f"[OK] plumbing eval ran across tasks: {tasks}")
        except Exception as exc:
            failures.append(f"plumbing eval failed: {exc}")

    if failures:
        print(f"\n[FAIL] {len(failures)} setup issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\n[OK] Colab smoke pass — ready to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
