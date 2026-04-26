"""Colab-ready orchestrator: prepare → state scope → train → eval.

Designed to be run end-to-end on a Colab Pro notebook with a CUDA GPU. The
script chains the existing modules (no new training logic here) and prints
an honest pre-run summary so the human driving Colab sees exactly:

    * what is being trained
    * what prompt version the dataset was generated under
    * which rubric axes SFT is expected to improve
    * which axes are structurally out of SFT's reach and why

After training, replays the held-out eval split and prints per-task
trained-vs-scripted deltas with the same caveat.

Colab recipe
------------
::

    !git clone <repo-url> /content/METAHackthon2026
    %cd /content/METAHackthon2026
    !pip install -e '.[train]'
    !python -m medibill.sft_colab \\
        --dataset datasets/sft_v2.jsonl \\
        --eval traces/eval.jsonl \\
        --out adapters/sft_v2/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medibill.models import AGENT_ACTION_TYPES
from medibill.prompting import PROMPT_VERSION


_DATASET_SCOPE_BANNER = (
    "─" * 72 + "\n"
    "SFT scope (spec v3 §7.6):\n"
    "  SFT will train on the core workflow backbone. Expect improvements on\n"
    "    - final_correctness\n"
    "    - policy_compliance\n"
    "    - process_auditability\n"
    "    - efficiency (partial)\n"
    "  SFT is NOT expected to improve:\n"
    "    - abstention_quality  (scripted never escalates; ambiguous-cell\n"
    "                           ground truth is hidden from the agent)\n"
    "    - drift_bonus         (scripted detects drift by schedule, not by\n"
    "                           policy staleness; true drift reasoning is\n"
    "                           an RL target)\n"
    "  On hard_drift specifically, the structural SFT ceiling is ≈ 0.80.\n"
    + "─" * 72
)


def _print_dataset_summary(dataset_path: Path) -> None:
    """Print coverage + PROMPT_VERSION for the dataset about to be trained."""
    print(f"PROMPT_VERSION being trained: {PROMPT_VERSION}")
    print(f"Dataset: {dataset_path}")
    if not dataset_path.exists():
        print(f"  [ERROR] dataset not found")
        raise SystemExit(1)
    n = 0
    action_counts: dict[str, int] = {}
    policies: set[str] = set()
    prompt_versions: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            at = row.get("action_type")
            if at is not None:
                action_counts[at] = action_counts.get(at, 0) + 1
            pol = row.get("policy")
            if pol is not None:
                policies.add(pol)
            pv = row.get("prompt_version")
            if pv is not None:
                prompt_versions.add(pv)
    print(f"  total examples:   {n}")
    print(f"  policies in data: {sorted(policies)}")
    print(f"  action_type coverage: {dict(sorted(action_counts.items()))}")

    if prompt_versions and prompt_versions != {PROMPT_VERSION}:
        print(
            f"  [ERROR] dataset prompt_version(s) = {sorted(prompt_versions)} "
            f"do not match installed PROMPT_VERSION = {PROMPT_VERSION}. "
            f"Regenerate dataset with the current prompt before training."
        )
        raise SystemExit(1)

    missing = [t for t in AGENT_ACTION_TYPES if t not in action_counts]
    if missing:
        print(
            f"  [WARN] tools with zero supervised examples: {missing}. "
            f"SFT cannot teach these. See spec v3 §7.6."
        )
    print(_DATASET_SCOPE_BANNER)


def _run_training(args: argparse.Namespace) -> int:
    from medibill.train_sft import train  # deferred — uses torch/unsloth on CUDA

    print("\n>>> Training LoRA adapter...")
    return train(
        dataset_path=args.dataset,
        model=args.model,
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
    )


def _run_eval(args: argparse.Namespace) -> int:
    try:
        import torch  # noqa: WPS433
    except ImportError:
        print(
            "[SKIP] torch not installed. Eval requires CUDA runtime.",
            file=sys.stderr,
        )
        return 2
    if not torch.cuda.is_available():
        print("[SKIP] No CUDA device for eval.", file=sys.stderr)
        return 3

    from medibill.evaluate_sft import evaluate_against_eval_split, make_hf_agent

    print(f"\n>>> Evaluating trained adapter on {args.eval}")

    # Adapter was trained with Unsloth's FastLanguageModel, which patches the
    # base attention layers (adds `apply_qkv` etc). Loading the LoRA adapter
    # onto a plain transformers model breaks at `.generate()`. Use Unsloth for
    # inference too so the adapter sees the same patched modules it was
    # trained on.
    try:
        from unsloth import FastLanguageModel  # noqa: WPS433
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.out),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[WARN] Unsloth inference path failed ({exc}); "
            f"falling back to plain transformers.",
            file=sys.stderr,
        )
        from peft import PeftModel  # noqa: WPS433
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433
        from medibill._gpu import preferred_torch_dtype  # noqa: WPS433
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=preferred_torch_dtype(),
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, args.out)
        model.eval()

    agent = make_hf_agent(model, tokenizer, max_new_tokens=args.max_new_tokens)
    report = evaluate_against_eval_split(args.eval, agent)

    print("\nPer-task trained vs scripted (held-out eval split):")
    for task_id, row in report["per_task"].items():
        direction = "↑" if row["delta"] > 0 else ("↓" if row["delta"] < 0 else "=")
        print(
            f"  {task_id:22s} n={row['n']}  "
            f"trained={row['trained_mean']:.3f}  "
            f"scripted={row['scripted_mean']:.3f}  "
            f"Δ={row['delta']:+.3f} {direction}"
        )
    pf = report["total_parse_failures"]
    if pf:
        print(
            f"\n  LM output parse failures: {pf} "
            f"(each counted as an invalid action; reduces score honestly)."
        )
    print(f"\nCaveat: {report['caveat']}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Colab orchestrator: prints scope, trains LoRA adapter, evaluates "
            "on held-out eval split. Requires CUDA."
        ),
    )
    parser.add_argument("--dataset", type=Path, default=Path("datasets/sft_v2.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("traces/eval.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("adapters/sft_v2/"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use an already-trained adapter).")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip eval (only prepare + train).")
    args = parser.parse_args()

    _print_dataset_summary(args.dataset)

    if not args.skip_train:
        exit_code = _run_training(args)
        if exit_code != 0:
            print(
                f"\n[ABORT] training returned {exit_code}; skipping eval.",
                file=sys.stderr,
            )
            raise SystemExit(exit_code)

    if not args.skip_eval:
        exit_code = _run_eval(args)
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
