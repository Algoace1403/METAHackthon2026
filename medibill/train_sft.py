"""SFT pipeline for MediBill-Env.

Two subcommands:

    prepare  Load one or more trajectory JSONL files, validate each record
             against the locked schema, REJECT any record whose
             ``prompt_version`` does not equal the installed package's
             ``medibill.prompting.PROMPT_VERSION``, optionally filter by
             final_score and/or policy, flatten each step into a
             chat-format training example, and write a dataset JSONL
             compatible with ``datasets.load_dataset("json", ...)``.

    train    Run Unsloth + TRL SFTTrainer on a prepared dataset. Guarded
             on CUDA availability; prints the equivalent Colab recipe
             when no GPU is present.

The prompt-version guard is the central safety property: we will NOT train
on trajectories whose system prompt has drifted from the one installed in
the package. The guard is a hard exception, not a warning.

Usage::

    python -m medibill.train_sft prepare \\
        --traces traces/scripted.jsonl traces/contrast.jsonl \\
        --out datasets/sft_v1.jsonl \\
        --min-score 0.70

    python -m medibill.train_sft train \\
        --dataset datasets/sft_v1.jsonl \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --out adapters/sft_v1/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from medibill._gpu import preferred_dtype_flags
from medibill.models import AGENT_ACTION_TYPES
from medibill.prompting import PROMPT_VERSION


# ---------------------------------------------------------------------------
# Schema contract (matches medibill.generate_trajectories output)
# ---------------------------------------------------------------------------

REQUIRED_TRAJ_KEYS: frozenset[str] = frozenset({
    "trajectory_id",
    "task_id",
    "seed",
    "policy",
    "prompt_version",
    "system_prompt",
    "final_score",
    "n_steps",
    "steps",
})

REQUIRED_STEP_KEYS: frozenset[str] = frozenset({
    "observation_step_number",
    "result_step_number",
    "observation_text",
    "action",
    "reward",
    "done",
})

REQUIRED_ACTION_KEYS: frozenset[str] = frozenset({"action_type", "params"})


class PromptVersionMismatch(RuntimeError):
    """Raised when a trajectory's prompt_version does not match the installed
    ``medibill.prompting.PROMPT_VERSION``. FATAL — never downgrade to a warning.
    """


class TrajectorySchemaError(RuntimeError):
    """Raised when a trajectory is missing required fields or has malformed
    structure. FATAL."""


# ---------------------------------------------------------------------------
# Load + validate
# ---------------------------------------------------------------------------


def _check_prompt_version(traj: Dict[str, Any], expected: str) -> None:
    got = traj.get("prompt_version")
    if got != expected:
        raise PromptVersionMismatch(
            f"Prompt version mismatch on trajectory "
            f"{traj.get('trajectory_id', '<no-id>')} "
            f"(task={traj.get('task_id', '?')}, seed={traj.get('seed', '?')}): "
            f"expected {expected!r}, got {got!r}. "
            f"The system prompt has changed since this trajectory was "
            f"generated. Regenerate with:\n"
            f"    python -m medibill.generate_trajectories --out <path>"
        )


def _check_trajectory_schema(traj: Dict[str, Any]) -> None:
    missing = REQUIRED_TRAJ_KEYS - set(traj.keys())
    if missing:
        raise TrajectorySchemaError(
            f"Trajectory {traj.get('trajectory_id', '<no-id>')} missing "
            f"required top-level keys: {sorted(missing)}"
        )
    steps = traj["steps"]
    if not isinstance(steps, list) or not steps:
        raise TrajectorySchemaError(
            f"Trajectory {traj.get('trajectory_id', '<no-id>')} has no steps."
        )
    for i, step in enumerate(steps):
        step_missing = REQUIRED_STEP_KEYS - set(step.keys())
        if step_missing:
            raise TrajectorySchemaError(
                f"Trajectory {traj.get('trajectory_id', '<no-id>')} step {i} "
                f"missing keys: {sorted(step_missing)}"
            )
        action = step["action"]
        if not isinstance(action, dict):
            raise TrajectorySchemaError(
                f"Trajectory {traj.get('trajectory_id', '<no-id>')} step {i} "
                f"has non-dict action."
            )
        action_missing = REQUIRED_ACTION_KEYS - set(action.keys())
        if action_missing:
            raise TrajectorySchemaError(
                f"Trajectory {traj.get('trajectory_id', '<no-id>')} step {i} "
                f"action missing keys: {sorted(action_missing)}"
            )
        if action["action_type"] not in AGENT_ACTION_TYPES:
            raise TrajectorySchemaError(
                f"Trajectory {traj.get('trajectory_id', '<no-id>')} step {i} "
                f"has invalid action_type {action['action_type']!r}. "
                f"Allowed: {list(AGENT_ACTION_TYPES)}"
            )


def load_trajectories(
    paths: Sequence[Path],
    *,
    expected_prompt_version: str,
) -> List[Dict[str, Any]]:
    """Load JSONL trajectories from *paths*.

    Every record is validated against the schema contract and against
    ``expected_prompt_version``. On any violation, raises (hard-fails).
    """
    out: List[Dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    traj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TrajectorySchemaError(
                        f"{path}:{line_no} is not valid JSON: {exc}"
                    ) from exc
                _check_prompt_version(traj, expected_prompt_version)
                _check_trajectory_schema(traj)
                out.append(traj)
    return out


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterSpec:
    min_score: Optional[float] = None
    policies: Optional[frozenset[str]] = None
    tasks: Optional[frozenset[str]] = None


def filter_trajectories(
    trajectories: Iterable[Dict[str, Any]],
    spec: FilterSpec,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in trajectories:
        if spec.min_score is not None and t["final_score"] < spec.min_score:
            continue
        if spec.policies is not None and t["policy"] not in spec.policies:
            continue
        if spec.tasks is not None and t["task_id"] not in spec.tasks:
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Flatten to chat messages
# ---------------------------------------------------------------------------


def flatten_to_chat(
    trajectories: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Turn trajectories into ``{"messages": [system, user, assistant]}`` rows.

    Each step of each trajectory becomes one training example. Metadata
    (task_id, seed, policy, trajectory_id, observation_step_number,
    final_score) is attached alongside ``messages`` so downstream code can
    subset or re-weight examples.
    """
    rows: List[Dict[str, Any]] = []
    for t in trajectories:
        system_prompt = t["system_prompt"]
        for step in t["steps"]:
            assistant_text = json.dumps(
                step["action"],
                separators=(",", ":"),
                sort_keys=True,
            )
            rows.append({
                "messages": [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": step["observation_text"]},
                    {"role": "assistant", "content": assistant_text},
                ],
                # Metadata — useful for analytics and for HF datasets
                # `.filter()` calls; never fed to the trainer directly.
                "trajectory_id":  t["trajectory_id"],
                "task_id":        t["task_id"],
                "seed":           t["seed"],
                "policy":         t["policy"],
                "prompt_version": t["prompt_version"],
                "final_score":    t["final_score"],
                "observation_step_number": step["observation_step_number"],
                "action_type":    step["action"]["action_type"],
            })
    return rows


def save_dataset(rows: Iterable[Dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
            n += 1
    return n


def summarise(rows: Sequence[Dict[str, Any]]) -> None:
    by_task = Counter(r["task_id"] for r in rows)
    by_policy = Counter(r["policy"] for r in rows)
    by_action = Counter(r["action_type"] for r in rows)
    print(f"  total examples:      {len(rows)}")
    print(f"  by task_id:          {dict(by_task)}")
    print(f"  by policy:           {dict(by_policy)}")
    print(f"  by action_type:      {dict(by_action)}")

    # Non-fatal coverage warning: surface every agent-visible action_type that
    # has ZERO supervised examples. The SFT-trained model will never emit such
    # an action type under imitation alone — this matters when the rubric
    # scores an axis that depends on it (e.g. `abstention_quality` depends on
    # `escalate_to_human`; a zero count means that axis is RL-territory, not
    # SFT-territory). Keep as WARN so it does not block bespoke datasets that
    # intentionally exclude a tool.
    missing_tools = [t for t in AGENT_ACTION_TYPES if by_action.get(t, 0) == 0]
    if missing_tools:
        print(
            f"  [WARN] tools with zero examples in prepared dataset: "
            f"{missing_tools}. An SFT-trained checkpoint will not emit these "
            f"actions. Any rubric axis that depends on them "
            f"(e.g. abstention_quality depends on escalate_to_human) must be "
            f"treated as RL-territory, not SFT-territory."
        )


# ---------------------------------------------------------------------------
# Train (CUDA-guarded)
# ---------------------------------------------------------------------------


def train(
    dataset_path: Path,
    model: str,
    output_dir: Path,
    *,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 2048,
) -> int:
    """Run Unsloth + TRL SFT. Requires CUDA. No-ops gracefully on non-GPU hosts.

    Returns 0 on success, a non-zero integer on intentional skip (no CUDA,
    missing deps), so callers can gate downstream steps on the exit code.
    """
    try:
        import torch  # noqa: WPS433  (runtime import is intentional)
    except ImportError:
        print(
            "[SKIP] torch not installed. On Colab: "
            "    pip install 'openenv-medibill-env[train]'",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print(
            "[SKIP] No CUDA device detected. SFT requires a GPU.\n"
            "       To run this on Colab Pro, use:\n"
            f"         !python -m medibill.train_sft train \\\n"
            f"             --dataset {dataset_path} \\\n"
            f"             --model {model} \\\n"
            f"             --out {output_dir}",
            file=sys.stderr,
        )
        return 3

    # Imports deferred until CUDA is confirmed so dev hosts without
    # these packages can still run `prepare`.
    from datasets import load_dataset  # noqa: WPS433
    from trl import SFTConfig, SFTTrainer  # noqa: WPS433
    from unsloth import FastLanguageModel  # noqa: WPS433
    from unsloth.chat_templates import train_on_responses_only  # noqa: WPS433

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    lora_model = FastLanguageModel.get_peft_model(
        base_model,
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )

    # Precision picks itself based on GPU capability. Ampere+ (A100, H100)
    # gets bf16 natively; Turing (T4) falls back to fp16 because its Tensor
    # Cores have no native bf16 and bf16=True there is slow/broken.
    use_bf16, use_fp16 = preferred_dtype_flags()

    # Pre-apply Qwen2.5's chat template so each row arrives at SFTTrainer as
    # a flat ``text`` string. trl's ``SFTTrainer._prepare_dataset`` then
    # tokenises ``text`` and hands token IDs to the collator. Without this
    # step the trainer crashes — its conversational-format auto-detection
    # only works on the very latest trl, and our pinned version expects a
    # ``dataset_text_field``. We also drop every other column so trl's
    # default collation does not try to batch metadata strings as tensors.
    ds = ds.map(
        lambda row: {
            "text": tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }
    ).select_columns(["text"])

    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=5,
        save_steps=50,
        # Cap checkpoint count: free Colab T4 has ~15GB disk and a rank-32
        # LoRA checkpoint at this dataset size is ~300-500MB; without a cap
        # ~13 checkpoints accumulate over 3 epochs and crowd the volume.
        save_total_limit=2,
        # Smooth the early loss spike when starting LoRA from random adapters
        # at LR 1e-4. ~3% of total steps is standard for LoRA SFT.
        warmup_ratio=0.03,
        # NOTE: ``max_seq_length`` is NOT passed to SFTConfig — it was removed
        # from SFTConfig in trl 0.18+ (the version Unsloth pulls in). The
        # value already reaches the model via FastLanguageModel.from_pretrained
        # above, which is where Unsloth uses it for forward-pass padding.
        # We pre-applied the chat template above; SFTTrainer will tokenise
        # ``text`` for us. Loss masking happens AFTER trainer construction,
        # via Unsloth's train_on_responses_only helper below.
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=lora_model,
        tokenizer=tokenizer,
        args=cfg,
        train_dataset=ds,
    )

    # Mask loss to assistant tokens only via Unsloth's helper. This rewrites
    # the trainer's internal collator to set labels=-100 on every token that
    # isn't inside an assistant turn, using the ChatML role-marker strings
    # passed below. Replaces the older ``DataCollatorForCompletionOnlyLM``
    # path which broke under newer trl/unsloth combinations because of the
    # removed top-level export and BPE-boundary fragility.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"[OK] SFT complete. Adapter saved to {output_dir}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_prepare(args: argparse.Namespace) -> int:
    print(f"Installed PROMPT_VERSION: {PROMPT_VERSION}")
    print(f"Loading from: {[str(p) for p in args.traces]}")
    trajectories = load_trajectories(
        args.traces, expected_prompt_version=PROMPT_VERSION,
    )
    print(f"  loaded: {len(trajectories)} trajectories (all schema-valid, "
          f"all prompt_version matches)")

    spec = FilterSpec(
        min_score=args.min_score,
        policies=frozenset(args.policies) if args.policies else None,
        tasks=frozenset(args.tasks) if args.tasks else None,
    )
    trajectories = filter_trajectories(trajectories, spec)
    print(f"  after filter: {len(trajectories)} trajectories")

    rows = flatten_to_chat(trajectories)
    n = save_dataset(rows, args.out)
    print(f"\nWrote {n} chat-format examples to {args.out}")
    summarise(rows)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    return train(
        dataset_path=args.dataset,
        model=args.model,
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFT pipeline for MediBill-Env.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare
    p = sub.add_parser("prepare", help="Load, validate, filter, flatten.")
    p.add_argument("--traces", nargs="+", type=Path, required=True,
                   help="One or more trajectory JSONL files.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output dataset path (JSONL).")
    p.add_argument("--min-score", type=float, default=None,
                   help="Keep only trajectories with final_score >= this.")
    p.add_argument("--policies", nargs="+", default=None,
                   help="Keep only these policies.")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Keep only these task_ids.")
    p.set_defaults(func=_cmd_prepare)

    # train
    t = sub.add_parser("train", help="Run SFT (requires CUDA).")
    t.add_argument("--dataset", type=Path, required=True,
                   help="Dataset JSONL produced by `prepare`.")
    t.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                   help="Base model id.")
    t.add_argument("--out", type=Path, required=True,
                   help="Output directory for the LoRA adapter.")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--grad-accum", type=int, default=4)
    t.set_defaults(func=_cmd_train)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
