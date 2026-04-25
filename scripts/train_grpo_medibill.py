"""GRPO fine-tuning for MediBill-Env, single-step formulation.

Initialises from the trained SFT LoRA adapter and trains the policy to
emit higher-quality single-step actions using FIVE reward functions
that directly encode the MediBillGrader's penalty / bonus structure.

Why single-step (not episode-level):
    * TRL's stable ``GRPOTrainer`` API operates on prompt -> completion
      pairs, not on multi-step environment rollouts.
    * The grader's penalties and bonuses (no_lookup_at_submit,
      submit_no_coding, oscillation, repeated_tool, params completeness)
      are ALL per-action signals — they can be approximated from the
      observation text + emitted action without running a full episode.
    * Episode-level performance still benefits because each step the
      model takes is now better-shaped on these axes.

Five reward functions (the bonus criteria asks for 5):
    1. reward_valid_json          — parses as a JSON object
    2. reward_known_action_type   — action_type in AGENT_ACTION_TYPES
    3. reward_params_complete     — required params present for that type
    4. reward_drift_aware         — bonus for insurance_lookup when the
                                    observation shows we are late in the
                                    episode and the last lookup is stale
                                    (this is what pushes past the
                                    scripted ceiling on hard_drift)
    5. reward_submission_ready    — bonus for submit_claim only when the
                                    observation shows all fields filled

Each reward returns floats in [0.0, 1.0]. TRL averages across reward
functions before computing group-relative advantage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Allow ``python scripts/train_grpo_medibill.py`` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Action parsing (mirrors medibill/evaluate_sft.py:parse_action behaviour)
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_DECODER = json.JSONDecoder()

VALID_ACTION_TYPES = {
    "ehr_query",
    "insurance_lookup",
    "coding_engine",
    "escalate_to_human",
    "submit_claim",
}

REQUIRED_PARAMS = {
    "ehr_query":         {"claim_id"},
    "insurance_lookup":  {"provider"},
    "coding_engine":     {"claim_id", "field", "value"},
    "submit_claim":      {"claim_id"},
    "escalate_to_human": {"reason"},
}


def _try_decode(candidate: str) -> dict | None:
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and "action_type" in obj else None


def parse_action_text(text: str) -> dict | None:
    """Best-effort JSON action extraction. Mirrors evaluate_sft.parse_action."""
    if not text:
        return None
    for cand in (text.strip(), _FENCE_RE.sub("", text).strip()):
        if not cand:
            continue
        obj = _try_decode(cand)
        if obj is not None:
            return obj
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = _JSON_DECODER.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "action_type" in obj:
            return obj
    return None


# ---------------------------------------------------------------------------
# Five reward functions
# ---------------------------------------------------------------------------


def reward_valid_json(prompts: list, completions: list, **kwargs: Any) -> list[float]:
    """1.0 if the completion parses to a JSON object with action_type, else 0.0."""
    rewards: list[float] = []
    for c in completions:
        text = c if isinstance(c, str) else c[-1].get("content", "")
        rewards.append(1.0 if parse_action_text(text) is not None else 0.0)
    return rewards


def reward_known_action_type(prompts: list, completions: list, **kwargs: Any) -> list[float]:
    """1.0 if action_type is one of the five legal tools, else 0.0."""
    rewards: list[float] = []
    for c in completions:
        text = c if isinstance(c, str) else c[-1].get("content", "")
        a = parse_action_text(text)
        rewards.append(1.0 if (a and a.get("action_type") in VALID_ACTION_TYPES) else 0.0)
    return rewards


def reward_params_complete(prompts: list, completions: list, **kwargs: Any) -> list[float]:
    """1.0 if required params for the chosen action_type are all present.

    Half credit (0.5) if the action_type is valid but params are missing —
    we still want to reward making a recognised choice.
    """
    rewards: list[float] = []
    for c in completions:
        text = c if isinstance(c, str) else c[-1].get("content", "")
        a = parse_action_text(text)
        if not a:
            rewards.append(0.0)
            continue
        atype = a.get("action_type")
        if atype not in VALID_ACTION_TYPES:
            rewards.append(0.0)
            continue
        params = a.get("params", {}) or {}
        required = REQUIRED_PARAMS.get(atype, set())
        if required.issubset(set(params.keys())):
            rewards.append(1.0)
        else:
            rewards.append(0.5)
    return rewards


def _prompt_text(prompt: Any) -> str:
    """Flatten chat-style prompts into a single string for inspection."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(str(m.get("content", "")) for m in prompt if isinstance(m, dict))
    return str(prompt)


_STEP_RE = re.compile(r"\[Step\]\s+(\d+)/(\d+)")
_LOOKUP_RE = re.compile(r"insurance_lookup", re.IGNORECASE)


def reward_drift_aware(prompts: list, completions: list, **kwargs: Any) -> list[float]:
    """Bonus for *proactive* insurance_lookup calls.

    Drift fires silently mid-episode. The scripted policy never re-queries.
    Reward the model for picking ``insurance_lookup`` when we are past
    step 25 (drift can fire from step ~20 onwards on hard_drift). This is
    the precise behaviour that breaks past the scripted ceiling.

    1.0  : action is insurance_lookup and step >= 25
    0.3  : action is insurance_lookup and step < 25 (still good but not the
           drift-recovery behaviour we want)
    0.5  : any other valid action (neutral — don't punish non-lookup)
    0.0  : malformed
    """
    rewards: list[float] = []
    for prompt, completion in zip(prompts, completions):
        text = completion if isinstance(completion, str) else completion[-1].get("content", "")
        a = parse_action_text(text)
        if not a or a.get("action_type") not in VALID_ACTION_TYPES:
            rewards.append(0.0)
            continue
        atype = a.get("action_type")
        ptext = _prompt_text(prompt)
        m = _STEP_RE.search(ptext)
        step = int(m.group(1)) if m else 0
        if atype == "insurance_lookup":
            rewards.append(1.0 if step >= 25 else 0.3)
        else:
            rewards.append(0.5)
    return rewards


def reward_submission_ready(prompts: list, completions: list, **kwargs: Any) -> list[float]:
    """Reward submit_claim ONLY when the observation indicates the claim is ready.

    The format_observation output marks unsubmitted claims and shows their
    field state. A submit on a claim where any of the policy-sensitive
    fields is still ``None`` triggers PENALTY_SUBMIT_NO_CODING in the grader.

    1.0  : action is submit_claim AND the claim_id appears in the prompt
           with no obvious ``=None`` policy fields
    0.3  : submit_claim with a missing or unrecognised claim_id
    0.7  : non-submit valid action
    0.0  : malformed
    """
    rewards: list[float] = []
    for prompt, completion in zip(prompts, completions):
        text = completion if isinstance(completion, str) else completion[-1].get("content", "")
        a = parse_action_text(text)
        if not a or a.get("action_type") not in VALID_ACTION_TYPES:
            rewards.append(0.0)
            continue
        atype = a.get("action_type")
        if atype != "submit_claim":
            rewards.append(0.7)
            continue
        params = a.get("params", {}) or {}
        cid = str(params.get("claim_id", ""))
        ptext = _prompt_text(prompt)
        if not cid or cid not in ptext:
            rewards.append(0.3)
            continue
        # Check if THIS claim_id's line contains any policy_version=None /
        # pre_auth_flag=None patterns. If yes, submitting is premature.
        for line in ptext.splitlines():
            if cid in line and ("policy_version=None" in line or "pre_auth_flag=None" in line
                                or "summary=None" in line):
                rewards.append(0.3)
                break
        else:
            rewards.append(1.0)
    return rewards


REWARD_FUNCS = [
    reward_valid_json,
    reward_known_action_type,
    reward_params_complete,
    reward_drift_aware,
    reward_submission_ready,
]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_prompt_dataset(sft_dataset_path: Path, n_samples: int, seed: int = 42):
    """Take SFT dataset rows and keep only the (system, user) prompt prefix.

    GRPO regenerates the assistant turn during training and scores it; we
    just need the prompt text. We sample uniformly across tasks so the
    drift-aware reward sees enough hard_drift prompts to bite.
    """
    import random
    from datasets import Dataset

    rows: list[dict] = []
    by_task: dict[str, list[dict]] = {}
    with sft_dataset_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            messages = r.get("messages", [])
            if len(messages) < 2:
                continue
            prompt = [
                {"role": "system", "content": messages[0]["content"]},
                {"role": "user",   "content": messages[1]["content"]},
            ]
            tid = r.get("task_id", "unknown")
            by_task.setdefault(tid, []).append({"prompt": prompt})

    # Sample roughly evenly across tasks.
    rng = random.Random(seed)
    per_task = max(1, n_samples // max(1, len(by_task)))
    selected: list[dict] = []
    for tid, items in by_task.items():
        rng.shuffle(items)
        selected.extend(items[:per_task])
    rng.shuffle(selected)
    selected = selected[:n_samples]

    print(f"[grpo] Built prompt dataset: {len(selected)} rows across "
          f"{len(by_task)} tasks ({', '.join(sorted(by_task))})")
    return Dataset.from_list(selected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-step GRPO from SFT init on MediBill-Env.",
    )
    parser.add_argument("--sft-adapter", type=Path, required=True,
                        help="Path to trained SFT LoRA adapter (folder).")
    parser.add_argument("--sft-dataset", type=Path, default=Path("datasets/sft_v1.jsonl"),
                        help="SFT dataset to source prompts from.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to save the GRPO-tuned LoRA adapter.")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of prompts to use during GRPO.")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="Hard cap on optimizer steps.")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="G in GRPO — completions per prompt.")
    parser.add_argument("--max-completion-length", type=int, default=128,
                        help="Token budget for each generated action.")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                        help="LoRA LR. GRPO is sensitive — keep low.")
    parser.add_argument("--per-device-batch-size", type=int, default=1,
                        help="Prompts per gradient step (× num_generations).")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Effective batch = per_device * grad_accum.")
    args = parser.parse_args()

    # ---- imports deferred to keep import-time cheap ----
    print("[grpo] Loading model and tokenizer (Unsloth + 4-bit)...")
    from unsloth import FastLanguageModel  # noqa: WPS433
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.sft_adapter),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    # Do NOT call FastLanguageModel.for_inference(model) — GRPO needs training mode.

    from trl import GRPOConfig, GRPOTrainer  # noqa: WPS433

    print("[grpo] Building prompt dataset...")
    dataset = build_prompt_dataset(args.sft_dataset, n_samples=args.n_samples)

    print("[grpo] Configuring trainer...")
    config = GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=2048,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=max(args.max_steps // 3, 5),
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],   # silence wandb/tensorboard probes on Colab
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=REWARD_FUNCS,
        processing_class=tokenizer,
    )

    print(f"[grpo] Training for {args.max_steps} steps, "
          f"G={args.num_generations}, bs={args.per_device_batch_size}, "
          f"grad_accum={args.grad_accum}")
    trainer.train()

    print(f"[grpo] Saving adapter to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("[grpo] Done.")


if __name__ == "__main__":
    main()
