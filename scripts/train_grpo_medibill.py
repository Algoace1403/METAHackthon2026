"""GRPO fine-tuning for MediBill-Env, single-step formulation.

v2 — fixes from brutal pre-deployment review:
    * CRITICAL: re-attaches LoRA via FastLanguageModel.get_peft_model
      after loading the SFT adapter, so trainable parameters actually
      exist (loading via from_pretrained alone leaves them frozen).
    * CRITICAL: removes gradient_checkpointing=True from GRPOConfig
      (Unsloth owns gradient checkpointing via use_gradient_checkpointing).
    * Disables wandb upfront (Colab default behaviour can crash GRPO).
    * Replaces broken reward_drift_aware (rewarded step >= 25 even when
      drift had not fired) with reward_no_oscillation (penalises
      coding_engine writes to fields that are already filled — directly
      proxies the grader's PENALTY_OSCILLATION).
    * Removes the 0.7 non-submit floor in reward_submission_ready (was
      training the model to AVOID submitting).
    * Defaults to ``--task-filter hard_drift`` only, the one task with
      headroom against scripted.
    * Conservative ``--max-steps=15`` default; saves every 5.

Five reward functions for the bonus criteria, all aligned with the
MediBillGrader's penalty/bonus structure (medibill/server/grader.py):
    1. reward_valid_json
    2. reward_known_action_type
    3. reward_params_complete
    4. reward_no_oscillation     — proxies PENALTY_OSCILLATION
    5. reward_submission_ready   — proxies PENALTY_SUBMIT_NO_CODING
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Disable wandb BEFORE any TRL import. TRL probes wandb on Colab regardless
# of report_to=[] on some 0.18-0.24 builds; this prevents the probe.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

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


def _completion_text(c: Any) -> str:
    """TRL passes completions as either str or list[dict] (chat format)."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return str(c[-1].get("content", ""))
    return str(c)


def _prompt_text(prompt: Any) -> str:
    """Flatten chat-style prompts into a single string for inspection."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(str(m.get("content", "")) for m in prompt if isinstance(m, dict))
    return str(prompt)


# ---------------------------------------------------------------------------
# Five reward functions (post-review fixes applied)
# ---------------------------------------------------------------------------


def reward_valid_json(prompts: list, completions: list, **kwargs: Any) -> list[float]:
    """1.0 if the completion parses to a JSON object with action_type, else 0.0."""
    return [1.0 if parse_action_text(_completion_text(c)) is not None else 0.0
            for c in completions]


def reward_known_action_type(prompts: list, completions: list, **kwargs: Any) -> list[float]:
    """1.0 if action_type is one of the five legal tools, else 0.0."""
    rewards: list[float] = []
    for c in completions:
        a = parse_action_text(_completion_text(c))
        rewards.append(1.0 if (a and a.get("action_type") in VALID_ACTION_TYPES) else 0.0)
    return rewards


def reward_params_complete(prompts: list, completions: list, **kwargs: Any) -> list[float]:
    """1.0 if required params for the chosen action_type are all present.

    Half credit (0.5) if the action_type is valid but params are missing —
    we still want to reward making a recognised choice.
    """
    rewards: list[float] = []
    for c in completions:
        a = parse_action_text(_completion_text(c))
        if not a:
            rewards.append(0.0)
            continue
        atype = a.get("action_type")
        if atype not in VALID_ACTION_TYPES:
            rewards.append(0.0)
            continue
        params = a.get("params", {}) or {}
        required = REQUIRED_PARAMS.get(atype, set())
        rewards.append(1.0 if required.issubset(set(params.keys())) else 0.5)
    return rewards


# Match a claim line like:
#   - CLM-CGHS-2024000  provider=CGHS  amount_inr=186  ...  policy_version=v2024.1  pre_auth_flag=False
# We need to extract the (claim_id, field, value) state from the prompt to
# detect oscillation: writing the same value, or writing to a field that
# already holds a non-None value the model is now overwriting.
_CLAIM_LINE_RE = re.compile(
    r"^\s*-\s+(CLM-[A-Z0-9-]+)\s+(.*)$",
    re.MULTILINE,
)


def _claim_state_from_prompt(ptext: str) -> dict[str, dict[str, str]]:
    """Extract per-claim field state from a format_observation rendering.

    Returns ``{claim_id: {field_name: value_text}}``. ``value_text`` is the
    raw textual value as it appeared in the prompt (``"None"``, ``"v2024.1"``,
    etc.). We use this to detect oscillation cheaply at reward time.
    """
    state: dict[str, dict[str, str]] = {}
    for m in _CLAIM_LINE_RE.finditer(ptext):
        cid = m.group(1)
        rest = m.group(2)
        fields: dict[str, str] = {}
        for pair in rest.split():
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            fields[k] = v
        state[cid] = fields
    return state


def reward_no_oscillation(prompts: list, completions: list, **kwargs: Any) -> list[float]:
    """Penalise coding_engine writes that look like oscillation.

    Proxy for PENALTY_OSCILLATION (grader.py:56, 561-573): if a coding_engine
    action targets a (claim_id, field) whose current value in the prompt is
    already a non-``None`` concrete value, the model is overwriting work
    already done — the actual grader penalises 3+ distinct values per
    (claim, field). We approximate at the per-step level.

    1.0  : coding_engine on a field whose prompt value is ``None`` (progress)
    0.3  : coding_engine on a field that already has a non-``None`` value
           (potential oscillation)
    0.7  : any other valid action_type (neutral — submit, lookup, etc.)
    0.0  : malformed
    """
    rewards: list[float] = []
    for prompt, c in zip(prompts, completions):
        a = parse_action_text(_completion_text(c))
        if not a or a.get("action_type") not in VALID_ACTION_TYPES:
            rewards.append(0.0)
            continue
        atype = a.get("action_type")
        if atype != "coding_engine":
            rewards.append(0.7)
            continue
        params = a.get("params", {}) or {}
        cid = str(params.get("claim_id", ""))
        field = str(params.get("field", ""))
        if not cid or not field:
            rewards.append(0.3)
            continue
        ptext = _prompt_text(prompt)
        state = _claim_state_from_prompt(ptext)
        cur = state.get(cid, {}).get(field, "None")
        # Treat "None" / "empty" as "still needs writing" → reward progress.
        if cur in ("None", "empty", "0"):
            rewards.append(1.0)
        else:
            rewards.append(0.3)
    return rewards


def reward_submission_ready(prompts: list, completions: list, **kwargs: Any) -> list[float]:
    """Reward submit_claim ONLY when the observation indicates the claim is ready.

    Proxy for PENALTY_SUBMIT_NO_CODING (grader.py:55, 540-554): submitting
    a claim whose policy-sensitive fields are still ``None`` triggers a
    penalty. We reward submits only when the prompt shows those fields
    populated.

    1.0  : submit_claim AND the named claim's policy_version + pre_auth
           fields are both non-``None`` in the prompt
    0.3  : submit_claim that the prompt shows isn't ready, OR the claim_id
           is missing / unrecognised
    0.5  : any other valid action_type (neutral — do NOT bias against
           non-submit actions; the model legitimately spends most steps
           on coding_engine / ehr_query)
    0.0  : malformed
    """
    rewards: list[float] = []
    for prompt, c in zip(prompts, completions):
        a = parse_action_text(_completion_text(c))
        if not a or a.get("action_type") not in VALID_ACTION_TYPES:
            rewards.append(0.0)
            continue
        atype = a.get("action_type")
        if atype != "submit_claim":
            rewards.append(0.5)
            continue
        params = a.get("params", {}) or {}
        cid = str(params.get("claim_id", ""))
        ptext = _prompt_text(prompt)
        if not cid or cid not in ptext:
            rewards.append(0.3)
            continue
        state = _claim_state_from_prompt(ptext)
        fields = state.get(cid, {})
        critical = ("policy_version", "pre_auth_flag")
        if all(fields.get(k, "None") not in ("None", "empty") for k in critical):
            rewards.append(1.0)
        else:
            rewards.append(0.3)
    return rewards


REWARD_FUNCS = [
    reward_valid_json,
    reward_known_action_type,
    reward_params_complete,
    reward_no_oscillation,
    reward_submission_ready,
]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_prompt_dataset(
    sft_dataset_path: Path,
    n_samples: int,
    task_filter: str | None,
    seed: int = 42,
):
    """Take SFT dataset rows and keep only the (system, user) prompt prefix.

    If ``task_filter`` is set (e.g., ``"hard_drift"``), only prompts from
    that task family are kept — concentrates GRPO updates on the task with
    actual headroom against scripted.
    """
    import random
    from datasets import Dataset

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
            tid = r.get("task_id", "unknown")
            if task_filter and tid != task_filter:
                continue
            prompt = [
                {"role": "system", "content": messages[0]["content"]},
                {"role": "user",   "content": messages[1]["content"]},
            ]
            by_task.setdefault(tid, []).append({"prompt": prompt})

    rng = random.Random(seed)
    selected: list[dict] = []
    if task_filter:
        items = by_task.get(task_filter, [])
        rng.shuffle(items)
        selected = items[:n_samples]
    else:
        per_task = max(1, n_samples // max(1, len(by_task)))
        for items in by_task.values():
            rng.shuffle(items)
            selected.extend(items[:per_task])
        rng.shuffle(selected)
        selected = selected[:n_samples]

    print(f"[grpo] Built prompt dataset: {len(selected)} rows "
          f"{'(task_filter=' + task_filter + ')' if task_filter else 'across ' + str(len(by_task)) + ' tasks'}")
    return Dataset.from_list(selected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-step GRPO from SFT init on MediBill-Env (v2, post-review).",
    )
    parser.add_argument("--sft-adapter", type=Path, required=True,
                        help="Path to trained SFT LoRA adapter (folder).")
    parser.add_argument("--sft-dataset", type=Path, default=Path("datasets/sft_v1.jsonl"),
                        help="SFT dataset to source prompts from.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to save the GRPO-tuned LoRA adapter.")
    parser.add_argument("--task-filter", default="hard_drift",
                        choices=("easy_cashless", "medium_multi_payer", "hard_drift", "all"),
                        help="Restrict prompts to a single task family. Default: hard_drift "
                             "(the only task with headroom vs scripted).")
    parser.add_argument("--n-samples", type=int, default=120,
                        help="Number of prompts to use during GRPO.")
    parser.add_argument("--max-steps", type=int, default=15,
                        help="Hard cap on optimizer steps. Conservative default.")
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
    parser.add_argument("--lora-r", type=int, default=32,
                        help="LoRA rank. MUST match the SFT adapter rank.")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA alpha. Match SFT for consistent scale.")
    args = parser.parse_args()

    task_filter = None if args.task_filter == "all" else args.task_filter

    # ---- model: load SFT adapter, then RE-ATTACH LoRA for training ----
    print("[grpo] Loading model and tokenizer (Unsloth + 4-bit, bf16)...")
    import torch  # noqa: WPS433
    from unsloth import FastLanguageModel  # noqa: WPS433
    # Explicit bfloat16 dtype — fixes "self and mat2 must have the same dtype,
    # got Half and Float" crash in Unsloth's apply_lora_mlp_swiglu kernel.
    # When dtype=None, Unsloth defaults to fp16 on Colab and the new LoRA
    # params from get_peft_model land in fp32, producing a Half/Float mix
    # the fast_lora kernel rejects. bf16 keeps everything in one precision
    # that the kernel accepts.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.sft_adapter),
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # CRITICAL FIX (review finding #1): without this, LoRA params remain
    # frozen and trainer.train() runs zero updates silently. Re-attach via
    # Unsloth's helper so all LoRA target_modules are marked trainable.
    print("[grpo] Re-attaching LoRA adapter as trainable via get_peft_model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",   # Unsloth owns this
        random_state=42,
    )

    # Sanity check: at least some params must be trainable.
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[grpo] Trainable params: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / max(1, n_total):.2f}%)")
    if n_trainable == 0:
        raise RuntimeError(
            "No trainable parameters detected after get_peft_model — "
            "GRPO would silently do nothing. Aborting."
        )

    from trl import GRPOConfig, GRPOTrainer  # noqa: WPS433

    print(f"[grpo] Building prompt dataset (task_filter={task_filter})...")
    dataset = build_prompt_dataset(
        args.sft_dataset,
        n_samples=args.n_samples,
        task_filter=task_filter,
    )

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
        save_total_limit=3,
        bf16=True,
        # NOTE: gradient_checkpointing intentionally OMITTED (review finding
        # #2). Unsloth manages it via use_gradient_checkpointing="unsloth"
        # in get_peft_model above. Setting it here causes a double-wrap
        # that can produce NaN gradients on some Unsloth builds in this
        # TRL range (0.18-0.24).
        report_to="none",
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
          f"grad_accum={args.grad_accum}, task={args.task_filter}")
    trainer.train()

    print(f"[grpo] Saving adapter to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("[grpo] Done.")


if __name__ == "__main__":
    main()
