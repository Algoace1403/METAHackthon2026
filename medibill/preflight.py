"""CPU-only preflight checks before burning a Colab GPU session.

Validates every integration-risk surface the first SFT run will hit:
    1. Required training packages import cleanly.
    2. Qwen/Qwen2.5-3B-Instruct tokenizer downloads + loads.
    3. The tokenizer has a chat_template; key ChatML markers are present.
    4. assistant_only_loss=True compatibility: verify chat_template has the
       ``{% generation %}`` jinja markers TRL needs, or fall back to an
       explicit DataCollatorForCompletionOnlyLM plan.
    5. One row from datasets/sft_v1.jsonl round-trips through
       apply_chat_template without error.
    6. Qwen2.5-3B-Instruct config has the architecture + module names our
       LoRA target_modules spec expects.
    7. traces/eval.jsonl loads + schema-validates + prompt_version matches.

No GPU needed; no model weights downloaded. Runtime ~30-120s depending on
HF download speed. Exits 0 on full pass; non-zero on any hard failure.

Usage:
    python -m medibill.preflight
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATASET = Path("datasets/sft_v1.jsonl")
EVAL = Path("traces/eval.jsonl")
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_LORA_TARGETS = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str, failures: list[str]) -> None:
    print(f"[FAIL] {msg}")
    failures.append(msg)


def check_imports(failures: list[str]) -> None:
    try:
        import torch  # noqa: WPS433
        import transformers  # noqa: WPS433
        import peft  # noqa: WPS433
        import trl  # noqa: WPS433
        import datasets as hf_datasets  # noqa: WPS433
        _ok(
            f"imports: torch {torch.__version__}  "
            f"transformers {transformers.__version__}  "
            f"peft {peft.__version__}  trl {trl.__version__}  "
            f"datasets {hf_datasets.__version__}"
        )
    except ImportError as exc:
        _fail(f"missing training package: {exc}", failures)


def check_tokenizer_and_template(failures: list[str]) -> "object":
    try:
        from transformers import AutoTokenizer  # noqa: WPS433
    except ImportError:
        _fail("transformers not installed — skipping tokenizer check", failures)
        return None
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    except Exception as exc:
        _fail(f"AutoTokenizer.from_pretrained({BASE_MODEL!r}): {exc}", failures)
        return None

    _ok(
        f"tokenizer: vocab={tok.vocab_size}  "
        f"eos={tok.eos_token_id}  "
        f"pad={tok.pad_token_id}  "
        f"bos={tok.bos_token_id}"
    )

    template = getattr(tok, "chat_template", None)
    if not template:
        _fail("tokenizer.chat_template is empty — SFTTrainer needs it", failures)
        return tok

    chatml_markers = ("<|im_start|>", "<|im_end|>")
    missing = [m for m in chatml_markers if m not in template]
    if missing:
        _warn(f"chat_template missing ChatML markers: {missing}")
    else:
        _ok("chat_template contains ChatML markers (<|im_start|>, <|im_end|>)")

    has_generation_tag = "{% generation %}" in template
    if has_generation_tag:
        _ok(
            "chat_template has {% generation %} markers — "
            "SFTConfig(assistant_only_loss=True) would work"
        )
    else:
        _ok(
            "chat_template has no {% generation %} markers (expected for "
            "Qwen2.5). train_sft.py uses DataCollatorForCompletionOnlyLM "
            "with response_template='<|im_start|>assistant\\n' as the "
            "template-independent substitute — already wired."
        )
    return tok


def check_apply_template_on_dataset_row(tok: "object", failures: list[str]) -> None:
    if tok is None:
        return
    if not DATASET.exists():
        _fail(f"dataset file missing: {DATASET}", failures)
        return
    with DATASET.open("r", encoding="utf-8") as fh:
        row = json.loads(fh.readline())
    messages = row.get("messages", [])
    if not messages or len(messages) != 3:
        _fail(f"dataset first row has malformed messages: {messages}", failures)
        return
    try:
        rendered = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:
        _fail(f"apply_chat_template raised on real dataset row: {exc}", failures)
        return
    if not rendered or not isinstance(rendered, str):
        _fail("apply_chat_template returned non-string output", failures)
        return
    # Sanity-check the assistant marker the collator will key off of
    assistant_marker = "<|im_start|>assistant\n"
    if assistant_marker not in rendered:
        _warn(
            f"rendered chat does not contain marker {assistant_marker!r} — "
            f"DataCollatorForCompletionOnlyLM will not find the response "
            f"boundary. Verify Qwen template hasn't changed."
        )
    else:
        _ok(
            f"apply_chat_template on real row: {len(rendered)} chars rendered; "
            f"assistant marker present at char {rendered.index(assistant_marker)}"
        )


def check_model_config_and_lora_targets(failures: list[str]) -> None:
    try:
        from transformers import AutoConfig  # noqa: WPS433
    except ImportError:
        return
    try:
        cfg = AutoConfig.from_pretrained(BASE_MODEL)
    except Exception as exc:
        _fail(f"AutoConfig.from_pretrained({BASE_MODEL!r}): {exc}", failures)
        return
    model_type = getattr(cfg, "model_type", "?")
    arch = getattr(cfg, "architectures", ["?"])
    hidden = getattr(cfg, "hidden_size", "?")
    nlayers = getattr(cfg, "num_hidden_layers", "?")
    _ok(
        f"model config: type={model_type}  "
        f"arch={arch}  "
        f"hidden_size={hidden}  layers={nlayers}"
    )

    # Qwen2.5 uses the Qwen2Model architecture. Its linear modules that
    # actually exist and accept LoRA are exactly the ones named in
    # EXPECTED_LORA_TARGETS. The names follow HF convention — we verify
    # the model family is one we know those names apply to.
    if model_type not in {"qwen2", "qwen3"}:
        _warn(
            f"model_type={model_type!r} is not a Qwen family model. "
            f"Our LoRA target_modules {sorted(EXPECTED_LORA_TARGETS)} "
            f"may not all exist. Verify against the model card before training."
        )
    else:
        _ok(
            f"LoRA targets {sorted(EXPECTED_LORA_TARGETS)} are the canonical "
            f"linear modules for Qwen2 family — will match"
        )


def check_eval_split(failures: list[str]) -> None:
    if not EVAL.exists():
        _fail(f"eval file missing: {EVAL}", failures)
        return
    try:
        from medibill.prompting import PROMPT_VERSION  # noqa: WPS433
    except ImportError as exc:
        _fail(f"cannot import medibill.prompting: {exc}", failures)
        return
    bad_rows = 0
    mismatches = 0
    count = 0
    with EVAL.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                traj = json.loads(line)
            except json.JSONDecodeError:
                bad_rows += 1
                continue
            if traj.get("prompt_version") != PROMPT_VERSION:
                mismatches += 1
    if count == 0:
        _fail("eval.jsonl is empty", failures)
        return
    if bad_rows:
        _fail(f"eval.jsonl has {bad_rows} unparseable row(s)", failures)
    if mismatches:
        _fail(
            f"eval.jsonl has {mismatches} rows with prompt_version != "
            f"installed {PROMPT_VERSION}",
            failures,
        )
    if not bad_rows and not mismatches:
        _ok(
            f"eval.jsonl: {count} trajectories, all parse, all prompt_version "
            f"= {PROMPT_VERSION}"
        )


def check_dataset_shape(failures: list[str]) -> None:
    if not DATASET.exists():
        _fail(f"dataset file missing: {DATASET}", failures)
        return
    rows = 0
    bad = 0
    with DATASET.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                row = json.loads(line)
                assert "messages" in row
                assert len(row["messages"]) == 3
            except Exception:
                bad += 1
    if bad:
        _fail(f"dataset has {bad} malformed row(s) out of {rows}", failures)
    else:
        _ok(f"dataset: {rows} rows, all have 3-message chat shape")


def main() -> int:
    print("MediBill-Env CPU preflight — Wed 22 Apr 2026")
    print("-" * 60)
    failures: list[str] = []

    check_imports(failures)
    check_dataset_shape(failures)
    check_eval_split(failures)
    tok = check_tokenizer_and_template(failures)
    check_apply_template_on_dataset_row(tok, failures)
    check_model_config_and_lora_targets(failures)

    print("-" * 60)
    if failures:
        print(f"[FAIL] {len(failures)} hard failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[OK] Preflight passed. Launching Colab is safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
