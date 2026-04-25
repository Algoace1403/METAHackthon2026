# MediBill-Env — Colab SFT Recipe

Canonical runbook for training and evaluating the first SFT adapter on Colab
Pro. Paste-ready; no code reading required.

**Prerequisites**

- A Colab Pro subscription (A100 preferred, T4 acceptable with slower runs).
- The repo is hosted somewhere git-clone can reach (GitHub, gist, or a Colab
  file upload).
- The frozen corpus (`traces/train_pool.jsonl`, `traces/eval.jsonl`) and the
  prepared dataset (`datasets/sft_v1.jsonl`) are committed into the repo. If
  they are missing you will regenerate them in step 3 below.

---

## 1. Clone the repo onto the Colab VM

```bash
!git clone <REPO_URL> /content/METAHackthon2026
%cd /content/METAHackthon2026
```

## 2. Install the package with training extras

```bash
!pip install -e '.[train]' -q
```

This installs `medibill` (editable), plus `trl`, `transformers`, `torch`,
`accelerate`, and `datasets`. Unsloth is installed separately because its
wheel is CUDA-version-specific:

```bash
!pip install -q "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```

## 3. (Optional) Regenerate the dataset

Only needed if `datasets/sft_v1.jsonl` is missing, or if `PROMPT_VERSION`
drifted since the dataset was last prepared. Otherwise skip.

```bash
# Generate trajectories — ~60s total
!python -m medibill.generate_trajectories --task all --seeds 16 --seed-offset 0 \
    --policy scripted_heuristic --out traces/train_scripted.jsonl
!python -m medibill.generate_trajectories --task all --seeds 16 --seed-offset 0 \
    --policy random --out traces/train_random.jsonl
!python -m medibill.generate_trajectories --task all --seeds 16 --seed-offset 0 \
    --policy no_op --out traces/train_no_op.jsonl
!cat traces/train_scripted.jsonl traces/train_random.jsonl traces/train_no_op.jsonl \
    > traces/train_pool.jsonl
!python -m medibill.generate_trajectories --task all --seeds 4 --seed-offset 16 \
    --policy scripted_heuristic --out traces/eval.jsonl

# Prepare SFT dataset — filters to scripted_heuristic only
!python -m medibill.train_sft prepare \
    --traces traces/train_pool.jsonl \
    --out datasets/sft_v1.jsonl \
    --policies scripted_heuristic
```

## 4. Run SFT + eval

`tee` does not create parent directories — make sure the `runs/` and
`adapters/` directories exist before piping log output into them.

```bash
!mkdir -p runs adapters

!python -m medibill.sft_colab \
    --dataset datasets/sft_v1.jsonl \
    --eval traces/eval.jsonl \
    --out adapters/sft_v1/ \
    2>&1 | tee runs/sft_v1_stdout.log
```

`tee` mirrors the full stdout (coverage summary, scope banner, training loss
lines, per-task eval table, parse-failure count) into `runs/sft_v1_stdout.log`.
Download that file at the end of the session — reconstructing it from memory
under pitch pressure is a failure mode worth preempting.

What you will see, in order:

1. `PROMPT_VERSION being trained: sp-<hash>` — confirms dataset / code agree.
2. Dataset coverage summary (example counts per task and per action type).
   A `[WARN] tools with zero supervised examples: ['escalate_to_human']`
   line is **expected** and not a failure — it signals the RL-only scope
   documented in `docs/round2-spec-v3.md §7.6`.
3. The SFT-scope banner — expect improvements on 4 axes, not on
   `abstention_quality` or `drift_bonus`.
4. TRL training output — one line per 5 optimiser steps.
5. A per-task eval table like:

   ```
   easy_cashless          n=4  sft_adapter=0.??  scripted=1.000  Δ=±0.???
   medium_multi_payer     n=4  sft_adapter=0.??  scripted=1.000  Δ=±0.???
   hard_drift             n=4  sft_adapter=0.??  scripted=0.764  Δ=±0.???
   ```

## 5. Where artefacts land

| Path | Contents |
|---|---|
| `adapters/sft_v1/` | LoRA adapter, tokenizer, and trainer state. Zip + download. |
| `stdout` | Training loss curve + eval table. Copy into the pitch deck. |
| `traces/eval.jsonl` | Unchanged. The baseline for the eval table. |

---

## Exit codes from `sft_colab.py`

| Code | Meaning | Fix |
|---|---|---|
| 0 | All stages succeeded. | — |
| 2 | `torch` not importable. | Re-run `pip install` step. |
| 3 | No CUDA device detected. | Runtime → Change runtime type → GPU. |
| 1 | Dataset prompt_version mismatch. | Regenerate dataset (step 3). |

Non-zero from training aborts eval by design — no point evaluating a
checkpoint that did not train.

---

## If the first run underperforms scripted on hard_drift

Do **not** immediately tune hyperparameters or rework the harness.
Inspection checklist first (per Codex 2026-04-21 review):

1. Look at `stdout` for `LM output parse failures: <N>`. If `N > 0`, the
   model is emitting malformed JSON under some observation shapes — that is
   a tokenizer/template issue, not a reward issue.
2. Sample two or three eval episodes and count the action mix. If the model
   calls `coding_engine` heavily and never re-calls `insurance_lookup` after
   drift, the SFT has imitated scripted's schedule without the reasoning —
   this is the expected structural limit. Abstention and drift calibration
   require RL; see §7.6.
3. Only after steps 1–2 come back clean should you adjust `--epochs`,
   `--lr`, or `--batch-size`.

## Memory and time budget, reference

On a Colab Pro A100 (40 GB), expected wall-clock for the `sft_v1` run:

- Model load (Qwen2.5-3B-Instruct, 4-bit): ~90 s
- Training (3 epochs × 3,632 examples, bs=4, ga=4): ~25-40 min
- Evaluation (12 episodes × ~75 steps × ~1 s generation): ~15 min

If a T4 is the only GPU available, halve the batch size to avoid OOM and
expect ~3× wall-clock.

---

## Running on Colab **free** (no Pro subscription)

Colab free gives a T4 16 GB when you request `Runtime → Change runtime type
→ GPU`. Our workload fits:

- Qwen2.5-3B 4-bit + LoRA + optimizer + activations ≈ 10 GB of 16 GB.
- Full SFT + eval wall-clock ≈ 70–80 min.
- Free-tier sessions cap at ~12 h but will disconnect after ~90 min of
  inactivity. Keep the Colab tab active; don't switch networks mid-run.

Two changes from the Pro recipe:

### A. Persist outputs through disconnects

Run this **before** step 4 so a disconnect doesn't cost you the adapter:

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/medibill/adapters /content/drive/MyDrive/medibill/runs
```

Then in step 4 point `--out` and the `tee` destination at Drive:

```bash
!python -m medibill.sft_colab \
    --dataset datasets/sft_v1.jsonl \
    --eval traces/eval.jsonl \
    --out /content/drive/MyDrive/medibill/adapters/sft_v1/ \
    --batch-size 2 --grad-accum 8 \
    2>&1 | tee /content/drive/MyDrive/medibill/runs/sft_v1_stdout.log
```

### B. Memory-safe T4 overrides

`--batch-size 2 --grad-accum 8` keeps the effective batch at 16 while
halving peak VRAM. If OOM still hits (unusual), add
`--max-new-tokens 96` to shrink the eval-time generation budget.

### If Colab free disconnects mid-training

1. Reconnect. Drive is still mounted from step A.
2. The HuggingFace base model is cached at `~/.cache/huggingface` — not on
   Drive, so the model redownload is the only real cost of a reconnect
   (~90 s).
3. Re-run step 4. `SFTTrainer` will start fresh unless you explicitly pass
   `--resume-from-checkpoint`, which we don't wire in by default. For a
   single-session train, restart-from-scratch is cheaper than wiring
   checkpoint resume.

### If free tier is too flaky

Fallback ranked by setup effort:

1. **Kaggle** — 30 GPU-hours/week free on P100 (faster than T4). `!git
   clone` works. The rest of the recipe is identical.
2. **Modal** — $30 new-user credit; A10 on-demand is ~$0.60/hr so one full
   run costs ~$0.80. Requires a one-time `modal token set` plus a thin
   wrapper around `sft_colab.py`. Defer unless Colab and Kaggle both fail.

---

## Pre-training smoke test (recommended)

Before step 4, run the 60-second setup check to catch packaging / path / GPU
issues cheaply rather than after a 30-minute training run:

```bash
!python -m medibill.colab_smoke                       # basic checks
!python -m medibill.colab_smoke --with-plumbing-eval  # also exercises eval harness
```

Exits non-zero on any failure (missing package, missing dataset, prompt
version drift, parse error). Train only after this exits `0`.

---

## Appendix: private repo or no git access

The main recipe assumes a public repo. Use one of the two paths below if the
repo is private.

### Option A — fine-grained Personal Access Token (keeps commit history)

1. GitHub → Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → **Generate new**.
2. Repository access: **Only select repositories** → pick
   `METAHackthon2026`.
3. Permissions: **Contents → Read-only** is enough.
4. Copy the token (`github_pat_…`).
5. On Colab, clone with the token inline:

   ```bash
   !git clone https://<username>:github_pat_XXXXXXXX@github.com/Algoace1403/METAHackthon2026 /content/METAHackthon2026
   ```

   **Never save the notebook with the token in it.** Delete or redact the
   cell before you close the tab. Colab notebooks auto-save, so the token
   leaks into history if you forget.

### Option B — zip upload (lowest friction; no auth)

1. Locally:

   ```bash
   zip -r medibill.zip METAHackthon2026 \
       -x '*.git*' '__pycache__*' 'adapters/*' '*.pyc'
   ```

2. On Colab: **Files panel → upload `medibill.zip`**.
3. In the notebook:

   ```bash
   !unzip -q medibill.zip -d /content/
   %cd /content/METAHackthon2026
   ```

Loses commit history but needs no auth. Fine for a single training run.
