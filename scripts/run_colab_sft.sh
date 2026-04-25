#!/bin/bash
# MediBill-Env — Colab SFT launcher
#
# Usage on Colab (paste a single cell):
#     !cd /content/METAHackthon2026 && git pull && bash scripts/run_colab_sft.sh
#
# Why this exists: pasting a long multi-line Python+shell command into a Colab
# cell is fragile because Colab's editor auto-indents lines after a `%cd` magic
# and any leading whitespace causes a Python IndentationError BEFORE the shell
# ever sees the command. Putting the actual training invocation in a tracked
# bash script removes that whole class of bug — the user pastes one short
# line, bash runs the file verbatim.
#
# What this does:
# 1. cd into the repo (assumes already cloned to /content/METAHackthon2026)
# 2. invoke medibill.sft_colab with the canonical flags
# 3. tee both stdout and stderr to a log on Drive so a free-tier disconnect
#    does not lose progress
#
# Wallclock on free Colab T4: ~75 min training + ~15 min eval

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/content/METAHackthon2026}"
ADAPTER_DIR="${ADAPTER_DIR:-/content/drive/MyDrive/medibill/adapters/sft_v1/}"
LOG_PATH="${LOG_PATH:-/content/drive/MyDrive/medibill/runs/sft_v1_stdout.log}"

mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$ADAPTER_DIR"

cd "$REPO_ROOT"
echo "[run_colab_sft] cwd=$(pwd)"
echo "[run_colab_sft] adapter_dir=$ADAPTER_DIR"
echo "[run_colab_sft] log_path=$LOG_PATH"
echo "[run_colab_sft] starting at $(date)"

python -m medibill.sft_colab \
    --dataset datasets/sft_v1.jsonl \
    --eval traces/eval.jsonl \
    --out "$ADAPTER_DIR" \
    --batch-size 2 \
    --grad-accum 8 \
    2>&1 | tee "$LOG_PATH"

EXIT=${PIPESTATUS[0]}
echo "[run_colab_sft] finished at $(date) (exit=$EXIT)"
exit "$EXIT"
