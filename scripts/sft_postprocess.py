"""Post-process an SFT training log into publication-ready plots.

Run this once the user pastes back the contents of
``runs/sft_v1_stdout.log`` (the file the Colab notebook ``tee``s into).

It produces two PNGs:

  * ``docs/img/sft_loss_curve.png``  — training loss per logging step
  * ``docs/img/sft_4bar.png``        — random / no_op / scripted / sft_adapter
                                       on hard_drift, side-by-side

Inputs
------
A path to the log file. The script is forgiving — if the eval table is
missing, it produces only the loss curve. If the loss lines are missing,
it produces only the 4-bar.

Usage
-----
    python -m scripts.sft_postprocess runs/sft_v1_stdout.log
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- Loss curve ----------------

LOSS_PATTERNS = [
    re.compile(r"'loss':\s*([0-9.]+)"),
    re.compile(r"\bloss\s*=\s*([0-9.]+)"),
    re.compile(r"\bstep\s+(\d+).*loss\s*[=:]\s*([0-9.]+)"),
]


def parse_loss(log_text: str) -> list[float]:
    """Return a list of training-loss values in temporal order."""
    losses: list[float] = []
    for line in log_text.splitlines():
        for pat in LOSS_PATTERNS:
            m = pat.search(line)
            if m:
                try:
                    val = float(m.group(m.lastindex))
                    if 0.0 < val < 100.0:
                        losses.append(val)
                except ValueError:
                    pass
                break
    return losses


def plot_loss(losses: list[float], out_path: Path) -> None:
    if not losses:
        print("[skip] loss curve — no loss values parsed from log")
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    xs = list(range(1, len(losses) + 1))
    ax.plot(xs, losses, color="#2a7fbf", lw=1.3)
    if len(losses) >= 5:
        # Light moving average for the eye
        win = max(3, len(losses) // 25)
        ma = [
            statistics.mean(losses[max(0, i - win) : i + 1]) for i in range(len(losses))
        ]
        ax.plot(xs, ma, color="red", lw=1.0, alpha=0.7, label=f"moving avg (window={win})")
        ax.legend(loc="upper right", framealpha=0.95)
    ax.set_xlabel("logging step")
    ax.set_ylabel("training loss")
    ax.set_title(
        f"SFT training loss (Qwen2.5-3B + LoRA, {len(losses)} logged steps)\n"
        "DataCollatorForCompletionOnlyLM, masked to assistant tokens"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[ok]   wrote {out_path} ({len(losses)} loss points)")


# ---------------- Eval table ----------------

EVAL_RE = re.compile(
    r"^\s*(?P<task>easy_cashless|medium_multi_payer|hard_drift)\s+"
    r"n=(?P<n>\d+)\s+trained=(?P<sft>[0-9.]+)\s+"
    r"scripted=(?P<scripted>[0-9.]+)",
    re.MULTILINE,
)


def parse_eval(log_text: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for m in EVAL_RE.finditer(log_text):
        out[m.group("task")] = {
            "sft_adapter": float(m.group("sft")),
            "scripted": float(m.group("scripted")),
            "n": int(m.group("n")),
        }
    return out


def plot_4bar(eval_table: dict[str, dict[str, float]], out_path: Path) -> None:
    """Render random/no_op/scripted/sft_adapter on hard_drift only."""
    if "hard_drift" not in eval_table:
        print("[skip] 4-bar — no hard_drift row found in eval table")
        return
    sft = eval_table["hard_drift"]["sft_adapter"]
    # Hard-locked 20-seed measured baselines on hard_drift after grader v3.1
    # (P6 oscillation penalty + B2/B3 bonus gating). See
    # docs/baseline_reproducibility.csv.
    baselines = {"random": 0.108, "no_op": 0.079, "scripted": 0.754}
    names = ["random", "no_op", "scripted", "sft_adapter"]
    means = [baselines["random"], baselines["no_op"], baselines["scripted"], sft]
    colors = ["#888888", "#bbbb44", "#2a7fbf", "#cc3399"]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bars = ax.bar(names, means, color=colors, edgecolor="black", linewidth=0.5)
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + 0.015,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.axhline(
        baselines["scripted"],
        ls="--",
        color="#2a7fbf",
        lw=1.0,
        alpha=0.7,
        label=f"scripted ceiling = {baselines['scripted']:.3f}",
    )
    delta = sft - baselines["scripted"]
    direction = "above" if delta > 0 else "≤"
    ax.set_title(
        f"Trained model on hard_drift: SFT adapter = {sft:.3f} "
        f"({direction} scripted by {abs(delta):.3f})"
    )
    ax.set_ylabel("Composite grader score")
    ax.set_ylim(0, max(0.95, max(means) + 0.10))
    ax.legend(loc="upper left", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[ok]   wrote {out_path}  (sft_adapter on hard_drift = {sft:.3f})")


# ---------------- Main ----------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Path to sft_v1_stdout.log")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/img"),
        help="Directory for output PNGs (default: docs/img)",
    )
    args = parser.parse_args()

    if not args.log_path.exists():
        print(f"[err] log not found: {args.log_path}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_text = args.log_path.read_text()

    losses = parse_loss(log_text)
    plot_loss(losses, args.out_dir / "sft_loss_curve.png")

    eval_table = parse_eval(log_text)
    plot_4bar(eval_table, args.out_dir / "sft_4bar.png")

    if not losses and not eval_table:
        print(
            "[warn] neither loss nor eval table parsed; check log format",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
