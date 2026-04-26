"""Generate the base_vs_sft.png chart for the README — the 20% Improvement axis evidence.

Run:
    python3 scripts/make_improvement_chart.py

Produces:
    docs/img/base_vs_sft.png
    docs/img/improvement_per_task.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#0A1628"
CORAL = "#FF5A4E"
GRAY = "#999999"
LIGHT_GRAY = "#E5E5E5"
GREEN = "#0A843D"


# ---------------------------------------------------------------------------
# Chart 1: Base / SFT v1 / GRPO / SFT v2 progression on hard_drift
# ---------------------------------------------------------------------------
def chart_progression() -> Path:
    labels = ["Base Qwen 2.5 3B\n(untrained)", "SFT v1\n(scripted teacher)", "GRPO over SFT v1\n(saturated)", "SFT v2\n(drift-aware teacher)"]
    scores = [0.0000, 0.7573, 0.7575, 0.996]
    # SFT v2 hard_drift mean: 0.996 ± 0.003 (n=4 seeds 16-19)
    colors = [GRAY, GRAY, GRAY, CORAL]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, scores, color=colors, edgecolor=NAVY, linewidth=1.2, width=0.65)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{score:.4f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=NAVY)

    ax.axhline(y=0.7611, color=GRAY, linestyle="--", linewidth=1, alpha=0.6)
    ax.text(3.45, 0.768, "scripted teacher ceiling 0.7611", ha="right", va="bottom",
            fontsize=9, color=GRAY, style="italic")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("composite score on hard_drift", fontsize=12, color=NAVY)
    ax.set_title("MediBill-Env training progression: 3 checkpoints to 0.996",
                 fontsize=15, fontweight="bold", color=NAVY, pad=18)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NAVY)
    ax.spines["bottom"].set_color(NAVY)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", linestyle=":", color=LIGHT_GRAY, alpha=0.7)
    ax.set_axisbelow(True)

    fig.text(0.5, 0.01,
             "n=4 held-out seeds (16–19) · 0 parse failures · "
             "Codex reproducibility protocol verified (sha256 + fresh subprocess × 2)",
             ha="center", fontsize=9, color=GRAY, style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT_DIR / "base_vs_sft.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Chart 2: Per-task lift (Base vs SFT v2) across all 3 tiers
# ---------------------------------------------------------------------------
def chart_per_task_lift() -> Path:
    tasks = ["easy_cashless", "medium_multi_payer", "hard_drift"]
    base = [0.0000, 0.0000, 0.0000]
    sft_v2 = [1.000, 1.000, 0.996]

    x = np.arange(len(tasks))
    w = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.2))
    b1 = ax.bar(x - w/2, base, w, label="Base Qwen 2.5 3B (untrained)",
                color=GRAY, edgecolor=NAVY, linewidth=1)
    b2 = ax.bar(x + w/2, sft_v2, w, label="SFT v2 (drift-aware teacher)",
                color=CORAL, edgecolor=NAVY, linewidth=1)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.4f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=NAVY)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("composite score (n=4 held-out seeds)", fontsize=12, color=NAVY)
    ax.set_title("Base → SFT v2: +0.999 average lift across all 3 task tiers",
                 fontsize=15, fontweight="bold", color=NAVY, pad=18)

    ax.legend(loc="upper left", frameon=False, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NAVY)
    ax.spines["bottom"].set_color(NAVY)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", linestyle=":", color=LIGHT_GRAY, alpha=0.7)
    ax.set_axisbelow(True)

    fig.text(0.5, 0.01,
             "Lift: easy +1.000 · medium +1.000 · hard_drift +0.996 · average +0.999",
             ha="center", fontsize=10, color=NAVY, style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT_DIR / "improvement_per_task.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = chart_progression()
    p2 = chart_per_task_lift()
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
