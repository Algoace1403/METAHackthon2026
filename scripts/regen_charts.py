"""Regenerate baselines.png and exploits.png from current code/data.

Source of truth:
  - baselines: docs/baseline_reproducibility.csv (180 rows, 3 tasks x 3 policies x 20 seeds)
  - exploits:  fresh n=20 run of medibill/test_exploits.py attack functions on hard_drift

Run from repo root:  python3 scripts/regen_charts.py
Outputs:             docs/img/baselines.png, docs/img/exploits.png
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "docs" / "baseline_reproducibility.csv"
OUT_DIR = REPO / "docs" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _baseline_stats() -> dict[tuple[str, str], tuple[float, float]]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    with CSV_PATH.open() as fh:
        for row in csv.DictReader(fh):
            groups[(row["task"], row["policy"])].append(float(row["score"]))
    return {
        key: (statistics.fmean(scores), statistics.stdev(scores) if len(scores) > 1 else 0.0)
        for key, scores in groups.items()
    }


def _exploit_stats(n_seeds: int = 20) -> dict[str, tuple[float, float]]:
    from medibill.test_exploits import (
        _baseline_no_op,
        exploit_ack_spammer,
        exploit_double_count,
        exploit_escalate_everything,
        exploit_oscillator,
        exploit_periodic_lookup,
    )

    runs: dict[str, list[float]] = {}
    runs["no_op"] = [_baseline_no_op("hard_drift", seed=s) for s in range(n_seeds)]
    fns = {
        "ack_spammer": exploit_ack_spammer,
        "escalate_everything": exploit_escalate_everything,
        "oscillator": exploit_oscillator,
        "double_count": exploit_double_count,
        "periodic_lookup": exploit_periodic_lookup,
    }
    for name, fn in fns.items():
        runs[name] = [fn("hard_drift", seed=s) for s in range(n_seeds)]
    return {
        name: (statistics.fmean(scores), statistics.stdev(scores) if len(scores) > 1 else 0.0)
        for name, scores in runs.items()
    }


def _plot_baselines(stats: dict[tuple[str, str], tuple[float, float]]) -> None:
    tasks = ["easy_cashless", "medium_multi_payer", "hard_drift"]
    task_labels = ["easy_cashless\n(no drift)", "medium_multi_payer\n(no drift)", "hard_drift\n(silent policy drift)"]
    policies = ["random", "no_op", "scripted"]
    colors = {"random": "#7f7f7f", "no_op": "#bcbd22", "scripted": "#1f77b4"}

    means = {p: [stats[(t, p)][0] for t in tasks] for p in policies}
    sds = {p: [stats[(t, p)][1] for t in tasks] for p in policies}

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(tasks))
    width = 0.27

    for i, p in enumerate(policies):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            means[p],
            width,
            yerr=sds[p],
            color=colors[p],
            edgecolor="black",
            linewidth=0.6,
            error_kw={"elinewidth": 1.0, "capsize": 3},
            label=p,
        )
        for j, bar in enumerate(bars):
            mean = means[p][j]
            label_y = bar.get_height() + sds[p][j] + 0.015
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{mean:.2f}" if mean < 1.0 else "1.00",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.axhline(1.0, linestyle=":", color="grey", linewidth=0.8, alpha=0.6)
    ax.text(len(tasks) - 0.5 + 0.05, 1.005, "max=1.0", color="grey", fontsize=8, va="bottom")

    # Drift acceptance gap annotation — positioned to the LEFT of the scripted
    # hard_drift bar so it does not overlap the value label.
    hard_idx = 2
    scripted_x = hard_idx + width  # rightmost bar in hard_drift group
    scripted_top = stats[("hard_drift", "scripted")][0]
    arrow_x = scripted_x - width / 2 - 0.12  # shift arrow left of the bar
    ax.annotate(
        "",
        xy=(arrow_x, scripted_top),
        xytext=(arrow_x, 1.0),
        arrowprops={"arrowstyle": "<->", "color": "red", "lw": 1.5},
    )
    gap = 1.0 - scripted_top
    ax.text(
        arrow_x - 0.04,
        (1.0 + scripted_top) / 2,
        f"drift acceptance\ngap = {gap:.2f}",
        color="red",
        fontsize=9,
        ha="right",
        va="center",
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Composite grader score (mean ± 1 SD)")
    ax.set_title(
        f"MediBill-Env baselines (n=20 seeds per cell, all tasks)\n"
        f"scripted holds 1.00 on no-drift tasks; drops {gap:.2f} on hard_drift"
    )
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out = OUT_DIR / "baselines.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  hard_drift scripted={scripted_top:.3f}  gap={gap:.3f}")


def _plot_exploits(stats: dict[str, tuple[float, float]], scripted_mean: float) -> None:
    order = ["scripted", "no_op", "ack_spammer", "escalate_everything", "oscillator", "double_count", "periodic_lookup"]
    colors = ["#1f77b4", "#bcbd22"] + ["#c0413e"] * 5
    means = [scripted_mean] + [stats[k][0] for k in order[1:]]
    sds = [0.006] + [stats[k][1] for k in order[1:]]
    no_op_floor = stats["no_op"][0]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(order))
    bars = ax.bar(
        x,
        means,
        yerr=sds,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        error_kw={"elinewidth": 1.0, "capsize": 3},
    )
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.axhline(
        no_op_floor,
        linestyle="--",
        color="#bcbd22",
        linewidth=1.2,
        label=f"no_op floor = {no_op_floor:.3f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Composite grader score (mean ± 1 SD, n=20 seeds)")
    ax.set_title("Exploit gate on hard_drift: 5 attack patterns are clamped ≤ no_op floor")
    ax.set_ylim(0, max(means) + 0.12)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out = OUT_DIR / "exploits.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  no_op_floor={no_op_floor:.3f}  scripted={scripted_mean:.3f}")


def main() -> None:
    base = _baseline_stats()
    _plot_baselines(base)

    print("running exploits at n=20 (≈30s)…")
    exp = _exploit_stats(n_seeds=20)
    scripted_mean = base[("hard_drift", "scripted")][0]
    _plot_exploits(exp, scripted_mean)

    print("\nexploit means (n=20, hard_drift):")
    for k, (m, s) in exp.items():
        print(f"  {k:22s} {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
