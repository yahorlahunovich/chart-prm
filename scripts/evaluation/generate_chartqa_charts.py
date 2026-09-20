#!/usr/bin/env python3
"""Generate publication-quality charts for the ChartQA holdout evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from visualization.style import setup_plot_style, get_model_color, METRIC_PALETTE

MODEL_ORDER = ["base", "sft", "dpo", "step_dpo", "kto", "sft_dpo", "simpo", "pareto_dpo"]
DISPLAY_NAMES = {
    "base": "Base",
    "sft": "SFT",
    "dpo": "Full DPO",
    "step_dpo": "Step-DPO",
    "kto": "KTO",
    "sft_dpo": "SFT→DPO",
    "simpo": "SimPO",
    "pareto_dpo": "Pareto-DPO",
}


def load_data(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_overall_comparison(data: dict, out_path: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 4.8))

    models = [m for m in MODEL_ORDER if m in data]
    labels = [DISPLAY_NAMES[m] for m in models]
    x = np.arange(len(models))
    width = 0.2

    metrics = [
        ("exact_official_pct", "Official Exact-Match (%)", METRIC_PALETTE["Official Exact-Match (%)"]),
        ("token_match_pct", "Token Match (%)", METRIC_PALETTE["Robust Token Match (%)"]),
        ("structure_score_pct", "Structure Score (%)", METRIC_PALETTE["Structured + Correct (%)"]),
        ("recall_gt_in_text_pct", "GT Mentioned in Text (%)", METRIC_PALETTE["GT Mentioned in Text (%)"]),
    ]

    for i, (key, label, color) in enumerate(metrics):
        offset = (i - 1.5) * width
        vals = [data[m][key] for m in models]
        rects = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="white", linewidth=0.5)
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{int(round(h))}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold" if "Exact" in label else "normal",
                )

    ax.set_title("Benchmark Performance Across Alignment Methods on ChartQA Holdout (N=30)", fontsize=11.5, fontweight="bold", pad=12)
    ax.set_ylabel("Percentage (%)", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_accuracy_vs_structure(data: dict, out_path: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    offsets = {
        "Pareto-DPO": (0.0, 3.8),
        "SFT": (-12.5, 5.5),
        "Base": (-12.5, -4.5),
        "SimPO": (12.0, 5.5),
        "SFT→DPO": (12.0, -4.5),
        "Full DPO": (0.0, -3.8),
        "Step-DPO": (0.0, 3.5),
        "KTO": (0.0, 3.5),
    }

    jitters = {
        "Base": (-0.35, 0.0),
        "SFT": (0.0, 0.0),
        "SimPO": (0.35, 0.0),
    }

    models = [m for m in MODEL_ORDER if m in data]
    for m in models:
        disp = DISPLAY_NAMES[m]
        color = get_model_color(disp)
        jx, jy = jitters.get(disp, (0.0, 0.0))
        x_val = data[m]["structure_score_pct"] + jx
        y_val = data[m]["exact_official_pct"] + jy
        raw_x = data[m]["structure_score_pct"]
        raw_y = data[m]["exact_official_pct"]
        recall = data[m]["recall_gt_in_text_pct"]
        size = recall * 5.0

        ax.scatter(x_val, y_val, s=size, color=color, alpha=0.85, edgecolor="black", linewidth=1.2, zorder=4)

        dx, dy = offsets.get(disp, (0, 2.0))
        ax.annotate(
            f"{disp}\n({raw_y:.0f}% Acc, {raw_x:.0f}% Struct)",
            xy=(x_val, y_val),
            xytext=(raw_x + dx, raw_y + dy),
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.92, linewidth=1.1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.8) if (dx != 0 or dy != 0) else None,
            zorder=5,
        )

    ax.set_title("ChartQA Holdout: Exact-Match Accuracy vs. Structural Compliance", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Instruction-Following & Structural Compliance Score (%)", fontsize=10.5)
    ax.set_ylabel("Official Exact-Match Accuracy (%)", fontsize=10.5)
    ax.set_xlim(22, 118)
    ax.set_ylim(24, 74)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")


def main() -> None:
    data_path = ROOT / "experiments/020_chartqa_transfer/holdout/chartqa_holdout_summary.json"
    data = load_data(data_path)
    out_dir = ROOT / "charts/chartqa"

    plot_overall_comparison(data, out_dir / "chartqa_01_overall_comparison.png")
    plot_accuracy_vs_structure(data, out_dir / "chartqa_03_accuracy_vs_structure_tradeoff.png")


if __name__ == "__main__":
    main()
