"""Generates a representative example plot demonstrating the NeurIPS/ICML research style.

Uses synthetic data for PRM training steps vs. process-level accuracy.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.visualization.style import PALETTE, setup_plot_style


def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """Generates synthetic process-level accuracy curves across training steps."""
    np.random.seed(seed)
    steps = np.linspace(0, 5000, 11, dtype=int)
    num_runs = 5
    records = []

    method_params = {
        "Base": {"start": 0.35, "end": 0.38, "rate": 0.0010},
        "SFT": {"start": 0.35, "end": 0.56, "rate": 0.0012},
        "DPO": {"start": 0.35, "end": 0.64, "rate": 0.0011},
        "PRM": {"start": 0.35, "end": 0.73, "rate": 0.0010},
        "Ours": {"start": 0.35, "end": 0.82, "rate": 0.0009},
    }

    for method, params in method_params.items():
        for run in range(num_runs):
            noise = np.random.normal(0, 0.012, size=len(steps))
            acc = (
                params["start"]
                + (params["end"] - params["start"])
                * (1.0 - np.exp(-params["rate"] * steps))
                + noise
            )
            acc = np.clip(acc, 0.0, 1.0)
            for s, a in zip(steps, acc):
                records.append(
                    {
                        "Training Step": s,
                        "Process Accuracy": a,
                        "Method": method,
                        "Run": run,
                    }
                )

    return pd.DataFrame(records)


def create_example_plot(output_png: str, output_pdf: str) -> None:
    """Renders and saves the representative example plot."""
    setup_plot_style()
    df = generate_synthetic_data()

    fig, ax = plt.subplots()

    color_map = {
        "Base": PALETTE["base"],
        "SFT": PALETTE["sft"],
        "DPO": PALETTE["dpo"],
        "PRM": PALETTE["prm"],
        "Ours": PALETTE["ours"],
    }

    sns.lineplot(
        data=df,
        x="Training Step",
        y="Process Accuracy",
        hue="Method",
        style="Method",
        palette=color_map,
        markers=True,
        dashes=False,
        markersize=6,
        linewidth=2.0,
        ax=ax,
        errorbar=("ci", 95),
    )

    ax.set_ylim(0.30, 0.90)
    ax.set_xlim(0, 5000)

    ax.set_ylabel("Process-Level Accuracy")
    ax.set_xlabel("Training Step")

    ax.set_title("Training Step vs. Process-Level Accuracy", loc="left", pad=10)

    # Clear subtitle indicator marking synthetic data
    ax.text(
        0.99,
        0.03,
        "[Synthetic Data for Style Preview]",
        transform=ax.transAxes,
        fontsize=8,
        color="#7A7A7A",
        ha="right",
        va="bottom",
        style="italic",
    )

    ax.legend(
        title="",
        loc="upper left",
        frameon=False,
        fontsize=9,
        handlelength=2.5,
    )

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out_png = "figures/example_prm_accuracy.png"
    out_pdf = "figures/example_prm_accuracy.pdf"
    create_example_plot(out_png, out_pdf)
    print(f"Plot saved successfully to:\n  - {out_png}\n  - {out_pdf}")
