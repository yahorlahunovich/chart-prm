"""generate_rollout_charts.py

Generates publication-quality charts 01 through 13 analyzing reasoning rollouts
evaluated by Meta PRM Judge on CharXiv 500 reasoning subset.

Uses SciencePlots scientific theme and Paul Tol color-blind safe palettes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.style import (
    EVAL_PALETTE,
    PALETTE,
    get_eval_color,
    setup_plot_style,
)


def load_rollout_data(
    evaluated_rollouts_path: Path,
    metadata_path: Path,
    chart_types_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and flattens evaluated rollouts joined with question metadata."""
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    with open(chart_types_path, "r") as f:
        chart_types = json.load(f)

    data = []
    with open(evaluated_rollouts_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    rows = []
    for item in data:
        q_id = str(item["question_id"])
        r_idx = item["rollout_index"]
        cat = meta.get(q_id, {}).get("category", "Unknown")

        ctype_list = chart_types.get(q_id, {}).get("chart_types", ["Unknown"])
        ctype = ctype_list[0] if len(ctype_list) > 0 else "Unknown"

        evals = item.get("evaluations") or []
        for i, step in enumerate(evals):
            rows.append(
                {
                    "question_id": q_id,
                    "rollout_index": r_idx,
                    "category": cat,
                    "chart_type": ctype,
                    "step_index": step.get("step_index", i),
                    "score": step.get("score"),
                    "analysis_len": len(step.get("analysis", "")),
                    "total_steps": len(evals),
                }
            )

    df_steps = pd.DataFrame(rows)

    df_steps_sorted = df_steps.sort_values(["question_id", "rollout_index", "step_index"]).copy()
    df_steps_sorted["prev_score"] = df_steps_sorted.groupby(["question_id", "rollout_index"])["score"].shift(1)
    cascade_data = df_steps_sorted.dropna(subset=["prev_score"]).copy()

    return df_steps, cascade_data


def plot_01_overall_accuracy(df_steps: pd.DataFrame, out_path: Path) -> None:
    """01: Overall Step-Level Accuracy Distribution."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    score_counts = df_steps["score"].value_counts().sort_index()
    colors = [get_eval_color(0), get_eval_color(1)]
    labels = ["0 (Incorrect)", "1 (Correct)"]

    bars = ax.bar(labels, score_counts.values, color=colors, width=0.45, edgecolor="none")
    ax.set_title("Overall Step-Level Accuracy Distribution", loc="center", pad=10)
    ax.set_xlabel("Step Score")
    ax.set_ylabel("Count of Steps")
    ax.set_ylim(0, max(score_counts.values) * 1.18)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_02_score_progression(df_steps: pd.DataFrame, out_path: Path) -> None:
    """02: Average Score Progression by Step Index."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    step_counts = df_steps["step_index"].value_counts()
    valid_steps = step_counts[step_counts > 10].index
    df_filtered = df_steps[df_steps["step_index"].isin(valid_steps)]

    sns.lineplot(
        data=df_filtered,
        x="step_index",
        y="score",
        color="#0077BB",
        marker="o",
        markersize=5,
        linewidth=1.8,
        errorbar=("ci", 95),
        ax=ax,
    )

    ax.set_title("Average Score Progression by Step Index", loc="center", pad=10)
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Mean Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(sorted(valid_steps))

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_03_rollout_success(df_steps: pd.DataFrame, out_path: Path) -> None:
    """03: Rollout Success Rate (Sequence Correctness)."""
    setup_plot_style()
    rollout_success = (
        df_steps.groupby(["question_id", "rollout_index"])["score"].min().reset_index()
    )
    rollout_success["status"] = rollout_success["score"].map(
        {1: "Perfect (All 1s)", 0: "Has Errors (Min 0)"}
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    status_counts = rollout_success["status"].value_counts()[
        ["Perfect (All 1s)", "Has Errors (Min 0)"]
    ]
    colors = [get_eval_color(1), get_eval_color(0)]

    bars = ax.bar(status_counts.index, status_counts.values, color=colors, width=0.45)
    ax.set_title("Rollout Success Rate (Sequence Correctness)", loc="center", pad=10)
    ax.set_xlabel("Rollout Status")
    ax.set_ylabel("Count of Rollouts")
    ax.set_ylim(0, max(status_counts.values) * 1.18)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_04_first_error_position(df_steps: pd.DataFrame, out_path: Path) -> None:
    """04: Position of First Error in Failed Rollouts."""
    setup_plot_style()
    errors = df_steps[df_steps["score"] == 0]
    first_errors = (
        errors.groupby(["question_id", "rollout_index"])["step_index"].min().reset_index()
    )
    err_counts = first_errors["step_index"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(
        err_counts.index.astype(str),
        err_counts.values,
        color="#CC6677",
        width=0.55,
        edgecolor="none",
    )
    ax.set_title("Position of First Error in Failed Rollouts", loc="center", pad=10)
    ax.set_xlabel("Step Index of First Error")
    ax.set_ylabel("Count of Rollouts")
    ax.set_ylim(0, max(err_counts.values) * 1.18)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_05_question_difficulty(df_steps: pd.DataFrame, out_path: Path) -> None:
    """05: Question Difficulty Distribution."""
    setup_plot_style()
    q_acc = df_steps.groupby("question_id")["score"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sns.histplot(
        data=q_acc,
        x="score",
        bins=15,
        kde=True,
        color="#0077BB",
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )
    ax.set_title("Question Difficulty Distribution", loc="center", pad=10)
    ax.set_xlabel("Mean Accuracy per Question")
    ax.set_ylabel("Number of Questions")
    ax.set_xlim(0.0, 1.0)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_06_length_vs_accuracy(df_steps: pd.DataFrame, out_path: Path) -> None:
    """06: Rollout Length vs. Accuracy."""
    setup_plot_style()
    rollout_stats = (
        df_steps.groupby(["question_id", "rollout_index"])
        .agg({"score": "mean", "total_steps": "first"})
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sns.boxplot(
        data=rollout_stats,
        x="total_steps",
        y="score",
        color="#88CCEE",
        width=0.45,
        fliersize=2.5,
        linewidth=1.0,
        ax=ax,
    )
    ax.set_title("Rollout Length vs. Mean Step Score", loc="center", pad=10)
    ax.set_xlabel("Total Steps in Rollout")
    ax.set_ylabel("Mean Step Score")
    ax.set_ylim(-0.05, 1.05)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_07_score_variance(df_steps: pd.DataFrame, out_path: Path) -> None:
    """07: Step Score Variance per Question."""
    setup_plot_style()
    q_var = df_steps.groupby("question_id")["score"].var().fillna(0).reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sns.histplot(
        data=q_var,
        x="score",
        bins=15,
        color="#EE7733",
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )
    ax.set_title("Step Score Variance per Question", loc="center", pad=10)
    ax.set_xlabel("Score Variance")
    ax.set_ylabel("Number of Questions")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_08_error_cascade(cascade_data: pd.DataFrame, out_path: Path) -> None:
    """08: Error Cascade Analysis (Step N+1 Score | Step N Score)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    palette_cascade = {0: get_eval_color(0), 1: get_eval_color(1)}

    sns.countplot(
        data=cascade_data,
        x="prev_score",
        hue="score",
        palette=palette_cascade,
        ax=ax,
    )
    ax.set_title("Error Cascade Analysis (Step N+1 Score | Step N Score)", loc="center", pad=10)
    ax.set_xlabel("Score at Step N")
    ax.set_ylabel("Count of Steps N+1")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0 (Incorrect)", "1 (Correct)"])

    leg = ax.legend(title="Score at N+1", loc="upper right", frameon=False)
    for text, label in zip(leg.get_texts(), ["0 (Incorrect)", "1 (Correct)"]):
        text.set_text(label)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_09_analysis_length(df_steps: pd.DataFrame, out_path: Path) -> None:
    """09: PRM Judge Explanation Length by Step Score."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    palette_box = {0: get_eval_color(0), 1: get_eval_color(1)}

    sns.boxplot(
        data=df_steps,
        x="score",
        y="analysis_len",
        hue="score",
        palette=palette_box,
        legend=False,
        width=0.45,
        fliersize=2.5,
        linewidth=1.0,
        ax=ax,
    )
    ax.set_title("PRM Judge Explanation Length by Step Score", loc="center", pad=10)
    ax.set_xlabel("Step Score")
    ax.set_ylabel("Analysis Length (characters)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0 (Incorrect)", "1 (Correct)"])

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_10_terminal_accuracy(df_steps: pd.DataFrame, out_path: Path) -> None:
    """10: Terminal State (Final Step) Accuracy."""
    setup_plot_style()
    terminal_steps = df_steps[df_steps["step_index"] == (df_steps["total_steps"] - 1)]
    term_counts = terminal_steps["score"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    colors = [get_eval_color(0), get_eval_color(1)]
    labels = ["0 (Incorrect)", "1 (Correct)"]

    bars = ax.bar(labels, term_counts.values, color=colors, width=0.45, edgecolor="none")
    ax.set_title("Terminal State (Final Step) Accuracy", loc="center", pad=10)
    ax.set_xlabel("Final Step Score")
    ax.set_ylabel("Count of Final Steps")
    ax.set_ylim(0, max(term_counts.values) * 1.18)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_11_domain_accuracy(df_steps: pd.DataFrame, out_path: Path) -> None:
    """11: Average Reasoning Accuracy by Academic Category."""
    setup_plot_style()
    cat_acc = (
        df_steps.groupby("category")["score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bars = ax.bar(cat_acc["category"], cat_acc["score"], color="#0077BB", width=0.55, edgecolor="none")
    ax.set_title("Average Reasoning Accuracy by Academic Category", loc="center", pad=10)
    ax.set_xlabel("Academic Category")
    ax.set_ylabel("Mean Step Score")
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=30, ha="right")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_12_chart_type_accuracy(df_steps: pd.DataFrame, out_path: Path) -> None:
    """12: Average Reasoning Accuracy by Chart Type (Top 15)."""
    setup_plot_style()
    top_chart_types = df_steps["chart_type"].value_counts().nlargest(15).index
    df_top_charts = df_steps[df_steps["chart_type"].isin(top_chart_types)]
    chart_acc = (
        df_top_charts.groupby("chart_type")["score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(chart_acc["chart_type"], chart_acc["score"], color="#0077BB", width=0.55, edgecolor="none")
    ax.set_title("Average Reasoning Accuracy by Chart Type (Top 15)", loc="center", pad=10)
    ax.set_xlabel("Chart Type")
    ax.set_ylabel("Mean Step Score")
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=35, ha="right")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_13_hallucinated_correctness(cascade_data: pd.DataFrame, out_path: Path) -> None:
    """13: Hallucinated Correctness (Recovery from Error)."""
    setup_plot_style()
    recovery_cases = cascade_data[
        (cascade_data["prev_score"] == 0) & (cascade_data["score"] == 1)
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    if len(recovery_cases) > 0:
        rec_counts = recovery_cases["step_index"].value_counts().sort_index()
        bars = ax.bar(
            rec_counts.index.astype(str),
            rec_counts.values,
            color="#44AA99",
            width=0.5,
            edgecolor="none",
        )
        ax.set_title("Error Recovery Step Index (0 -> 1 Transition)", loc="center", pad=10)
        ax.set_xlabel("Step Index where Score became 1")
        ax.set_ylabel("Count of Occurrences")
        ax.set_ylim(0, max(rec_counts.values) * 1.18)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    else:
        ax.text(0.5, 0.5, "No recovery cases found", ha="center", va="center")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rollout analysis charts 01-13")
    parser.add_argument(
        "--eval-jsonl",
        type=Path,
        default=REPO_ROOT / "experiments/001_500_reasoning/data/evaluated_rollouts.jsonl",
    )
    parser.add_argument(
        "--meta-json",
        type=Path,
        default=REPO_ROOT / "data/CharXiv/data/image_metadata_val.json",
    )
    parser.add_argument(
        "--chart-types-json",
        type=Path,
        default=REPO_ROOT / "data/CharXiv/data/chart_types_val.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "charts",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading evaluated rollouts from {args.eval_jsonl}...")
    df_steps, cascade_data = load_rollout_data(
        args.eval_jsonl, args.meta_json, args.chart_types_json
    )
    print(f"Loaded {len(df_steps)} step records across {df_steps['question_id'].nunique()} questions.")

    print(f"Generating charts 01 through 13 in {out_dir}...")
    plot_01_overall_accuracy(df_steps, out_dir / "01_overall_accuracy.png")
    plot_02_score_progression(df_steps, out_dir / "02_score_progression.png")
    plot_03_rollout_success(df_steps, out_dir / "03_rollout_success.png")
    plot_04_first_error_position(df_steps, out_dir / "04_first_error_position.png")
    plot_05_question_difficulty(df_steps, out_dir / "05_question_difficulty.png")
    plot_06_length_vs_accuracy(df_steps, out_dir / "06_length_vs_accuracy.png")
    plot_07_score_variance(df_steps, out_dir / "07_score_variance.png")
    plot_08_error_cascade(cascade_data, out_dir / "08_error_cascade.png")
    plot_09_analysis_length(df_steps, out_dir / "09_analysis_length.png")
    plot_10_terminal_accuracy(df_steps, out_dir / "10_terminal_accuracy.png")
    plot_11_domain_accuracy(df_steps, out_dir / "11_domain_accuracy.png")
    plot_12_chart_type_accuracy(df_steps, out_dir / "12_chart_type_accuracy.png")
    plot_13_hallucinated_correctness(cascade_data, out_dir / "13_hallucinated_correctness.png")

    print("Charts 01-13 generated successfully!")


if __name__ == "__main__":
    main()
