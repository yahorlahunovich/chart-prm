"""
generate_finetuning_result_charts.py

Generates publication-quality charts summarizing all fine-tuning experiments,
training dynamics, and benchmark evaluation results on the CharXiv holdout set.

Visualizations:
1. Overall Model Comparison (Exact Match, Token Match, Structured Correct, GT Recall)
2. Reasoning Structure & Instruction-Following Fidelity
3. Error Mode Composition & Hallucination Proxy Breakdown
4. Training Dynamics & Convergence Curves (SFT, DPO, Step-DPO, KTO, SFT->DPO)
5. Disciplinary Domain Performance Heatmap (8 CharXiv Domains)
6. Visual Chart Type Performance Heatmap (All Holdout Chart Types)
7. Pairwise Head-to-Head Net Win Matrix
8. Accuracy vs. Structural Compliance Trade-off Frontier

Outputs saved to charts/ matching NeurIPS/ICML research publication standards.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setup root path for project imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.visualization.style import setup_plot_style, PALETTE

# Canonical display order and color palette for all 6 models
MODEL_ORDER = ["Base", "SFT", "Full DPO", "Step-DPO", "KTO", "SFT→DPO"]
MODEL_PALETTE = {
    "Base": PALETTE.get("Base", "#7A7A7A"),
    "SFT": PALETTE.get("SFT", "#4C78A8"),
    "Full DPO": PALETTE.get("DPO", "#F58518"),
    "Step-DPO": PALETTE.get("Step-DPO", "#B279A2"),
    "KTO": PALETTE.get("KTO", "#72B7B2"),
    "SFT→DPO": PALETTE.get("SFT-DPO", "#E45756"),
}


def load_holdout_data(
    quality_summary_path: Path,
    all_answers_path: Path,
    metadata_path: Path,
    chart_types_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads quality summary and full predictions joined with metadata."""
    df_summary = pd.read_csv(quality_summary_path)

    # Standardize model names in summary
    name_map = {
        "Base": "Base",
        "SFT": "SFT",
        "Full DPO": "Full DPO",
        "DPO": "Full DPO",
        "Step-DPO": "Step-DPO",
        "KTO": "KTO",
        "SFT→DPO": "SFT→DPO",
        "SFT-DPO": "SFT→DPO",
    }
    df_summary["model"] = df_summary["model"].map(name_map).fillna(df_summary["model"])

    # Load metadata
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    with open(chart_types_path, "r") as f:
        chart_types = json.load(f)

    df_answers = pd.read_csv(all_answers_path)
    df_answers["question_id_str"] = df_answers["question_id"].astype(str)
    df_answers["domain"] = df_answers["question_id_str"].map(
        lambda qid: meta.get(qid, {}).get("category", "Unknown").upper()
    )
    df_answers["chart_type"] = df_answers["question_id_str"].map(
        lambda qid: chart_types.get(qid, {}).get("chart_types", ["Unknown"])[0]
        if chart_types.get(qid, {}).get("chart_types")
        else "Unknown"
    )

    return df_summary, df_answers


def plot_01_overall_model_comparison(df_summary: pd.DataFrame, out_path: Path) -> None:
    """Plot 1: Grouped bar chart of core benchmark metrics across the 6 models."""
    setup_plot_style()

    df_plot = df_summary.set_index("model").reindex(MODEL_ORDER).reset_index()

    metrics = [
        ("exact_official", "Official Exact-Match (%)", "#4C78A8"),
        ("token_extracted", "Robust Token Match (%)", "#72B7B2"),
        ("structured_correct", "Structured + Correct (%)", "#54A24B"),
        ("gt_in_response", "GT Mentioned in Text (%)", "#F58518"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.19

    for i, (col, label, color) in enumerate(metrics):
        offset = (i - 1.5) * width
        rects = ax.bar(
            x + offset,
            df_plot[col],
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
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
                    fontsize=8,
                    fontweight="bold" if col == "exact_official" else "normal",
                )

    ax.set_title("Benchmark Performance Across Alignment Methods on CharXiv Holdout (N=100)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 80)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=9.5)

    # Highlight box for key takeaway
    fig.text(
        0.13,
        -0.03,
        "Key Findings: Full DPO (Instruct→DPO) achieves highest accuracy (29% exact, 30% token match).\n"
        "SFT achieves 100% structure compliance. KTO recalls GT in 66% of texts but collapses in structured output.",
        fontsize=9,
        style="italic",
        color="#333",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_02_structure_instruction_following(df_summary: pd.DataFrame, out_path: Path) -> None:
    """Plot 2: Structure compliance & instruction-following fidelity metrics."""
    setup_plot_style()

    df_plot = df_summary.set_index("model").reindex(MODEL_ORDER).reset_index()

    categories = [
        ("structure_score", "Overall Structure Score"),
        ("starts_step1", "Starts with 'Step 1:'"),
        ("has_step2", "Contains 'Step 2:'"),
        ("has_final_answer_plain", "Plain 'Final Answer:'"),
        ("has_preamble", "Conversational Preamble (Penalty)"),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x = np.arange(len(MODEL_ORDER))
    width = 0.16

    colors = ["#4C78A8", "#54A24B", "#72B7B2", "#B279A2", "#E45756"]

    for i, ((col, label), color) in enumerate(zip(categories, colors)):
        offset = (i - 2) * width
        rects = ax.bar(
            x + offset,
            df_plot[col],
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{int(round(h))}%",
                xy=(rect.get_x() + rect.get_width() / 2, max(h, 1.5)),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#222" if h > 0 else "#888",
            )

    ax.set_title("Reasoning Structure and Instruction-Following Fidelity (%)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Compliance Rate (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_03_error_mode_hallucination_breakdown(df_summary: pd.DataFrame, out_path: Path) -> None:
    """Plot 3: 100% stacked bar chart of error modes and hallucination proxies."""
    setup_plot_style()

    df_plot = df_summary.set_index("model").reindex(MODEL_ORDER).reset_index()

    components = [
        ("token_extracted", "Correct Extracted Answer", "#54A24B"),
        ("correct_unextracted", "Correct in text, unextracted", "#72B7B2"),
        ("mentions_gt_wrong_commit", "Mentions GT, commits wrong", "#F58518"),
        ("wrong_committed", "Wrong committed (Hallucination proxy)", "#E45756"),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y_pos = np.arange(len(MODEL_ORDER))

    lefts = np.zeros(len(MODEL_ORDER))

    for col, label, color in components:
        values = df_plot[col].values
        rects = ax.barh(
            y_pos,
            values,
            left=lefts,
            height=0.62,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for i, val in enumerate(values):
            if val >= 5:
                ax.text(
                    lefts[i] + val / 2,
                    y_pos[i],
                    f"{int(round(val))}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white",
                    fontweight="bold",
                )
        lefts += values

    ax.set_yticks(y_pos)
    ax.set_yticklabels(MODEL_ORDER, fontsize=10.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Share of Holdout Questions (%)", fontsize=10.5)
    ax.set_title("Holdout Error Mode Composition & Hallucination Breakdown", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center", ncol=2, frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")


def parse_training_logs(
    sft_log_path: Path,
    dpo_log_path: Path,
    kto_log_path: Path,
    sft_dpo_history_path: Path,
) -> Dict[str, pd.DataFrame]:
    """Parses training loss and margin curves from logs and history files."""
    data = {}

    # Parse SFT log
    if sft_log_path.exists():
        with open(sft_log_path) as f:
            text = f.read()
        sft_matches = re.findall(r'\[Step (\d+)\] SFT Loss: ([\d\.]+)', text)
        if sft_matches:
            steps, losses = zip(*sft_matches)
            df_sft = pd.DataFrame({"step": [int(s) for s in steps], "loss": [float(l) for l in losses]}).drop_duplicates(subset=["step"])
            data["SFT"] = df_sft

    # Parse DPO / Step-DPO log
    if dpo_log_path.exists():
        with open(dpo_log_path) as f:
            text = f.read()
        dpo_matches = re.findall(r'\[Step (\d+)\] Loss: ([\d\.]+) \| Margin: ([\-\d\.]+) \| Acc: ([\d\.]+)%', text)
        if dpo_matches:
            steps, losses, margins, accs = zip(*dpo_matches)
            df_dpo = pd.DataFrame({
                "step": [int(s) for s in steps],
                "loss": [float(l) for l in losses],
                "margin": [float(m) for m in margins],
                "acc": [float(a) for a in accs],
            }).drop_duplicates(subset=["step"])
            data["Step-DPO"] = df_dpo

    # Parse KTO log
    if kto_log_path.exists():
        with open(kto_log_path) as f:
            text = f.read()
        kto_matches = re.findall(r'\[Step (\d+)\] KTO Loss: ([\d\.]+) \| Margin: ([\-\d\.]+)', text)
        if kto_matches:
            steps, losses, margins = zip(*kto_matches)
            df_kto = pd.DataFrame({
                "step": [int(s) for s in steps],
                "loss": [float(l) for l in losses],
                "margin": [float(m) for m in margins],
            }).drop_duplicates(subset=["step"])
            data["KTO"] = df_kto

    # Parse SFT->DPO json history
    if sft_dpo_history_path.exists():
        with open(sft_dpo_history_path) as f:
            hist = json.load(f)
        if isinstance(hist, list) and len(hist) > 0:
            df_sft_dpo = pd.DataFrame(hist)
            data["SFT→DPO"] = df_sft_dpo

    return data


def plot_04_training_dynamics_loss_rewards(logs_data: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """Plot 4: 2x2 grid of training dynamics across all alignment algorithms."""
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_sft, ax_dpo_loss = axes[0, 0], axes[0, 1]
    ax_margins, ax_kto = axes[1, 0], axes[1, 1]

    # Panel A: SFT Loss
    if "SFT" in logs_data:
        df = logs_data["SFT"]
        ax_sft.plot(df["step"], df["loss"], color=MODEL_PALETTE["SFT"], lw=1.8, label="SFT Loss (1 epoch)")
        smooth = df["loss"].rolling(window=10, min_periods=1).mean()
        ax_sft.plot(df["step"], smooth, color="#204060", lw=2.5, label="10-step Moving Avg")
        ax_sft.set_title("(A) Supervised Fine-Tuning (SFT) Loss", fontsize=11, fontweight="bold")
        ax_sft.set_xlabel("Training Step", fontsize=10)
        ax_sft.set_ylabel("Cross-Entropy Loss", fontsize=10)
        ax_sft.grid(True, linestyle="--", alpha=0.3)
        ax_sft.legend(frameon=False, fontsize=8.5)

    # Panel B: DPO / SFT->DPO Loss Trajectory
    if "Step-DPO" in logs_data:
        df_dpo = logs_data["Step-DPO"]
        ax_dpo_loss.plot(df_dpo["step"], df_dpo["loss"].rolling(8, min_periods=1).mean(), color=MODEL_PALETTE["Step-DPO"], lw=2.0, label="Step-DPO Loss")
    if "SFT→DPO" in logs_data:
        df_sft_dpo = logs_data["SFT→DPO"]
        ax_dpo_loss.plot(df_sft_dpo["step"], df_sft_dpo["loss"].rolling(8, min_periods=1).mean(), color=MODEL_PALETTE["SFT→DPO"], lw=2.0, label="SFT→DPO Loss")
    ax_dpo_loss.set_title("(B) Preference Optimization Loss Decay", fontsize=11, fontweight="bold")
    ax_dpo_loss.set_xlabel("Training Step", fontsize=10)
    ax_dpo_loss.set_ylabel("DPO Loss", fontsize=10)
    ax_dpo_loss.grid(True, linestyle="--", alpha=0.3)
    ax_dpo_loss.legend(frameon=False, fontsize=8.5)

    # Panel C: Preference Margin Growth
    if "Step-DPO" in logs_data:
        df_dpo = logs_data["Step-DPO"]
        ax_margins.plot(df_dpo["step"], df_dpo["margin"].rolling(8, min_periods=1).mean(), color=MODEL_PALETTE["Step-DPO"], lw=2.0, label="Step-DPO Margin (r_c - r_r)")
    if "SFT→DPO" in logs_data:
        df_sft_dpo = logs_data["SFT→DPO"]
        ax_margins.plot(df_sft_dpo["step"], df_sft_dpo["reward_margin"].rolling(8, min_periods=1).mean(), color=MODEL_PALETTE["SFT→DPO"], lw=2.0, label="SFT→DPO Margin")
    ax_margins.set_title("(C) Implicit Reward Margin Growth", fontsize=11, fontweight="bold")
    ax_margins.set_xlabel("Training Step", fontsize=10)
    ax_margins.set_ylabel("Reward Margin", fontsize=10)
    ax_margins.grid(True, linestyle="--", alpha=0.3)
    ax_margins.legend(frameon=False, fontsize=8.5)

    # Panel D: KTO Dynamics
    if "KTO" in logs_data:
        df_kto = logs_data["KTO"]
        ax_kto.plot(df_kto["step"], df_kto["loss"].rolling(15, min_periods=1).mean(), color=MODEL_PALETTE["KTO"], lw=2.0, label="KTO Loss")
        ax_kto.plot(df_kto["step"], df_kto["margin"].rolling(15, min_periods=1).mean(), color="#306060", lw=2.0, linestyle="--", label="KTO Margin")
        ax_kto.set_title("(D) KTO Training Stability", fontsize=11, fontweight="bold")
        ax_kto.set_xlabel("Training Step", fontsize=10)
        ax_kto.set_ylabel("Loss / Margin", fontsize=10)
        ax_kto.grid(True, linestyle="--", alpha=0.3)
        ax_kto.legend(frameon=False, fontsize=8.5)

    fig.suptitle("Training Dynamics & Loss Trajectories Across Fine-Tuning Paradigms", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_05_domain_performance_heatmap(df_answers: pd.DataFrame, out_path: Path) -> None:
    """Plot 5: Heatmap of model accuracy across the 8 CharXiv academic disciplines."""
    setup_plot_style()

    model_cols = {
        "base_correct": "Base",
        "sft_correct": "SFT",
        "dpo_correct": "Full DPO",
        "step_dpo_correct": "Step-DPO",
        "kto_correct": "KTO",
        "sft_dpo_correct": "SFT→DPO",
    }

    df_sub = df_answers.copy()
    domain_acc = (df_sub.groupby("domain")[list(model_cols.keys())].mean() * 100).round(1)
    domain_acc.columns = [model_cols[c] for c in domain_acc.columns]
    domain_acc = domain_acc.reindex(columns=MODEL_ORDER)

    domain_order = ["CS", "MATH", "PHYSICS", "ECON", "Q-FIN", "Q-BIO", "EESS", "STAT"]
    domain_acc = domain_acc.reindex(index=[d for d in domain_order if d in domain_acc.index])

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.heatmap(
        domain_acc,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "Official Exact-Match Accuracy (%)"},
        ax=ax,
        linewidths=0.6,
        annot_kws={"size": 10, "weight": "bold"},
    )

    ax.set_title("Model Accuracy Across 8 CharXiv Scientific Disciplines (%)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Alignment Method", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Scientific Domain", fontsize=10.5, labelpad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_06_chart_type_performance(df_answers: pd.DataFrame, out_path: Path) -> None:
    """Plot 6: Heatmap of model accuracy across all visual chart types."""
    setup_plot_style()

    model_cols = {
        "base_correct": "Base",
        "sft_correct": "SFT",
        "dpo_correct": "Full DPO",
        "step_dpo_correct": "Step-DPO",
        "kto_correct": "KTO",
        "sft_dpo_correct": "SFT→DPO",
    }

    # All chart types present in holdout
    ct_order = [
        "Line Chart",
        "Scatter Plot",
        "Bar Chart",
        "Heatmap",
        "Histogram",
        "Box Plot",
        "3D Surface Plot",
    ]
    df_sub = df_answers[df_answers["chart_type"].isin(ct_order)].copy()

    ct_acc = (df_sub.groupby("chart_type")[list(model_cols.keys())].mean() * 100).round(1)
    ct_acc.columns = [model_cols[c] for c in ct_acc.columns]
    ct_acc = ct_acc.reindex(index=ct_order, columns=MODEL_ORDER)

    # Add sample counts to index
    counts = df_sub["chart_type"].value_counts()
    ct_acc.index = [f"{ct} (N={counts.get(ct, 0)})" for ct in ct_acc.index]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.heatmap(
        ct_acc,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "Official Exact-Match Accuracy (%)"},
        ax=ax,
        linewidths=0.6,
        annot_kws={"size": 10, "weight": "bold"},
    )

    ax.set_title("Model Accuracy Across Visual Chart Types on Holdout (%)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Alignment Method", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Chart Type (Sample Size)", fontsize=10.5, labelpad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_07_pairwise_model_head_to_head(df_answers: pd.DataFrame, out_path: Path) -> None:
    """Plot 7: Pairwise Net Win matrix comparing model correctness on identical questions."""
    setup_plot_style()

    model_cols = [
        ("base_correct", "Base"),
        ("sft_correct", "SFT"),
        ("dpo_correct", "Full DPO"),
        ("step_dpo_correct", "Step-DPO"),
        ("kto_correct", "KTO"),
        ("sft_dpo_correct", "SFT→DPO"),
    ]

    n_models = len(model_cols)
    win_matrix = np.zeros((n_models, n_models))

    for i, (col_i, name_i) in enumerate(model_cols):
        for j, (col_j, name_j) in enumerate(model_cols):
            wins_i = ((df_answers[col_i] == 1) & (df_answers[col_j] == 0)).sum()
            wins_j = ((df_answers[col_j] == 1) & (df_answers[col_i] == 0)).sum()
            win_matrix[i, j] = wins_i - wins_j

    df_win = pd.DataFrame(
        win_matrix,
        index=[name for _, name in model_cols],
        columns=[name for _, name in model_cols],
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    mask = np.eye(n_models, dtype=bool)

    sns.heatmap(
        df_win,
        annot=True,
        fmt=".0f",
        cmap="vlag",
        center=0,
        mask=mask,
        cbar_kws={"label": "Net Advantage (Wins - Losses)"},
        ax=ax,
        linewidths=0.6,
        annot_kws={"size": 11, "weight": "bold"},
    )

    ax.set_title("Pairwise Net Win Advantage on Holdout Set (N=100)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_08_accuracy_vs_structure_tradeoff(df_summary: pd.DataFrame, out_path: Path) -> None:
    """Plot 8: 2D Pareto frontier mapping Accuracy vs. Structural Compliance."""
    setup_plot_style()

    df_plot = df_summary.set_index("model").reindex(MODEL_ORDER).reset_index()

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    # Smart text placement with explicit offsets in data coordinates
    offsets = {
        "Base": (-10.5, 0.0),
        "SFT": (0.0, 1.2),
        "Full DPO": (0.0, 1.1),
        "Step-DPO": (0.0, -1.2),
        "KTO": (0.0, 1.2),
        "SFT→DPO": (0.0, -1.2),
    }

    for _, row in df_plot.iterrows():
        model_name = row["model"]
        color = MODEL_PALETTE.get(model_name, "#4C78A8")
        x_val = row["structure_score"]
        y_val = row["exact_official"]
        gt_val = row["gt_in_response"]

        # Bubble size proportional to GT Mention Recall
        size = gt_val * 5.0

        ax.scatter(
            x_val,
            y_val,
            s=size,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )

        dx, dy = offsets.get(model_name, (0, 0.8))

        ax.annotate(
            f"{model_name}\n({y_val:.0f}% Acc, {x_val:.0f}% Struct)",
            xy=(x_val, y_val),
            xytext=(x_val + dx, y_val + dy),
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=color,
                alpha=0.92,
                linewidth=1.1,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=0.9,
                alpha=0.7,
            ) if dx != 0 else None,
            zorder=5,
        )

    # Shaded optimal quadrant
    ax.axvspan(90, 105, color="#54A24B", alpha=0.06, zorder=1)
    ax.text(97.5, 31.8, "High Structure Zone (≥90%)", color="#2E7D32", fontsize=9, fontweight="bold", ha="center")

    ax.set_title("Accuracy vs. Structural Compliance Trade-Off Across Alignment Methods", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Instruction Following / Structure Score (%)", fontsize=11, labelpad=8)
    ax.set_ylabel("Official Exact-Match Accuracy (%)", fontsize=11, labelpad=8)
    ax.set_xlim(8, 107)
    ax.set_ylim(20, 32.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.text(
        0.13,
        -0.03,
        "Note: Bubble diameter is proportional to Ground Truth Mention Recall in text (47% - 66%).\n"
        "Full DPO achieves the optimal Pareto position (29% accuracy, 97.2% structure).",
        fontsize=8.5,
        style="italic",
        color="#333",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT / "experiments/007_sft_dpo_holdout/data/holdout_quality_summary.csv",
    )
    parser.add_argument(
        "--answers-csv",
        type=Path,
        default=REPO_ROOT / "data/test_predictions/all_models_test_answers.csv",
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

    print("Loading holdout benchmark and metadata...")
    df_summary, df_answers = load_holdout_data(
        args.summary_csv, args.answers_csv, args.meta_json, args.chart_types_json
    )

    print("Parsing training dynamics logs...")
    logs_data = parse_training_logs(
        REPO_ROOT / "logs/qwen-vl-sft-custom.log",
        REPO_ROOT / "logs/qwen-vl-step-dpo-custom.log",
        REPO_ROOT / "logs/qwen-vl-kto-custom.log",
        REPO_ROOT / "experiments/006_sft_then_dpo/data/training_history.json",
    )

    print("Generating fine-tuning result charts in:", out_dir)
    plot_01_overall_model_comparison(df_summary, out_dir / "results_01_overall_model_comparison.png")
    plot_02_structure_instruction_following(df_summary, out_dir / "results_02_structure_instruction_following.png")
    plot_03_error_mode_hallucination_breakdown(df_summary, out_dir / "results_03_error_mode_hallucination_breakdown.png")
    plot_04_training_dynamics_loss_rewards(logs_data, out_dir / "results_04_training_dynamics_loss_rewards.png")
    plot_05_domain_performance_heatmap(df_answers, out_dir / "results_05_domain_performance_heatmap.png")
    plot_06_chart_type_performance(df_answers, out_dir / "results_06_chart_type_performance.png")
    plot_07_pairwise_model_head_to_head(df_answers, out_dir / "results_07_pairwise_model_head_to_head.png")
    plot_08_accuracy_vs_structure_tradeoff(df_summary, out_dir / "results_08_accuracy_vs_structure_tradeoff.png")
    print("All fine-tuning result charts generated successfully!")


if __name__ == "__main__":
    main()
