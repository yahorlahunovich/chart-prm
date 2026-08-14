#!/usr/bin/env python3
"""
Score holdout generations on structure, answer tiers, and error types.

Goes beyond official exact-match: token-level answer match, ground-truth
mentioned in the full response, instruction-following format, and a
wrong-committed-answer proxy (extracted answer is wrong and GT never appears).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from chart_prm.holdout_metrics import score_generation
from src.visualization.style import setup_plot_style

MODELS = ["base", "sft", "dpo", "step_dpo", "kto", "sft_dpo"]
DISPLAY = {
    "base": "Base",
    "sft": "SFT",
    "dpo": "Full DPO",
    "step_dpo": "Step-DPO",
    "kto": "KTO",
    "sft_dpo": "SFT→DPO",
}
ERROR_ORDER = [
    "correct_extracted",
    "correct_unextracted",
    "mentions_gt_wrong_commit",
    "wrong_committed",
    "no_answer",
]
ERROR_LABELS = {
    "correct_extracted": "Correct extracted answer",
    "correct_unextracted": "GT in text, not extracted",
    "mentions_gt_wrong_commit": "Mentions GT, commits wrong",
    "wrong_committed": "Wrong committed answer",
    "no_answer": "No extractable answer",
}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_all(rows: list[dict]) -> pd.DataFrame:
    scored = []
    for row in rows:
        gt = row["ground_truth"]
        for model in MODELS:
            text = row["responses"].get(model, "") or ""
            official = (row.get("predicted_answers") or {}).get(model, "") or ""
            metrics = score_generation(text, gt, official_pred=official)
            scored.append(
                {
                    "question_id": str(row["question_id"]),
                    "model": model,
                    "ground_truth": gt,
                    **metrics,
                }
            )
    return pd.DataFrame(scored)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        sub = df[df["model"] == model]
        n = len(sub)
        rows.append(
            {
                "model": DISPLAY[model],
                "n": n,
                "exact_official": 100 * sub["exact_official"].mean(),
                "token_extracted": 100 * sub["token_pred"].mean(),
                "gt_in_response": 100 * sub["token_body"].mean(),
                "structured_correct": 100 * sub["structured_correct"].mean(),
                "structure_score": 100 * sub["structure_score"].mean(),
                "starts_step1": 100 * sub["starts_step1"].mean(),
                "has_step2": 100 * sub["has_step2"].mean(),
                "has_final_answer_plain": 100 * sub["has_final_answer_plain"].mean(),
                "has_preamble": 100 * sub["has_preamble"].mean(),
                "format_gap": 100 * ((sub["token_pred"]) & (~sub["exact_official"].astype(bool))).mean(),
                "wrong_committed": 100 * (sub["error_type"] == "wrong_committed").mean(),
                "mentions_gt_wrong_commit": 100 * (sub["error_type"] == "mentions_gt_wrong_commit").mean(),
                "correct_unextracted": 100 * (sub["error_type"] == "correct_unextracted").mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_answer_tiers(summary: pd.DataFrame, out: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    metrics = [
        ("exact_official", "Official exact-match"),
        ("token_extracted", "Token match on extracted"),
        ("format_gap", "Correct, not exact"),
        ("structured_correct", "Structured + correct"),
    ]
    x = range(len(summary))
    width = 0.18
    colors = ["#7A7A7A", "#4C78A8", "#54A24B", "#F58518"]
    for i, ((key, label), color) in enumerate(zip(metrics, colors)):
        offsets = [xi + (i - 1.5) * width for xi in x]
        ax.bar(offsets, summary[key], width=width, label=label, color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["model"])
    ax.set_ylabel("Share of 100 holdout questions (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", ncol=2)
    fig.savefig(out)
    plt.close(fig)


def plot_structure(summary: pd.DataFrame, out: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    metrics = [
        ("starts_step1", "Starts with Step 1:"),
        ("has_step2", "Has Step 2:"),
        ("has_final_answer_plain", "Plain Final Answer:"),
        ("has_preamble", "Conversational preamble"),
    ]
    x = range(len(summary))
    width = 0.18
    colors = ["#4C78A8", "#54A24B", "#F58518", "#E45756"]
    for i, ((key, label), color) in enumerate(zip(metrics, colors)):
        offsets = [xi + (i - 1.5) * width for xi in x]
        ax.bar(offsets, summary[key], width=width, label=label, color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["model"])
    ax.set_ylabel("Share of 100 holdout questions (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", ncol=2)
    fig.savefig(out)
    plt.close(fig)


def plot_error_breakdown(df: pd.DataFrame, out: Path) -> None:
    setup_plot_style()
    counts = []
    for model in MODELS:
        sub = df[df["model"] == model]
        c = Counter(sub["error_type"])
        counts.append({key: 100 * c.get(key, 0) / len(sub) for key in ERROR_ORDER})
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = range(len(MODELS))
    bottoms = [0.0] * len(MODELS)
    colors = ["#54A24B", "#B279A2", "#F58518", "#E45756", "#7A7A7A"]
    for key, color in zip(ERROR_ORDER, colors):
        vals = [row[key] for row in counts]
        ax.bar(list(x), vals, bottom=bottoms, label=ERROR_LABELS[key], color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels([DISPLAY[m] for m in MODELS])
    ax.set_ylabel("Share of 100 holdout questions (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    fig.savefig(out)
    plt.close(fig)


def example_format_gap(df: pd.DataFrame, raw: list[dict], model: str, n: int = 3) -> list[str]:
    by_id = {str(r["question_id"]): r for r in raw}
    hits = df[(df["model"] == model) & df["token_pred"] & ~df["exact_official"].astype(bool)]
    lines = []
    for _, hit in hits.head(n).iterrows():
        rec = by_id[str(hit["question_id"])]
        text = rec["responses"][model]
        snippet = (text or "").replace("\n", " | ")
        if len(snippet) > 280:
            snippet = snippet[:140] + " … " + snippet[-140:]
        lines.append(
            f"- `{hit['question_id']}` GT=`{hit['ground_truth']}` "
            f"official=`{hit['official_pred']}` robust=`{hit['robust_pred']}`: {snippet}"
        )
    return lines


def example_rows(df: pd.DataFrame, raw: list[dict], error_type: str, model: str, n: int = 3) -> list[str]:
    by_id = {str(r["question_id"]): r for r in raw}
    hits = df[(df["model"] == model) & (df["error_type"] == error_type)]
    lines = []
    for _, hit in hits.head(n).iterrows():
        rec = by_id[str(hit["question_id"])]
        text = rec["responses"][model]
        snippet = (text or "").replace("\n", " | ")
        if len(snippet) > 280:
            snippet = snippet[:140] + " … " + snippet[-140:]
        lines.append(
            f"- `{hit['question_id']}` GT=`{hit['ground_truth']}` pred=`{hit['pred']}`: {snippet}"
        )
    return lines


def write_markdown(summary: pd.DataFrame, df: pd.DataFrame, raw: list[dict], out: Path) -> None:
    lines = [
        "# Holdout quality beyond exact-match",
        "",
        "Official exact-match only scores the parsed `Final Answer:` string. "
        "This report adds instruction-following structure, token-level answer match, "
        "and whether the ground truth appears anywhere in the generated text.",
        "",
        "**Correct, not exact** counts extracted answers that match the GT after stripping "
        "markdown/unicode/parentheses, or with a short unit/label (`S = 25`, `10 µA`). "
        "It does not count extra entities (`No News` vs `news`).",
        "",
        "**Wrong committed answer** is a hallucination *proxy*: the model emits a final "
        "answer that does not match the ground truth *and* never mentions the ground truth. "
        "It is not a visual judge of chart entities.",
        "",
        "## Answer quality (% of 100 questions)",
        "",
        "| Model | Official exact | Token match (extracted) | Correct, not exact | GT in full text | Structured + correct |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model']} | {row['exact_official']:.0f} | {row['token_extracted']:.0f} | "
            f"{row['format_gap']:.0f} | {row['gt_in_response']:.0f} | {row['structured_correct']:.0f} |"
        )
    lines += [
        "",
        "## Structure / instruction following (%)",
        "",
        "| Model | Starts `Step 1:` | Has `Step 2:` | Plain `Final Answer:` | Conversational preamble | Structure score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model']} | {row['starts_step1']:.0f} | {row['has_step2']:.0f} | "
            f"{row['has_final_answer_plain']:.0f} | {row['has_preamble']:.0f} | {row['structure_score']:.0f} |"
        )
    lines += [
        "",
        "## Error breakdown (%)",
        "",
        "| Model | Correct extracted | GT in text, not extracted | Mentions GT, commits wrong | Wrong committed | No answer |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        sub = df[df["model"] == model]
        c = Counter(sub["error_type"])
        n = len(sub)
        vals = [100 * c.get(key, 0) / n for key in ERROR_ORDER]
        lines.append(
            f"| {DISPLAY[model]} | " + " | ".join(f"{v:.0f}" for v in vals) + " |"
        )
    by_struct = summary.sort_values("structure_score", ascending=False)
    best_struct = by_struct.iloc[0]
    worst_struct = by_struct.iloc[-1]
    by_wrong = summary.sort_values("wrong_committed")
    least_wrong = by_wrong.iloc[0]
    most_wrong = by_wrong.iloc[-1]
    by_token = summary.sort_values("token_extracted", ascending=False)
    best_token = by_token.iloc[0]
    by_gap = summary.sort_values("format_gap", ascending=False)
    most_gap = by_gap.iloc[0]
    kto = summary[summary["model"] == "KTO"].iloc[0]
    step = summary[summary["model"] == "Step-DPO"].iloc[0]
    lines += [
        "",
        "## How to read this",
        "",
        f"- **Structure:** {best_struct['model']} follows `Step N:` + plain `Final Answer:` best "
        f"(structure score {best_struct['structure_score']:.0f}%). "
        f"{worst_struct['model']} is worst ({worst_struct['structure_score']:.0f}%; "
        f"starts `Step 1:` {worst_struct['starts_step1']:.0f}%, preamble {worst_struct['has_preamble']:.0f}%).",
        f"- **Extracted correctness:** {best_token['model']} has the highest token-match on the extracted "
        f"answer ({best_token['token_extracted']:.0f}%). Official exact-match still uses the holdout "
        f"notebook's whitespace+lowercase equality, so markdown leaks and extra punctuation do not count.",
        f"- **Correct but not exact:** {most_gap['model']} has the largest format gap "
        f"({most_gap['format_gap']:.0f} pp). KTO {kto['format_gap']:.0f} pp; "
        f"Step-DPO {step['format_gap']:.0f} pp.",
        f"- **Wrong committed answer** (hallucination proxy): {least_wrong['model']} commits a wrong "
        f"final value with GT never mentioned least often ({least_wrong['wrong_committed']:.0f}%). "
        f"{most_wrong['model']} does this most ({most_wrong['wrong_committed']:.0f}%). "
        "This is not a visual judge of chart entities.",
        f"- **KTO** mentions GT in the full text most often ({kto['gt_in_response']:.0f}%) but has "
        f"{kto['structured_correct']:.0f}% structured+correct — it writes the answer in prose/markdown.",
        "",
        "## Examples: correct answer, failed official exact-match (KTO)",
    ]
    lines += example_format_gap(df, raw, "kto", n=4) or ["- none"]
    lines += ["", "## Examples: correct answer, failed official exact-match (SFT)"]
    lines += example_format_gap(df, raw, "sft", n=3) or ["- none"]
    lines += ["", "## Examples: correct content, no extractable answer (KTO)"]
    lines += example_rows(df, raw, "correct_unextracted", "kto", n=4) or ["- none"]
    lines += ["", "## Examples: mentions GT but commits a different answer (KTO)"]
    lines += example_rows(df, raw, "mentions_gt_wrong_commit", "kto", n=3) or ["- none"]
    lines += ["", "## Examples: wrong committed answer with GT never mentioned (Full DPO)"]
    lines += example_rows(df, raw, "wrong_committed", "dpo", n=3) or ["- none"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations",
        type=Path,
        default=ROOT / "experiments/005_holdout_eval_suffix_step_dpo/data/holdout_generations.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments/005_holdout_eval_suffix_step_dpo",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = load_rows(args.generations)
    df = score_all(raw)
    summary = summarize(df)
    df.to_csv(out_dir / "data" / "holdout_quality_by_example.csv", index=False)
    summary.to_csv(out_dir / "data" / "holdout_quality_summary.csv", index=False)

    plot_answer_tiers(summary, fig_dir / "answer_tiers.png")
    plot_structure(summary, fig_dir / "structure.png")
    plot_error_breakdown(df, fig_dir / "error_breakdown.png")
    write_markdown(summary, df, raw, out_dir / "quality_metrics.md")
    print(summary.to_string(index=False))
    print(f"Wrote {out_dir / 'quality_metrics.md'}")


if __name__ == "__main__":
    main()
