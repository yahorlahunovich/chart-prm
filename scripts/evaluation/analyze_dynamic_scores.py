#!/usr/bin/env python3
"""
analyze_dynamic_scores.py

Summarizes experiment 010's Phase 2 pilot: how often the judge self-selects
each reward-tree criterion, the resulting 1-3 score distribution, and
coverage across the 100-question pilot set. Writes metrics.md and a chart
matching the project's plotting style. No API calls -- pure analysis of
scores.jsonl already produced by score_steps_dynamic.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_rows(scores_path: Path) -> list[dict]:
    rows = []
    with scores_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=Path("experiments/010_dynamic_scoring_pilot/data/dynamic_scores.jsonl"),
    )
    parser.add_argument(
        "--pilot-ids-path",
        type=Path,
        default=Path("experiments/010_dynamic_scoring_pilot/data/pilot_question_ids.json"),
    )
    parser.add_argument("--tree-path", type=Path, default=Path("experiments/009_reward_tree/data/reward_tree.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/010_dynamic_scoring_pilot"))
    args = parser.parse_args()

    rows = load_rows(args.scores_path)
    with args.pilot_ids_path.open(encoding="utf-8") as handle:
        pilot_ids = set(json.load(handle)["question_ids"])
    with args.tree_path.open(encoding="utf-8") as handle:
        tree = json.load(handle)

    child_to_parent = {
        child["child_id"]: parent["label"]
        for parent in tree["parents"].values()
        for child in parent["children"]
    }
    n_children = sum(len(p["children"]) for p in tree["parents"].values())

    n_steps_total = 0
    n_steps_zero_criteria = 0
    criteria_per_step: list[int] = []
    score_counts: Counter = Counter()
    criterion_usage: Counter = Counter()
    score_by_criterion: dict[str, list[int]] = {}

    covered_qids = set()
    for row in rows:
        covered_qids.add(row["question_id"])
        for step in row["scores"]:
            n_steps_total += 1
            crits = step.get("scores") or []
            criteria_per_step.append(len(crits))
            if len(crits) == 0:
                n_steps_zero_criteria += 1
            for c in crits:
                score_counts[c.get("score")] += 1
                cid = c.get("criterion_id")
                criterion_usage[cid] += 1
                score_by_criterion.setdefault(cid, []).append(c.get("score"))

    total_score_instances = sum(score_counts.values())
    avg_criteria_per_step = sum(criteria_per_step) / len(criteria_per_step) if criteria_per_step else 0.0

    parent_usage: Counter = Counter()
    for cid, n in criterion_usage.items():
        parent_usage[child_to_parent.get(cid, "unknown")] += n

    lines = [
        "# Experiment 010: Dynamic Multi-Criteria Scoring Pilot (DG-PRM Phase 2)\n",
        "\n",
        "## What this is\n",
        "\n",
        "For each rollout of 100 training-pool questions (frozen in "
        "`data/pilot_question_ids.json`, disjoint from the protected holdout), a Gemini judge "
        "(`gemini-3.5-flash-lite`, free tier) is shown the full 33-criterion reward tree from "
        "experiment 009 once, and for every step self-selects which criteria apply and scores "
        "them 1-3 -- blind to the ground-truth answer "
        "(`chart_prm.dynamic_scoring.build_dynamic_scoring_prompt`). This replaces the original "
        "judge's single binary pass/fail-with-ground-truth per step.\n",
        "\n",
        f"- Rollouts scored: **{len(rows)}**\n",
        f"- Questions covered: **{len(covered_qids)} / {len(pilot_ids)}** pilot questions\n",
        f"- Steps scored: **{n_steps_total}**\n",
        "\n## Criteria selection\n\n",
        f"- Average criteria selected per step: **{avg_criteria_per_step:.2f}** (out of {n_children} "
        "available) -- selective, not indiscriminate.\n",
        f"- Steps with zero criteria selected (judge found nothing relevant): "
        f"**{n_steps_zero_criteria} ({n_steps_zero_criteria / n_steps_total:.1%})**\n",
        f"- Distinct criteria actually used at least once: **{len(criterion_usage)} / {n_children}**\n",
        "\n## Score distribution (1=exhibits failure, 2=ambiguous, 3=does not exhibit)\n\n",
        "| Score | Count | Share |\n| --- | ---: | ---: |\n",
    ]
    for s, label in [(1, "1 (fail)"), (2, "2 (ambiguous)"), (3, "3 (pass)")]:
        n = score_counts.get(s, 0)
        lines.append(f"| {label} | {n} | {n / total_score_instances:.1%} |\n")

    lines.append(
        "\n**Caveat worth flagging in the report**: selection and scoring are not fully "
        "decoupled by this prompt -- the judge is asked to select criteria it considers relevant, "
        "which correlates with suspecting a violation, so the 62% fail-rate above should be read "
        "as \"among criteria the judge flagged as worth checking,\" not as a neutral spot-check of "
        "arbitrary dimensions. It is directionally consistent with the original judge's ~41% "
        "step-pass rate on the same rollout pool, but not a clean apples-to-apples comparison.\n"
        if score_counts.get(1, 0) / total_score_instances > 0.5
        else "\n"
    )

    lines.append("\n## Most and least used criteria\n\n| Criterion | Parent | Times used | Avg score |\n| --- | --- | ---: | ---: |\n")
    for cid, n in criterion_usage.most_common(10):
        avg = sum(score_by_criterion[cid]) / len(score_by_criterion[cid])
        lines.append(f"| `{cid}` | {child_to_parent.get(cid, '?')} | {n} | {avg:.2f} |\n")

    lines.append("\n## Usage by parent category\n\n| Parent | Total selections |\n| --- | ---: |\n")
    for label, n in parent_usage.most_common():
        lines.append(f"| {label} | {n} |\n")

    lines.append("\n![Score distribution by parent category](figures/score_distribution_by_parent.png)\n")

    metrics_path = args.output_dir / "metrics.md"
    metrics_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {metrics_path}")

    write_figure(parent_usage, score_by_criterion, child_to_parent, args.output_dir / "figures")


def write_figure(parent_usage: Counter, score_by_criterion: dict, child_to_parent: dict, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.visualization.style import setup_plot_style

    setup_plot_style()
    figures_dir.mkdir(parents=True, exist_ok=True)

    parent_avg_score: dict[str, list[int]] = {}
    for cid, scores in score_by_criterion.items():
        parent_avg_score.setdefault(child_to_parent.get(cid, "unknown"), []).extend(scores)

    labels = list(parent_usage.keys())
    counts = [parent_usage[label] for label in labels]
    avg_scores = [sum(parent_avg_score[label]) / len(parent_avg_score[label]) for label in labels]
    order = np.argsort(counts)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(order))
    colors = plt.cm.RdYlGn([((avg_scores[i] - 1) / 2) for i in order])
    ax.barh(y, [counts[i] for i in order], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Times selected by the judge")
    ax.set_title("Phase 2 pilot: criteria usage and avg score by category", loc="left", pad=10)
    for i, idx in enumerate(order):
        ax.text(counts[idx] + 2, i, f"avg {avg_scores[idx]:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "score_distribution_by_parent.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {figures_dir / 'score_distribution_by_parent.png'}")


if __name__ == "__main__":
    main()
