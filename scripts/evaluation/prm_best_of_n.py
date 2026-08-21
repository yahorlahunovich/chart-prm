#!/usr/bin/env python3
"""
prm_best_of_n.py

Answers a question nothing else in this repo asks: is the PRM judge actually
useful as a *verifier*, not just as a training-data label?

Every question in the 500-question pool has up to 5 rollouts with per-step
judge scores already sitting in `evaluated_rollouts.jsonl`. For every question
with >=2 scored rollouts, this script compares four ways of picking a final
answer out of the candidates that already exist:

  - random   : expected accuracy of picking one candidate uniformly at random
  - PRM      : pick the candidate with the highest mean step-pass rate
  - majority : pick the most common final answer across candidates
  - oracle   : upper bound — 1 if *any* candidate is correct

No new generation and no new API calls: this is pure re-analysis of data
`scripts/evaluation/evaluate_rollouts_meta.py` already produced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chart_prm.verifier import build_candidate, evaluate_question_group, summarize  # noqa: E402


def load_rollout_meta(cleaned_path: Path) -> dict[tuple[str, int], dict]:
    meta: dict[tuple[str, int], dict] = {}
    with cleaned_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            key = (str(data["question_id"]), data["rollout_index"])
            meta[key] = data
    return meta


def group_candidates_by_question(cleaned_path: Path, evals_path: Path) -> dict[str, list[dict]]:
    rollout_meta = load_rollout_meta(cleaned_path)
    grouped: dict[str, list[dict]] = {}
    with evals_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            qid = str(data["question_id"])
            key = (qid, data["rollout_index"])
            meta = rollout_meta.get(key)
            if meta is None:
                continue
            candidate = build_candidate(meta, data.get("evaluations") or [])
            if candidate is None:
                continue
            grouped.setdefault(qid, []).append(candidate)
    return grouped


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    cleaned_path = base_dir / "experiments/001_500_reasoning/data/001_500_reasoning_cleaned.jsonl"
    evals_path = base_dir / "experiments/001_500_reasoning/data/evaluated_rollouts.jsonl"
    out_dir = base_dir / "experiments/008_prm_best_of_n"
    data_dir = out_dir / "data"
    figures_dir = out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not cleaned_path.exists() or not evals_path.exists():
        raise FileNotFoundError(f"Missing inputs: {cleaned_path} or {evals_path}")

    grouped = group_candidates_by_question(cleaned_path, evals_path)
    eligible = {qid: cands for qid, cands in grouped.items() if len(cands) >= 2}

    n_questions_total = len(grouped)
    n_questions_eligible = len(eligible)
    candidate_counts = [len(c) for c in eligible.values()]

    per_question_results = {qid: evaluate_question_group(cands) for qid, cands in eligible.items()}
    summary = summarize(list(per_question_results.values()))
    summary["n_questions_with_any_scored_rollout"] = n_questions_total
    summary["n_questions_eligible_ge2_candidates"] = n_questions_eligible
    summary["mean_candidates_per_eligible_question"] = (
        sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
    )

    results_path = data_dir / "verifier_results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"summary": summary, "per_question": per_question_results},
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Eligible questions (>=2 scored rollouts): {n_questions_eligible} / {n_questions_total}")
    print(f"Mean candidates per eligible question: {summary['mean_candidates_per_eligible_question']:.2f}")
    print()
    print(f"{'Strategy':<28}{'Accuracy':>10}")
    print(f"{'random (expected)':<28}{summary['random_baseline_accuracy']:>9.1%}")
    print(f"{'majority vote':<28}{summary['majority_vote_accuracy']:>9.1%}")
    print(f"{'PRM best-of-N':<28}{summary['prm_best_of_n_accuracy']:>9.1%}")
    print(f"{'oracle (upper bound)':<28}{summary['oracle_accuracy']:>9.1%}")
    print()
    print(
        "PRM accuracy on questions where a correct candidate exists: "
        f"{summary['prm_accuracy_when_oracle_positive']:.1%}"
    )
    print(f"\nWrote {results_path}")

    write_figure(summary, figures_dir)
    write_metrics_md(summary, n_questions_total, out_dir)


def write_figure(summary: dict, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.visualization.style import PALETTE, setup_plot_style

    setup_plot_style()

    order = [
        ("random\n(expected)", summary["random_baseline_accuracy"], PALETTE["base"]),
        ("majority\nvote", summary["majority_vote_accuracy"], PALETTE["sft"]),
        ("PRM\nbest-of-N", summary["prm_best_of_n_accuracy"], PALETTE["prm"]),
        ("oracle\n(upper bound)", summary["oracle_accuracy"], "#B0B0B0"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = range(len(order))
    bars = ax.bar(
        x,
        [v for _, v, _ in order],
        color=[c for _, _, c in order],
        width=0.6,
    )
    for bar, (_, v, _) in zip(bars, order):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.012,
            f"{v:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([label for label, _, _ in order])
    ax.set_ylabel("Final-answer accuracy")
    ax.set_ylim(0, min(1.0, max(v for _, v, _ in order) + 0.12))
    ax.set_title(
        f"Selecting among existing rollouts (n={summary['n_questions_eligible_ge2_candidates']} questions)",
        loc="left",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "best_of_n_accuracy.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "best_of_n_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {figures_dir / 'best_of_n_accuracy.png'}")


def write_metrics_md(summary: dict, n_questions_total: int, out_dir: Path) -> None:
    lines = [
        "# Experiment 008: PRM Best-of-N Verifier\n",
        "\n",
        "## Question\n",
        "\n",
        "Every other use of the PRM judge in this project is offline: it only shapes "
        "SFT/DPO/KTO training data. Does the judge's step-level score also work as a "
        "*verifier* — picking the best of several already-generated rollouts — better "
        "than picking one at random or taking a majority vote?\n",
        "\n",
        "## Data\n",
        "\n",
        f"- {n_questions_total} questions have at least one judged rollout.\n",
        f"- {summary['n_questions_eligible_ge2_candidates']} questions have >=2 judged rollouts "
        "(needed for best-of-N to mean anything); those are what this report scores.\n",
        f"- Mean candidates per eligible question: {summary['mean_candidates_per_eligible_question']:.2f} "
        "(max 5, since generation used 5 rollouts/question).\n",
        "- No new generation or judge calls: this reuses "
        "`experiments/001_500_reasoning/data/{001_500_reasoning_cleaned,evaluated_rollouts}.jsonl`.\n",
        "\n",
        "## Selection strategies\n",
        "\n",
        "| Strategy | Rule |\n",
        "| --- | --- |\n",
        "| Random | Expected accuracy of picking one candidate uniformly at random (mean correctness "
        "across all candidates for the question) |\n",
        "| Majority vote | Most common final answer across candidates wins (self-consistency) |\n",
        "| PRM best-of-N | Candidate with the highest mean step-pass rate wins |\n",
        "| Oracle | Upper bound: 1 if *any* candidate for the question is correct |\n",
        "\n",
        "## Results\n",
        "\n",
        "| Strategy | Accuracy |\n",
        "| --- | ---: |\n",
        f"| Random (expected) | {summary['random_baseline_accuracy']:.1%} |\n",
        f"| Majority vote | {summary['majority_vote_accuracy']:.1%} |\n",
        f"| **PRM best-of-N** | **{summary['prm_best_of_n_accuracy']:.1%}** |\n",
        f"| Oracle (upper bound) | {summary['oracle_accuracy']:.1%} |\n",
        "\n",
        f"PRM accuracy restricted to questions where a correct candidate exists among the pool: "
        f"**{summary['prm_accuracy_when_oracle_positive']:.1%}** "
        "(this is the judge's precision as a verifier when there is something correct to find).\n",
        "\n",
        "![Best-of-N accuracy](figures/best_of_n_accuracy.png)\n",
        "\n",
        "## Reading\n",
        "\n",
        "Fill in after inspecting the numbers above: does PRM best-of-N beat random and majority "
        "vote, and how far below oracle does it land? A gap between PRM and oracle quantifies "
        "how much judge-quality — not generation quality — is currently costing the pipeline.\n",
    ]
    metrics_path = out_dir / "metrics.md"
    metrics_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
