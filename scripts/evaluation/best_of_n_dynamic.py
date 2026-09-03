#!/usr/bin/env python3
"""
best_of_n_dynamic.py

The actual payoff question for all of Phase 1/Phase 2: does the new
dynamic multi-criteria score (experiment 010) pick better rollouts than
the original binary judge score (experiment 008) did? Reruns the exact
best-of-N methodology from experiment 008, but restricted to the same
100-question pilot set for both scores so the comparison is apples-to-
apples (008's headline 27.5% was measured on all 309 eligible questions,
not just this 100-question subset).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chart_prm.dynamic_scoring import dynamic_process_score  # noqa: E402
from chart_prm.text_match import answers_match  # noqa: E402
from chart_prm.verifier import build_candidate, evaluate_question_group, summarize  # noqa: E402


def load_rollout_meta(cleaned_path: Path) -> dict[tuple[str, int], dict]:
    meta = {}
    with cleaned_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            key = (str(data["question_id"]), data["rollout_index"])
            meta[key] = data
    return meta


def build_dynamic_candidate(rollout_meta: dict, dynamic_row: dict, step_aggregation: str = "mean") -> dict | None:
    score = dynamic_process_score(dynamic_row["scores"], step_aggregation=step_aggregation)
    if score is None:
        return None
    final_answer = str(rollout_meta.get("model_final_answer", "")).strip()
    if not final_answer:
        return None
    return {
        "rollout_index": rollout_meta.get("rollout_index"),
        "final_answer": final_answer,
        "process_score": score,
        "correct": answers_match(rollout_meta.get("ground_truth", ""), final_answer),
    }


def group_and_summarize(grouped: dict[str, list[dict]]) -> dict:
    eligible = {qid: c for qid, c in grouped.items() if len(c) >= 2}
    per_question = [evaluate_question_group(c) for c in eligible.values()]
    if not per_question:
        raise RuntimeError("No eligible (>=2 candidate) questions to summarize")
    summary = summarize(per_question)
    summary["n_eligible_questions"] = len(eligible)
    return summary


def main() -> None:
    import argparse

    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dynamic-path",
        type=Path,
        default=base_dir / "experiments/010_dynamic_scoring_pilot/data/dynamic_scores.jsonl",
    )
    parser.add_argument(
        "--pilot-ids-path",
        type=Path,
        default=base_dir / "experiments/010_dynamic_scoring_pilot/data/pilot_question_ids.json",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=base_dir / "experiments/010_dynamic_scoring_pilot/data/best_of_n_comparison.json",
    )
    parser.add_argument(
        "--step-aggregation",
        choices=["mean", "min"],
        default="mean",
        help="How dynamic_process_score collapses multiple flagged criteria within one step",
    )
    args = parser.parse_args()

    cleaned_path = base_dir / "experiments/001_500_reasoning/data/001_500_reasoning_cleaned.jsonl"
    evals_path = base_dir / "experiments/001_500_reasoning/data/evaluated_rollouts.jsonl"
    dynamic_path = args.dynamic_path
    pilot_ids_path = args.pilot_ids_path

    with pilot_ids_path.open(encoding="utf-8") as handle:
        pilot_ids = set(json.load(handle)["question_ids"])

    rollout_meta = load_rollout_meta(cleaned_path)

    # Old (v0) binary-judge candidates, restricted to the pilot set
    old_grouped: dict[str, list[dict]] = {}
    with evals_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            qid = str(data["question_id"])
            if qid not in pilot_ids:
                continue
            meta = rollout_meta.get((qid, data["rollout_index"]))
            if meta is None:
                continue
            candidate = build_candidate(meta, data.get("evaluations") or [])
            if candidate is None:
                continue
            old_grouped.setdefault(qid, []).append(candidate)

    # New (v1) dynamic multi-criteria candidates, same pilot set
    new_grouped: dict[str, list[dict]] = {}
    with dynamic_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            qid = str(data["question_id"])
            meta = rollout_meta.get((qid, data["rollout_index"]))
            if meta is None:
                continue
            candidate = build_dynamic_candidate(meta, data, step_aggregation=args.step_aggregation)
            if candidate is None:
                continue
            new_grouped.setdefault(qid, []).append(candidate)

    old_summary = group_and_summarize(old_grouped)
    new_summary = group_and_summarize(new_grouped)

    print(f"{'Metric':<38}{'Old (v0 binary)':>18}{'New (v1 dynamic)':>18}")
    for key, label in [
        ("n_eligible_questions", "Eligible questions"),
        ("random_baseline_accuracy", "Random baseline"),
        ("majority_vote_accuracy", "Majority vote"),
        ("prm_best_of_n_accuracy", "PRM best-of-N"),
        ("oracle_accuracy", "Oracle (upper bound)"),
        ("prm_accuracy_when_oracle_positive", "PRM | oracle positive"),
    ]:
        old_v = old_summary[key]
        new_v = new_summary[key]
        if key == "n_eligible_questions":
            print(f"{label:<38}{old_v:>18d}{new_v:>18d}")
        else:
            print(f"{label:<38}{old_v:>17.1%}{new_v:>18.1%}")

    args.out_path.write_text(
        json.dumps({"old_v0_binary": old_summary, "new_v1_dynamic": new_summary}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out_path}")


if __name__ == "__main__":
    main()
