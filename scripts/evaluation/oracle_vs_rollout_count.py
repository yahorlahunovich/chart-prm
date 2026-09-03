#!/usr/bin/env python3
"""
oracle_vs_rollout_count.py

Cheap diagnostic before spending Kaggle GPU time generating more rollouts:
using only the 5 rollouts already generated per question, compute the oracle
ceiling (does *any* rollout have the correct final answer) as a function of
how many rollouts are considered, K=1..5. If the curve is still rising
steeply at K=5, more rollouts are likely worth generating; if it's already
flattening, the ceiling is more about which questions the student never
gets right in any attempt, not about needing more attempts.

Restricted to questions with all 5 rollout indices (0-4) present in the
cleaned data, so the comparison across K is apples-to-apples (not just
counting more questions as K grows). No GPU, no API calls -- pure
re-analysis of experiment 001's existing cleaned rollouts.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chart_prm.text_match import answers_match  # noqa: E402


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    cleaned_path = base_dir / "experiments/001_500_reasoning/data/001_500_reasoning_cleaned.jsonl"

    by_question: dict[str, dict[int, tuple[str, str]]] = defaultdict(dict)
    with cleaned_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            qid = str(data["question_id"])
            ridx = data["rollout_index"]
            by_question[qid][ridx] = (data.get("ground_truth", ""), data.get("model_final_answer", ""))

    full_questions = {qid: rollouts for qid, rollouts in by_question.items() if set(rollouts) == {0, 1, 2, 3, 4}}
    n = len(full_questions)
    print(f"Questions with all 5 rollout indices present: {n}")

    print(f"\n{'K (rollouts considered)':<28}{'Oracle accuracy':>18}{'Marginal gain':>16}")
    prev = None
    results = {}
    for k in range(1, 6):
        correct = 0
        for qid, rollouts in full_questions.items():
            for idx in range(k):
                gt, pred = rollouts[idx]
                if answers_match(gt, pred):
                    correct += 1
                    break
        acc = correct / n
        results[k] = acc
        gain = f"{acc - prev:+.1%}" if prev is not None else "--"
        print(f"{k:<28}{acc:>17.1%}{gain:>16}")
        prev = acc

    out_path = base_dir / "experiments/013_oracle_vs_rollout_count.json"
    out_path.write_text(json.dumps({"n_questions": n, "oracle_by_k": results}, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
