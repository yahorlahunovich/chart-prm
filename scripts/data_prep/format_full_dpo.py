#!/usr/bin/env python3
"""
Build full-trajectory DPO pairs for holdout-style generation training.

Unlike Step-DPO fragments, each pair is a complete assistant response with
Step N: lines and Final Answer:, so preference learning matches eval decoding.
"""

from __future__ import annotations

import json
import random
import re
import sys
import unicodedata
from pathlib import Path

SEED = 42
MAX_PAIRS_PER_CHART = 3


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("\\%", "%")
    return " ".join(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?|[%+\-]", text))


def answers_match(ground_truth: str, model_answer: str) -> bool:
    expected = normalize_text(ground_truth)
    actual = normalize_text(model_answer)
    if not expected or not actual:
        return False
    if expected == actual:
        return True
    expected_tokens = expected.split()
    actual_tokens = actual.split()
    width = len(expected_tokens)
    return any(
        actual_tokens[index : index + width] == expected_tokens
        for index in range(len(actual_tokens) - width + 1)
    )


def reconstruct_solution(parsed_steps: list, model_final_answer: str) -> str:
    body = "\n".join(str(step).strip() for step in parsed_steps if str(step).strip())
    if not body:
        return ""
    if re.search(r"Final Answer:\s*", body, flags=re.IGNORECASE):
        return body
    answer = str(model_final_answer or "").strip()
    if not answer:
        return body
    return f"{body}\nFinal Answer: {answer}"


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    cleaned_path = base_dir / "experiments/001_500_reasoning/data/001_500_reasoning_cleaned.jsonl"
    evals_path = base_dir / "experiments/001_500_reasoning/data/evaluated_rollouts.jsonl"
    output_path = base_dir / "experiments/001_500_reasoning/data/dpo_pairs.jsonl"

    if not cleaned_path.exists() or not evals_path.exists():
        raise FileNotFoundError(f"Missing inputs: {cleaned_path} or {evals_path}")

    rollout_meta = {}
    with cleaned_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            key = (str(data["question_id"]), data["rollout_index"])
            rollout_meta[key] = data

    chosen_per_chart: dict[str, list] = {}
    rejected_per_chart: dict[str, list] = {}

    with evals_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            qid = str(data["question_id"])
            ridx = data["rollout_index"]
            meta = rollout_meta.get((qid, ridx))
            if meta is None:
                continue
            evaluations = data.get("evaluations") or []
            if not evaluations:
                continue

            solution = reconstruct_solution(
                meta.get("parsed_steps") or [],
                meta.get("model_final_answer", ""),
            )
            if not solution or "Final Answer:" not in solution:
                continue

            record = {
                "rollout_index": ridx,
                "question": meta.get("question", ""),
                "ground_truth": meta.get("ground_truth", ""),
                "solution": solution,
                "model_final_answer": meta.get("model_final_answer", ""),
            }

            all_pass = all(step.get("score") == 1 for step in evaluations)
            has_fail = any(step.get("score") == 0 for step in evaluations)
            correct = answers_match(meta["ground_truth"], meta["model_final_answer"])

            chosen_per_chart.setdefault(qid, [])
            rejected_per_chart.setdefault(qid, [])
            if all_pass and correct:
                chosen_per_chart[qid].append(record)
            elif has_fail:
                rejected_per_chart[qid].append(record)

    pairs = []
    for qid, chosen_list in chosen_per_chart.items():
        rejected_list = rejected_per_chart.get(qid) or []
        if not chosen_list or not rejected_list:
            continue
        combinations = [(c, r) for c in chosen_list for r in rejected_list]
        random.Random(f"{SEED}:{qid}").shuffle(combinations)
        for chosen, rejected in combinations[:MAX_PAIRS_PER_CHART]:
            if normalize_text(chosen["solution"]) == normalize_text(rejected["solution"]):
                continue
            pairs.append(
                {
                    "question_id": qid,
                    "image_path": f"data/CharXiv/images/{qid}.jpg",
                    "question": chosen["question"],
                    "prefix": "",
                    "chosen": chosen["solution"],
                    "rejected": rejected["solution"],
                    "metadata": {
                        "pair_type": "full_trajectory",
                        "chosen_rollout_index": chosen["rollout_index"],
                        "rejected_rollout_index": rejected["rollout_index"],
                        "ground_truth": chosen["ground_truth"],
                        "chosen_final_answer": chosen["model_final_answer"],
                        "rejected_final_answer": rejected["model_final_answer"],
                    },
                }
            )

    if not pairs:
        raise RuntimeError("No full DPO pairs produced.")

    with output_path.open("w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fa_chosen = sum("Final Answer:" in p["chosen"] for p in pairs)
    print(f"Wrote {len(pairs)} full DPO pairs to {output_path}")
    print(f"Chosen Final Answer rate: {fa_chosen}/{len(pairs)}")
    print(f"Mean chosen chars: {sum(len(p['chosen']) for p in pairs) / len(pairs):.1f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — CLI entrypoint
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
