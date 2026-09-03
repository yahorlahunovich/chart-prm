#!/usr/bin/env python3
"""
Build Phase 3 DPO pairs: Pareto-dominance-filtered preference pairs.

Reuses experiment 011's full-scale dynamic multi-criteria scores (309 training-
pool questions, 1199 rollouts, already computed -- no new Gemini calls needed)
and the reward tree's 9 parent categories to score each rollout on a
9-dimensional vector instead of one scalar. A pair is kept only when the
correct rollout Pareto-dominates the incorrect one -- at least as good on
every category, strictly better on one -- filtering out pairs where the two
rollouts trade off against each other on different failure axes, which would
otherwise be an ambiguous preference signal for DPO.

Output schema matches `format_full_dpo.py`'s exactly, so it drops straight
into the existing `train_dpo.py` / `chart_prm.dpo` training pipeline unchanged.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chart_prm.pareto import build_criterion_to_parent, build_pareto_pairs, per_parent_vector  # noqa: E402

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


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    cleaned_path = base_dir / "experiments/001_500_reasoning/data/001_500_reasoning_cleaned.jsonl"
    dynamic_path = base_dir / "experiments/011_dynamic_scoring_full/data/dynamic_scores.jsonl"
    tree_path = base_dir / "experiments/009_reward_tree/data/reward_tree.json"
    output_dir = base_dir / "experiments/014_pareto_dpo/data"
    output_path = output_dir / "pareto_dpo_pairs.jsonl"

    for path in (cleaned_path, dynamic_path, tree_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    rollout_meta = load_rollout_meta(cleaned_path)
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    criterion_to_parent = build_criterion_to_parent(tree)
    parent_ids = list(tree["parents"].keys())

    candidates_by_question: dict[str, list[dict]] = {}
    n_rows = 0
    n_skipped_no_meta = 0
    n_skipped_no_solution = 0
    with dynamic_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n_rows += 1
            data = json.loads(line)
            qid = str(data["question_id"])
            ridx = data["rollout_index"]
            meta = rollout_meta.get((qid, ridx))
            if meta is None:
                n_skipped_no_meta += 1
                continue

            solution = reconstruct_solution(meta.get("parsed_steps") or [], meta.get("model_final_answer", ""))
            if not solution or "Final Answer:" not in solution:
                n_skipped_no_solution += 1
                continue

            vector = per_parent_vector(data.get("scores") or [], criterion_to_parent, parent_ids)
            correct = answers_match(meta.get("ground_truth", ""), meta.get("model_final_answer", ""))

            candidates_by_question.setdefault(qid, []).append(
                {
                    "solution": solution,
                    "correct": correct,
                    "vector": vector,
                    "rollout_index": ridx,
                    "question": meta.get("question", ""),
                    "ground_truth": meta.get("ground_truth", ""),
                    "model_final_answer": meta.get("model_final_answer", ""),
                }
            )

    pairs = []
    for qid, candidates in candidates_by_question.items():
        for chosen, rejected in build_pareto_pairs(candidates, seed_key=f"pareto:{qid}", max_pairs=MAX_PAIRS_PER_CHART):
            pairs.append(
                {
                    "question_id": qid,
                    "image_path": f"data/CharXiv/images/{qid}.jpg",
                    "question": chosen["question"],
                    "prefix": "",
                    "chosen": chosen["solution"],
                    "rejected": rejected["solution"],
                    "metadata": {
                        "pair_type": "pareto_dominance",
                        "chosen_rollout_index": chosen["rollout_index"],
                        "rejected_rollout_index": rejected["rollout_index"],
                        "ground_truth": chosen["ground_truth"],
                        "chosen_final_answer": chosen["model_final_answer"],
                        "rejected_final_answer": rejected["model_final_answer"],
                        "chosen_vector": chosen["vector"],
                        "rejected_vector": rejected["vector"],
                    },
                }
            )

    if not pairs:
        raise RuntimeError("No Pareto DPO pairs produced.")

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_questions_with_candidates = len(candidates_by_question)
    n_questions_with_pairs = len({p["question_id"] for p in pairs})
    fa_chosen = sum("Final Answer:" in p["chosen"] for p in pairs)
    print(f"Dynamic-score rows read: {n_rows} (skipped: {n_skipped_no_meta} no-meta, {n_skipped_no_solution} no-solution)")
    print(f"Questions with >=1 scored rollout: {n_questions_with_candidates}")
    print(f"Questions that produced >=1 Pareto pair: {n_questions_with_pairs}")
    print(f"Wrote {len(pairs)} Pareto DPO pairs to {output_path}")
    print(f"Chosen Final Answer rate: {fa_chosen}/{len(pairs)}")
    print(f"Mean chosen chars: {sum(len(p['chosen']) for p in pairs) / len(pairs):.1f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — CLI entrypoint
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
