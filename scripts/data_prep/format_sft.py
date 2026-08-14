#!/usr/bin/env python3
"""
Build full-trajectory SFT targets from PRM-evaluated CharXiv rollouts.

Keeps only rollouts where every step scored 1 and the final answer matches
ground truth. Writes intact Step N: / Final Answer: text for supervised
fine-tuning (not Step-DPO fragments).
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

SEED_NOTE = "deterministic: one row per (question_id, rollout_index), no shuffle"


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
    output_path = base_dir / "experiments/001_500_reasoning/data/sft_samples.jsonl"

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

    samples = []
    with evals_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            key = (str(data["question_id"]), data["rollout_index"])
            meta = rollout_meta.get(key)
            if meta is None:
                continue
            evaluations = data.get("evaluations") or []
            if not evaluations:
                continue
            if not all(step.get("score") == 1 for step in evaluations):
                continue
            if not answers_match(meta["ground_truth"], meta["model_final_answer"]):
                continue

            solution = reconstruct_solution(
                meta.get("parsed_steps") or [],
                meta.get("model_final_answer", ""),
            )
            if not solution or "Final Answer:" not in solution:
                continue

            samples.append(
                {
                    "question_id": key[0],
                    "rollout_index": key[1],
                    "image_path": f"data/CharXiv/images/{key[0]}.jpg",
                    "question": meta.get("question", ""),
                    "prefix": "",
                    "solution": solution,
                    "metadata": {
                        "ground_truth": meta.get("ground_truth", ""),
                        "model_final_answer": meta.get("model_final_answer", ""),
                        "n_steps": len(meta.get("parsed_steps") or []),
                        "note": SEED_NOTE,
                    },
                }
            )

    if not samples:
        raise RuntimeError("No SFT samples produced; check rollout evaluations.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    mean_len = sum(len(row["solution"]) for row in samples) / len(samples)
    print(f"Wrote {len(samples)} SFT samples to {output_path}")
    print(f"Mean solution chars: {mean_len:.1f}")
    print(f"Final Answer rate: {sum('Final Answer:' in r['solution'] for r in samples)}/{len(samples)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — CLI entrypoint
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
