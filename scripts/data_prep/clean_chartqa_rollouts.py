"""
clean_chartqa_rollouts.py

Parses the raw ChartQA rollouts (generated on Kaggle, data/ChartQA/data/
generated_chartqa_rollouts.jsonl) into the same structured schema
clean_dataset.py produces for CharXiv (question_id, rollout_index, question,
ground_truth, parsed_steps, model_final_answer) -- reuses parse_model_output
from clean_dataset.py instead of duplicating the Step-N/Final-Answer regex,
so this is a source-dataset swap, not a second implementation of the parser.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_dataset import parse_model_output  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data/ChartQA/data/generated_chartqa_rollouts.jsonl"
OUTPUT_PATH = BASE_DIR / "data/ChartQA/data/generated_chartqa_rollouts_cleaned.jsonl"


def main():
    total_raw = 0
    discarded_missing_delimiters = 0
    discarded_too_long = 0
    total_clean = 0
    unique_questions = set()

    with INPUT_PATH.open("r", encoding="utf-8") as f_in, OUTPUT_PATH.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue

            total_raw += 1
            data = json.loads(line)
            model_output = data.get("model_output", "")

            if "Step 1:" not in model_output or "Final Answer:" not in model_output:
                discarded_missing_delimiters += 1
                continue

            if len(model_output) > 4000:
                discarded_too_long += 1
                continue

            parsed_steps, final_answer = parse_model_output(model_output)
            if not parsed_steps or final_answer is None:
                discarded_missing_delimiters += 1
                continue

            q_id = str(data["question_id"])
            clean_record = {
                "question_id": q_id,
                "rollout_index": int(data.get("rollout_index", 0)),
                "question": data.get("question", ""),
                "ground_truth": str(data.get("ground_truth", "")),
                "parsed_steps": parsed_steps,
                "model_final_answer": final_answer,
            }

            f_out.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
            total_clean += 1
            unique_questions.add(q_id)

    print("=" * 40)
    print("      CHARTQA DATA CLEANING REPORT")
    print("=" * 40)
    print(f"Total raw rollouts processed: {total_raw}")
    print(f"Total discarded (missing delimiters): {discarded_missing_delimiters}")
    print(f"Total discarded (>4000 chars loops): {discarded_too_long}")
    print(f"Total clean rollouts saved: {total_clean}")
    print(f"Total unique questions remaining: {len(unique_questions)}")
    print("=" * 40)


if __name__ == "__main__":
    main()
