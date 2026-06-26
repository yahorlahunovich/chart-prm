"""Evaluator for VLM inference using CharXiv benchmark."""
import sys
import json
import time
import traceback
import os
from pathlib import Path

CHARXIV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "CharXiv"
sys.path.insert(0, str(CHARXIV_PATH / "src"))

from descriptive_utils import build_descriptive_quries  # noqa: E402


def load_charxiv_data(num_samples=128):
    with open(CHARXIV_PATH / "data" / "descriptive_val.json") as f:
        data = json.load(f)
    queries = build_descriptive_quries(data, str(CHARXIV_PATH / "images"))
    if num_samples is not None:
        queries = dict(list(queries.items())[:num_samples])
    return queries, data


def evaluate(program):
    NUM_SAMPLES = int(os.environ.get("EVAL_NUM_SAMPLES", "128"))
    LOG_PATH = os.environ.get("EVAL_LOG_PATH", None)

    queries, ground_truth_data = load_charxiv_data(NUM_SAMPLES)

    num_errors = 0
    records = []

    start_time = time.time()

    for query_key, query in queries.items():
        query_start = time.time()
        error_message = None

        try:
            response = program.vlm_inference(
                image_path=query["figure_path"],
                question=query["question"],
            )
            query["response"] = response

        except Exception as e:
            error_message = str(e)
            print(f"Error on {query_key}: {e}")
            traceback.print_exc()
            query["response"] = "ERROR"
            num_errors += 1

        query_time = time.time() - query_start
        query["latency"] = query_time
        query["error"] = error_message

    total_time = time.time() - start_time

    correct = 0
    total = 0

    for query_key, query in queries.items():
        if "response" not in query:
            continue

        figure_id, subq_idx = query_key.split("_")
        gt_entry = ground_truth_data.get(figure_id)

        if gt_entry is None:
            continue

        gt_answer = gt_entry["answers"][int(subq_idx)]
        model_response = str(query["response"]).strip()

        is_correct = model_response.lower() == gt_answer.lower()

        if is_correct:
            correct += 1

        total += 1

        records.append({
            "query_key": query_key,
            "figure_id": figure_id,
            "subq_idx": int(subq_idx),
            "image_path": query.get("figure_path"),
            "question": query.get("question"),
            "ground_truth": gt_answer,
            "model_response": model_response,
            "correct": is_correct,
            "latency": query.get("latency"),
            "error": query.get("error"),
        })

    accuracy = correct / total if total > 0 else 0.0

    results = {
        "accuracy": accuracy,
        "num_evaluated": total,
        "num_errors": num_errors,
        "total_time": total_time,
        "avg_time_per_query": total_time / total if total > 0 else 0.0,
    }

    if LOG_PATH is not None:
        log_path = Path(LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary_path = log_path.with_suffix(".summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"Saved per-example log to: {log_path}")
        print(f"Saved summary to: {summary_path}")

    return results


if __name__ == "__main__":
    import importlib

    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <module_name_without_py>")
        print("Example: python evaluate.py starting_scripts")
        sys.exit(1)

    module_name = sys.argv[1].replace(".py", "")
    program = importlib.import_module(module_name)

    results = evaluate(program)

    print("\n=== Evaluation results ===")
    print(json.dumps(results, indent=2))
