"""
clean_chartqa_dataset.py

Data-quality pass over the sampled ChartQA pool (data/ChartQA/data/chartqa_reasoning.json)
before it enters the generation pipeline: verifies every image decodes, flags duplicate
(image, query) pairs, and flags degenerate answers (empty, or a bare "N/A"). Mirrors the
role clean_dataset.py plays for model rollouts, but at the source-data stage instead.

Drops nothing silently -- a dataset issue here should be a visible decision, not a
silent filter, since it changes exactly which questions the alignment methods are
compared on.
"""
import json
from collections import Counter
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "ChartQA" / "data" / "chartqa_reasoning.json"
IMAGES_DIR = BASE_DIR / "data" / "ChartQA" / "images"


def main():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    n_total = len(questions)
    broken_images = []
    empty_answers = []
    duplicate_queries = []

    query_counts = Counter(q["query"].strip().lower() for q in questions.values())

    for qid, q in questions.items():
        image_path = IMAGES_DIR / f"{qid}.jpg"
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as exc:
            broken_images.append((qid, str(exc)))

        answer = q["answer"].strip()
        if not answer or answer.lower() in ("n/a", "na", "none"):
            empty_answers.append(qid)

        if query_counts[q["query"].strip().lower()] > 1:
            duplicate_queries.append(qid)

    print(f"Total questions: {n_total}")
    print(f"Broken images: {len(broken_images)}")
    for qid, err in broken_images:
        print(f"  {qid}: {err}")
    print(f"Empty/degenerate answers: {len(empty_answers)} -> {empty_answers}")
    print(f"Questions sharing a duplicate query text: {len(duplicate_queries)} -> {duplicate_queries}")

    if not broken_images and not empty_answers and not duplicate_queries:
        print("Clean: no broken images, no empty answers, no duplicate query text.")


if __name__ == "__main__":
    main()
