"""
sample_chartqa_questions.py

Samples a ChartQA (Masry et al., 2022) reasoning pool + holdout split for the
cross-dataset generalization test: the same rollout -> judge -> alignment
pipeline built for CharXiv, applied unmodified to a second chart-QA benchmark.

Mirrors CharXiv's own reasoning/descriptive split: ChartQA's `human_or_machine`
field distinguishes human-authored questions (open-ended, requires reading and
comparing chart values) from machine-generated ones (templated, closer to
CharXiv's excluded "descriptive" category). We keep only human-authored,
non-yes/no questions, so the pool matches CharXiv's reasoning-only pool in
kind, not just in name.

Source split is ChartQA's official `test` split (1,250 human-authored
questions after filtering) -- the same rationale CharXiv's own `val` split was
used for: an official held-out split, not the training split.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
TRAIN_POOL_SIZE = 150
HOLDOUT_SIZE = 30
SOURCE_SPLIT = "test"

BASE_DIR = Path(__file__).resolve().parents[2]
IMAGES_DIR = BASE_DIR / "data" / "ChartQA" / "images"
DATA_DIR = BASE_DIR / "data" / "ChartQA" / "data"
SPLITS_DIR = BASE_DIR / "data" / "splits"


def is_yes_no(answer: str) -> bool:
    return answer.strip().lower() in ("yes", "no")


def main():
    random.seed(SEED)

    ds = load_dataset("HuggingFaceM4/ChartQA")[SOURCE_SPLIT]
    human_rows = [
        (idx, row)
        for idx, row in enumerate(ds)
        if row["human_or_machine"] == 0 and not is_yes_no(row["label"][0])
    ]
    print(f"Human-authored, non-yes/no rows in {SOURCE_SPLIT}: {len(human_rows)} / {len(ds)}")

    total_needed = TRAIN_POOL_SIZE + HOLDOUT_SIZE
    if len(human_rows) < total_needed:
        raise ValueError(f"Need {total_needed} eligible rows, only found {len(human_rows)}")

    sampled = random.sample(human_rows, total_needed)
    random.shuffle(sampled)
    holdout = sampled[:HOLDOUT_SIZE]
    train_pool = sampled[HOLDOUT_SIZE:]

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    questions = {}
    for idx, row in sampled:
        qid = f"chartqa_{SOURCE_SPLIT}_{idx}"
        image_path = IMAGES_DIR / f"{qid}.jpg"
        row["image"].convert("RGB").save(image_path, "JPEG", quality=95)
        questions[qid] = {
            "query": row["query"],
            "answer": row["label"][0],
            "source_split": SOURCE_SPLIT,
            "source_index": idx,
        }

    with (DATA_DIR / "chartqa_reasoning.json").open("w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)

    train_ids = [f"chartqa_{SOURCE_SPLIT}_{idx}" for idx, _ in train_pool]
    holdout_ids = [f"chartqa_{SOURCE_SPLIT}_{idx}" for idx, _ in holdout]

    with (SPLITS_DIR / "chartqa_main_ids.json").open("w", encoding="utf-8") as f:
        json.dump(train_ids, f, indent=2)
    with (SPLITS_DIR / "chartqa_eval_ids.json").open("w", encoding="utf-8") as f:
        json.dump(holdout_ids, f, indent=2)

    print(f"Saved {len(train_ids)} train-pool IDs to data/splits/chartqa_main_ids.json")
    print(f"Saved {len(holdout_ids)} holdout IDs to data/splits/chartqa_eval_ids.json")
    print(f"Saved {len(questions)} question images to data/ChartQA/images/")


if __name__ == "__main__":
    main()
