#!/usr/bin/env python3
"""
Merge the 8 separate per-system ChartQA holdout generation files (one per
Kaggle eval kernel: base + 7 trained methods) and score them with the same
chart_prm.holdout_metrics.score_generation used for CharXiv's holdout table
(structure score, token match, hallucination proxy), producing a summary
directly comparable to README.md's CharXiv table.

Unlike analyze_holdout_quality.py, this doesn't require a single merged
multi-system row file first -- score_generation operates on one (text, gt)
pair at a time, so each per-system file is scored independently and then
combined into one summary table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chart_prm.holdout_metrics import score_generation  # noqa: E402

SYSTEMS = ["base", "sft", "dpo", "step_dpo", "kto", "simpo", "pareto_dpo", "sft_dpo"]
DISPLAY = {
    "base": "Base (Instruct)",
    "sft": "SFT",
    "dpo": "Full DPO",
    "step_dpo": "Step-DPO",
    "kto": "KTO",
    "simpo": "SimPO",
    "pareto_dpo": "Pareto-DPO",
    "sft_dpo": "SFT→DPO",
}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-dir", type=Path,
        default=ROOT / "experiments/020_chartqa_transfer/holdout",
    )
    parser.add_argument(
        "--output-path", type=Path,
        default=ROOT / "experiments/020_chartqa_transfer/holdout/chartqa_holdout_summary.json",
    )
    args = parser.parse_args()

    summary = {}
    for system in SYSTEMS:
        gen_path = args.holdout_dir / f"{system}_chartqa_holdout_generations.jsonl"
        if not gen_path.exists():
            raise FileNotFoundError(f"Missing {gen_path}")
        rows = load_rows(gen_path)

        scored = []
        for row in rows:
            text = row["responses"].get(system, "") or ""
            official = (row.get("predicted_answers") or {}).get(system, "") or ""
            metrics = score_generation(text, row["ground_truth"], official_pred=official)
            scored.append(metrics)

        n = len(scored)
        summary[system] = {
            "n": n,
            "exact_official_pct": 100 * sum(m["exact_official"] for m in scored) / n,
            "token_match_pct": 100 * sum(m["token_pred"] for m in scored) / n,
            "structure_score_pct": 100 * sum(m["structure_score"] for m in scored) / n,
            "starts_step1_pct": 100 * sum(m["starts_step1"] for m in scored) / n,
            "has_preamble_pct": 100 * sum(m["has_preamble"] for m in scored) / n,
            "recall_gt_in_text_pct": 100 * sum(m["token_body"] for m in scored) / n,
            "wrong_committed_pct": 100 * sum(m["error_type"] == "wrong_committed" for m in scored) / n,
            "no_answer_pct": 100 * sum(m["error_type"] == "no_answer" for m in scored) / n,
        }

    args.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    header = f"{'System':<18}{'EM':>7}{'Token':>8}{'Struct':>8}{'Step1':>8}{'Preamb':>8}{'Recall':>8}{'Wrong':>8}"
    print(header)
    print("-" * len(header))
    for system in SYSTEMS:
        s = summary[system]
        print(
            f"{DISPLAY[system]:<18}"
            f"{s['exact_official_pct']:>6.1f}%"
            f"{s['token_match_pct']:>7.1f}%"
            f"{s['structure_score_pct']:>7.1f}%"
            f"{s['starts_step1_pct']:>7.1f}%"
            f"{s['has_preamble_pct']:>7.1f}%"
            f"{s['recall_gt_in_text_pct']:>7.1f}%"
            f"{s['wrong_committed_pct']:>7.1f}%"
        )
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
