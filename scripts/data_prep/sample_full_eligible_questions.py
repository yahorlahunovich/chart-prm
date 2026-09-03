#!/usr/bin/env python3
"""
sample_full_eligible_questions.py

Scales the Phase 2 pilot (experiment 010, 100 questions) up to all 309
"eligible" (>=2 judged rollouts) questions from experiment 008's pool --
the full set the 100-question pilot was itself sampled from. No sampling
involved here, unlike sample_pilot_questions.py: every eligible question
_is_ included, since the point is resolving the statistical-power gap the
100-question pilot's bootstrap CIs couldn't close.

Still the training pool, not the protected 100-question holdout.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    verifier_results_path = base_dir / "experiments/008_prm_best_of_n/data/verifier_results.json"
    output_path = base_dir / "experiments/011_dynamic_scoring_full/data/full_question_ids.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with verifier_results_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    all_ids = sorted(data["per_question"].keys())

    output_path.write_text(
        json.dumps(
            {
                "n": len(all_ids),
                "source": "experiments/008_prm_best_of_n/data/verifier_results.json (all per_question keys)",
                "note": "All 309 eligible (>=2 judged rollouts) training-pool questions, not a sample "
                "-- superset of experiments/010_dynamic_scoring_pilot's 100-question pilot. "
                "Training pool, not the protected holdout (data/splits/eval_reasoning_ids.json).",
                "question_ids": all_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(all_ids)} question ids to {output_path}")


if __name__ == "__main__":
    main()
