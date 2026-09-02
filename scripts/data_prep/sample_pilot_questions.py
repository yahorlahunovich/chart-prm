#!/usr/bin/env python3
"""
sample_pilot_questions.py

Freezes a deterministic ~100-question pilot set for Phase 2 dynamic
multi-criteria scoring, reused from experiment 008's already-judged
"eligible" pool (questions with >=2 judged rollouts) rather than the
protected 100-question holdout. Keeping the same pool as experiment 008
means a future best-of-N re-check with the new scores is directly
comparable to the existing 27.5%/21.0%/18.4% numbers.

No API calls, no GPU. Pure data selection over data already in the repo.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
PILOT_SIZE = 100


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    verifier_results_path = base_dir / "experiments/008_prm_best_of_n/data/verifier_results.json"
    output_path = base_dir / "experiments/010_dynamic_scoring_pilot/data/pilot_question_ids.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with verifier_results_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    eligible_ids = sorted(data["per_question"].keys())
    if len(eligible_ids) < PILOT_SIZE:
        raise SystemExit(f"Only {len(eligible_ids)} eligible questions, need {PILOT_SIZE}")

    pilot_ids = sorted(random.Random(SEED).sample(eligible_ids, PILOT_SIZE))

    output_path.write_text(
        json.dumps(
            {
                "seed": SEED,
                "n": len(pilot_ids),
                "source": "experiments/008_prm_best_of_n/data/verifier_results.json (per_question keys)",
                "note": "Sampled from the training-pool 'eligible' set, not the protected holdout "
                "(data/splits/eval_reasoning_ids.json). Reusing experiment 008's pool keeps a "
                "future best-of-N re-check comparable to the existing 27.5%/21.0%/18.4% numbers.",
                "question_ids": pilot_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Sampled {len(pilot_ids)} pilot questions from {len(eligible_ids)} eligible.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
