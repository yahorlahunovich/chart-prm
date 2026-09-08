"""
dynamic_scores_to_binary.py

Converts dynamic multi-criteria judge scores (score_steps_dynamic.py's output,
1-3 per flagged criterion) into the binary per-step schema the original
formatters (format_sft.py, format_full_dpo.py, format_step_dpo.py,
format_kto.py) expect -- the same schema evaluate_rollouts_meta.py (Meta's
muse-spark-1.1 judge) produces, so those formatters run unmodified on a
dynamic-judge-only dataset like ChartQA.

Binary rule: a step passes (score=1) only if every criterion flagged for it
scored the maximum (3/3); any lower score fails the step. A step with no
criteria flagged passes (absence of a flagged issue is evidence the step is
fine, not missing data). This mirrors dynamic_scoring.dynamic_process_score's
own "min" aggregation mode, whose docstring names this exact equivalence:
"any single failure fails the step" is how the original binary judge worked.
"""
import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def step_passes(step: dict) -> bool:
    criteria = step.get("scores") or []
    if not criteria:
        return True
    return all(c.get("score") == 3 for c in criteria)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-scores-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    n_rollouts = 0
    n_steps = 0
    n_pass = 0
    with args.dynamic_scores_path.open(encoding="utf-8") as f_in, args.output_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            if not line.strip():
                continue
            data = json.loads(line)
            evaluations = []
            for step in data["scores"]:
                passed = step_passes(step)
                n_steps += 1
                n_pass += int(passed)
                evaluations.append(
                    {
                        "step_index": step["step_index"],
                        "analysis": "; ".join(c.get("note", "") for c in (step.get("scores") or [])),
                        "score": int(passed),
                    }
                )
            record = {
                "question_id": data["question_id"],
                "rollout_index": data["rollout_index"],
                "evaluations": evaluations,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_rollouts += 1

    print(f"Converted {n_rollouts} rollouts, {n_steps} steps ({n_pass} pass, {n_steps - n_pass} fail, "
          f"{100 * n_pass / n_steps:.1f}% pass rate) -> {args.output_path}")


if __name__ == "__main__":
    main()
