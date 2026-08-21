"""Best-of-N PRM verifier: use step-level judge scores to select among rollouts.

Every other use of the PRM judge in this repo is offline — it only shapes
SFT/DPO/KTO training data (see `scripts/data_prep/format_*.py`). This module
asks the more classical PRM question instead: at inference time, when several
rollouts already exist for the same question, does picking the one the judge
scored highest actually beat picking one at random, or beat a plain
majority vote over final answers?

Pure functions only; `scripts/evaluation/prm_best_of_n.py` wires this to the
real experiment-001 rollout files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from chart_prm.text_match import answers_match, normalize_text


def process_score(evaluations: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Mean per-step pass rate in [0, 1]; None if no step carries a score."""
    scores = [step.get("score") for step in evaluations if step.get("score") is not None]
    if not scores:
        return None
    return sum(1 for s in scores if s == 1) / len(scores)


def build_candidate(
    rollout_meta: Dict[str, Any], evaluations: Sequence[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """One scored rollout, or None if it has no judge score or no final answer."""
    score = process_score(evaluations)
    if score is None:
        return None
    final_answer = str(rollout_meta.get("model_final_answer", "")).strip()
    if not final_answer:
        return None
    ground_truth = rollout_meta.get("ground_truth", "")
    return {
        "rollout_index": rollout_meta.get("rollout_index"),
        "final_answer": final_answer,
        "process_score": score,
        "correct": answers_match(ground_truth, final_answer),
    }


def select_by_process_score(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Highest process score wins; ties broken by lowest rollout_index."""
    return sorted(candidates, key=lambda c: (-c["process_score"], c["rollout_index"]))[0]


def select_by_majority_vote(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Most common normalized final answer wins; ties broken by lowest rollout_index."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(normalize_text(candidate["final_answer"]), []).append(candidate)
    ranked_groups = sorted(
        groups.values(),
        key=lambda group: (-len(group), min(c["rollout_index"] for c in group)),
    )
    winning_group = ranked_groups[0]
    return sorted(winning_group, key=lambda c: c["rollout_index"])[0]


def evaluate_question_group(candidates: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Per-question outcome for each selection strategy. Needs >=2 candidates."""
    if len(candidates) < 2:
        raise ValueError("Best-of-N selection needs at least 2 scored candidates")
    random_expected = sum(c["correct"] for c in candidates) / len(candidates)
    prm_pick = select_by_process_score(candidates)
    majority_pick = select_by_majority_vote(candidates)
    oracle_correct = any(c["correct"] for c in candidates)
    return {
        "n_candidates": len(candidates),
        "random_expected_correct": random_expected,
        "prm_correct": float(prm_pick["correct"]),
        "majority_correct": float(majority_pick["correct"]),
        "oracle_correct": float(oracle_correct),
    }


def summarize(question_results: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-question outcomes into headline accuracy numbers."""
    n = len(question_results)
    if n == 0:
        raise ValueError("No question groups to summarize")

    def mean(key: str) -> float:
        return sum(r[key] for r in question_results) / n

    oracle_positive = [r for r in question_results if r["oracle_correct"] == 1.0]
    prm_accuracy_when_oracle_positive = (
        sum(r["prm_correct"] for r in oracle_positive) / len(oracle_positive)
        if oracle_positive
        else 0.0
    )

    return {
        "n_questions": n,
        "random_baseline_accuracy": mean("random_expected_correct"),
        "prm_best_of_n_accuracy": mean("prm_correct"),
        "majority_vote_accuracy": mean("majority_correct"),
        "oracle_accuracy": mean("oracle_correct"),
        "prm_accuracy_when_oracle_positive": prm_accuracy_when_oracle_positive,
    }
