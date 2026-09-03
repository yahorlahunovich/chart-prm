"""Phase 3 of the DG-PRM adaptation: Pareto-dominance-filtered preference pairs.

`dynamic_process_score` (see `dynamic_scoring.py`) collapses a rollout's per-step,
per-criterion scores into one scalar. This module keeps the multi-dimensional
signal instead: one score per reward-tree *parent category* (9 categories --
hallucinated entities, wrong numeric reads, logic inconsistency, etc.), and only
treats one rollout as preferred over another when it is at least as good on
every category and strictly better on at least one (Pareto dominance) -- rather
than just having a higher average. This filters out preference pairs where the
two rollouts trade off against each other on different failure axes (e.g. one
hallucinates less but misreads the axis more), since those are ambiguous
training signal for DPO, not a clean "this one is better" pair.

Pure functions only; `scripts/data_prep/format_pareto_dpo.py` wires this to the
real experiment 011 dynamic-scores and rollout-metadata files.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple

TIE_EPSILON = 1e-9


def build_criterion_to_parent(tree: Dict[str, Any]) -> Dict[str, str]:
    """Map every reward-tree child criterion id to its parent category id."""
    mapping: Dict[str, str] = {}
    for parent_id, parent in tree.get("parents", {}).items():
        for child in parent.get("children", []):
            mapping[child["child_id"]] = parent_id
    return mapping


def per_parent_vector(
    parsed_scores: Sequence[Dict[str, Any]],
    criterion_to_parent: Dict[str, str],
    parent_ids: Sequence[str],
) -> Dict[str, float]:
    """One rollout's per-step judge scores, collapsed to one [0, 1] score per parent category.

    Within a step, criteria sharing a parent are averaged first (so a step that trips
    several criteria under the same category doesn't count more than one that trips
    just one). Across steps, a parent's contributions are averaged with equal weight.
    A parent category never flagged anywhere in the rollout defaults to 1.0 -- same
    "absence of a flagged issue is evidence the step is fine" convention as
    `dynamic_process_score`, just applied per category instead of to one overall score.
    Unrecognized criterion ids (the judge occasionally invents one not in the tree) are
    skipped rather than raising, mirroring how they're already dropped in analysis.
    """
    per_step_by_parent: Dict[str, List[float]] = {pid: [] for pid in parent_ids}
    for step in parsed_scores:
        criteria = step.get("scores") or []
        step_group: Dict[str, List[float]] = {}
        for criterion in criteria:
            score = criterion.get("score")
            if score not in (1, 2, 3):
                continue
            parent_id = criterion_to_parent.get(criterion.get("criterion_id"))
            if parent_id is None or parent_id not in per_step_by_parent:
                continue
            step_group.setdefault(parent_id, []).append((score - 1) / 2)
        for parent_id, values in step_group.items():
            per_step_by_parent[parent_id].append(sum(values) / len(values))
    return {
        parent_id: (sum(values) / len(values) if values else 1.0)
        for parent_id, values in per_step_by_parent.items()
    }


def pareto_dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """True if `a` is at least as good as `b` on every shared dimension and strictly better on one.

    Equal vectors, or vectors that trade off (better on some dims, worse on others),
    both return False -- dominance is a strict, not total, order.
    """
    keys = a.keys()
    at_least_as_good = all(a[k] >= b[k] - TIE_EPSILON for k in keys)
    strictly_better_somewhere = any(a[k] > b[k] + TIE_EPSILON for k in keys)
    return at_least_as_good and strictly_better_somewhere


def build_pareto_pairs(
    candidates: Sequence[Dict[str, Any]],
    seed_key: str,
    max_pairs: int = 3,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """(chosen, rejected) pairs for one question's candidate rollouts.

    Each candidate needs `vector` (per-parent score dict), `correct` (bool), and
    `solution` (the reconstructed text, for dedup). `chosen` must be a correct rollout
    that Pareto-dominates a `rejected` incorrect rollout -- so the preference is
    grounded in both the final answer and unambiguous process quality, not just one
    or the other. Deterministically shuffled by `seed_key` before capping to
    `max_pairs`, matching `format_full_dpo.py`'s per-question capping so heavily
    over-sampled questions don't dominate the training set.
    """
    chosen_pool = [c for c in candidates if c.get("correct")]
    rejected_pool = [c for c in candidates if not c.get("correct")]
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for chosen in chosen_pool:
        for rejected in rejected_pool:
            if chosen["solution"] == rejected["solution"]:
                continue
            if pareto_dominates(chosen["vector"], rejected["vector"]):
                pairs.append((chosen, rejected))
    random.Random(seed_key).shuffle(pairs)
    return pairs[:max_pairs]
