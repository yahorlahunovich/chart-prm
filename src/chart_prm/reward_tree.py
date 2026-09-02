"""Reward tree: parent/child evaluation criteria for step-level PRM grading.

Adapts the reward-tree half of Yin et al., "Dynamic and Generalizable Process
Reward Modeling" (DG-PRM) to ChartPRM. DG-PRM discovers coarse parent
criteria from scratch via clustering; ChartPRM already has 9 human-validated
parent categories from `categorize_judge_errors.py`'s regex taxonomy on 2,920
real judge failure explanations, so this module only builds the missing
child (fine-grained) layer underneath them, and the retrieval mechanism
(Phase 2 of the paper) that selects relevant children for a new step.

Pure, testable logic only. `scripts/evaluation/build_reward_tree.py` wires
this to the real categorized fail-analysis data and embeddings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    similarity = float(np.dot(a, b) / denom)
    return 1.0 - similarity


def choose_child_count(n_members: int, min_k: int = 2, max_k: int = 6, per_cluster: int = 60) -> int:
    """How many sub-clusters (children) a parent category should split into.

    Scales with category size (~1 child per `per_cluster` members) but is
    clamped to [min_k, max_k], and never exceeds the number of members.
    """
    if n_members <= 0:
        return 0
    if n_members < min_k:
        return n_members
    k = round(n_members / per_cluster)
    k = max(min_k, min(max_k, k))
    return min(k, n_members)


def merge_by_distance(
    candidates: Sequence[Dict[str, Any]], threshold: float, embedding_key: str = "embedding"
) -> List[Dict[str, Any]]:
    """Greedily merge near-duplicate criteria (cosine distance <= threshold).

    Mirrors DG-PRM's merge hyperparameter `xi`: candidates closer than the
    threshold to an already-kept representative are folded into it instead
    of kept as a separate child. Deterministic given a fixed input order.
    """
    kept: List[Dict[str, Any]] = []
    for candidate in candidates:
        merged_into = None
        for representative in kept:
            if cosine_distance(candidate[embedding_key], representative[embedding_key]) <= threshold:
                merged_into = representative
                break
        if merged_into is None:
            kept.append(dict(candidate, merged_member_counts=[candidate.get("member_count", 1)]))
        else:
            merged_into["merged_member_counts"].append(candidate.get("member_count", 1))
            merged_into["member_count"] = merged_into.get("member_count", 0) + candidate.get(
                "member_count", 1
            )
    return kept


def select_relevant_children(
    step_embedding: np.ndarray, tree: Dict[str, Any], zeta: float = 0.2
) -> List[Dict[str, Any]]:
    """Retrieve child criteria whose centroid is within `zeta` cosine distance of a step.

    Implements DG-PRM's Phase 2 fine-grained matching (delta_k <= zeta) over
    every child in the tree, regardless of parent — the parent-selection
    step (DG-PRM's coarse-grained R) is left to the judge prompt in Phase 2,
    since chart-step context is short enough to just show all parents.
    """
    hits: List[Dict[str, Any]] = []
    for parent_key, parent in tree.get("parents", {}).items():
        for child in parent.get("children", []):
            distance = cosine_distance(step_embedding, np.asarray(child["embedding"]))
            if distance <= zeta:
                hits.append({**child, "parent": parent_key, "distance": distance})
    hits.sort(key=lambda c: c["distance"])
    return hits
