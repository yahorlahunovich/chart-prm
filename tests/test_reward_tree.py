import numpy as np

from chart_prm.reward_tree import (
    choose_child_count,
    cosine_distance,
    merge_by_distance,
    select_relevant_children,
)


def test_cosine_distance_identical_vectors_is_zero():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_distance(v, v) == 0.0


def test_cosine_distance_orthogonal_vectors_is_one():
    assert cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0


def test_cosine_distance_handles_zero_vector():
    assert cosine_distance(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 1.0


def test_choose_child_count_scales_with_size_and_clamps():
    assert choose_child_count(33) == 2  # small category -> floor
    assert choose_child_count(702) == 6  # large category -> ceiling
    assert choose_child_count(180, per_cluster=60) == 3  # mid-size -> scales
    assert choose_child_count(1) == 1  # never exceeds member count
    assert choose_child_count(0) == 0


def test_merge_by_distance_folds_near_duplicates():
    candidates = [
        {"text": "a", "embedding": np.array([1.0, 0.0]), "member_count": 10},
        {"text": "a-dup", "embedding": np.array([0.99, 0.01]), "member_count": 5},  # near-dup of a
        {"text": "b", "embedding": np.array([0.0, 1.0]), "member_count": 7},  # distinct
    ]
    merged = merge_by_distance(candidates, threshold=0.05)
    assert len(merged) == 2
    assert merged[0]["member_count"] == 15  # 10 + 5 folded in
    assert merged[0]["merged_member_counts"] == [10, 5]
    assert merged[1]["member_count"] == 7


def test_merge_by_distance_keeps_all_when_threshold_is_tight():
    candidates = [
        {"text": "a", "embedding": np.array([1.0, 0.0]), "member_count": 1},
        {"text": "b", "embedding": np.array([0.0, 1.0]), "member_count": 1},
    ]
    merged = merge_by_distance(candidates, threshold=0.0)
    assert len(merged) == 2


def test_select_relevant_children_filters_by_zeta_and_sorts_by_distance():
    tree = {
        "parents": {
            "axis_or_layout_misread": {
                "label": "Axis / layout misread",
                "children": [
                    {"child_id": "axis_0", "embedding": [1.0, 0.0]},
                    {"child_id": "axis_1", "embedding": [0.9, 0.1]},
                ],
            },
            "wrong_series_or_color": {
                "label": "Wrong series / color",
                "children": [
                    {"child_id": "series_0", "embedding": [0.0, 1.0]},
                ],
            },
        }
    }
    step_embedding = np.array([1.0, 0.0])
    hits = select_relevant_children(step_embedding, tree, zeta=0.2)
    ids = [h["child_id"] for h in hits]
    assert ids[0] == "axis_0"  # exact match comes first
    assert "series_0" not in ids  # orthogonal, distance 1.0 > zeta
    assert all(h["distance"] <= 0.2 for h in hits)
    assert all("parent" in h for h in hits)


def test_select_relevant_children_empty_tree_returns_empty():
    assert select_relevant_children(np.array([1.0, 0.0]), {"parents": {}}, zeta=0.2) == []
