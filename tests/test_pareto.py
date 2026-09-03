from chart_prm.pareto import (
    build_criterion_to_parent,
    build_pareto_pairs,
    pareto_dominates,
    per_parent_vector,
)

TREE = {
    "parents": {
        "axis_or_layout_misread": {
            "label": "Axis / layout misread",
            "children": [
                {"child_id": "axis_0"},
                {"child_id": "axis_1"},
            ],
        },
        "wrong_series_or_color": {
            "label": "Wrong series / color",
            "children": [
                {"child_id": "series_0"},
            ],
        },
    }
}
PARENT_IDS = ["axis_or_layout_misread", "wrong_series_or_color"]


def test_build_criterion_to_parent_maps_every_child():
    mapping = build_criterion_to_parent(TREE)
    assert mapping == {
        "axis_0": "axis_or_layout_misread",
        "axis_1": "axis_or_layout_misread",
        "series_0": "wrong_series_or_color",
    }


def test_per_parent_vector_defaults_to_one_when_never_flagged():
    mapping = build_criterion_to_parent(TREE)
    vector = per_parent_vector([], mapping, PARENT_IDS)
    assert vector == {"axis_or_layout_misread": 1.0, "wrong_series_or_color": 1.0}


def test_per_parent_vector_averages_within_step_then_across_steps():
    mapping = build_criterion_to_parent(TREE)
    parsed = [
        {
            "step_index": 0,
            "scores": [
                {"criterion_id": "axis_0", "score": 1},  # -> 0.0
                {"criterion_id": "axis_1", "score": 3},  # -> 1.0, avg with axis_0 = 0.5
            ],
        },
        {
            "step_index": 1,
            "scores": [
                {"criterion_id": "axis_0", "score": 3},  # -> 1.0
            ],
        },
    ]
    vector = per_parent_vector(parsed, mapping, PARENT_IDS)
    # axis: step0 contributes 0.5, step1 contributes 1.0 -> mean 0.75
    assert vector["axis_or_layout_misread"] == 0.75
    # never flagged -> default 1.0
    assert vector["wrong_series_or_color"] == 1.0


def test_per_parent_vector_skips_unknown_criterion_ids():
    mapping = build_criterion_to_parent(TREE)
    parsed = [{"step_index": 0, "scores": [{"criterion_id": "made_up_id", "score": 1}]}]
    vector = per_parent_vector(parsed, mapping, PARENT_IDS)
    assert vector == {"axis_or_layout_misread": 1.0, "wrong_series_or_color": 1.0}


def test_pareto_dominates_requires_at_least_as_good_everywhere_and_better_somewhere():
    a = {"x": 1.0, "y": 0.5}
    b = {"x": 1.0, "y": 0.0}
    assert pareto_dominates(a, b) is True
    assert pareto_dominates(b, a) is False


def test_pareto_dominates_false_on_tie():
    a = {"x": 1.0, "y": 0.5}
    b = {"x": 1.0, "y": 0.5}
    assert pareto_dominates(a, b) is False


def test_pareto_dominates_false_on_tradeoff():
    a = {"x": 1.0, "y": 0.0}
    b = {"x": 0.0, "y": 1.0}
    assert pareto_dominates(a, b) is False
    assert pareto_dominates(b, a) is False


def _candidate(solution: str, correct: bool, vector: dict) -> dict:
    return {"solution": solution, "correct": correct, "vector": vector}


def test_build_pareto_pairs_only_pairs_correct_over_incorrect_with_dominance():
    good = _candidate("good", True, {"x": 1.0, "y": 1.0})
    bad_dominated = _candidate("bad", False, {"x": 0.0, "y": 0.0})
    bad_tradeoff = _candidate("tradeoff", False, {"x": 1.0, "y": 0.0})
    also_correct = _candidate("also_correct", True, {"x": 0.0, "y": 0.0})

    pairs = build_pareto_pairs([good, bad_dominated, bad_tradeoff, also_correct], seed_key="q1")

    # good dominates bad_dominated (1.0>=0.0 and 1.0>=0.0, strictly better both) -> included
    # good vs bad_tradeoff: x tie, y better -> dominates -> included
    # also_correct is correct, never paired as rejected
    pair_solutions = {(c["solution"], r["solution"]) for c, r in pairs}
    assert ("good", "bad") in pair_solutions
    assert ("good", "tradeoff") in pair_solutions
    assert all(r["solution"] != "also_correct" for _, r in pairs)


def test_build_pareto_pairs_excludes_incomparable_pairs():
    correct_but_worse_on_one_axis = _candidate("c", True, {"x": 1.0, "y": 0.0})
    incorrect_better_on_other_axis = _candidate("i", False, {"x": 0.0, "y": 1.0})
    pairs = build_pareto_pairs(
        [correct_but_worse_on_one_axis, incorrect_better_on_other_axis], seed_key="q2"
    )
    assert pairs == []


def test_build_pareto_pairs_dedupes_identical_solution_text():
    same_text_correct = _candidate("same", True, {"x": 1.0, "y": 1.0})
    same_text_incorrect = _candidate("same", False, {"x": 0.0, "y": 0.0})
    pairs = build_pareto_pairs([same_text_correct, same_text_incorrect], seed_key="q3")
    assert pairs == []


def test_build_pareto_pairs_caps_and_is_deterministic():
    chosen_list = [_candidate(f"good_{i}", True, {"x": 1.0, "y": 1.0}) for i in range(3)]
    rejected_list = [_candidate(f"bad_{i}", False, {"x": 0.0, "y": 0.0}) for i in range(3)]
    pairs_a = build_pareto_pairs(chosen_list + rejected_list, seed_key="qX", max_pairs=3)
    pairs_b = build_pareto_pairs(chosen_list + rejected_list, seed_key="qX", max_pairs=3)
    assert len(pairs_a) == 3
    assert [(c["solution"], r["solution"]) for c, r in pairs_a] == [
        (c["solution"], r["solution"]) for c, r in pairs_b
    ]
