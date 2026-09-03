from chart_prm.dynamic_scoring import (
    build_dynamic_scoring_prompt,
    build_step_block,
    dynamic_process_score,
    format_criterion,
    format_tree_criteria_list,
    parse_dynamic_scores,
)

TREE = {
    "parents": {
        "axis_or_layout_misread": {
            "label": "Axis / layout misread",
            "children": [
                {"child_id": "axis_0", "top_terms": ["axis", "gridline"], "exemplars": ["Wrong axis."]},
                {"child_id": "axis_1", "top_terms": ["subplot"], "exemplars": []},
            ],
        },
        "wrong_series_or_color": {
            "label": "Wrong series / color",
            "children": [
                {"child_id": "series_0", "top_terms": ["color"], "exemplars": ["Wrong color."]},
            ],
        },
        "empty_parent": {"label": "Nothing here", "children": []},
    }
}


def test_format_criterion_includes_id_terms_and_exemplar():
    child = {
        "child_id": "axis_or_layout_misread_2",
        "top_terms": ["axis", "gridline", "subplot"],
        "exemplars": ["The x-axis label is wrong."],
    }
    text = format_criterion(child)
    assert "[axis_or_layout_misread_2]" in text
    assert "axis, gridline, subplot" in text
    assert "The x-axis label is wrong." in text


def test_format_criterion_handles_missing_exemplars_and_terms():
    text = format_criterion({"child_id": "x_0", "top_terms": [], "exemplars": []})
    assert "[x_0]" in text
    assert "no keywords" in text


def test_format_criterion_prefers_distilled_rubric_text_when_present():
    child = {
        "child_id": "axis_0",
        "top_terms": ["axis"],
        "exemplars": ["Wrong axis."],
        "rubric_text": "Axis values must be read from the labeled gridline nearest the point.",
    }
    text = format_criterion(child)
    assert "[axis_0]" in text
    assert "Axis values must be read from the labeled gridline nearest the point." in text
    assert "failure pattern keywords" not in text  # v1 phrasing replaces v0, not appended to it


def test_format_tree_criteria_list_groups_by_parent_and_skips_empty():
    text = format_tree_criteria_list(TREE)
    assert "Axis / layout misread:" in text
    assert "[axis_0]" in text and "[axis_1]" in text
    assert "Wrong series / color:" in text
    assert "[series_0]" in text
    assert "Nothing here" not in text  # empty_parent has no children, should be skipped


def test_build_step_block_is_plain_step_text():
    assert build_step_block(1, "Step 1: read the axis.") == "Step 1: Step 1: read the axis."


def test_prompt_shows_full_tree_once_and_lists_all_steps():
    prompt = build_dynamic_scoring_prompt(
        "Which model is best?",
        [(0, "Step 0: read the chart."), (1, "Step 1: compare values.")],
        TREE,
    )
    assert "Which model is best?" in prompt
    assert "Step 0: read the chart." in prompt
    assert "Step 1: compare values." in prompt
    assert "[axis_0]" in prompt
    assert "[series_0]" in prompt
    # tree criteria block appears exactly once, not repeated per step
    assert prompt.count("[axis_0]") == 1


def test_prompt_with_ground_truth_shows_it_and_acknowledges_it_instead_of_denying_it():
    blind = build_dynamic_scoring_prompt("Q?", [(0, "Step 0: text.")], TREE)
    informed = build_dynamic_scoring_prompt("Q?", [(0, "Step 0: text.")], TREE, ground_truth="42")

    assert "Ground Truth Answer: 42" in informed
    assert "Ground Truth Answer:" not in blind
    # the blind closing sentence ("not told the ground truth") must not survive into the
    # informed prompt, since it would be a direct contradiction
    assert "not told the ground truth" not in informed.lower()
    assert "given the correct final answer" in informed.lower()


def test_prompt_defaults_to_blind_with_no_ground_truth_passed():
    # Regression guard for the deliberate default: calling without ground_truth (the normal
    # Phase 2 path) never mentions an answer, and explicitly tells the judge it has none.
    # (ground_truth is an explicit, opt-in parameter for the sighted-vs-blind ablation only
    # -- see test_prompt_with_ground_truth_shows_it_and_acknowledges_it_instead_of_denying_it.)
    prompt = build_dynamic_scoring_prompt("Q?", [(0, "Step 0: text.")], TREE)
    assert "Ground Truth Answer:" not in prompt
    assert "not told the ground truth" in prompt.lower()  # judge is explicitly told not to use it


def test_parse_dynamic_scores_handles_markdown_fence():
    text = '```json\n[{"step_index": 0, "scores": [{"criterion_id": "a", "score": 2}]}]\n```'
    parsed = parse_dynamic_scores(text)
    assert parsed[0]["step_index"] == 0
    assert parsed[0]["scores"][0]["score"] == 2


def test_parse_dynamic_scores_handles_plain_json():
    text = '[{"step_index": 1, "scores": []}]'
    assert parse_dynamic_scores(text)[0]["step_index"] == 1


def test_parse_dynamic_scores_rejects_malformed_shape():
    assert parse_dynamic_scores('{"not": "a list"}') is None
    assert parse_dynamic_scores('[{"missing_scores_key": true}]') is None
    assert parse_dynamic_scores(None) is None
    assert parse_dynamic_scores("not json at all") is None


def test_dynamic_process_score_perfect_and_worst_cases():
    perfect = [{"step_index": 0, "scores": [{"criterion_id": "a", "score": 3}]}]
    assert dynamic_process_score(perfect) == 1.0
    worst = [{"step_index": 0, "scores": [{"criterion_id": "a", "score": 1}]}]
    assert dynamic_process_score(worst) == 0.0


def test_dynamic_process_score_step_with_no_criteria_counts_as_fine():
    steps = [
        {"step_index": 0, "scores": [{"criterion_id": "a", "score": 1}]},  # worst: 0.0
        {"step_index": 1, "scores": []},  # nothing flagged -> treated as 1.0
    ]
    assert dynamic_process_score(steps) == 0.5  # mean(0.0, 1.0)


def test_dynamic_process_score_averages_criteria_within_a_step_first():
    # One step with two criteria (avg 0.5) should count the same as any other single
    # step, not let a step with more flagged criteria dominate the rollout average.
    steps = [
        {"step_index": 0, "scores": [{"criterion_id": "a", "score": 1}, {"criterion_id": "b", "score": 3}]},
        {"step_index": 1, "scores": [{"criterion_id": "c", "score": 3}]},
    ]
    assert dynamic_process_score(steps) == 0.75  # mean(0.5, 1.0)


def test_dynamic_process_score_empty_input_is_none():
    assert dynamic_process_score([]) is None


def test_dynamic_process_score_min_mode_uses_worst_criterion_per_step():
    # Same fixture as the mean test above, but the worst criterion (score=1) should
    # decide step 0 entirely under "min" mode, not get averaged against score=3.
    steps = [
        {"step_index": 0, "scores": [{"criterion_id": "a", "score": 1}, {"criterion_id": "b", "score": 3}]},
        {"step_index": 1, "scores": [{"criterion_id": "c", "score": 3}]},
    ]
    assert dynamic_process_score(steps, step_aggregation="min") == 0.5  # mean(0.0, 1.0)


def test_dynamic_process_score_min_mode_matches_mean_when_one_criterion_per_step():
    steps = [{"step_index": 0, "scores": [{"criterion_id": "a", "score": 3}]}]
    assert dynamic_process_score(steps, step_aggregation="min") == dynamic_process_score(steps)


def test_dynamic_process_score_min_mode_no_criteria_step_still_scores_fine():
    steps = [{"step_index": 0, "scores": []}]
    assert dynamic_process_score(steps, step_aggregation="min") == 1.0


def test_dynamic_process_score_rejects_unknown_aggregation():
    steps = [{"step_index": 0, "scores": [{"criterion_id": "a", "score": 1}]}]
    try:
        dynamic_process_score(steps, step_aggregation="max")
        assert False, "expected ValueError"
    except ValueError:
        pass
