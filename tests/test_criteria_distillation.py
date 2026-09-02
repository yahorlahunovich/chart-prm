from chart_prm.criteria_distillation import (
    build_distillation_prompt,
    format_child_for_distillation,
    parse_distillation_response,
)

CHILDREN = [
    {"child_id": "axis_0", "top_terms": ["axis", "gridline"], "exemplars": ["The axis is wrong."]},
    {"child_id": "axis_1", "top_terms": ["subplot"], "exemplars": []},
]


def test_format_child_for_distillation_includes_id_terms_and_exemplar():
    text = format_child_for_distillation(CHILDREN[0])
    assert "[axis_0]" in text
    assert "axis, gridline" in text
    assert "The axis is wrong." in text


def test_format_child_for_distillation_handles_missing_exemplar():
    text = format_child_for_distillation(CHILDREN[1])
    assert "(no example)" in text


def test_build_distillation_prompt_lists_all_children_and_expected_ids():
    prompt = build_distillation_prompt("Axis / layout misread", CHILDREN)
    assert "Axis / layout misread" in prompt
    assert "[axis_0]" in prompt and "[axis_1]" in prompt
    assert '"axis_0"' in prompt and '"axis_1"' in prompt
    assert "positive requirement" in prompt  # regression guard: framed as requirement, not failure


def test_parse_distillation_response_handles_markdown_fence():
    text = '```json\n{"axis_0": "Read axis values from the gridline.", "axis_1": "Match the correct subplot."}\n```'
    parsed = parse_distillation_response(text, ["axis_0", "axis_1"])
    assert parsed["axis_0"] == "Read axis values from the gridline."
    assert parsed["axis_1"] == "Match the correct subplot."


def test_parse_distillation_response_handles_plain_json():
    text = '{"axis_0": "Rubric one.", "axis_1": "Rubric two."}'
    assert parse_distillation_response(text, ["axis_0", "axis_1"])["axis_0"] == "Rubric one."


def test_parse_distillation_response_rejects_missing_expected_id():
    text = '{"axis_0": "Only one present."}'
    assert parse_distillation_response(text, ["axis_0", "axis_1"]) is None


def test_parse_distillation_response_rejects_blank_value():
    text = '{"axis_0": "", "axis_1": "Fine."}'
    assert parse_distillation_response(text, ["axis_0", "axis_1"]) is None


def test_parse_distillation_response_rejects_malformed():
    assert parse_distillation_response(None, ["axis_0"]) is None
    assert parse_distillation_response("not json", ["axis_0"]) is None
    assert parse_distillation_response("[1, 2, 3]", ["axis_0"]) is None
