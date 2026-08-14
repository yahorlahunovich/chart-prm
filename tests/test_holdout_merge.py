from chart_prm.holdout_merge import exact_match_summary, merge_system_rows


def test_merge_copies_only_sft_dpo():
    base = [
        {
            "question_id": "1",
            "ground_truth": "A",
            "responses": {"base": "Final Answer: A", "sft": "Final Answer: B"},
            "predicted_answers": {"base": "A", "sft": "B"},
        }
    ]
    extra = [
        {
            "question_id": "1",
            "ground_truth": "A",
            "responses": {"sft_dpo": "Final Answer: A", "sft": "should not overwrite"},
            "predicted_answers": {"sft_dpo": "A", "sft": "Z"},
        }
    ]
    merged = merge_system_rows(base, extra)
    assert merged[0]["responses"]["sft"] == "Final Answer: B"
    assert merged[0]["predicted_answers"]["sft"] == "B"
    assert merged[0]["responses"]["sft_dpo"] == "Final Answer: A"
    assert merged[0]["predicted_answers"]["sft_dpo"] == "A"


def test_merge_rejects_id_mismatch():
    try:
        merge_system_rows(
            [{"question_id": "1", "responses": {}, "predicted_answers": {}}],
            [{"question_id": "2", "responses": {"sft_dpo": "x"}, "predicted_answers": {}}],
        )
    except ValueError as exc:
        assert "do not match" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_exact_match_summary_scores_sft_dpo():
    rows = [
        {
            "ground_truth": "10",
            "predicted_answers": {"sft": "10", "sft_dpo": "10 µA"},
        },
        {
            "ground_truth": "10",
            "predicted_answers": {"sft": "11", "sft_dpo": "10"},
        },
    ]
    summary = exact_match_summary(rows, systems=["sft", "sft_dpo"])
    assert summary["exact_match"]["sft"]["correct"] == 1
    assert summary["exact_match"]["sft_dpo"]["correct"] == 1
    assert summary["extracted_answer_rate"]["sft_dpo"] == 1.0
