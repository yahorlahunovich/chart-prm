"""Tests for holdout structure / answer-tier scoring."""

from chart_prm.holdout_metrics import extract_final_answer, extracted_match, score_generation, token_match


def test_official_exact_rejects_markdown_leak():
    row = score_generation("Final Answer: Reverse", "Reverse", official_pred="** Reverse")
    assert row["exact_official"] is False
    assert row["token_pred"] is True


def test_extracted_match_allows_format_not_extra_entities():
    assert extracted_match("Reverse", "** Reverse")
    assert extracted_match("25", "S = 25")
    assert extracted_match("10", "10 µA")
    assert extracted_match("(c)", "c")
    assert extracted_match("Theory NU N=64", "Theory NU N = 64")
    assert not extracted_match("news", "No News")
    assert not extracted_match("1", "Layer_1")
    assert not extracted_match("Decreases", "Decreases then increases")
    assert not extracted_match("max", "max+max")


def test_token_match_rejects_substring_digits():
    assert token_match("4", "The answer is 4")
    assert not token_match("4", "94")


def test_extracts_markdown_final_answer():
    text = (
        "To determine the subplot...\n"
        "the curves are symmetric along the horizontal axis is labeled \"Reverse\".\n"
        "**Final Answer:** Reverse"
    )
    assert extract_final_answer(text) == "Reverse"


def test_score_unstructured_but_mentions_gt():
    text = (
        "To determine the name, look at the plots.\n"
        "The subplot labeled Reverse is symmetric.\n"
        "**Final Answer:** Reverse"
    )
    row = score_generation(text, "Reverse", official_pred="** Reverse")
    assert row["token_body"] is True
    assert row["token_pred"] is True
    assert row["starts_step1"] is False
    assert row["has_preamble"] is True
    assert row["robust_pred"] == "Reverse"
    assert row["structured_correct"] is False


def test_score_structured_correct():
    text = "Step 1: Read the legend.\nStep 2: Compare bars.\nFinal Answer: Model B"
    row = score_generation(text, "Model B", official_pred="Model B")
    assert row["structured_correct"] is True
    assert row["error_type"] == "correct_extracted"
    assert row["structure_score"] == 1.0
