import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from format_step_dpo import (
    answers_match,
    normalize_step,
    shared_prefix_length,
    valid_step,
)


def test_answers_match_whole_tokens_not_substrings():
    assert answers_match("(c)", "The answer is subplot (c).")
    assert answers_match("Top institutions", "Top institutions have the higher value.")
    assert not answers_match("4", "The answer is 94.")
    assert not answers_match("", "anything")


def test_shared_prefix_requires_same_reasoning_context():
    chosen = ["Step 1: Read the x-axis.", "Step 2: Read the blue bar."]
    rejected = ["Step 1: Read the x-axis.", "Step 2: Read the red bar."]

    assert shared_prefix_length(chosen, rejected) == 1
    assert normalize_step(chosen[0]) == "Read the x-axis."


def test_malformed_steps_are_rejected():
    assert valid_step("Step 1: Compare the two values.")
    assert not valid_step("Step Step analysis repeats malformed output.")
    assert not valid_step("   ")
