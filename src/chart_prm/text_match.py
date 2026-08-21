"""Shared whole-token answer matching.

`format_sft.py`, `format_full_dpo.py`, and `format_step_dpo.py` each carry an
identical copy of this logic. New code (the best-of-N verifier) reuses this
module instead of adding a fourth copy; the existing formatters are left as
they are since they are already tested and used to build frozen datasets.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("\\%", "%")
    return " ".join(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?|[%+\-]", text))


def answers_match(ground_truth: object, model_answer: object) -> bool:
    """Whole-token containment; a raw substring like "4" never matches "94"."""
    expected = normalize_text(ground_truth)
    actual = normalize_text(model_answer)
    if not expected or not actual:
        return False
    if expected == actual:
        return True
    expected_tokens = expected.split()
    actual_tokens = actual.split()
    width = len(expected_tokens)
    return any(
        actual_tokens[index : index + width] == expected_tokens
        for index in range(len(actual_tokens) - width + 1)
    )
