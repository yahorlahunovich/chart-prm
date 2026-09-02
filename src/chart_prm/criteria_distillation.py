"""Distill v0 reward-tree children (TF-IDF terms + exemplar sentences) into
clean, general rubric statements -- the LLM-distillation step DG-PRM's Phase 1
specifies and experiment 009 deliberately deferred (documented there as
"v0: ... not yet LLM-distilled ... pending judge API access").

One call per parent (9 total for the current tree), not one per child (33),
mirroring the project's established rollout-batching cost pattern. Pure
prompt-building/parsing here; `scripts/evaluation/distill_reward_tree_criteria.py`
wires this to a real (text-only) Gemini call and writes `rubric_text` back
onto the tree.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

DISTILLATION_INSTRUCTIONS = (
    "You are writing evaluation rubric statements for grading chart-reading reasoning steps. "
    "For each criterion below, you are given characteristic keywords and one real example of a "
    "step that violates it. Write ONE short (max ~20 words), general, self-contained rubric "
    "statement describing what a reasoning step must do to AVOID this failure -- phrase it as a "
    "positive requirement (e.g. \"Axis values must be read from the labeled gridline nearest the "
    "data point, not estimated\"), not as a restatement of the failure. It must generalize beyond "
    "the one example shown, since it will be reused to grade different steps on different charts."
)


def format_child_for_distillation(child: Dict[str, Any]) -> str:
    terms = ", ".join(child.get("top_terms", [])[:6]) or "(no keywords)"
    exemplars = child.get("exemplars") or []
    example = exemplars[0] if exemplars else "(no example)"
    return f'[{child["child_id"]}]\n  keywords: {terms}\n  example failure: "{example}"'


def build_distillation_prompt(parent_label: str, children: List[Dict[str, Any]]) -> str:
    child_blocks = "\n".join(format_child_for_distillation(child) for child in children)
    child_ids = ", ".join(f'"{child["child_id"]}"' for child in children)
    return (
        f"{DISTILLATION_INSTRUCTIONS}\n\n"
        f"Category: {parent_label}\n\n"
        f"{child_blocks}\n\n"
        "Respond strictly as a JSON object mapping every criterion id to its rubric statement, "
        f"e.g. {{{child_ids}: \"...\"}}. No markdown formatting, no text outside the JSON object."
    )


def parse_distillation_response(
    response_text: Optional[str], expected_child_ids: List[str]
) -> Optional[Dict[str, str]]:
    """Extract {child_id: rubric_text}. None if unparseable or missing any expected id."""
    if not response_text:
        return None
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if not all(cid in parsed and isinstance(parsed[cid], str) and parsed[cid].strip() for cid in expected_child_ids):
        return None
    return {cid: parsed[cid].strip() for cid in expected_child_ids}
