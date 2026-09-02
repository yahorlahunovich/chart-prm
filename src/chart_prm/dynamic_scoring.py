"""Phase 2 of the DG-PRM adaptation: dynamic, multi-criteria step scoring.

Builds the judge prompt and parses its response. Unlike `evaluate_rollouts_meta.py`'s
prompt, this one never shows the ground-truth answer -- it grades whether a step
exhibits the failure pattern described by each criterion, not whether the rollout
reached the right answer.

Earlier design showed the judge a small, embedding-retrieved subset of the reward
tree per step (mirroring DG-PRM's Phase 2 retrieval). That broke empirically: the
tree's children are centroids of *judge critique sentences*, while a step's own text
is written in a different register (a reasoning claim, not a critique), so cosine
similarity between the two was near-zero at any reasonable threshold -- see
experiment 010's notes. DG-PRM's own Phase 2 avoids this by having an LLM generate
an intermediate "aspect" phrase per step before matching (`Phi`); replicating that
would cost another LLM call per step. Simpler fix at our scale: the tree only has
33 children total, small enough to show the judge the *whole* tree once per rollout
and let it self-select which criteria apply to each step in the same call that
scores them -- which is arguably closer to what an LLM-mediated `Phi` is doing
anyway, just folded into one call instead of two.

Pure text-in/text-out functions only; no network calls, no embedding model, so
fully testable without an API key.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCORE_SCALE_NOTE = (
    "Score each on-topic requirement 1-3:\n"
    "  1 = the step clearly fails to satisfy this requirement\n"
    "  2 = ambiguous, or not enough information in the step/image to tell\n"
    "  3 = the step clearly satisfies this requirement\n"
)

TASK_INSTRUCTIONS = (
    "You are grading individual reasoning steps from a vision-language model's chart-reading "
    "solution against a list of specific requirements (grouped by category below), each derived "
    "from real error examples in this domain.\n\n"
    "For EACH step, do this in two SEPARATE passes, in order:\n"
    "  1. TOPIC CHECK, before judging correctness at all: list every requirement whose subject "
    "matter the step touches on -- e.g. a step that reads a number off an axis is on-topic for "
    "axis-reading requirements no matter whether it read the number correctly. Decide this purely "
    "by what kind of action the step is performing, NOT by whether you suspect the step got it "
    "right; do not skip a requirement just because the step looks correct to you, and do not "
    "include one just because the step looks wrong. Only skip requirements about a genuinely "
    "different kind of action than what this step is doing.\n"
    "  2. VERDICT, only after step 1 is complete: for every requirement that passed the topic "
    "check, score 1-3 whether the step actually satisfies it.\n\n"
    + SCORE_SCALE_NOTE
    + "\nJudge each step ONLY against the chart image, the question, and the step's own text -- do "
    "not use any outside knowledge of what the correct final answer is; you are not told the "
    "ground truth and should not guess at it."
)

RESPONSE_FORMAT_INSTRUCTIONS = (
    "Respond strictly as a JSON array, one object per step, in this exact schema:\n"
    "[\n"
    '  {"step_index": 0, "scores": [{"criterion_id": "...", "score": 1, "note": "..."}]}\n'
    "]\n"
    "Only include requirements that passed the topic check in step 1 above.\n"
    "Do not include markdown formatting or any text outside the JSON array."
)


def format_criterion(child: Dict[str, Any]) -> str:
    """One-line description of a reward-tree child criterion for the prompt.

    Prefers the LLM-distilled `rubric_text` (v1, from distill_reward_tree_criteria.py)
    when present; falls back to the v0 TF-IDF-terms + exemplar description otherwise,
    so a partially-distilled tree (some children failed distillation) still works.
    """
    rubric_text = child.get("rubric_text")
    if rubric_text:
        return f'[{child["child_id"]}] requirement: {rubric_text}'
    terms = ", ".join(child.get("top_terms", [])[:6]) or "(no keywords)"
    exemplars = child.get("exemplars") or []
    example = f' e.g. "{exemplars[0]}"' if exemplars else ""
    return f'[{child["child_id"]}] failure pattern keywords: {terms}.{example}'


def format_tree_criteria_list(tree: Dict[str, Any]) -> str:
    """Every parent's children, grouped under the parent's label, shown once per prompt."""
    blocks = []
    for parent in tree.get("parents", {}).values():
        children = parent.get("children", [])
        if not children:
            continue
        lines = [f"{parent['label']}:"]
        lines.extend(f"  {format_criterion(child)}" for child in children)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_step_block(step_index: int, step_text: str) -> str:
    return f"Step {step_index}: {step_text}"


def build_dynamic_scoring_prompt(
    question: str, steps: Sequence[Tuple[int, str]], tree: Dict[str, Any]
) -> str:
    """Full judge prompt. Deliberately excludes the ground-truth answer -- see module docstring."""
    criteria_list = format_tree_criteria_list(tree)
    step_blocks = "\n".join(build_step_block(idx, text) for idx, text in steps)
    return (
        f"{TASK_INSTRUCTIONS}\n"
        f"Failure-pattern criteria (grouped by category):\n\n{criteria_list}\n\n"
        f"Chart Question: {question}\n\n"
        f"Steps:\n{step_blocks}\n\n"
        f"{RESPONSE_FORMAT_INSTRUCTIONS}"
    )


def parse_dynamic_scores(response_text: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Extract the judge's per-step, per-criterion scores. None if unparseable."""
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
    if not isinstance(parsed, list):
        return None
    for entry in parsed:
        if not isinstance(entry, dict) or "step_index" not in entry or "scores" not in entry:
            return None
    return parsed


def dynamic_process_score(parsed_scores: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Aggregate a rollout's per-step, per-criterion 1-3 scores into one process score in [0, 1].

    Per step: mean of (score-1)/2 across that step's flagged criteria; a step with no
    criteria flagged (no on-topic failure pattern found) scores 1.0 -- absence of a
    flagged issue is itself evidence the step is fine, not missing data. Steps are then
    averaged with equal weight, so a step that happened to trip more criteria doesn't
    dominate the rollout's score. None if there are no steps to score.
    """
    if not parsed_scores:
        return None
    step_scores = []
    for step in parsed_scores:
        criteria = step.get("scores") or []
        if not criteria:
            step_scores.append(1.0)
            continue
        normalized = [(c["score"] - 1) / 2 for c in criteria if c.get("score") in (1, 2, 3)]
        if normalized:
            step_scores.append(sum(normalized) / len(normalized))
    if not step_scores:
        return None
    return sum(step_scores) / len(step_scores)
