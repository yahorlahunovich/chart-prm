#!/usr/bin/env python3
"""
distill_reward_tree_criteria.py

Finishes the part of DG-PRM Phase 1 that experiment 009 deliberately left
undone: turns each reward-tree child's TF-IDF terms + exemplar sentence
into one clean, general rubric statement, via a text-only Gemini call
(one call per parent category, batching all its children together -- 9
calls total for the current tree, not 33). Writes the result back onto
`experiments/009_reward_tree/data/reward_tree.json` as a `rubric_text`
field on each child; `chart_prm.dynamic_scoring.format_criterion` prefers
`rubric_text` when present.

No image needed (text-only), so this is cheap relative to the Phase 2
step-scoring calls.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from score_steps_dynamic import build_gemini_text_payload, call_gemini_api, DEFAULT_MODEL  # noqa: E402
from chart_prm.criteria_distillation import (  # noqa: E402
    build_distillation_prompt,
    parse_distillation_response,
)


async def distill_tree(tree_path: Path, model: str = DEFAULT_MODEL) -> dict:
    with tree_path.open(encoding="utf-8") as handle:
        tree = json.load(handle)

    async with aiohttp.ClientSession() as session:
        for parent_key, parent in tree["parents"].items():
            children = parent.get("children", [])
            if not children:
                continue
            child_ids = [c["child_id"] for c in children]
            prompt = build_distillation_prompt(parent["label"], children)
            payload = build_gemini_text_payload(prompt)
            response_text = await call_gemini_api(session, payload, model=model)
            rubric_by_id = parse_distillation_response(response_text, child_ids)
            if rubric_by_id is None:
                print(f"FAILED to distill {parent_key} ({len(children)} children) -- left as v0")
                continue
            for child in children:
                child["rubric_text"] = rubric_by_id[child["child_id"]]
            print(f"Distilled {parent_key}: {len(children)} rubric statements")

    tree["child_criteria_status"] = (
        "v1: LLM-distilled rubric statements (rubric_text) via distill_reward_tree_criteria.py, "
        "falling back to v0 TF-IDF/exemplar description for any child a distillation call failed on"
    )
    return tree


async def main_async(tree_path: Path, model: str) -> None:
    tree = await distill_tree(tree_path, model=model)
    tree_path.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")
    n_distilled = sum(
        1
        for parent in tree["parents"].values()
        for child in parent["children"]
        if "rubric_text" in child
    )
    n_total = sum(len(p["children"]) for p in tree["parents"].values())
    print(f"\n{n_distilled}/{n_total} children now have rubric_text. Wrote {tree_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    tree_path = base_dir / "experiments/009_reward_tree/data/reward_tree.json"
    asyncio.run(main_async(tree_path, DEFAULT_MODEL))
