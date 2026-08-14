"""Merge a single-system holdout jsonl onto experiment 005 generations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from chart_prm.holdout_metrics import normalize_answer

SFT_DPO_KEY = "sft_dpo"


def merge_system_rows(
    base_rows: Sequence[Dict[str, Any]],
    extra_rows: Sequence[Dict[str, Any]],
    system: str = SFT_DPO_KEY,
) -> List[Dict[str, Any]]:
    extra = {str(row["question_id"]): row for row in extra_rows}
    base_ids = [str(row["question_id"]) for row in base_rows]
    extra_ids = set(extra)
    missing = [qid for qid in base_ids if qid not in extra_ids]
    unexpected = extra_ids - set(base_ids)
    if missing or unexpected:
        raise ValueError(
            f"{system} holdout IDs do not match the existing generations. "
            f"missing={missing[:8]} unexpected={sorted(unexpected)[:8]}"
        )
    merged: List[Dict[str, Any]] = []
    for row in base_rows:
        qid = str(row["question_id"])
        src = extra[qid]
        src_responses = src.get("responses") or {}
        if system not in src_responses:
            raise KeyError(f"{system} response missing for question_id={qid}")
        out = dict(row)
        responses = dict(row.get("responses") or {})
        predicted = dict(row.get("predicted_answers") or {})
        responses[system] = src_responses[system]
        predicted[system] = (src.get("predicted_answers") or {}).get(system, "")
        out["responses"] = responses
        out["predicted_answers"] = predicted
        merged.append(out)
    return merged


def exact_match_summary(
    rows: Sequence[Dict[str, Any]],
    systems: Iterable[str] | None = None,
) -> Dict[str, Any]:
    if systems is None:
        systems = []
        seen = set()
        for row in rows:
            for name in row.get("predicted_answers") or {}:
                if name not in seen:
                    seen.add(name)
                    systems.append(name)
    systems = list(systems)
    summary: Dict[str, Any] = {"n": len(rows), "exact_match": {}, "extracted_answer_rate": {}}
    for system in systems:
        correct = 0
        extracted = 0
        for row in rows:
            pred = (row.get("predicted_answers") or {}).get(system, "")
            if pred:
                extracted += 1
            if normalize_answer(pred) == normalize_answer(row.get("ground_truth")):
                correct += 1
        n = len(rows) or 1
        summary["exact_match"][system] = {"correct": correct, "accuracy": correct / n}
        summary["extracted_answer_rate"][system] = extracted / n
    return summary
