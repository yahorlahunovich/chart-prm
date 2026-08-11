"""
Dataset sanity checks for multimodal SFT / preference training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _text_fields(item: Dict[str, Any]) -> List[str]:
    values = []
    for key in ("solution", "response", "chosen", "rejected", "completion"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value)
    return values


def validate_training_dataset(
    dataset: Sequence[Dict[str, Any]],
    *,
    require_final_answer: bool = True,
    min_final_answer_rate: float = 0.8,
    min_mean_chars: float = 120.0,
    min_examples: int = 1,
    name: str = "dataset",
) -> Dict[str, float]:
    """
    Fail closed on fragment-style targets that previously collapsed holdout eval.

    Returns summary metrics for logging.
    """
    if len(dataset) < min_examples:
        raise ValueError(f"{name}: expected at least {min_examples} examples, got {len(dataset)}")

    texts: List[str] = []
    for item in dataset:
        fields = _text_fields(dict(item))
        if not fields:
            raise ValueError(f"{name}: example missing completion text: {item.get('question_id')}")
        texts.extend(fields)

    mean_chars = sum(len(text) for text in texts) / max(len(texts), 1)
    final_answer_rate = sum("Final Answer:" in text for text in texts) / max(len(texts), 1)
    step_rate = sum(text.lstrip().startswith("Step ") for text in texts) / max(len(texts), 1)

    if require_final_answer and final_answer_rate < min_final_answer_rate:
        raise ValueError(
            f"{name}: Final Answer rate {final_answer_rate:.2%} < {min_final_answer_rate:.0%}. "
            "Refusing fragment/step-only targets for full-response training. "
            "Use sft_samples.jsonl / dpo_pairs.jsonl / kto_samples.jsonl."
        )
    if mean_chars < min_mean_chars:
        raise ValueError(
            f"{name}: mean completion length {mean_chars:.1f} chars < {min_mean_chars}. "
            "This looks like Step-DPO fragments, not full trajectories."
        )

    return {
        "n_examples": float(len(dataset)),
        "n_text_fields": float(len(texts)),
        "mean_chars": float(mean_chars),
        "final_answer_rate": float(final_answer_rate),
        "step_prefix_rate": float(step_rate),
    }


def collapse_guard(
    history_row: Dict[str, Any],
    *,
    max_logp_drop_vs_ref: float = 40.0,
    logp_key: str = "chosen_logp",
    ref_key: str = "chosen_ref_logp",
) -> Optional[str]:
    """
    Return a warning/error string when policy log-probabilities collapse vs reference.
    """
    if logp_key not in history_row or ref_key not in history_row:
        return None
    policy = float(history_row[logp_key])
    ref = float(history_row[ref_key])
    drop = ref - policy
    if drop > max_logp_drop_vs_ref:
        return (
            f"policy {logp_key}={policy:.1f} is {drop:.1f} nats below {ref_key}={ref:.1f}; "
            "likely generative collapse"
        )
    return None
