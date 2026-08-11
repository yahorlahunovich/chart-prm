"""
Shared causal-LM label masking helpers for Qwen2.5-VL trainers.
Assumes tokenizer padding_side='right' (set in train_*.py).
"""

from __future__ import annotations

from typing import Optional

import torch


def join_prefix_and_completion(prefix: str, completion: str) -> str:
    """Join shared reasoning prefix with the next-step / full completion."""
    prefix = (prefix or "").rstrip()
    completion = (completion or "").strip()
    if not prefix:
        return completion
    return f"{prefix}\n{completion}"


def mask_response_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    prefix_lengths: Optional[torch.Tensor] = None,
    label_pad_token_id: int = -100,
) -> torch.Tensor:
    """
    Mask prompt tokens (and optional in-response prefix) plus padding.

    Sequences must be right-padded so content starts at index 0.
    If pad_token_id == eos_token_id, padding is masked via attention_mask only
    so the real response EOS remains supervised.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"Expected 2D input_ids, got {tuple(input_ids.shape)}")
    labels = input_ids.clone()
    batch_size = labels.shape[0]
    for i in range(batch_size):
        prompt_len = int(prompt_lengths[i].item())
        labels[i, :prompt_len] = label_pad_token_id
        if prefix_lengths is not None:
            prefix_len = int(prefix_lengths[i].item())
            if prefix_len > 0:
                labels[i, prompt_len : prompt_len + prefix_len] = label_pad_token_id

    labels = labels.masked_fill(attention_mask == 0, label_pad_token_id)

    if pad_token_id is not None:
        if eos_token_id is not None and pad_token_id == eos_token_id:
            # Pads already cleared via attention_mask; do not strip response EOS.
            pass
        else:
            labels = labels.masked_fill(input_ids == pad_token_id, label_pad_token_id)

    return labels
