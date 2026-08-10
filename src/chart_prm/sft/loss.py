"""
Core mathematical SFT loss implementation.
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_pad_token_id: int = -100,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes masked sequence cross-entropy loss for Supervised Fine-Tuning (SFT).

    L_SFT = -1/N * sum_{t: y_t != -100} log p_theta(y_t | x, y_<t)

    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len) with prompt and pad positions set to -100

    Returns:
        loss: Scalar cross-entropy loss tensor
        metrics: Dictionary containing loss value and token count
    """
    assert logits.ndim == 3, f"Expected 3D logits (batch, seq_len, vocab), got {logits.shape}"
    assert labels.ndim == 2, f"Expected 2D labels (batch, seq_len), got {labels.shape}"
    assert logits.shape[0] == labels.shape[0], f"Batch size mismatch: {logits.shape[0]} vs {labels.shape[0]}"
    assert logits.shape[1] == labels.shape[1], f"Sequence length mismatch: {logits.shape[1]} vs {labels.shape[1]}"

    # Shift logits and labels for autoregressive alignment:
    # logits[:, :-1] predicts labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Flatten tensors for CrossEntropyLoss calculation
    shift_logits_flat = shift_logits.view(-1, shift_logits.shape[-1])
    shift_labels_flat = shift_labels.view(-1)

    loss = F.cross_entropy(
        shift_logits_flat,
        shift_labels_flat,
        ignore_index=label_pad_token_id,
        reduction="mean",
    )

    assert torch.isfinite(loss), f"Non-finite SFT loss computed: {loss}"

    valid_tokens = (shift_labels_flat != label_pad_token_id).sum().item()
    metrics = {
        "loss": loss.item(),
        "valid_tokens": float(valid_tokens),
    }

    return loss, metrics
