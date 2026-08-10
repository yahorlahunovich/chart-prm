"""
Custom SFT Trainer engine for Qwen2.5-VL.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from chart_prm.sft.loss import sft_loss
from chart_prm.sft.utils import build_qwen_sft_batch


def train_sft_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    max_grad_norm: Optional[float] = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Executes a single minimal SFT training step with VRAM cache clearing.

    Args:
        model: Qwen2.5-VL policy model
        optimizer: PyTorch optimizer
        batch: Formatted SFT batch dict containing input_ids, labels, attention_mask, etc.
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        loss: Loss tensor
        metrics: Dictionary containing loss value and token count
    """
    model.train()

    model_inputs = {k: v for k, v in batch.items() if k != "labels"}
    outputs = model(**model_inputs)

    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    labels = batch["labels"]

    loss, metrics = sft_loss(logits, labels)

    optimizer.zero_grad()
    loss.backward()

    if max_grad_norm is not None:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)

    optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return loss, metrics


def fit_sft(
    model: Any,
    dataset: List[Dict[str, Any]],
    processor: Any,
    lr: float = 1e-5,
    epochs: int = 1,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    on_step_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> List[Dict[str, float]]:
    """
    Fits policy model on target solution dataset using SFT.

    Args:
        model: Trainable Qwen2.5-VL policy model
        dataset: SFT dataset items
        processor: Qwen2.5-VL processor
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size per step
        device: Torch compute device
        on_step_end: Optional callback executed at step end

    Returns:
        history: List of metric dicts for each step
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("Model has no trainable parameters! Ensure LoRA adapters are attached.")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    history = []
    global_step = 0

    for epoch in range(epochs):
        print(f"--- Epoch {epoch + 1}/{epochs} ---")
        for i in range(0, len(dataset), batch_size):
            batch_items = dataset[i : i + batch_size]

            batch = build_qwen_sft_batch(
                processor=processor,
                items=batch_items,
                device=device,
            )

            loss, metrics = train_sft_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
            )

            global_step += 1
            step_metrics = {"step": global_step, "epoch": epoch + 1, **metrics}
            history.append(step_metrics)

            if on_step_end is not None:
                on_step_end(global_step, metrics)

    return history
