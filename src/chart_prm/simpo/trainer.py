"""
Custom SimPO Trainer engine for Qwen2.5-VL.

No reference model anywhere in this file, and no reference forward pass --
that's SimPO's whole point. `chart_prm.dpo.trainer.train_dpo_step` runs two
forwards per side (policy and reference); this runs one, per side.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from chart_prm.dpo.utils import build_qwen_dpo_batch, sequence_logprob
from chart_prm.simpo.loss import simpo_loss


def compute_policy_logprobs(model: Any, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Length-normalized (mean per-token) log-probability under the policy model."""
    model_inputs = {k: v for k, v in batch.items() if k != "labels"}
    outputs = model(**model_inputs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    return sequence_logprob(logits, batch["labels"], average=True)


def train_simpo_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    chosen_batch: Dict[str, torch.Tensor],
    rejected_batch: Dict[str, torch.Tensor],
    beta: float = 2.0,
    gamma_beta_ratio: float = 0.5,
    max_grad_norm: Optional[float] = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Executes a single SimPO training step with memory efficient cache clearing.
    """
    model.train()

    chosen_logp = compute_policy_logprobs(model, chosen_batch)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rejected_logp = compute_policy_logprobs(model, rejected_batch)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loss, metrics = simpo_loss(
        chosen_logp, rejected_logp, beta=beta, gamma_beta_ratio=gamma_beta_ratio
    )

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


def fit_simpo(
    model: Any,
    dataset: List[Dict[str, Any]],
    processor: Any,
    lr: float = 1e-6,
    beta: float = 2.0,
    gamma_beta_ratio: float = 0.5,
    epochs: int = 1,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    on_step_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> List[Dict[str, float]]:
    """
    Fits policy model on preference dataset using SimPO.
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

            chosen_batch, rejected_batch = build_qwen_dpo_batch(
                processor=processor,
                items=batch_items,
                device=device,
            )

            loss, metrics = train_simpo_step(
                model=model,
                optimizer=optimizer,
                chosen_batch=chosen_batch,
                rejected_batch=rejected_batch,
                beta=beta,
                gamma_beta_ratio=gamma_beta_ratio,
            )

            global_step += 1
            step_metrics = {"step": global_step, "epoch": epoch + 1, **metrics}
            history.append(step_metrics)

            if on_step_end is not None:
                on_step_end(global_step, metrics)

    return history
