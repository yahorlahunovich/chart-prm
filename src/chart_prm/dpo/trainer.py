"""
Minimal custom DPO trainer for Qwen2.5-VL and causal language models.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim

from chart_prm.dpo.loss import dpo_loss
from chart_prm.dpo.utils import sequence_logprob


def compute_sequence_logprobs(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    is_reference: bool = False,
) -> torch.Tensor:
    """
    Computes sequence log probabilities for a batch of model inputs.
    """
    # Extract only valid forward kwargs
    model_inputs = {
        k: v for k, v in batch.items()
        if k not in ("labels", "question_id", "image_path")
    }

    if is_reference:
        if hasattr(model, "disable_adapter"):
            with torch.no_grad(), model.disable_adapter():
                outputs = model(**model_inputs)
        else:
            with torch.no_grad():
                outputs = model(**model_inputs)
    else:
        outputs = model(**model_inputs)

    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    labels = batch["labels"]

    return sequence_logprob(logits, labels)


def train_dpo_step(
    model: nn.Module,
    chosen_batch: Dict[str, torch.Tensor],
    rejected_batch: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    ref_model: Optional[nn.Module] = None,
    beta: float = 0.1,
    max_grad_norm: Optional[float] = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Executes a single minimal DPO training step.
    """
    model.train()

    # Policy model log-probabilities
    chosen_logp = compute_sequence_logprobs(model, chosen_batch, is_reference=False)
    rejected_logp = compute_sequence_logprobs(model, rejected_batch, is_reference=False)

    # Reference model log-probabilities
    if ref_model is not None:
        ref_model.eval()
        with torch.no_grad():
            chosen_ref_logp = compute_sequence_logprobs(ref_model, chosen_batch, is_reference=False)
            rejected_ref_logp = compute_sequence_logprobs(ref_model, rejected_batch, is_reference=False)
    else:
        with torch.no_grad():
            chosen_ref_logp = compute_sequence_logprobs(model, chosen_batch, is_reference=True)
            rejected_ref_logp = compute_sequence_logprobs(model, rejected_batch, is_reference=True)

    # Compute DPO loss
    loss, metrics = dpo_loss(
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta=beta
    )

    # Gradient step
    optimizer.zero_grad()
    loss.backward()

    if max_grad_norm is not None:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)

    optimizer.step()

    return loss, metrics


def fit_dpo(
    model: nn.Module,
    dataset: List[Dict[str, Any]],
    processor: Any,
    ref_model: Optional[nn.Module] = None,
    lr: float = 1e-5,
    beta: float = 0.1,
    epochs: int = 1,
    batch_size: int = 1,
    max_grad_norm: float = 1.0,
    device: Optional[torch.device] = None,
    on_step_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> List[Dict[str, float]]:
    """
    Minimal training loop for DPO fine-tuning.
    """
    from chart_prm.dpo.utils import build_qwen_dpo_batch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    if ref_model is not None:
        ref_model.to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr)

    history = []
    global_step = 0

    for epoch in range(epochs):
        # Create mini-batches
        for i in range(0, len(dataset), batch_size):
            batch_items = dataset[i : i + batch_size]
            chosen_batch, rejected_batch = build_qwen_dpo_batch(
                processor=processor, items=batch_items, device=device
            )

            loss, metrics = train_dpo_step(
                model=model,
                chosen_batch=chosen_batch,
                rejected_batch=rejected_batch,
                optimizer=optimizer,
                ref_model=ref_model,
                beta=beta,
                max_grad_norm=max_grad_norm,
            )

            global_step += 1
            step_metrics = {"epoch": epoch + 1, "step": global_step, **metrics}
            history.append(step_metrics)

            if on_step_end is not None:
                on_step_end(global_step, step_metrics)

    return history
