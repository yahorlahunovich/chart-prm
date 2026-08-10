"""
Custom DPO Trainer engine for Qwen2.5-VL.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from chart_prm.dpo.loss import dpo_loss
from chart_prm.dpo.utils import build_qwen_dpo_batch


def compute_sequence_logprobs(
    model: Any,
    batch: Dict[str, torch.Tensor],
    is_reference: bool = False,
) -> torch.Tensor:
    """
    Computes sequence log-probabilities for a given model and batch.

    Args:
        model: Trainable policy model (or base model with disabled adapters)
        batch: Formatted Qwen2.5-VL batch containing input_ids, labels, attention_mask, etc.
        is_reference: If True, evaluates in reference mode (using model.disable_adapter() if PEFT model)

    Returns:
        seq_logprobs: (batch_size,) tensor of sequence log-probabilities
    """
    from chart_prm.dpo.utils import sequence_logprob

    # Construct model forward inputs excluding labels
    model_inputs = {k: v for k, v in batch.items() if k != "labels"}

    if is_reference and hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            outputs = model(**model_inputs)
    else:
        outputs = model(**model_inputs)

    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    labels = batch["labels"]

    seq_logps = sequence_logprob(logits, labels)
    return seq_logps


def train_dpo_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    chosen_batch: Dict[str, torch.Tensor],
    rejected_batch: Dict[str, torch.Tensor],
    ref_model: Optional[Any] = None,
    beta: float = 0.1,
    max_grad_norm: Optional[float] = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Executes a single minimal DPO training step with memory efficient cache clearing.
    """
    model.train()

    # Policy model chosen log-probabilities
    chosen_logp = compute_sequence_logprobs(model, chosen_batch, is_reference=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reference model chosen log-probabilities
    if ref_model is not None:
        ref_model.eval()
        with torch.no_grad():
            chosen_ref_logp = compute_sequence_logprobs(ref_model, chosen_batch, is_reference=False)
    else:
        with torch.no_grad():
            chosen_ref_logp = compute_sequence_logprobs(model, chosen_batch, is_reference=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Policy model rejected log-probabilities
    rejected_logp = compute_sequence_logprobs(model, rejected_batch, is_reference=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reference model rejected log-probabilities
    if ref_model is not None:
        ref_model.eval()
        with torch.no_grad():
            rejected_ref_logp = compute_sequence_logprobs(ref_model, rejected_batch, is_reference=False)
    else:
        with torch.no_grad():
            rejected_ref_logp = compute_sequence_logprobs(model, rejected_batch, is_reference=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return loss, metrics


def fit_dpo(
    model: Any,
    dataset: List[Dict[str, Any]],
    processor: Any,
    ref_model: Optional[Any] = None,
    lr: float = 1e-5,
    beta: float = 0.1,
    epochs: int = 1,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    on_step_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> List[Dict[str, float]]:
    """
    Fits policy model on preference dataset using DPO.
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

            loss, metrics = train_dpo_step(
                model=model,
                optimizer=optimizer,
                chosen_batch=chosen_batch,
                rejected_batch=rejected_batch,
                ref_model=ref_model,
                beta=beta,
            )

            global_step += 1
            step_metrics = {"step": global_step, "epoch": epoch + 1, **metrics}
            history.append(step_metrics)

            if on_step_end is not None:
                on_step_end(global_step, metrics)

    return history
