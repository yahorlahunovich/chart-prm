"""
Core mathematical KTO (Kahneman-Tversky Optimization) loss implementation.
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def kto_loss(
    policy_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    kto_labels: torch.Tensor,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    kl_baseline: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes Kahneman-Tversky Optimization (KTO) loss for unpaired binary completions.

    Reward r_theta(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x))

    L_KTO(y | z=+1) = w_D * (1 - sigma( r_theta(x, y) - z_0 ))
    L_KTO(y | z=-1) = w_U * (1 - sigma( z_0 - r_theta(x, y) ))

    Args:
        policy_logp: (batch_size,) log-probabilities under policy model
        ref_logp: (batch_size,) log-probabilities under frozen reference model
        kto_labels: (batch_size,) binary labels: +1 for desirable, -1 for undesirable
        beta: Scaling factor for implicit reward
        desirable_weight: Weight multiplier for desirable completions (w_D)
        undesirable_weight: Weight multiplier for undesirable completions (w_U)
        kl_baseline: Reference baseline value z_0 (typically 0.0 or running KL divergence)

    Returns:
        loss: Scalar KTO loss tensor
        metrics: Dictionary of metric values
    """
    assert policy_logp.shape == ref_logp.shape, (
        f"Shape mismatch: policy_logp {policy_logp.shape} vs ref_logp {ref_logp.shape}"
    )
    assert policy_logp.shape == kto_labels.shape, (
        f"Shape mismatch: policy_logp {policy_logp.shape} vs kto_labels {kto_labels.shape}"
    )

    # Compute implicit reward
    rewards = beta * (policy_logp - ref_logp)

    is_desirable = kto_labels > 0
    is_undesirable = kto_labels < 0

    # Desirable loss: w_D * (1 - sigmoid(r - z_0)) = w_D * sigmoid(z_0 - r)
    desirable_loss = desirable_weight * torch.sigmoid(kl_baseline - rewards)

    # Undesirable loss: w_U * (1 - sigmoid(z_0 - r)) = w_U * sigmoid(r - z_0)
    undesirable_loss = undesirable_weight * torch.sigmoid(rewards - kl_baseline)

    per_sample_loss = torch.where(is_desirable, desirable_loss, undesirable_loss)
    loss = per_sample_loss.mean()

    assert torch.isfinite(loss), f"Non-finite KTO loss computed: {loss}"

    desirable_mask_float = is_desirable.float()
    undesirable_mask_float = is_undesirable.float()

    mean_desirable_reward = (
        (rewards * desirable_mask_float).sum() / (desirable_mask_float.sum() + 1e-8)
    ).item()
    mean_undesirable_reward = (
        (rewards * undesirable_mask_float).sum() / (undesirable_mask_float.sum() + 1e-8)
    ).item()

    metrics = {
        "loss": loss.item(),
        "mean_desirable_reward": mean_desirable_reward,
        "mean_undesirable_reward": mean_undesirable_reward,
        "reward_margin": mean_desirable_reward - mean_undesirable_reward,
        "policy_logp": policy_logp.mean().item(),
        "ref_logp": ref_logp.mean().item(),
    }

    return loss, metrics
