"""
Core mathematical DPO loss implementation.
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def dpo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes the Direct Preference Optimization (DPO) loss.

    L_DPO = -E [ log sigma ( beta * ( (log pi_theta(y_w|x) - log pi_ref(y_w|x))
                                      - (log pi_theta(y_l|x) - log pi_ref(y_l|x)) ) ) ]
    """
    assert chosen_logp.shape == rejected_logp.shape, (
        f"Shape mismatch: chosen_logp {chosen_logp.shape} vs rejected_logp {rejected_logp.shape}"
    )
    assert chosen_ref_logp.shape == chosen_logp.shape, (
        f"Shape mismatch: chosen_ref_logp {chosen_ref_logp.shape} vs chosen_logp {chosen_logp.shape}"
    )
    assert rejected_ref_logp.shape == rejected_logp.shape, (
        f"Shape mismatch: rejected_ref_logp {rejected_ref_logp.shape} vs rejected_logp {rejected_logp.shape}"
    )

    chosen_reward = beta * (chosen_logp - chosen_ref_logp)
    rejected_reward = beta * (rejected_logp - rejected_ref_logp)
    margin = chosen_reward - rejected_reward

    loss = -F.logsigmoid(margin).mean()

    assert torch.isfinite(loss), f"Non-finite DPO loss computed: {loss}"

    metrics = {
        "loss": loss.item(),
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
        "reward_margin": margin.mean().item(),
        "preference_accuracy": (chosen_reward > rejected_reward).float().mean().item(),
        "chosen_logp": chosen_logp.mean().item(),
        "rejected_logp": rejected_logp.mean().item(),
        "chosen_ref_logp": chosen_ref_logp.mean().item(),
        "rejected_ref_logp": rejected_ref_logp.mean().item(),
    }

    return loss, metrics
