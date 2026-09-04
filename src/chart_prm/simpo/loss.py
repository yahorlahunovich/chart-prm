"""
Core mathematical SimPO loss implementation.

Reimplemented from princeton-nlp/SimPO's simpo_loss (scripts/simpo_trainer.py)
against this project's existing DPO trainer conventions, not copied verbatim --
same formula, this codebase's tensor/metrics shape.
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def simpo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    beta: float = 2.0,
    gamma_beta_ratio: float = 0.5,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes the SimPO (Simple Preference Optimization) loss.

    L_SimPO = -E[ log sigma( beta * (logp_w/|y_w| - logp_l/|y_l|) - gamma ) ]

    Unlike `chart_prm.dpo.loss.dpo_loss`, there is no reference model anywhere
    in this formula -- the reward is the policy's own length-normalized
    log-probability (`chosen_logp`/`rejected_logp` must already be computed
    with `chart_prm.dpo.utils.sequence_logprob(..., average=True)`, i.e. mean
    log-probability per response token, not summed). That length
    normalization is what stands in for DPO's reference-relative KL term, and
    is what removes DPO's structural bias toward longer chosen responses.

    `gamma_beta_ratio` is gamma/beta (the target reward margin expressed as a
    ratio, matching the reference implementation's parameterization so beta
    and gamma can be tuned semi-independently, per the SimPO paper).
    """
    assert chosen_logp.shape == rejected_logp.shape, (
        f"Shape mismatch: chosen_logp {chosen_logp.shape} vs rejected_logp {rejected_logp.shape}"
    )

    chosen_reward = beta * chosen_logp.detach()
    rejected_reward = beta * rejected_logp.detach()
    margin = chosen_reward - rejected_reward

    pi_logratios = chosen_logp - rejected_logp
    logits = pi_logratios - gamma_beta_ratio

    loss = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    ).mean()

    assert torch.isfinite(loss), f"Non-finite SimPO loss computed: {loss}"

    metrics = {
        "loss": loss.item(),
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
        "reward_margin": margin.mean().item(),
        "preference_accuracy": (chosen_reward > rejected_reward).float().mean().item(),
        "chosen_logp": chosen_logp.mean().item(),
        "rejected_logp": rejected_logp.mean().item(),
    }

    return loss, metrics
