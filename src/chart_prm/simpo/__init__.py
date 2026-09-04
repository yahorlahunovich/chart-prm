"""
Minimal SimPO package for Qwen2.5-VL.
"""

from chart_prm.simpo.loss import simpo_loss
from chart_prm.simpo.trainer import compute_policy_logprobs, fit_simpo, train_simpo_step

__all__ = [
    "simpo_loss",
    "compute_policy_logprobs",
    "train_simpo_step",
    "fit_simpo",
]
