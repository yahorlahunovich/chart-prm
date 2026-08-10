"""
Minimal DPO package for Qwen2.5-VL.
"""

from chart_prm.dpo.loss import dpo_loss
from chart_prm.dpo.trainer import fit_dpo, train_dpo_step
from chart_prm.dpo.utils import build_qwen_dpo_batch, sequence_logprob

__all__ = [
    "dpo_loss",
    "sequence_logprob",
    "build_qwen_dpo_batch",
    "train_dpo_step",
    "fit_dpo",
]
