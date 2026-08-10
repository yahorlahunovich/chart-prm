"""
Unit tests for reference model freezing, trainable model gradients, and synthetic DPO optimization.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pytest

from chart_prm.dpo.trainer import train_dpo_step, compute_sequence_logprobs
from chart_prm.dpo.loss import dpo_loss


class ToyCausalLM(nn.Module):
    """Tiny toy causal language model for fast unit testing."""
    def __init__(self, vocab_size=16, hidden_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        logits = self.linear(x)
        return logits


def test_reference_model_frozen_and_policy_gets_gradients():
    """
    Tests 7 & 8: Verify that reference model parameters remain frozen (no grad, unchanged)
    and trainable policy model parameters receive non-zero gradients.
    """
    torch.manual_seed(42)
    vocab_size = 10
    policy_model = ToyCausalLM(vocab_size=vocab_size)
    ref_model = copy.deepcopy(policy_model)

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    ref_params_before = [p.clone() for p in ref_model.parameters()]

    optimizer = optim.SGD(policy_model.parameters(), lr=0.1)

    chosen_batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
    }
    rejected_batch = {
        "input_ids": torch.tensor([[1, 2, 5, 6]]),
        "labels": torch.tensor([[-100, -100, 5, 6]]),
    }

    loss, metrics = train_dpo_step(
        model=policy_model,
        chosen_batch=chosen_batch,
        rejected_batch=rejected_batch,
        optimizer=optimizer,
        ref_model=ref_model,
        beta=0.1,
    )

    # Test 7: Reference model must receive NO gradients and parameters must remain unchanged
    for p in ref_model.parameters():
        assert p.grad is None, "Reference model parameter received a gradient!"

    for p_before, p_after in zip(ref_params_before, ref_model.parameters()):
        assert torch.equal(p_before, p_after), "Reference model parameter values changed!"

    # Test 8: Policy model MUST receive non-zero gradients
    has_nonzero_grad = False
    for p in policy_model.parameters():
        if p.requires_grad and p.grad is not None:
            if (p.grad != 0).any():
                has_nonzero_grad = True
                break
    assert has_nonzero_grad, "Trainable policy model received no non-zero gradients!"


def test_synthetic_dpo_preference_learning():
    """
    Test 9: Verify synthetic preference optimization improves reward margin over training steps.
    """
    torch.manual_seed(123)
    vocab_size = 12
    policy_model = ToyCausalLM(vocab_size=vocab_size)
    ref_model = copy.deepcopy(policy_model)

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(policy_model.parameters(), lr=0.05)

    # Synthetic batch: chosen response tokens [7, 8], rejected response tokens [9, 10]
    chosen_batch = {
        "input_ids": torch.tensor([[1, 2, 7, 8]]),
        "labels": torch.tensor([[-100, -100, 7, 8]]),
    }
    rejected_batch = {
        "input_ids": torch.tensor([[1, 2, 9, 10]]),
        "labels": torch.tensor([[-100, -100, 9, 10]]),
    }

    # Initial margin measurement before optimization
    policy_model.eval()
    with torch.no_grad():
        c_logp_init = compute_sequence_logprobs(policy_model, chosen_batch)
        r_logp_init = compute_sequence_logprobs(policy_model, rejected_batch)
        ref_c_logp = compute_sequence_logprobs(ref_model, chosen_batch)
        ref_r_logp = compute_sequence_logprobs(ref_model, rejected_batch)

    _, metrics_init = dpo_loss(c_logp_init, r_logp_init, ref_c_logp, ref_r_logp, beta=0.5)
    margin_before = metrics_init["reward_margin"]

    # Run 15 DPO optimization steps
    for step in range(15):
        loss, metrics = train_dpo_step(
            model=policy_model,
            chosen_batch=chosen_batch,
            rejected_batch=rejected_batch,
            optimizer=optimizer,
            ref_model=ref_model,
            beta=0.5,
        )

    margin_after = metrics["reward_margin"]

    # Verify that the DPO optimization increased the reward margin significantly
    assert margin_after > margin_before, (
        f"DPO optimization failed to improve reward margin: before={margin_before:.4f}, after={margin_after:.4f}"
    )
    assert metrics["preference_accuracy"] == 1.0
