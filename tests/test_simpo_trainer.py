"""
Unit tests for SimPO trainer: gradient flow, length normalization, and synthetic
preference learning without any reference model.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from chart_prm.simpo.trainer import train_simpo_step, compute_policy_logprobs
from chart_prm.simpo.loss import simpo_loss


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


def test_policy_gets_gradients_with_no_reference_model():
    """
    Test 1: train_simpo_step never touches a second model -- policy parameters
    get non-zero gradients from a single-model training step.
    """
    torch.manual_seed(42)
    vocab_size = 10
    policy_model = ToyCausalLM(vocab_size=vocab_size)
    optimizer = optim.SGD(policy_model.parameters(), lr=0.1)

    chosen_batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
    }
    rejected_batch = {
        "input_ids": torch.tensor([[1, 2, 5, 6]]),
        "labels": torch.tensor([[-100, -100, 5, 6]]),
    }

    loss, metrics = train_simpo_step(
        model=policy_model,
        chosen_batch=chosen_batch,
        rejected_batch=rejected_batch,
        optimizer=optimizer,
        beta=2.0,
        gamma_beta_ratio=0.5,
    )

    has_nonzero_grad = False
    for p in policy_model.parameters():
        if p.requires_grad and p.grad is not None:
            if (p.grad != 0).any():
                has_nonzero_grad = True
                break
    assert has_nonzero_grad, "Trainable policy model received no non-zero gradients!"


def test_compute_policy_logprobs_matches_length_normalized_sequence_logprob():
    """
    Test 2: compute_policy_logprobs is exactly sequence_logprob(..., average=True)
    on the model's own logits -- a direct wiring check, not a claim about model
    behavior (length-invariance through a real model depends on which tokens
    appear, not just how many -- that's covered precisely, without a model in
    the loop, by test_logprob.py's test_sequence_logprob_average_divides_by_response_length).
    """
    torch.manual_seed(7)
    from chart_prm.dpo.utils import sequence_logprob

    model = ToyCausalLM(vocab_size=10)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 3, 3]]),
        "labels": torch.tensor([[-100, -100, 3, 3, 3]]),
    }

    with torch.no_grad():
        actual = compute_policy_logprobs(model, batch)
        expected = sequence_logprob(model(batch["input_ids"]), batch["labels"], average=True)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_synthetic_simpo_preference_learning():
    """
    Test 3: Verify synthetic preference optimization improves reward margin
    over training steps, with no reference model involved anywhere.
    """
    torch.manual_seed(123)
    vocab_size = 12
    policy_model = ToyCausalLM(vocab_size=vocab_size)
    optimizer = optim.AdamW(policy_model.parameters(), lr=0.05)

    chosen_batch = {
        "input_ids": torch.tensor([[1, 2, 7, 8]]),
        "labels": torch.tensor([[-100, -100, 7, 8]]),
    }
    rejected_batch = {
        "input_ids": torch.tensor([[1, 2, 9, 10]]),
        "labels": torch.tensor([[-100, -100, 9, 10]]),
    }

    policy_model.eval()
    with torch.no_grad():
        c_logp_init = compute_policy_logprobs(policy_model, chosen_batch)
        r_logp_init = compute_policy_logprobs(policy_model, rejected_batch)
    _, metrics_init = simpo_loss(c_logp_init, r_logp_init, beta=2.0, gamma_beta_ratio=0.5)
    margin_before = metrics_init["reward_margin"]

    for _ in range(15):
        loss, metrics = train_simpo_step(
            model=policy_model,
            chosen_batch=chosen_batch,
            rejected_batch=rejected_batch,
            optimizer=optimizer,
            beta=2.0,
            gamma_beta_ratio=0.5,
        )

    margin_after = metrics["reward_margin"]

    assert margin_after > margin_before, (
        f"SimPO optimization failed to improve reward margin: before={margin_before:.4f}, after={margin_after:.4f}"
    )
    assert metrics["preference_accuracy"] == 1.0
