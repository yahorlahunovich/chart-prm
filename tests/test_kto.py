"""
Unit tests for Kahneman-Tversky Optimization (KTO) loss, batch formatting, and trainer step.
"""

import copy
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from chart_prm.kto.loss import kto_loss
from chart_prm.kto.trainer import train_kto_step


class ToyKTOModel(nn.Module):
    """Simple toy model for unit testing KTO optimization without heavy VLM loading."""

    def __init__(self, vocab_size: int = 10, hidden_dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.embedding(input_ids)
        logits = self.linear(h)
        return logits


def test_kto_loss_manual_math():
    """
    Test 1: Verify kto_loss against manual Prospect Theory math.
    """
    policy_logp = torch.tensor([-20.0, -30.0])
    ref_logp = torch.tensor([-22.0, -28.0])
    kto_labels = torch.tensor([1, -1])  # Item 0 is desirable (+1), Item 1 is undesirable (-1)
    beta = 0.1

    loss, metrics = kto_loss(
        policy_logp=policy_logp,
        ref_logp=ref_logp,
        kto_labels=kto_labels,
        beta=beta,
    )

    # Implicit rewards:
    # r0 = 0.1 * (-20 - (-22)) = +0.2
    # r1 = 0.1 * (-30 - (-28)) = -0.2
    # Item 0 desirable: loss = 1 - sigmoid(0.2) = sigmoid(-0.2)
    # Item 1 undesirable: loss = 1 - sigmoid(0.2) = sigmoid(-0.2)
    expected_loss = (F.sigmoid(torch.tensor(-0.2)) + F.sigmoid(torch.tensor(-0.2))) / 2.0

    assert torch.isclose(loss, expected_loss, atol=1e-6)
    assert metrics["mean_desirable_reward"] == pytest.approx(0.2, abs=1e-4)
    assert metrics["mean_undesirable_reward"] == pytest.approx(-0.2, abs=1e-4)
    assert metrics["reward_margin"] == pytest.approx(0.4, abs=1e-4)


def test_kto_trainer_synthetic_optimization():
    """
    Test 2: Verify KTO training step updates model weights and increases desirable reward relative to undesirable reward.
    """
    torch.manual_seed(42)
    vocab_size = 10
    model = ToyKTOModel(vocab_size=vocab_size)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=0.05)

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]]),
        "labels": torch.tensor([[-100, -100, 3, 4], [-100, -100, 5, 6]]),
        "kto_labels": torch.tensor([1, -1]),
    }

    # Run KTO training steps
    for _ in range(15):
        loss, metrics = train_kto_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            ref_model=ref_model,
            beta=0.1,
        )

    assert metrics["reward_margin"] > 0.0
