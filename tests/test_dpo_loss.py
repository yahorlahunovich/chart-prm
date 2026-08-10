"""
Unit tests for mathematical DPO loss function.
"""

import math
import torch
import pytest
from chart_prm.dpo.loss import dpo_loss


def test_dpo_loss_exact_value():
    """
    Test 1: Verify dpo_loss matches exact manual calculation.
    """
    chosen_logp = torch.tensor([[-2.0]])
    rejected_logp = torch.tensor([[-4.0]])
    chosen_ref_logp = torch.tensor([[-3.0]])
    rejected_ref_logp = torch.tensor([[-3.0]])
    beta = 0.1

    # Manual calculation:
    # chosen_reward = 0.1 * (-2.0 - (-3.0)) = 0.1 * 1.0 = 0.1
    # rejected_reward = 0.1 * (-4.0 - (-3.0)) = 0.1 * (-1.0) = -0.1
    # margin = 0.1 - (-0.1) = 0.2
    # expected_loss = -log_sigmoid(0.2) = log(1 + exp(-0.2))
    expected_margin = 0.2
    expected_loss = math.log(1.0 + math.exp(-expected_margin))

    loss, metrics = dpo_loss(
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta=beta
    )

    assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-6)
    assert math.isclose(metrics["reward_margin"], expected_margin, rel_tol=1e-5)
    assert metrics["preference_accuracy"] == 1.0


def test_perfect_preference():
    """
    Test 2: When chosen reward >> rejected reward, loss should be close to zero.
    """
    chosen_logp = torch.tensor([[0.0]])
    rejected_logp = torch.tensor([[-100.0]])
    chosen_ref_logp = torch.tensor([[0.0]])
    rejected_ref_logp = torch.tensor([[0.0]])
    beta = 1.0

    loss, metrics = dpo_loss(
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta=beta
    )

    assert loss.item() < 1e-5
    assert metrics["preference_accuracy"] == 1.0


def test_wrong_preference():
    """
    Test 3: When chosen reward << rejected reward, loss should be large.
    """
    chosen_logp = torch.tensor([[-100.0]])
    rejected_logp = torch.tensor([[0.0]])
    chosen_ref_logp = torch.tensor([[0.0]])
    rejected_ref_logp = torch.tensor([[0.0]])
    beta = 1.0

    loss, metrics = dpo_loss(
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta=beta
    )

    assert loss.item() > 90.0
    assert metrics["preference_accuracy"] == 0.0


def test_equal_preference():
    """
    Test 4: When chosen reward == rejected reward, loss should equal log(2).
    """
    chosen_logp = torch.tensor([[-2.5]])
    rejected_logp = torch.tensor([[-2.5]])
    chosen_ref_logp = torch.tensor([[-2.5]])
    rejected_ref_logp = torch.tensor([[-2.5]])
    beta = 0.1

    loss, metrics = dpo_loss(
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta=beta
    )

    expected_loss = math.log(2.0)
    assert math.isclose(loss.item(), expected_loss, abs_tol=1e-6)
    assert metrics["reward_margin"] == 0.0
