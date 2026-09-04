"""
Unit tests for mathematical SimPO loss function.
"""

import math
import torch
from chart_prm.simpo.loss import simpo_loss


def test_simpo_loss_exact_value():
    """
    Test 1: Verify simpo_loss matches exact manual calculation. No reference
    logps anywhere -- inputs are already length-normalized policy logps.
    """
    chosen_logp = torch.tensor([[-0.5]])
    rejected_logp = torch.tensor([[-1.0]])
    beta = 2.0
    gamma_beta_ratio = 0.5

    # Manual calculation:
    # pi_logratios = -0.5 - (-1.0) = 0.5
    # logits = 0.5 - 0.5 = 0.0
    # loss = -log_sigmoid(2.0 * 0.0) = -log_sigmoid(0.0) = log(2)
    expected_loss = math.log(2.0)

    loss, metrics = simpo_loss(chosen_logp, rejected_logp, beta=beta, gamma_beta_ratio=gamma_beta_ratio)

    assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-6)
    # chosen_reward = 2.0 * -0.5 = -1.0, rejected_reward = 2.0 * -1.0 = -2.0, margin = 1.0
    assert math.isclose(metrics["reward_margin"], 1.0, rel_tol=1e-5)
    assert metrics["preference_accuracy"] == 1.0


def test_perfect_preference():
    """
    Test 2: When chosen logp >> rejected logp, loss should be close to zero.
    """
    chosen_logp = torch.tensor([[0.0]])
    rejected_logp = torch.tensor([[-100.0]])

    loss, metrics = simpo_loss(chosen_logp, rejected_logp, beta=2.0, gamma_beta_ratio=0.5)

    assert loss.item() < 1e-5
    assert metrics["preference_accuracy"] == 1.0


def test_wrong_preference():
    """
    Test 3: When chosen logp << rejected logp, loss should be large.
    """
    chosen_logp = torch.tensor([[-100.0]])
    rejected_logp = torch.tensor([[0.0]])

    loss, metrics = simpo_loss(chosen_logp, rejected_logp, beta=2.0, gamma_beta_ratio=0.5)

    assert loss.item() > 90.0
    assert metrics["preference_accuracy"] == 0.0


def test_no_reference_model_inputs_needed():
    """
    Test 4: simpo_loss only ever takes two tensors -- unlike dpo_loss, there is
    no reference-model logp argument to pass at all, by construction.
    """
    import inspect

    params = list(inspect.signature(simpo_loss).parameters)
    assert "chosen_logp" in params
    assert "rejected_logp" in params
    assert not any("ref" in p for p in params)


def test_gamma_beta_ratio_raises_the_bar():
    """
    Test 5: A larger gamma_beta_ratio (bigger required margin) should not
    decrease the loss for a fixed logratio -- the sigmoid argument shrinks.
    """
    chosen_logp = torch.tensor([[-0.5]])
    rejected_logp = torch.tensor([[-1.0]])  # pi_logratios = 0.5

    loss_small_gamma, _ = simpo_loss(chosen_logp, rejected_logp, beta=2.0, gamma_beta_ratio=0.1)
    loss_large_gamma, _ = simpo_loss(chosen_logp, rejected_logp, beta=2.0, gamma_beta_ratio=0.9)

    assert loss_large_gamma.item() > loss_small_gamma.item()


def test_equal_logp_with_zero_margin_gives_log_two():
    """
    Test 6: When chosen == rejected and gamma_beta_ratio == 0, loss == log(2),
    mirroring dpo_loss's equal-preference case.
    """
    chosen_logp = torch.tensor([[-2.5]])
    rejected_logp = torch.tensor([[-2.5]])

    loss, metrics = simpo_loss(chosen_logp, rejected_logp, beta=2.0, gamma_beta_ratio=0.0)

    assert math.isclose(loss.item(), math.log(2.0), abs_tol=1e-6)
    assert metrics["reward_margin"] == 0.0
