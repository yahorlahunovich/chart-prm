"""
Unit tests for sequence log-probability calculation and prompt masking.
"""

import torch
import torch.nn.functional as F
import pytest
from chart_prm.dpo.utils import sequence_logprob


def test_sequence_logprob_deterministic_values():
    """
    Test 5: Verify sequence_logprob against manual log_softmax calculations.
    """
    # Batch size = 1, seq_len = 4, vocab_size = 3
    # Tokens: [Prompt1, Prompt2, Resp1, Resp2]
    logits = torch.tensor([
        [
            [2.0, 1.0, 0.0],  # Output after Prompt1 (predicts Prompt2)
            [0.0, 3.0, 1.0],  # Output after Prompt2 (predicts Resp1 = token 2)
            [1.0, 0.0, 4.0],  # Output after Resp1 (predicts Resp2 = token 0)
            [0.0, 0.0, 0.0],  # Output after Resp2 (end)
        ]
    ])

    labels = torch.tensor([
        [-100, -100, 2, 0]  # Prompt tokens masked out, Resp1=2, Resp2=0
    ])

    # Shifted alignment:
    # shift_logits[0, 1] (after Prompt2) predicts labels[0, 2] = 2.
    #   logits at pos 1: [0.0, 3.0, 1.0]. Log-softmax for token 2:
    #   log_softmax([0, 3, 1])[2] = 1 - log(exp(0) + exp(3) + exp(1))
    logps_pos1 = F.log_softmax(logits[0, 1], dim=-1)[2].item()

    # shift_logits[0, 2] (after Resp1) predicts labels[0, 3] = 0.
    #   logits at pos 2: [1.0, 0.0, 4.0]. Log-softmax for token 0:
    #   log_softmax([1, 0, 4])[0] = 1 - log(exp(1) + exp(0) + exp(4))
    logps_pos2 = F.log_softmax(logits[0, 2], dim=-1)[0].item()

    expected_sum = logps_pos1 + logps_pos2

    actual_logp = sequence_logprob(logits, labels)

    assert torch.allclose(actual_logp, torch.tensor([expected_sum]), atol=1e-5)


def test_prompt_masking_independence():
    """
    Test 6: Verify that logits/labels for predicting prompt tokens (-100) do NOT alter calculated response log-probability.
    """
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 6
    vocab_size = 10

    logits1 = torch.randn(batch_size, seq_len, vocab_size)
    labels1 = torch.tensor([
        [-100, -100, -100, 5, 2, 8],
        [-100, -100, -100, 4, 9, 1]
    ])

    # Calculate logprob for original logits and labels
    logp1 = sequence_logprob(logits1, labels1)

    # Mutate prompt-predicting logits (indices 0 and 1, which predict prompt tokens at indices 1 and 2)
    logits2 = logits1.clone()
    logits2[:, :2, :] = torch.randn(batch_size, 2, vocab_size) * 100.0

    labels2 = labels1.clone()

    logp2 = sequence_logprob(logits2, labels2)

    # The computed response log-probability must be IDENTICAL because prompt token predictions are masked out with -100
    assert torch.allclose(logp1, logp2, atol=1e-6)
