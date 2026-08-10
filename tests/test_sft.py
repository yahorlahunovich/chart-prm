"""
Unit tests for Supervised Fine-Tuning (SFT) loss, batch formatting, and trainer step.
"""

import copy
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from chart_prm.sft.loss import sft_loss
from chart_prm.sft.trainer import train_sft_step
from chart_prm.sft.utils import format_qwen_vlm_sft_messages, load_sft_dataset


class ToySFTModel(nn.Module):
    """Simple toy model for unit testing SFT optimization without heavy VLM loading."""

    def __init__(self, vocab_size: int = 10, hidden_dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.embedding(input_ids)
        logits = self.linear(h)
        return logits


def test_sft_loss_manual_math():
    """
    Test 1: Verify sft_loss against manual PyTorch F.cross_entropy.
    """
    batch_size = 2
    seq_len = 4
    vocab_size = 5

    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.tensor([
        [-100, -100, 2, 3],
        [-100, 1, 4, -100],
    ])

    loss, metrics = sft_loss(logits, labels)

    # Shift logits and labels
    shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = labels[:, 1:].contiguous().view(-1)

    expected_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

    assert torch.isclose(loss, expected_loss, atol=1e-6)
    assert metrics["loss"] == pytest.approx(expected_loss.item())
    assert metrics["valid_tokens"] == 4  # 4 valid target tokens in shift_labels


def test_sft_prompt_masking_independence():
    """
    Test 2: Verify changing prompt logits/labels (-100) does not alter computed SFT loss.
    """
    vocab_size = 8
    logits1 = torch.randn(1, 5, vocab_size)
    logits2 = logits1.clone()
    # Modify prompt positions (indices 0, 1)
    logits2[:, :2, :] += 10.0

    labels = torch.tensor([[-100, -100, -100, 3, 4]])

    loss1, _ = sft_loss(logits1, labels)
    loss2, _ = sft_loss(logits2, labels)

    assert torch.isclose(loss1, loss2, atol=1e-6)


def test_sft_trainer_synthetic_optimization():
    """
    Test 3: Verify SFT training step updates model weights and decreases loss over iterations.
    """
    torch.manual_seed(42)
    vocab_size = 10
    model = ToySFTModel(vocab_size=vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=0.05)

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "labels": torch.tensor([[-100, -100, 3, 4, 5]]),
    }

    initial_loss, _ = sft_loss(model(batch["input_ids"]), batch["labels"])

    for _ in range(15):
        loss, _ = train_sft_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
        )

    final_loss, _ = sft_loss(model(batch["input_ids"]), batch["labels"])

    assert final_loss.item() < initial_loss.item()
