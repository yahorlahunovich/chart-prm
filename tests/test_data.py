"""
Unit tests for data preprocessing, Qwen message formatting, and label masking.
"""

from pathlib import Path
import torch
from PIL import Image
import pytest

from chart_prm.dpo.utils import (
    format_qwen_vlm_messages,
    create_masked_labels,
    build_qwen_dpo_batch,
)


class MockProcessor:
    """Mock processor for testing multimodal batch construction without downloading weights."""
    def __init__(self):
        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 2
        self.tokenizer = MockTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        user_text = messages[0]["content"][1]["text"]
        if len(messages) == 1:
            return f"<user> {user_text} <assistant>"
        assistant_text = messages[1]["content"]
        return f"<user> {user_text} <assistant> {assistant_text}"

    def __call__(self, images, text, padding=True, return_tensors="pt"):
        batch_input_ids = []
        for t in text:
            tokens = [abs(hash(w)) % 1000 + 10 for w in t.split()]
            batch_input_ids.append(tokens)

        max_len = max(len(toks) for toks in batch_input_ids)
        padded_ids = []
        attn_masks = []
        for toks in batch_input_ids:
            pad_len = max_len - len(toks)
            padded = toks + [self.tokenizer.pad_token_id] * pad_len
            mask = [1] * len(toks) + [0] * pad_len
            padded_ids.append(padded)
            attn_masks.append(mask)

        return {
            "input_ids": torch.tensor(padded_ids),
            "attention_mask": torch.tensor(attn_masks),
            "pixel_values": torch.randn(len(images), 3, 32, 32),
        }


def test_format_qwen_vlm_messages():
    """
    Test 10a: Verify prompt, chosen, and rejected messages contain same prompt/image structure.
    """
    question = "What is the peak value in 2020?"
    chosen = "Step 1: Locate 2020. Final Answer: 42"
    rejected = "Step 1: Locate 2020. Final Answer: 10"
    prefix = "Let's analyze."

    prompt_msgs, chosen_msgs, rejected_msgs = format_qwen_vlm_messages(
        question=question, chosen=chosen, rejected=rejected, prefix=prefix
    )

    # Verify conditioning user content is identical
    assert prompt_msgs[0]["content"] == chosen_msgs[0]["content"]
    assert prompt_msgs[0]["content"] == rejected_msgs[0]["content"]

    # Verify chosen and rejected responses are appended
    assert prefix in chosen_msgs[1]["content"]
    assert chosen in chosen_msgs[1]["content"]
    assert rejected in rejected_msgs[1]["content"]


def test_create_masked_labels():
    """
    Test 10b: Verify prompt tokens and pad tokens are masked with -100.
    """
    input_ids = torch.tensor([[10, 20, 30, 40, 50, 0, 0]])
    prompt_len = 3
    pad_token_id = 0

    labels = create_masked_labels(input_ids, prompt_len=prompt_len, pad_token_id=pad_token_id)

    # Prompt tokens (indices 0, 1, 2) must be -100
    assert (labels[0, :3] == -100).all()
    # Response tokens (indices 3, 4) must match input_ids
    assert (labels[0, 3:5] == input_ids[0, 3:5]).all()
    # Pad tokens (indices 5, 6) must be -100
    assert (labels[0, 5:] == -100).all()


def test_build_qwen_dpo_batch_with_mock_processor():
    """
    Test 10c: Verify build_qwen_dpo_batch creates valid chosen/rejected batches.
    """
    processor = MockProcessor()
    mock_img = Image.new("RGB", (32, 32), color="blue")
    items = [{
        "image": mock_img,
        "question": "Which bar is tallest?",
        "chosen": "Bar A is tallest.",
        "rejected": "Bar B is tallest.",
        "prefix": "",
    }]

    chosen_batch, rejected_batch = build_qwen_dpo_batch(processor, items)

    assert "input_ids" in chosen_batch
    assert "labels" in chosen_batch
    assert "attention_mask" in chosen_batch
    assert "pixel_values" in chosen_batch

    # Verify prompt tokens in labels are masked (-100)
    assert (chosen_batch["labels"][0, :5] == -100).all()
    # Verify response tokens are unmasked
    assert (chosen_batch["labels"][0, 5:] != -100).any()
