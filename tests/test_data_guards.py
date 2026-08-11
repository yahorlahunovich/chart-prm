"""Tests for training data guards and label masking helpers."""

import torch
import pytest

from chart_prm.data_guards import collapse_guard, validate_training_dataset
from chart_prm.label_mask import join_prefix_and_completion, mask_response_labels
from chart_prm.kto.utils import load_kto_dataset


def test_validate_rejects_fragment_targets():
    dataset = [
        {"question_id": "1", "chosen": "The x-axis represents Epochs."},
        {"question_id": "2", "chosen": "The x-axis represents Time."},
    ]
    with pytest.raises(ValueError, match="Final Answer"):
        validate_training_dataset(dataset, name="toy")


def test_validate_accepts_full_trajectories():
    dataset = [
        {
            "question_id": "1",
            "solution": "Step 1: Read the legend.\nStep 2: Compare bars.\nFinal Answer: A",
        },
        {
            "question_id": "2",
            "solution": "Step 1: Find epoch 10.\nFinal Answer: Model B",
        },
    ]
    stats = validate_training_dataset(dataset, name="toy", min_mean_chars=20.0)
    assert stats["final_answer_rate"] == 1.0


def test_collapse_guard_triggers_on_large_drop():
    warning = collapse_guard(
        {"chosen_logp": -120.0, "chosen_ref_logp": -40.0},
        max_logp_drop_vs_ref=40.0,
    )
    assert warning is not None
    assert "collapse" in warning


def test_join_prefix_keeps_step_boundaries():
    text = join_prefix_and_completion("Step 1: Look at x.\n", "Step 2: Look at y.")
    assert text == "Step 1: Look at x.\nStep 2: Look at y."


def test_mask_response_labels_right_padded():
    input_ids = torch.tensor([[10, 11, 12, 13, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    prompt_lengths = torch.tensor([2])
    labels = mask_response_labels(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        pad_token_id=0,
        eos_token_id=2,
    )
    assert labels.tolist() == [[-100, -100, 12, 13, -100]]


def test_load_kto_reads_completion_and_bool_label(tmp_path):
    path = tmp_path / "kto.jsonl"
    rows = [
        {
            "question_id": "1",
            "question": "Q?",
            "image_path": "missing.jpg",
            "prefix": "",
            "completion": "Step 1: A\nFinal Answer: 1",
            "label": True,
        },
        {
            "question_id": "2",
            "question": "Q?",
            "image_path": "missing.jpg",
            "prefix": "",
            "completion": "Step 1: B\nFinal Answer: 2",
            "label": False,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(__import__("json").dumps(row) + "\n")

    items = load_kto_dataset(path, images_dir=tmp_path, unpack_pairs=True)
    assert len(items) == 2
    assert items[0]["kto_label"] == 1
    assert items[1]["kto_label"] == -1
    assert items[0]["response"].startswith("Step 1:")
