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


def test_mask_response_labels_with_prefix():
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
    prompt_lengths = torch.tensor([2])
    prefix_lengths = torch.tensor([2])
    labels = mask_response_labels(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        prefix_lengths=prefix_lengths,
        pad_token_id=0,
        eos_token_id=99,
    )
    assert labels.tolist() == [[-100, -100, -100, -100, 14, 15]]


def test_balance_kto_dataset_subsamples_negatives(tmp_path):
    from chart_prm.kto.utils import balance_kto_dataset

    items = []
    for idx in range(4):
        items.append({
            "question_id": str(idx),
            "response": f"Step 1: A\nStep 2: B\nFinal Answer: {idx}",
            "kto_label": 1,
        })
    for idx in range(20):
        short = idx % 2 == 0
        response = (
            "Step 1: The x-axis represents time."
            if short
            else f"Step 1: A\nStep 2: B\nFinal Answer: wrong-{idx}"
        )
        items.append({
            "question_id": f"neg-{idx}",
            "response": response,
            "kto_label": -1,
        })

    balanced, stats = balance_kto_dataset(items, max_undesirable_per_desirable=3.0)
    assert stats["n_desirable"] == 4
    assert stats["n_undesirable"] == 10
    assert len(balanced) == 14
    assert stats["recommended_desirable_weight"] == pytest.approx(2.5)


def test_shared_prefix_assistant_text():
    from chart_prm.dpo.utils import shared_prefix_assistant_text

    assert shared_prefix_assistant_text("") == ""
    assert shared_prefix_assistant_text("Step 1: A") == "Step 1: A\n"


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
