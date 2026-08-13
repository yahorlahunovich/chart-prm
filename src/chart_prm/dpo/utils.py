"""
Utility functions for sequence log-probabilities and Qwen2.5-VL dataset preprocessing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image

from chart_prm.generator import build_generation_prompt
from chart_prm.label_mask import join_prefix_and_completion, mask_response_labels


def sequence_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_pad_token_id: int = -100,
) -> torch.Tensor:
    """
    Computes sequence log-probability for response tokens.

    log pi(y|x) = sum_{t: y_t != -100} log p(y_t | x, y_{<t})

    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
        label_pad_token_id: Label value to ignore (-100)

    Returns:
        seq_logprob: (batch_size,) sequence log-probabilities
    """
    assert logits.ndim == 3, f"Expected 3D logits (batch, seq_len, vocab), got {logits.shape}"
    assert labels.ndim == 2, f"Expected 2D labels (batch, seq_len), got {labels.shape}"
    assert logits.shape[0] == labels.shape[0], f"Batch size mismatch: {logits.shape[0]} vs {labels.shape[0]}"
    assert logits.shape[1] == labels.shape[1], f"Sequence length mismatch: {logits.shape[1]} vs {labels.shape[1]}"

    # Shift logits and labels for autoregressive alignment:
    # logits[:, :-1] predicts labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Mask for target response tokens
    mask = shift_labels != label_pad_token_id
    assert (mask.sum(dim=-1) > 0).all(), "Every example in batch must contain at least 1 valid response token"

    # Compute token log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Clamp labels so masked -100 values don't index out-of-bounds in gather
    clamped_labels = shift_labels.masked_fill(~mask, 0)

    # Gather log-probabilities corresponding to target tokens
    per_token_logps = torch.gather(log_probs, dim=-1, index=clamped_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out prompt and padding positions
    masked_logps = per_token_logps * mask.float()
    seq_logprob = masked_logps.sum(dim=-1)

    return seq_logprob


def shared_prefix_assistant_text(prefix: str) -> str:
    """Assistant text through the shared prefix (exclusive of the diverging step)."""
    prefix = (prefix or "").rstrip()
    if not prefix:
        return ""
    return f"{prefix}\n"


def format_qwen_vlm_messages(
    question: str,
    chosen: str,
    rejected: str,
    prefix: str = "",
    use_generation_prompt: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Formats prompt, chosen, and rejected messages for Qwen2.5-VL.
    """
    prompt_text = build_generation_prompt(question)
    user_content = [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]

    prompt_messages = [{"role": "user", "content": user_content}]

    chosen_response = join_prefix_and_completion(prefix, chosen)
    rejected_response = join_prefix_and_completion(prefix, rejected)

    chosen_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": chosen_response},
    ]

    rejected_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": rejected_response},
    ]

    return prompt_messages, chosen_messages, rejected_messages


def create_masked_labels(
    input_ids: torch.Tensor,
    prompt_len: int,
    pad_token_id: Optional[int] = None,
    label_pad_token_id: int = -100,
    attention_mask: Optional[torch.Tensor] = None,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Creates target labels mask where prompt tokens and pad tokens are set to label_pad_token_id (-100).
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    prompt_lengths = torch.full(
        (input_ids.shape[0],),
        int(prompt_len),
        dtype=torch.long,
        device=input_ids.device,
    )
    return mask_response_labels(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        label_pad_token_id=label_pad_token_id,
    )


def build_qwen_dpo_batch(
    processor: Any,
    items: List[Dict[str, Any]],
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Processes a list of raw preference items into chosen and rejected multimodal batch dicts for Qwen2.5-VL.

    Each item in items must contain:
      - "image" or "image_path": PIL.Image.Image or path to image
      - "question": str
      - "chosen": str
      - "rejected": str
      - "prefix": str (optional)

    Returns:
        chosen_batch, rejected_batch (each formatted with input_ids, labels, attention_mask, pixel_values, etc.)
    """
    images = []
    chosen_texts = []
    rejected_texts = []
    prompt_texts = []

    for item in items:
        img = item.get("image") or item.get("image_path")
        if img is None:
            raise KeyError(f"Item is missing required 'image' or 'image_path' key: {item}")

        if isinstance(img, (str, Path)):
            with Image.open(img) as opened_img:
                img = opened_img.convert("RGB").copy()
        elif hasattr(img, "convert"):
            img = img.convert("RGB")
        images.append(img)

        prompt_msgs, chosen_msgs, rejected_msgs = format_qwen_vlm_messages(
            question=item["question"],
            chosen=item["chosen"],
            rejected=item["rejected"],
            prefix=item.get("prefix", ""),
        )

        p_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        c_text = processor.apply_chat_template(chosen_msgs, tokenize=False, add_generation_prompt=False)
        r_text = processor.apply_chat_template(rejected_msgs, tokenize=False, add_generation_prompt=False)

        prompt_texts.append(p_text)
        chosen_texts.append(c_text)
        rejected_texts.append(r_text)

    # Process prompt batches to obtain prompt lengths
    prompt_inputs = processor(
        images=images,
        text=prompt_texts,
        padding=True,
        return_tensors="pt",
    )

    # Process chosen batch
    chosen_inputs = processor(
        images=images,
        text=chosen_texts,
        padding=True,
        return_tensors="pt",
    )

    # Process rejected batch
    rejected_inputs = processor(
        images=images,
        text=rejected_texts,
        padding=True,
        return_tensors="pt",
    )

    # Construct labels for chosen / rejected (right-padded prompt+response)
    pad_id = processor.tokenizer.pad_token_id
    eos_id = processor.tokenizer.eos_token_id
    prompt_lengths = (prompt_inputs["attention_mask"] == 1).sum(dim=1)

    prefix_lengths = torch.zeros(len(items), dtype=torch.long)
    prefix_indices = [
        idx
        for idx, item in enumerate(items)
        if shared_prefix_assistant_text(item.get("prefix", ""))
    ]
    if prefix_indices:
        shared_images = [images[idx] for idx in prefix_indices]
        shared_texts = []
        for idx in prefix_indices:
            item = items[idx]
            prompt_text = build_generation_prompt(item["question"])
            user_content = [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]
            shared_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": shared_prefix_assistant_text(item.get("prefix", ""))},
            ]
            shared_texts.append(
                processor.apply_chat_template(
                    shared_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        shared_inputs = processor(
            images=shared_images,
            text=shared_texts,
            padding=True,
            return_tensors="pt",
        )
        shared_seq_lens = (shared_inputs["attention_mask"] == 1).sum(dim=1)
        for batch_idx, item_idx in enumerate(prefix_indices):
            prefix_lengths[item_idx] = max(
                0,
                int(shared_seq_lens[batch_idx].item()) - int(prompt_lengths[item_idx].item()),
            )

    chosen_inputs["labels"] = mask_response_labels(
        input_ids=chosen_inputs["input_ids"],
        attention_mask=chosen_inputs["attention_mask"],
        prompt_lengths=prompt_lengths,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        prefix_lengths=prefix_lengths,
    )
    rejected_inputs["labels"] = mask_response_labels(
        input_ids=rejected_inputs["input_ids"],
        attention_mask=rejected_inputs["attention_mask"],
        prompt_lengths=prompt_lengths,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        prefix_lengths=prefix_lengths,
    )

    if device is not None:
        chosen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in chosen_inputs.items()}
        rejected_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in rejected_inputs.items()}

    return chosen_inputs, rejected_inputs


def load_step_dpo_dataset(
    jsonl_path: Union[str, Path],
    images_dir: Union[str, Path] = "data/CharXiv/images",
) -> List[Dict[str, Any]]:
    """
    Loads Step-DPO dataset from jsonl file.
    """
    jsonl_path = Path(jsonl_path)
    images_dir = Path(images_dir)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {jsonl_path}")

    items = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            img_path = Path(row.get("image_path", ""))
            if not img_path.is_absolute() and not img_path.exists():
                img_path = images_dir / f"{row['question_id']}.jpg"

            items.append({
                "question_id": row["question_id"],
                "image": str(img_path),
                "image_path": str(img_path),
                "question": row["question"],
                "prefix": row.get("prefix", ""),
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            })
    return items
