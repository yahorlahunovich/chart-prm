"""
Utility functions for Qwen2.5-VL KTO dataset loading and batch processing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from chart_prm.generator import build_generation_prompt


def format_qwen_vlm_kto_messages(
    question: str,
    response: str,
    prefix: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Formats prompt and response messages for Qwen2.5-VL KTO.

    Args:
        question: Chart question
        response: Single completion text (desirable or undesirable)
        prefix: Optional prior reasoning steps

    Returns:
        prompt_messages: User prompt messages (for computing prompt token length)
        response_messages: User prompt + assistant response messages
    """
    prompt_text = build_generation_prompt(question)
    user_content = [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]

    prompt_messages = [{"role": "user", "content": user_content}]

    full_response = (prefix + " " + response).strip() if prefix else response.strip()
    response_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": full_response},
    ]

    return prompt_messages, response_messages


def build_qwen_kto_batch(
    processor: Any,
    items: List[Dict[str, Any]],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Processes a list of KTO items into a multimodal batch dict for Qwen2.5-VL.

    Each item in items must contain:
      - "image" or "image_path": PIL.Image.Image or path to image
      - "question": str
      - "response": str
      - "kto_label": int (+1 for desirable, -1 for undesirable)

    Returns:
        batch dict formatted with input_ids, labels (prompt masked with -100), kto_labels, attention_mask, pixel_values, etc.
    """
    images = []
    prompt_texts = []
    response_texts = []
    kto_labels = []

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

        response = item.get("response") or item.get("solution") or item.get("chosen")
        if not response:
            raise KeyError(f"Item is missing required completion 'response' key: {item}")

        kto_label = item.get("kto_label", 1)
        kto_labels.append(int(kto_label))

        prompt_msgs, resp_msgs = format_qwen_vlm_kto_messages(
            question=item["question"],
            response=response,
            prefix=item.get("prefix", ""),
        )

        p_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        r_text = processor.apply_chat_template(resp_msgs, tokenize=False, add_generation_prompt=False)

        prompt_texts.append(p_text)
        response_texts.append(r_text)

    # Process prompt batch to get prompt token lengths
    prompt_inputs = processor(
        images=images,
        text=prompt_texts,
        padding=True,
        return_tensors="pt",
    )

    # Process full response batch
    batch_inputs = processor(
        images=images,
        text=response_texts,
        padding=True,
        return_tensors="pt",
    )

    pad_id = processor.tokenizer.pad_token_id
    prompt_lengths = (prompt_inputs["attention_mask"] == 1).sum(dim=1)

    labels = batch_inputs["input_ids"].clone()
    for i, p_len in enumerate(prompt_lengths):
        labels[i, :p_len] = -100
    if pad_id is not None:
        labels[labels == pad_id] = -100

    batch_inputs["labels"] = labels
    batch_inputs["kto_labels"] = torch.tensor(kto_labels, dtype=torch.long)

    if device is not None:
        batch_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}

    return batch_inputs


def load_kto_dataset(
    jsonl_path: Union[str, Path],
    images_dir: Union[str, Path] = "data/CharXiv/images",
    unpack_pairs: bool = True,
) -> List[Dict[str, Any]]:
    """
    Loads KTO dataset from jsonl file.
    If jsonl contains chosen/rejected pairs and unpack_pairs=True, it unpacks each pair into:
      - 1 desirable sample (kto_label = +1)
      - 1 undesirable sample (kto_label = -1)
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
                img_path = images_dir / f"{row.get('question_id', line_idx)}.jpg"

            # Check if row is a preference pair (chosen & rejected)
            if unpack_pairs and "chosen" in row and "rejected" in row:
                # Add desirable sample
                items.append({
                    "question_id": row.get("question_id", line_idx),
                    "image": str(img_path),
                    "image_path": str(img_path),
                    "question": row["question"],
                    "prefix": row.get("prefix", ""),
                    "response": row["chosen"],
                    "kto_label": 1,  # Desirable (+1)
                })
                # Add undesirable sample
                items.append({
                    "question_id": row.get("question_id", line_idx),
                    "image": str(img_path),
                    "image_path": str(img_path),
                    "question": row["question"],
                    "prefix": row.get("prefix", ""),
                    "response": row["rejected"],
                    "kto_label": -1,  # Undesirable (-1)
                })
            else:
                label = row.get("kto_label", 1 if row.get("is_correct", True) else -1)
                response = row.get("response") or row.get("solution") or row.get("chosen", "")
                items.append({
                    "question_id": row.get("question_id", line_idx),
                    "image": str(img_path),
                    "image_path": str(img_path),
                    "question": row["question"],
                    "prefix": row.get("prefix", ""),
                    "response": response,
                    "kto_label": label,
                })

    return items
