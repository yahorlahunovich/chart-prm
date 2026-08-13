"""
Utility functions for Qwen2.5-VL KTO dataset loading and batch processing.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from chart_prm.generator import build_generation_prompt
from chart_prm.label_mask import join_prefix_and_completion, mask_response_labels


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

    full_response = join_prefix_and_completion(prefix, response)
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

        response = (
            item.get("response")
            or item.get("solution")
            or item.get("completion")
            or item.get("chosen")
        )
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
    eos_id = processor.tokenizer.eos_token_id
    prompt_lengths = (prompt_inputs["attention_mask"] == 1).sum(dim=1)

    batch_inputs["labels"] = mask_response_labels(
        input_ids=batch_inputs["input_ids"],
        attention_mask=batch_inputs["attention_mask"],
        prompt_lengths=prompt_lengths,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )
    batch_inputs["kto_labels"] = torch.tensor(kto_labels, dtype=torch.long)

    if device is not None:
        batch_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}

    return batch_inputs


def _coerce_kto_label(row: Dict[str, Any]) -> int:
    if "kto_label" in row:
        value = row["kto_label"]
    elif "label" in row:
        value = row["label"]
    elif "is_correct" in row:
        value = row["is_correct"]
    else:
        value = 1

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "desirable", "positive"}:
            return 1
        if lowered in {"0", "-1", "false", "no", "undesirable", "negative"}:
            return -1
        raise ValueError(f"Unrecognized KTO label string: {value!r}")
    if isinstance(value, bool):
        return 1 if value else -1
    ivalue = int(value)
    if ivalue >= 0 and ivalue != -1:
        # Treat 0 as undesirable, >0 as desirable
        return 1 if ivalue > 0 else -1
    return -1


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

    Also accepts format_kto.py rows with `completion` + boolean `label`.
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

            # Preference pairs only when completion-style KTO fields are absent
            is_pair = (
                unpack_pairs
                and "chosen" in row
                and "rejected" in row
                and "completion" not in row
                and "label" not in row
            )
            if is_pair:
                items.append({
                    "question_id": row.get("question_id", line_idx),
                    "image": str(img_path),
                    "image_path": str(img_path),
                    "question": row["question"],
                    "prefix": row.get("prefix", ""),
                    "response": row["chosen"],
                    "kto_label": 1,
                })
                items.append({
                    "question_id": row.get("question_id", line_idx),
                    "image": str(img_path),
                    "image_path": str(img_path),
                    "question": row["question"],
                    "prefix": row.get("prefix", ""),
                    "response": row["rejected"],
                    "kto_label": -1,
                })
                continue

            response = (
                row.get("response")
                or row.get("completion")
                or row.get("solution")
                or row.get("chosen", "")
            )
            if not response:
                continue
            items.append({
                "question_id": row.get("question_id", line_idx),
                "image": str(img_path),
                "image_path": str(img_path),
                "question": row["question"],
                "prefix": row.get("prefix", ""),
                "response": response,
                "kto_label": _coerce_kto_label(row),
            })

    return items


def is_hard_negative_kto_sample(item: Dict[str, Any]) -> bool:
    """
    Drop short first-step-only rollouts that dominate KTO negatives but lack useful signal.
    """
    response = (item.get("response") or "").strip()
    if not response:
        return True
    if "Final Answer:" not in response:
        return True
    if response.lstrip().startswith("Step 1:") and "Step 2:" not in response and len(response) < 220:
        return True
    return False


def balance_kto_dataset(
    items: List[Dict[str, Any]],
    *,
    max_undesirable_per_desirable: float = 3.0,
    require_final_answer: bool = True,
    filter_hard_negatives: bool = True,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Rebalance KTO samples toward ~1:3 desirable:undesirable for stable training.

    Keeps all desirable samples, filters low-signal negatives, and subsamples the rest.
    """
    rng = random.Random(seed)
    desirable = [item for item in items if int(item.get("kto_label", 0)) == 1]
    undesirable = [item for item in items if int(item.get("kto_label", 0)) == -1]

    if require_final_answer:
        desirable = [item for item in desirable if "Final Answer:" in item.get("response", "")]
        undesirable = [item for item in undesirable if "Final Answer:" in item.get("response", "")]
    if filter_hard_negatives:
        undesirable = [item for item in undesirable if not is_hard_negative_kto_sample(item)]

    n_desirable = len(desirable)
    if n_desirable == 0:
        raise ValueError("balance_kto_dataset: no desirable samples left after filtering")

    max_undesirable = max(1, int(round(n_desirable * max_undesirable_per_desirable)))
    if len(undesirable) > max_undesirable:
        undesirable = rng.sample(undesirable, max_undesirable)

    balanced = desirable + undesirable
    rng.shuffle(balanced)
    stats = {
        "n_input": len(items),
        "n_desirable": n_desirable,
        "n_undesirable": len(undesirable),
        "ratio_undesirable_to_desirable": len(undesirable) / n_desirable,
        "recommended_desirable_weight": len(undesirable) / n_desirable,
    }
    return balanced, stats
