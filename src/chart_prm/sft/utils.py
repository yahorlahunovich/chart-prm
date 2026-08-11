"""
Utility functions for Qwen2.5-VL SFT dataset loading and batch processing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from chart_prm.generator import build_generation_prompt
from chart_prm.label_mask import join_prefix_and_completion, mask_response_labels


def format_qwen_vlm_sft_messages(
    question: str,
    solution: str,
    prefix: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Formats prompt and full solution messages for Qwen2.5-VL SFT.

    Args:
        question: Chart question
        solution: Target correct solution text
        prefix: Optional prior reasoning steps

    Returns:
        prompt_messages: User prompt messages (for computing prompt token length)
        solution_messages: User prompt + assistant response messages (for full target token sequence)
    """
    prompt_text = build_generation_prompt(question)
    user_content = [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]

    prompt_messages = [{"role": "user", "content": user_content}]

    full_response = join_prefix_and_completion(prefix, solution)
    solution_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": full_response},
    ]

    return prompt_messages, solution_messages


def build_qwen_sft_batch(
    processor: Any,
    items: List[Dict[str, Any]],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Processes a list of raw SFT items into a multimodal batch dict for Qwen2.5-VL.

    Each item in items must contain:
      - "image" or "image_path": PIL.Image.Image or path to image
      - "question": str
      - "solution" or "chosen": str

    Returns:
        batch dict formatted with input_ids, labels (prompt masked with -100), attention_mask, pixel_values, etc.
    """
    images = []
    prompt_texts = []
    solution_texts = []

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

        solution = item.get("solution") or item.get("chosen")
        if not solution:
            raise KeyError(f"Item is missing required 'solution' or 'chosen' key: {item}")

        prompt_msgs, solution_msgs = format_qwen_vlm_sft_messages(
            question=item["question"],
            solution=solution,
            prefix=item.get("prefix", ""),
        )

        p_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        s_text = processor.apply_chat_template(solution_msgs, tokenize=False, add_generation_prompt=False)

        prompt_texts.append(p_text)
        solution_texts.append(s_text)

    # Process prompt batch to get prompt token lengths
    prompt_inputs = processor(
        images=images,
        text=prompt_texts,
        padding=True,
        return_tensors="pt",
    )

    # Process full solution batch
    solution_inputs = processor(
        images=images,
        text=solution_texts,
        padding=True,
        return_tensors="pt",
    )

    pad_id = processor.tokenizer.pad_token_id
    eos_id = processor.tokenizer.eos_token_id
    prompt_lengths = (prompt_inputs["attention_mask"] == 1).sum(dim=1)

    solution_inputs["labels"] = mask_response_labels(
        input_ids=solution_inputs["input_ids"],
        attention_mask=solution_inputs["attention_mask"],
        prompt_lengths=prompt_lengths,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )

    if device is not None:
        solution_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in solution_inputs.items()}

    return solution_inputs


def load_sft_dataset(
    jsonl_path: Union[str, Path],
    images_dir: Union[str, Path] = "data/CharXiv/images",
) -> List[Dict[str, Any]]:
    """
    Loads SFT dataset from jsonl file. Supports preference files (using 'chosen') or target solution files.
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

            solution = (
                row.get("solution")
                or row.get("completion")
                or row.get("chosen")
                or row.get("response", "")
            )
            if not solution:
                continue

            items.append({
                "question_id": row.get("question_id", line_idx),
                "image": str(img_path),
                "image_path": str(img_path),
                "question": row["question"],
                "prefix": row.get("prefix", ""),
                "solution": solution,
            })
    return items
