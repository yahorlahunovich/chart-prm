#!/usr/bin/env python3
"""
Minimal Custom SimPO Training Script for Qwen2.5-VL.

Reference-free: unlike train_dpo.py (which forwards the base model twice more
per step -- once per side, with the LoRA adapter disabled -- to get a
reference log-probability), this script never computes a reference
log-probability at all. Half the forward passes per step, no reference-model
plumbing (no --sft-dpo / --init-adapter / collapse-guard, none of which apply
to a method with no reference to collapse toward).

Usage:
  python scripts/train/train_simpo.py --smoke-only
  python scripts/train/train_simpo.py --dataset-path experiments/001_500_reasoning/data/dpo_pairs.jsonl --epochs 1
"""

import argparse
import json
from pathlib import Path
import sys
import torch

from chart_prm.data_guards import validate_training_dataset
from chart_prm.simpo.trainer import fit_simpo
from chart_prm.dpo.utils import load_step_dpo_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Custom SimPO Trainer for Qwen2.5-VL")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="experiments/001_500_reasoning/data/dpo_pairs.jsonl",
        help="Path to full-trajectory preference jsonl -- same schema as train_dpo.py",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/CharXiv/images",
        help="Directory containing chart images",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="adapters/qwen_vl_simpo_adapter",
        help="Directory to save final trained adapter/model",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument(
        "--beta", type=float, default=2.0, help="SimPO beta (much larger than DPO's ~0.1 -- see princeton-nlp/SimPO's tuning guidance)"
    )
    parser.add_argument(
        "--gamma-beta-ratio",
        type=float,
        default=0.5,
        help="Target reward margin gamma, expressed as gamma/beta",
    )
    parser.add_argument("--load-in-4bit", action="store_true", default=False, help="Explicitly force 4-bit NF4 QLoRA quantization")
    parser.add_argument("--smoke-only", action="store_true", help="Run a 2-step smoke test")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA peft adapter")
    parser.add_argument(
        "--skip-data-guard",
        action="store_true",
        help="Allow fragment/step-only targets (not recommended for full-generation eval)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.dataset_path)
    if not data_path.exists():
        print(f"Dataset path {data_path} not found. Exiting.")
        sys.exit(1)

    print(f"Loading SimPO dataset from {data_path}...")
    dataset = load_step_dpo_dataset(data_path, images_dir=args.images_dir)
    print(f"Loaded {len(dataset)} preference pairs.")
    if not args.skip_data_guard:
        stats = validate_training_dataset(dataset, name="SimPO")
        print(
            f"SimPO data guard OK: mean_chars={stats['mean_chars']:.1f} "
            f"final_answer_rate={stats['final_answer_rate']:.1%}"
        )
    else:
        print("WARNING: --skip-data-guard set; fragment targets allowed.")

    if args.smoke_only:
        print("Running in SMOKE-ONLY mode (limiting dataset to 2 samples)...")
        dataset = dataset[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    use_4bit = False
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"Detected {gpu_count} GPU(s): {gpu_name} (Compute Capability: {capability})")
        if capability[0] >= 7 or args.load_in_4bit:
            use_4bit = True

    print(f"Loading processor and model for {args.model_id} (use_4bit={use_4bit})...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor_kwargs = {}
    if torch.cuda.is_available():
        processor_kwargs["min_pixels"] = 64 * 28 * 28
        processor_kwargs["max_pixels"] = 128 * 28 * 28

    processor = AutoProcessor.from_pretrained(args.model_id, **processor_kwargs)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else ({"": 0} if torch.cuda.is_available() else None)

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    if device_map is not None:
        model_kwargs["device_map"] = device_map
        model_kwargs["attn_implementation"] = "sdpa"

    if use_4bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            print("Enabling 4-bit NF4 QLoRA quantization...")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        except Exception as e:
            print(f"BitsAndBytes unavailable ({e}), falling back to native FP16.")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        **model_kwargs,
    )
    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing for VRAM optimization...")
        model.gradient_checkpointing_enable()

    if not args.no_lora:
        print("Applying PEFT LoRA adapter...")
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    def print_step_log(step, metrics):
        print(
            f"[Step {step:03d}] Loss: {metrics['loss']:.4f} | "
            f"Margin: {metrics['reward_margin']:.4f} | "
            f"Acc: {metrics['preference_accuracy']*100:.1f}% | "
            f"Chosen Reward: {metrics['chosen_reward']:.4f} | "
            f"Rejected Reward: {metrics['rejected_reward']:.4f}"
        )

    print(
        f"Starting SimPO fine-tuning... epochs={args.epochs} lr={args.lr} "
        f"beta={args.beta} gamma_beta_ratio={args.gamma_beta_ratio} output={args.output_dir}"
    )
    history = fit_simpo(
        model=model,
        dataset=dataset,
        processor=processor,
        lr=args.lr,
        beta=args.beta,
        gamma_beta_ratio=args.gamma_beta_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        on_step_end=print_step_log,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving trained model and processor to {out_dir}...")
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))

    history_path = out_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("SimPO training complete!")


if __name__ == "__main__":
    main()
