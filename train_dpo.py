#!/usr/bin/env python3
"""
Minimal Custom DPO Training Script for Qwen2.5-VL.

Usage:
  python train_dpo.py --smoke-only
  python train_dpo.py --dataset-path experiments/001_500_reasoning/data/step_dpo_pairs.jsonl --epochs 3
"""

import argparse
import json
from pathlib import Path
import sys
import torch

from chart_prm.dpo.trainer import fit_dpo, train_dpo_step
from chart_prm.dpo.utils import load_step_dpo_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Custom DPO Trainer for Qwen2.5-VL")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="experiments/001_500_reasoning/data/step_dpo_pairs.jsonl",
        help="Path to Step-DPO jsonl preference file",
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
        default="qwen_vl_dpo_adapter",
        help="Directory to save final trained adapter/model",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Load model in 4-bit NF4 quantization")
    parser.add_argument("--smoke-only", action="store_true", help="Run a 2-step smoke test")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA peft adapter")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.dataset_path)
    if not data_path.exists():
        print(f"Dataset path {data_path} not found. Exiting.")
        sys.exit(1)

    print(f"Loading Step-DPO dataset from {data_path}...")
    dataset = load_step_dpo_dataset(data_path, images_dir=args.images_dir)
    print(f"Loaded {len(dataset)} preference pairs.")

    if args.smoke_only:
        print("Running in SMOKE-ONLY mode (limiting dataset to 2 samples)...")
        dataset = dataset[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu_name} (Compute Capability: {capability})")
        if capability[0] < 7:
            raise RuntimeError(
                f"Unsupported GPU '{gpu_name}' with compute capability {capability} (sm_60). "
                f"PyTorch 2.x requires CUDA compute capability >= 7.0 (Nvidia T4 or newer)."
            )

    print(f"Loading processor and model for {args.model_id}...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor_kwargs = {}
    if torch.cuda.is_available():
        processor_kwargs["min_pixels"] = 96 * 28 * 28
        processor_kwargs["max_pixels"] = 192 * 28 * 28

    processor = AutoProcessor.from_pretrained(args.model_id, **processor_kwargs)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_kwargs = {
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
        model_kwargs["attn_implementation"] = "sdpa"

    if args.load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        print("Enabling 4-bit NF4 QLoRA quantization...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        **model_kwargs,
    )
    model.config.use_cache = False

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

    print("Starting DPO fine-tuning...")
    history = fit_dpo(
        model=model,
        dataset=dataset,
        processor=processor,
        ref_model=None,  # Using in-line model.disable_adapter()
        lr=args.lr,
        beta=args.beta,
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

    print("DPO training complete!")


if __name__ == "__main__":
    main()
