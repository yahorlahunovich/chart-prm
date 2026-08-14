#!/usr/bin/env python3
"""
Minimal Custom Kahneman-Tversky Optimization (KTO) Training Script for Qwen2.5-VL.

Usage:
  python scripts/train/train_kto.py --smoke-only
  python scripts/train/train_kto.py --dataset-path experiments/001_500_reasoning/data/kto_samples.jsonl --epochs 2
"""

import argparse
import json
from pathlib import Path
import sys
import torch

from chart_prm.data_guards import collapse_guard, validate_training_dataset
from chart_prm.kto.trainer import fit_kto
from chart_prm.kto.utils import balance_kto_dataset, load_kto_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Custom KTO Trainer for Qwen2.5-VL")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="experiments/001_500_reasoning/data/kto_samples.jsonl",
        help="Path to sequence-level KTO jsonl (completion + label)",
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
        default="adapters/qwen_vl_kto_adapter",
        help="Directory to save final trained KTO adapter/model",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="KTO beta parameter")
    parser.add_argument("--desirable-weight", type=float, default=1.0, help="Weight for desirable completions")
    parser.add_argument("--undesirable-weight", type=float, default=1.0, help="Weight for undesirable completions")
    parser.add_argument(
        "--max-logp-drop",
        type=float,
        default=40.0,
        help="Abort if policy_logp falls this many nats below ref_logp",
    )
    parser.add_argument(
        "--collapse-guard-warn-only",
        action="store_true",
        help="Log collapse-guard warnings without aborting training",
    )
    parser.add_argument(
        "--balance-kto",
        action="store_true",
        help="Filter hard negatives and subsample to ~1:3 desirable:undesirable ratio",
    )
    parser.add_argument(
        "--max-undesirable-per-desirable",
        type=float,
        default=3.0,
        help="Target undesirable:desirable ratio when --balance-kto is set",
    )
    parser.add_argument(
        "--auto-desirable-weight",
        action="store_true",
        help="Set desirable_weight to match the post-balance class ratio",
    )
    parser.add_argument("--load-in-4bit", action="store_true", default=False, help="Explicitly force 4-bit NF4 QLoRA quantization")
    parser.add_argument("--smoke-only", action="store_true", help="Run a 2-step smoke test")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA peft adapter")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.dataset_path)
    if not data_path.exists():
        print(f"Dataset path {data_path} not found. Exiting.")
        sys.exit(1)

    print(f"Loading KTO dataset from {data_path}...")
    dataset = load_kto_dataset(data_path, images_dir=args.images_dir, unpack_pairs=True)
    print(f"Loaded {len(dataset)} unpacked KTO training samples.")
    if args.balance_kto:
        dataset, balance_stats = balance_kto_dataset(
            dataset,
            max_undesirable_per_desirable=args.max_undesirable_per_desirable,
        )
        print(
            "Balanced KTO dataset: "
            f"{balance_stats['n_desirable']} desirable / {balance_stats['n_undesirable']} undesirable "
            f"(ratio={balance_stats['ratio_undesirable_to_desirable']:.2f})"
        )
        if args.auto_desirable_weight or args.desirable_weight == 1.0:
            args.desirable_weight = balance_stats["recommended_desirable_weight"]
            print(f"Auto-set desirable_weight={args.desirable_weight:.3f}")
    stats = validate_training_dataset(dataset, name="KTO")
    print(
        f"KTO data guard OK: mean_chars={stats['mean_chars']:.1f} "
        f"final_answer_rate={stats['final_answer_rate']:.1%}"
    )

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
            f"[Step {step:03d}] KTO Loss: {metrics['loss']:.4f} | "
            f"Margin: {metrics['reward_margin']:.4f} | "
            f"Desirable R: {metrics['mean_desirable_reward']:.4f} | "
            f"Undesirable R: {metrics['mean_undesirable_reward']:.4f}"
        )
        warning = collapse_guard(
            metrics,
            max_logp_drop_vs_ref=args.max_logp_drop,
            logp_key="desirable_policy_logp",
            ref_key="desirable_ref_logp",
        )
        if warning:
            if args.collapse_guard_warn_only:
                print(f"[WARN step {step}] KTO collapse guard: {warning}")
            else:
                raise RuntimeError(f"KTO collapse guard tripped at step {step}: {warning}")

    print("Starting KTO fine-tuning...")
    history = fit_kto(
        model=model,
        dataset=dataset,
        processor=processor,
        ref_model=None,  # Using in-line model.disable_adapter()
        lr=args.lr,
        beta=args.beta,
        desirable_weight=args.desirable_weight,
        undesirable_weight=args.undesirable_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        on_step_end=print_step_log,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving trained KTO model and processor to {out_dir}...")
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))

    history_path = out_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("KTO training complete!")


if __name__ == "__main__":
    main()
