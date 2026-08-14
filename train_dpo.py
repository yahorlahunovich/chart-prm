#!/usr/bin/env python3
"""
Minimal Custom DPO Training Script for Qwen2.5-VL.

Usage:
  python train_dpo.py --smoke-only
  python train_dpo.py --dataset-path experiments/001_500_reasoning/data/dpo_pairs.jsonl --epochs 2
"""

import argparse
import json
from pathlib import Path
import sys
import torch

from chart_prm.data_guards import collapse_guard, validate_training_dataset
from chart_prm.dpo.trainer import fit_dpo
from chart_prm.dpo.utils import load_step_dpo_dataset
from chart_prm.sft_dpo_init import (
    POLICY_ADAPTER,
    REFERENCE_ADAPTER,
    init_peft_from_sft,
    resolve_sft_init_adapter,
    save_policy_adapter,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Custom DPO Trainer for Qwen2.5-VL")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="experiments/001_500_reasoning/data/dpo_pairs.jsonl",
        help="Path to full-trajectory (or Step-DPO) preference jsonl",
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
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument(
        "--max-logp-drop",
        type=float,
        default=40.0,
        help="Abort if chosen_logp falls this many nats below chosen_ref_logp",
    )
    parser.add_argument(
        "--collapse-guard-warn-only",
        action="store_true",
        help="Log collapse-guard warnings without aborting training",
    )
    parser.add_argument("--load-in-4bit", action="store_true", default=False, help="Explicitly force 4-bit NF4 QLoRA quantization")
    parser.add_argument("--smoke-only", action="store_true", help="Run a 2-step smoke test")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA peft adapter")
    parser.add_argument(
        "--skip-data-guard",
        action="store_true",
        help="Allow fragment/step-only targets (not recommended for full-generation eval)",
    )
    parser.add_argument(
        "--step-dpo",
        action="store_true",
        help="Step-DPO mode: train on step_dpo_pairs.jsonl with prefix masking",
    )
    parser.add_argument(
        "--sft-dpo",
        action="store_true",
        help="Full-trajectory DPO initialized from the SFT adapter (SFT is the DPO reference)",
    )
    parser.add_argument(
        "--init-adapter",
        type=str,
        default="",
        help="Path to the SFT LoRA directory used when --sft-dpo is set",
    )
    return parser.parse_args()


def _apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    default_dataset = "experiments/001_500_reasoning/data/dpo_pairs.jsonl"
    default_output = "qwen_vl_dpo_adapter"
    if args.step_dpo and args.sft_dpo:
        raise SystemExit("Use either --step-dpo or --sft-dpo, not both.")
    if args.step_dpo:
        if args.dataset_path == default_dataset:
            args.dataset_path = "experiments/001_500_reasoning/data/step_dpo_pairs.jsonl"
        if args.output_dir == default_output:
            args.output_dir = "qwen_vl_step_dpo_adapter"
        # Suffix-from-divergence pairs include Final Answer; keep the fragment guard on.
    if args.sft_dpo:
        if args.output_dir == default_output:
            args.output_dir = "qwen_vl_sft_dpo_adapter"
        if args.lr == 1e-5:
            args.lr = 2e-6
        if args.epochs == 2:
            args.epochs = 1
    return args


def main():
    args = _apply_mode_defaults(parse_args())

    data_path = Path(args.dataset_path)
    if not data_path.exists():
        print(f"Dataset path {data_path} not found. Exiting.")
        sys.exit(1)

    mode_label = "Step-DPO" if args.step_dpo else ("SFT→DPO" if args.sft_dpo else "DPO")
    print(f"Loading {mode_label} dataset from {data_path}...")
    dataset = load_step_dpo_dataset(data_path, images_dir=args.images_dir)
    print(f"Loaded {len(dataset)} preference pairs.")
    if not args.skip_data_guard:
        stats = validate_training_dataset(dataset, name="DPO")
        print(
            f"DPO data guard OK: mean_chars={stats['mean_chars']:.1f} "
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

    reference_adapter = None
    if args.sft_dpo or args.init_adapter:
        if args.no_lora:
            raise SystemExit("SFT→DPO requires LoRA (--no-lora is incompatible).")
        sft_dir = resolve_sft_init_adapter(args.init_adapter or None)
        print(f"SFT init adapter: {sft_dir}")
        model = init_peft_from_sft(model, sft_dir)
        reference_adapter = REFERENCE_ADAPTER
        print(
            f"DPO reference is frozen SFT adapter {REFERENCE_ADAPTER!r}; "
            f"training adapter {POLICY_ADAPTER!r}."
        )
    elif not args.no_lora:
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
        warning = collapse_guard(metrics, max_logp_drop_vs_ref=args.max_logp_drop)
        if warning:
            if args.collapse_guard_warn_only:
                print(f"[WARN step {step}] DPO collapse guard: {warning}")
            else:
                raise RuntimeError(f"DPO collapse guard tripped at step {step}: {warning}")

    print(
        f"Starting DPO fine-tuning... mode={mode_label} epochs={args.epochs} "
        f"lr={args.lr} beta={args.beta} output={args.output_dir}"
    )
    history = fit_dpo(
        model=model,
        dataset=dataset,
        processor=processor,
        ref_model=None,
        lr=args.lr,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        on_step_end=print_step_log,
        reference_adapter=reference_adapter,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving trained model and processor to {out_dir}...")
    if reference_adapter is not None:
        save_policy_adapter(model, out_dir, adapter_name=POLICY_ADAPTER)
    else:
        model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))

    history_path = out_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("DPO training complete!")


if __name__ == "__main__":
    main()
