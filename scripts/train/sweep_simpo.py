#!/usr/bin/env python3
"""
Small local hyperparameter sweep for SimPO, sized to this project's actual scale.

Experiment 017's single SimPO run (paper-default beta=2.0, gamma_beta_ratio=0.5,
lr=1e-6) landed at 26% -- below both DPO variants -- with a noisier training
curve than DPO's runs (preference accuracy only ~44.8% -> ~52.2% across 134
steps, no clean downward loss trend). Those defaults were tuned by SimPO's
authors on tens-of-thousands-of-pairs datasets with multi-epoch training and
learning-rate warmup; none of that was re-validated for this project's much
smaller scale (134 pairs, 1 epoch, batch size 1, no schedule). This sweeps
lr and beta the same way experiment 009 swept the reward tree's merge
threshold (xi) locally instead of trusting a paper default -- using the
cheap training-time signal (does loss trend down, does preference accuracy
trend up, no collapse) to pick a promising config before spending a full
holdout-eval GPU run on just one of them.

Reloads a fresh LoRA adapter around the SAME already-loaded base model for
each config (no repeated multi-GB model downloads), trains each config for
one full epoch on the fixed dpo_pairs.jsonl (same file Full DPO and the
original SimPO run trained on), and picks a winner by mean preference
accuracy over the last 25% of steps (tie-broken by lowest mean loss over
that same window). Saves every config's adapter to a numbered subdirectory
and copies the winner to --output-dir for the eval kernel to pick up.
"""

import argparse
import json
from pathlib import Path
import sys
import torch

from chart_prm.data_guards import validate_training_dataset
from chart_prm.simpo.trainer import fit_simpo
from chart_prm.dpo.utils import load_step_dpo_dataset

GRID = [
    {"lr": 5e-7, "beta": 2.0, "gamma_beta_ratio": 0.5},
    {"lr": 1e-6, "beta": 2.0, "gamma_beta_ratio": 0.5},
    {"lr": 1e-5, "beta": 2.0, "gamma_beta_ratio": 0.5},
    {"lr": 5e-7, "beta": 5.0, "gamma_beta_ratio": 0.5},
    {"lr": 1e-6, "beta": 5.0, "gamma_beta_ratio": 0.5},
    {"lr": 1e-5, "beta": 5.0, "gamma_beta_ratio": 0.5},
]


def summarize_history(history: list) -> dict:
    """Cheap, GPU-free summary of one config's training run -- no eval needed."""
    n = len(history)
    if n == 0:
        return {
            "n_steps": 0,
            "final_loss": None,
            "mean_loss_tail": None,
            "mean_preference_accuracy_tail": None,
            "mean_reward_margin_tail": None,
            "all_finite": False,
        }
    tail_start = max(0, n - max(1, n // 4))
    tail = history[tail_start:]
    losses = [h["loss"] for h in history]
    all_finite = all(l == l and abs(l) != float("inf") for l in losses)  # l == l is False for NaN
    return {
        "n_steps": n,
        "final_loss": history[-1]["loss"],
        "mean_loss_tail": sum(h["loss"] for h in tail) / len(tail),
        "mean_preference_accuracy_tail": sum(h["preference_accuracy"] for h in tail) / len(tail),
        "mean_reward_margin_tail": sum(h["reward_margin"] for h in tail) / len(tail),
        "all_finite": all_finite,
    }


def pick_best_config(results: list) -> dict:
    """Highest mean preference_accuracy over the last 25% of steps; ties broken by lowest mean loss (tail)."""
    if not results:
        raise ValueError("No sweep results to pick from.")
    finite_results = [r for r in results if r["summary"]["all_finite"]]
    pool = finite_results or results
    return sorted(
        pool,
        key=lambda r: (-r["summary"]["mean_preference_accuracy_tail"], r["summary"]["mean_loss_tail"]),
    )[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Small local hyperparameter sweep for SimPO")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="experiments/001_500_reasoning/data/dpo_pairs.jsonl",
        help="Same pairs file every SimPO/DPO run this project has used, for a fair comparison",
    )
    parser.add_argument("--images-dir", type=str, default="data/CharXiv/images")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", type=str, default="adapters/qwen_vl_simpo_tuned_adapter")
    parser.add_argument("--sweep-output-dir", type=str, default="/kaggle/working/simpo_sweep")
    parser.add_argument("--skip-data-guard", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.dataset_path)
    if not data_path.exists():
        print(f"Dataset path {data_path} not found. Exiting.")
        sys.exit(1)

    print(f"Loading SimPO sweep dataset from {data_path}...")
    dataset = load_step_dpo_dataset(data_path, images_dir=args.images_dir)
    print(f"Loaded {len(dataset)} preference pairs.")
    if not args.skip_data_guard:
        stats = validate_training_dataset(dataset, name="SimPO sweep")
        print(f"Data guard OK: mean_chars={stats['mean_chars']:.1f} final_answer_rate={stats['final_answer_rate']:.1%}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    processor_kwargs = {}
    if torch.cuda.is_available():
        processor_kwargs["min_pixels"] = 64 * 28 * 28
        processor_kwargs["max_pixels"] = 128 * 28 * 28
    processor = AutoProcessor.from_pretrained(args.model_id, **processor_kwargs)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
        model_kwargs["attn_implementation"] = "sdpa"

    print(f"Loading base model {args.model_id} once for the whole sweep...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    sweep_out = Path(args.sweep_output_dir)
    sweep_out.mkdir(parents=True, exist_ok=True)

    results = []
    peft_model = None
    for i, config in enumerate(GRID):
        print(f"\n=== Config {i + 1}/{len(GRID)}: {config} ===")
        adapter_name = f"cfg_{i}"
        # get_peft_model wraps base_model in a NEW PeftModel object the first time; every
        # later config reuses that same wrapper via add_adapter/set_adapter instead of
        # calling get_peft_model again, which would try to re-wrap an already-PEFT-ified
        # base_model and fail.
        if peft_model is None:
            peft_model = get_peft_model(base_model, peft_config, adapter_name=adapter_name)
        else:
            peft_model.add_adapter(adapter_name, peft_config)
            peft_model.set_adapter(adapter_name)
        model = peft_model

        def print_step_log(step, metrics, cfg=config):
            if step % 25 == 0 or step == 1:
                print(
                    f"[cfg={cfg} step {step:03d}] Loss: {metrics['loss']:.4f} | "
                    f"Margin: {metrics['reward_margin']:.4f} | Acc: {metrics['preference_accuracy']*100:.1f}%"
                )

        history = fit_simpo(
            model=model,
            dataset=dataset,
            processor=processor,
            lr=config["lr"],
            beta=config["beta"],
            gamma_beta_ratio=config["gamma_beta_ratio"],
            epochs=1,
            batch_size=1,
            device=device,
            on_step_end=print_step_log,
        )

        summary = summarize_history(history)
        print(f"Config {i + 1} summary: {summary}")

        cfg_dir = sweep_out / f"config_{i:02d}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(cfg_dir))
        with (cfg_dir / "training_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        results.append({"index": i, "config": config, "summary": summary, "adapter_dir": str(cfg_dir)})

        # Delete this config's adapter (on the PeftModel wrapper, not base_model -- base_model
        # itself is never a PeftModel and has no delete_adapter method) before the next one.
        peft_model.delete_adapter(adapter_name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sweep_results_path = sweep_out / "sweep_results.json"
    with sweep_results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {sweep_results_path}")

    best = pick_best_config(results)
    print(f"\nBest config: {best['config']} (summary: {best['summary']})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    for item in Path(best["adapter_dir"]).iterdir():
        shutil.copy2(item, out_dir / item.name)
    processor.save_pretrained(str(out_dir))
    with (out_dir / "winning_config.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"Copied winning config's adapter to {out_dir}")
    print("SimPO sweep complete!")


if __name__ == "__main__":
    main()
