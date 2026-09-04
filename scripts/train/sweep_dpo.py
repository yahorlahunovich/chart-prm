#!/usr/bin/env python3
"""
Small local hyperparameter sweep for DPO, built after the Pareto-DPO v2 run
(298 pairs from the full 0-9 rollout pool, roughly double v1's 154) hit the
trainer's own collapse guard at step 215/298: "policy chosen_logp=-148.8 is
42.0 nats below chosen_ref_logp=-106.8; likely generative collapse", using
the exact same hyperparameters (lr=1e-5, beta=0.1) that trained cleanly on
every prior DPO run in this project at the smaller v1/original scale. The
larger, more diverse v2 pair pool (which includes rollouts 5-9, judged at a
lower 66.5% clean-parse rate than the original 0-4 batch -- see experiment
016) evidently doesn't tolerate the same aggressive update size.

Sweeps lr (lower = smaller steps) and beta (higher = tighter trust region
around the reference model, i.e. more resistant to the policy diverging)
around the known-collapsing baseline, including that baseline itself in the
grid for a documented before/after comparison. A fresh base model is loaded
for every config (no adapter reuse across configs -- see sweep_simpo.py's
docstring for why that was worth avoiding after a similar sweep there hit an
unexplained NaN under adapter reuse that a real config never produced
standalone). A single config's collapse is caught and recorded rather than
taking down the whole sweep, and sweep_results.json is rewritten after every
config so a later failure never discards earlier results.
"""

import argparse
import json
from pathlib import Path
import sys
import torch

from chart_prm.data_guards import collapse_guard, validate_training_dataset
from chart_prm.dpo.trainer import fit_dpo
from chart_prm.dpo.utils import load_step_dpo_dataset

GRID = [
    {"lr": 1e-5, "beta": 0.1},  # the baseline that collapsed on the v2 pair set -- kept for comparison
    {"lr": 1e-5, "beta": 0.3},
    {"lr": 5e-6, "beta": 0.1},
    {"lr": 5e-6, "beta": 0.3},
    {"lr": 2e-6, "beta": 0.1},
    {"lr": 2e-6, "beta": 0.3},
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
    """Highest mean preference_accuracy over the last 25% of steps; ties broken by lowest
    mean loss (tail), excluding non-finite runs. `results` is expected to already be
    restricted to configs that finished (fit_dpo either completes the full dataset or
    raises partway through with no partial history returned -- there is no "partially
    completed" case to separately filter for here)."""
    if not results:
        raise ValueError("No sweep results to pick from.")
    finite_results = [r for r in results if r["summary"]["all_finite"]]
    pool = finite_results or results
    return sorted(
        pool,
        key=lambda r: (-r["summary"]["mean_preference_accuracy_tail"], r["summary"]["mean_loss_tail"]),
    )[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Small local hyperparameter sweep for DPO")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Preference pairs jsonl (e.g. experiments/018_pareto_dpo_v2_extra_rollouts/data/pareto_dpo_pairs_v2.jsonl)",
    )
    parser.add_argument("--images-dir", type=str, default="data/CharXiv/images")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", type=str, default="adapters/qwen_vl_dpo_tuned_adapter")
    parser.add_argument("--sweep-output-dir", type=str, default="/kaggle/working/dpo_sweep")
    parser.add_argument("--max-logp-drop", type=float, default=40.0)
    parser.add_argument("--skip-data-guard", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.dataset_path)
    if not data_path.exists():
        print(f"Dataset path {data_path} not found. Exiting.")
        sys.exit(1)

    print(f"Loading DPO sweep dataset from {data_path}...")
    dataset = load_step_dpo_dataset(data_path, images_dir=args.images_dir)
    print(f"Loaded {len(dataset)} preference pairs.")
    if not args.skip_data_guard:
        stats = validate_training_dataset(dataset, name="DPO sweep")
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
    sweep_results_path = sweep_out / "sweep_results.json"

    results = []
    for i, config in enumerate(GRID):
        print(f"\n=== Config {i + 1}/{len(GRID)}: {config} ===")

        print(f"Loading a fresh base model for config {i + 1}...")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
        base_model.config.use_cache = False
        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()
        model = get_peft_model(base_model, peft_config)

        def print_step_log(step, metrics, cfg=config):
            if step % 25 == 0 or step == 1:
                print(
                    f"[cfg={cfg} step {step:03d}] Loss: {metrics['loss']:.4f} | "
                    f"Margin: {metrics['reward_margin']:.4f} | Acc: {metrics['preference_accuracy']*100:.1f}%"
                )
            warning = collapse_guard(metrics, max_logp_drop_vs_ref=args.max_logp_drop)
            if warning:
                raise RuntimeError(f"DPO collapse guard tripped at step {step}: {warning}")

        cfg_dir = sweep_out / f"config_{i:02d}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        error = None
        history = []
        try:
            history = fit_dpo(
                model=model,
                dataset=dataset,
                processor=processor,
                ref_model=None,
                lr=config["lr"],
                beta=config["beta"],
                epochs=1,
                batch_size=1,
                device=device,
                on_step_end=print_step_log,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad config (e.g. collapse) must not sink the sweep
            error = f"{type(exc).__name__}: {exc}"
            print(f"Config {i + 1} FAILED: {error}")

        summary = summarize_history(history)
        summary["error"] = error
        print(f"Config {i + 1} summary: {summary}")

        if history:
            model.save_pretrained(str(cfg_dir))
            with (cfg_dir / "training_history.json").open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        results.append(
            {
                "index": i,
                "config": config,
                "summary": summary,
                "adapter_dir": str(cfg_dir) if history else None,
            }
        )

        with sweep_results_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        del model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nWrote {sweep_results_path}")

    completed_results = [r for r in results if r["adapter_dir"] is not None]
    if not completed_results:
        raise RuntimeError("Every config in the sweep failed -- no adapter to select as a winner.")
    best = pick_best_config(completed_results)
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
    print("DPO sweep complete!")


if __name__ == "__main__":
    main()
