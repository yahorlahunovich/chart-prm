#!/usr/bin/env python3
"""Preflight checks for SFT→DPO before a Kaggle launch.

Fails closed if data, hyperparameters, kernel wiring, or the SFT source
adapter look wrong. Does not load the 3B VLM.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chart_prm.data_guards import validate_training_dataset
from chart_prm.sft_dpo_init import read_adapter_config, require_adapter_dir


DPO_PATH = ROOT / "experiments/001_500_reasoning/data/dpo_pairs.jsonl"
SFT_PATH = ROOT / "experiments/001_500_reasoning/data/sft_samples.jsonl"
KERNEL_META = ROOT / "scripts/kaggle/kaggle_train_sft_dpo/kernel-metadata.json"
SFT_CONFIG = ROOT / "adapters/qwen_vl_sft_adapter/adapter_config.json"
EXPECTED_PAIRS = 134
EXPECTED_SFT = 70


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check_preference_data() -> None:
    pairs = _load_jsonl(DPO_PATH)
    if len(pairs) != EXPECTED_PAIRS:
        raise SystemExit(f"dpo_pairs.jsonl has {len(pairs)} rows, expected {EXPECTED_PAIRS}")
    stats = validate_training_dataset(pairs, name="DPO")
    if stats["final_answer_rate"] < 1.0:
        raise SystemExit(f"DPO Final Answer rate {stats['final_answer_rate']:.1%} < 100%")
    pair_types = {row.get("metadata", {}).get("pair_type") for row in pairs}
    if pair_types != {"full_trajectory"}:
        raise SystemExit(f"DPO pair_type values {pair_types}; expected only full_trajectory")
    missing_images = [
        row["image_path"]
        for row in pairs
        if not (ROOT / row["image_path"]).is_file()
    ]
    if missing_images:
        raise SystemExit(f"Missing {len(missing_images)} DPO images, e.g. {missing_images[0]}")
    print(
        f"OK DPO data: n={int(stats['n_examples'])} mean_chars={stats['mean_chars']:.0f} "
        f"final_answer={stats['final_answer_rate']:.0%} step_prefix={stats['step_prefix_rate']:.0%}"
    )

    sft = _load_jsonl(SFT_PATH)
    if len(sft) != EXPECTED_SFT:
        raise SystemExit(f"sft_samples.jsonl has {len(sft)} rows, expected {EXPECTED_SFT}")
    dpo_ids = {str(row["question_id"]) for row in pairs}
    sft_ids = {str(row["question_id"]) for row in sft}
    missing = dpo_ids - sft_ids
    if missing:
        raise SystemExit(f"{len(missing)} DPO questions have no SFT gold trace")
    sft_by_key = {
        (str(row["question_id"]), row.get("rollout_index")): row["solution"] for row in sft
    }
    matched = 0
    for row in pairs:
        key = (str(row["question_id"]), row.get("metadata", {}).get("chosen_rollout_index"))
        if sft_by_key.get(key) == row["chosen"]:
            matched += 1
    if matched != len(pairs):
        raise SystemExit(f"Only {matched}/{len(pairs)} DPO chosen completions match SFT gold traces")
    print(
        f"OK SFT data: n={len(sft)} unique_q={len(sft_ids)}; "
        f"DPO unique_q={len(dpo_ids)}; all DPO questions in SFT; "
        f"chosen==SFT on {matched}/{len(pairs)} pairs"
    )


def check_sft_adapter_config() -> None:
    if not SFT_CONFIG.is_file():
        print(
            "WARN local SFT adapter_config missing (adapters/ is gitignored); "
            "Kaggle must mount qwen-vl-sft-custom"
        )
        return
    config = read_adapter_config(SFT_CONFIG.parent)
    if config.get("r") != 16 or config.get("lora_alpha") != 32:
        raise SystemExit(f"SFT LoRA config unexpected: r={config.get('r')} alpha={config.get('lora_alpha')}")
    targets = set(config.get("target_modules") or [])
    expected = {"q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if targets != expected:
        raise SystemExit(f"SFT target_modules {targets} != {expected}")
    try:
        require_adapter_dir(SFT_CONFIG.parent, require_weights=True)
        print(f"OK local SFT adapter weights at {SFT_CONFIG.parent}")
    except FileNotFoundError:
        print(
            "OK SFT adapter_config (r=16); weights are gitignored — "
            "Kaggle must mount qwen-vl-sft-custom"
        )


def check_kernel_metadata() -> None:
    meta = json.loads(KERNEL_META.read_text(encoding="utf-8"))
    if meta.get("id") != "egorlagunovich/qwen-vl-sft-dpo":
        raise SystemExit(f"Kernel id {meta.get('id')} would overwrite another run")
    if "qwen-vl-step-dpo-custom" in str(meta.get("id")):
        raise SystemExit("Refusing to reuse the 29% full-DPO kernel")
    sources = meta.get("kernel_sources") or []
    if "egorlagunovich/qwen-vl-sft-custom" not in sources:
        raise SystemExit(f"kernel_sources missing SFT kernel: {sources}")
    if meta.get("accelerator") != "gpuT4x2":
        raise SystemExit(f"accelerator {meta.get('accelerator')} != gpuT4x2")
    notebook = (KERNEL_META.parent / meta["code_file"]).read_text(encoding="utf-8")
    required = [
        "--sft-dpo",
        "qwen_vl_sft_dpo_adapter",
        "2e-6",
        "'--epochs', '1'",
        "--init-adapter",
        "--collapse-guard-warn-only",
        "'--max-logp-drop', '70'",
    ]
    missing = [item for item in required if item not in notebook]
    if missing:
        raise SystemExit(f"Kaggle notebook missing {missing}")
    if "--step-dpo" in notebook:
        raise SystemExit("Kaggle notebook contains --step-dpo")
    if "/kaggle/working/qwen_vl_dpo_adapter" in notebook:
        raise SystemExit("Kaggle notebook writes the Instruct→DPO adapter path")
    print(
        f"OK kernel {meta['id']} mounts {sources} accelerator={meta['accelerator']}"
    )


def check_sft_kaggle_kernel() -> None:
    try:
        proc = subprocess.run(
            [
                "kaggle",
                "kernels",
                "status",
                "egorlagunovich/qwen-vl-sft-custom",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("WARN kaggle CLI not found; skip remote SFT kernel status")
        return
    text = (proc.stdout or "") + (proc.stderr or "")
    print(text.strip() or f"kaggle status exit {proc.returncode}")
    if proc.returncode != 0:
        raise SystemExit("Could not read qwen-vl-sft-custom status; SFT adapter mount would fail")
    if "complete" not in text.lower() and "COMPLETE" not in text:
        # kaggle status format varies; accept hasOutput / complete
        if "hasOutput" not in text and "has_output" not in text.lower():
            raise SystemExit(
                "qwen-vl-sft-custom does not look COMPLETE. "
                "SFT→DPO needs that kernel output as kernel_sources."
            )
    print("OK qwen-vl-sft-custom kernel is available as an SFT weight source")


def main() -> None:
    check_preference_data()
    check_sft_adapter_config()
    check_kernel_metadata()
    check_sft_kaggle_kernel()
    print("SFT→DPO preflight passed.")


if __name__ == "__main__":
    main()
