#!/usr/bin/env python3
"""Merge SFT→DPO-only holdout generations onto experiment 005 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chart_prm.holdout_merge import SFT_DPO_KEY, exact_match_summary, merge_system_rows

DEFAULT_BASE = ROOT / "experiments/005_holdout_eval_suffix_step_dpo/data/holdout_generations.jsonl"
DEFAULT_OUT = ROOT / "experiments/007_sft_dpo_holdout"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--sft-dpo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    base_rows = load_jsonl(args.base)
    extra_rows = load_jsonl(args.sft_dpo)
    merged = merge_system_rows(base_rows, extra_rows, system=SFT_DPO_KEY)
    summary = exact_match_summary(
        merged,
        systems=["base", "sft", "dpo", "step_dpo", "kto", SFT_DPO_KEY],
    )

    data_dir = args.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = data_dir / "holdout_generations.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (data_dir / "holdout_accuracy.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_jsonl}")


if __name__ == "__main__":
    main()
