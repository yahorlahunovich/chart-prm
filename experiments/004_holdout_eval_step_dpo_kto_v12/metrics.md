# Experiment 004 — Holdout v8 (invalid DPO comparison)

Holdout kernel `qwen-vl-holdout-eval` v8 completed after fragment Step-DPO v2 and balanced KTO v12.

## Exact-match (100 holdout questions)

| System | Accuracy | Extracted-answer rate | Notes |
| --- | --- | --- | --- |
| Base | 26% | 100% | Unchanged |
| SFT | 23% | 100% | Existing 1-epoch/3-epoch full-trajectory adapter |
| DPO | 25% | 90% | **Invalid** — loaded fragment Step-DPO adapter |
| Step-DPO | 25% | 90% | Valid fragment-step adapter; identical generations to DPO column |
| KTO | 25% | 97% | Balanced 84/252, lr=1e-6, 1 epoch; format recovered vs 16%/73% |

DPO and Step-DPO generations are bit-identical (`0/100` differing). Log:

```
Available /kaggle/input entries: ['notebooks', 'qwen-vl-fragment-step-dpo']
dpo: /kaggle/input/qwen-vl-fragment-step-dpo/qwen_vl_step_dpo_adapter
step_dpo: /kaggle/input/qwen-vl-fragment-step-dpo/qwen_vl_step_dpo_adapter
```

Cause: last-resort resolver used `"dpo" in path`, which matches `qwen-vl-fragment-step-dpo` / `qwen_vl_step_dpo_adapter`. Full-trajectory DPO (`qwen_vl_dpo_adapter`, 29% in experiment 003) was never loaded.

## KTO v12 training

- Dataset after `--balance-kto`: 84 desirable / 252 undesirable
- `lr=1e-6`, `beta=0.05`, 1 epoch, 336 steps
- Rewards barely moved (final desirable reward −0.06). Underfit, not collapse.
- Holdout format recovered: `Final Answer:` 96% vs 73% previously.

## Fragment Step-DPO v2 training

- 54 pairs × 3 epochs = 162 steps
- Loss 0.693 → 0.011, preference accuracy 100%, margin +4.50
- Chosen/rejected were **single first-error steps with no Final Answer** (mean 126/172 chars)
- Holdout format degraded vs base: Step labels 59% (base 99%), Final Answer 90%

Do not use the DPO column from this run in the paper comparison.
