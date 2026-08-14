# Experiment 007 — SFT→DPO holdout (merged onto 005)

Holdout kernel `qwen-vl-holdout-eval` **v11** generated **only** `sft_dpo`. The other five systems are copied from experiment 005 (kernel v9). Adapter path is unique:

| Slot | Mount |
| --- | --- |
| sft_dpo | `/kaggle/input/qwen-vl-sft-dpo/qwen_vl_sft_dpo_adapter` |

Checks: 100/100 rows, IDs identical to 005, response keys `{sft_dpo}` only, 0 empty generations, 0/100 traces identical to Base, SFT, or Instruct→DPO. GPU: P100. Extracted-answer rate 100%.

## Exact-match (100 holdout questions)

| System | Accuracy | Extracted-answer rate | Starts `Step 1:` | Wrong committed |
| --- | --- | --- | --- | --- |
| Base | 26% | 100% | 93% | 37% |
| SFT | 23% | 100% | **100%** | 43% |
| Full DPO (Instruct→DPO) | **29%** | 100% | 95% | 49% |
| Step-DPO (suffix) | 25% | 99% | 42% | 36% |
| KTO v14 | 26% | 90% | 0% | **30%** |
| SFT→DPO | 22% | 100% | **100%** | 53% |

SFT→DPO does **not** beat Instruct→DPO (29%) or even SFT (23%). It kept SFT’s `Step 1:` format (100%, no preamble) but exact-match dropped 1 pp vs SFT and 7 pp vs Instruct→DPO, and the wrong-committed proxy is the worst of the six (53%).

See `quality_metrics.md` for token-match and error breakdowns.
