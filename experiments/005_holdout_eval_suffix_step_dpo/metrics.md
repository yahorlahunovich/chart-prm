# Experiment 005 — Valid 5-system holdout (suffix Step-DPO + KTO v14)

Holdout kernel `qwen-vl-holdout-eval` v9. Adapter paths are unique (v8 collision is gone):

| Slot | Mount |
| --- | --- |
| sft | `/kaggle/input/notebooks/egorlagunovich/qwen-vl-sft-custom/qwen_vl_sft_adapter` |
| dpo | `/kaggle/input/notebooks/egorlagunovich/qwen-vl-step-dpo-custom/qwen_vl_dpo_adapter` |
| step_dpo | `/kaggle/input/qwen-vl-fragment-step-dpo/qwen_vl_step_dpo_adapter` |
| kto | `/kaggle/input/notebooks/egorlagunovich/qwen-vl-kto-custom/qwen_vl_kto_adapter` |

DPO vs Step-DPO generations differ on **100/100** questions.

## Exact-match (100 holdout questions)

| System | Accuracy | Extracted-answer rate | `Final Answer:` | `Step N:` |
| --- | --- | --- | --- | --- |
| Base | **26%** | 100% | 100% | 99% |
| SFT | 23% | 100% | 100% | 100% |
| Full DPO | **29%** | 100% | 100% | 99% |
| Step-DPO (suffix) | 25% | 99% | 99% | 78% |
| KTO v14 | 26% | 90% | 84% | 15% |

Full DPO matches experiment 003 (29%) and is the only method that beats base. Suffix Step-DPO recovered format vs the fragment-only v2 run (extract 90% → 99%, Step labels 59% → 78%) but did not improve accuracy. KTO v14 ties base on exact-match while dropping step-by-step formatting (v12 had 96% FA / 79% Step labels at 25%).

See `quality_metrics.md` for structure, token-match, and wrong-committed-answer breakdowns beyond official exact-match.
