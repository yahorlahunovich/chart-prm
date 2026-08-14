# Experiment 006 — SFT→DPO training (kernel v3)

Kernel `qwen-vl-sft-dpo` v3 on Tesla P100. Policy LoRA copied from `qwen-vl-sft-custom`; frozen SFT is π_ref.

| | |
| --- | --- |
| Pairs | 134 full-trajectory |
| LR | 2e-6 |
| Epochs | 1 |
| Beta | 0.1 |
| Collapse guard | warn-only, max drop 70 nats |
| Init copy diff | 7.63e-06 |
| Steps completed | **134 / 134** |
| Mean pref acc | 97.8% |
| Step 1 margin | +0.014 |
| Last-10 mean margin | +0.85 |
| Last-10 mean loss | 0.39 |
| Collapse warnings | **0** (no chosen-logp drop > 40 nats) |
| Adapter | `qwen_vl_sft_dpo_adapter` (149 MB) |

v1 aborted on the copy-diff guard. v2 aborted at step 109 (`lr=5e-6`, hard collapse guard). v3 finished; milder LR avoided the 66-nat drop.

Holdout comparison vs Instruct→DPO (29%) is not run yet.
