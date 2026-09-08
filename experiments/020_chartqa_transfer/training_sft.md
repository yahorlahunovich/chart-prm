# ChartQA Training Log: SFT (method 1/7)

Kaggle kernel `qwen-vl-sft-chartqa` (T4x2 requested). Same trainer, hyperparameters,
and LoRA config as the CharXiv SFT run (`chart_prm.sft`, r=16/alpha=32, 1 epoch,
lr=1e-5, batch size 1) -- only the dataset changed.

- Dataset: `experiments/020_chartqa_transfer/data/sft_samples.jsonl` (136 gold
  ChartQA traces, vs. CharXiv's 70 -- ChartQA's higher step pass rate under the
  transferred reward tree, 64.5% vs. CharXiv's 41.0%, yields more fully-verified
  rollouts to train on).
- 136 training steps (1 sample/step, 1 epoch), ~10.4 min wall clock.
- Loss: mean first 20 steps 0.733 -> mean last 20 steps 0.575 (noisy, batch size 1,
  single epoch -- same shape as the original CharXiv SFT run). Min 0.127, max 2.174.
- Adapter saved (148MB, `adapters/qwen_vl_sft_chartqa_adapter/`, gitignored like all
  adapters in this repo) -- will serve as the init/reference for SFT->DPO later.

No holdout eval yet -- that happens once all 7 methods are trained, evaluated
together on the same 30-question ChartQA holdout for a fair comparison.
