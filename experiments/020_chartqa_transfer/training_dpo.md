# ChartQA Training Log: Full DPO (method 2/7)

Kaggle kernel `qwen-vl-dpo-chartqa` (T4x2 requested). Same trainer, hyperparameters
(lr=1e-5, beta=0.1, 1 epoch, batch size 1), and collapse guard (max_logp_drop=40
nats) as CharXiv's Full DPO run -- only the dataset and image dir changed.

- Dataset: `experiments/020_chartqa_transfer/data/dpo_pairs.jsonl` (127 pairs, vs.
  CharXiv's 134).
- 127 training steps, 1 epoch.
- Loss: mean first 20 steps 0.679 -> mean last 20 steps 0.664 (DPO loss is noisier
  than SFT's; the more informative signal is preference accuracy and reward margin
  below).
- Preference accuracy (tail 50% of steps): 65.6%.
- Reward margin: mean first 20 steps 0.032 -> mean last 20 steps 0.151 (chosen
  rollouts pulling ahead of rejected ones as training progresses -- the intended
  direction).
- Collapse guard: max chosen_logp drop below reference was 17.9 nats, well under
  the 40-nat abort threshold. No collapse, no abort.
- Adapter saved (`adapters/qwen_vl_dpo_chartqa_adapter/`, gitignored).

No holdout eval yet -- happens once all 7 methods are trained.
