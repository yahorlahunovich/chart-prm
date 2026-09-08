# ChartQA Training Log: Pareto-DPO (method 6/7)

Kaggle kernel `qwen-vl-pareto-dpo-chartqa` (T4x2 requested). Same trainer,
hyperparameters (lr=1e-5, beta=0.1, 1 epoch, batch size 1, collapse guard
aborting at 40 nats) as CharXiv's Pareto-DPO run -- only the dataset and image
dir changed.

- Dataset: `experiments/020_chartqa_transfer/data/pareto_dpo_pairs.jsonl` (92
  pairs, Pareto-dominance-filtered across the reward tree's 9 parent categories
  from the transfer-only dynamic scores, vs. CharXiv's 154).
- 92 training steps, 1 epoch.
- Loss: mean first 20 steps 0.688 -> mean last 20 steps 0.656.
- Preference accuracy (tail 50% of steps): 60.9%.
- Reward margin: mean first 20 steps 0.011 -> mean last 20 steps 0.085.
- Collapse guard: max chosen_logp drop 3.4 nats, far below the 40-nat abort
  threshold -- the cleanest run of the DPO-family methods so far, consistent
  with Pareto-DPO pairs being the most stringently filtered (must dominate on
  all 9 categories at once).
- Adapter saved (`adapters/qwen_vl_pareto_dpo_chartqa_adapter/`, gitignored).

No holdout eval yet -- happens once all 7 methods are trained. This is method 6/7;
only SFT->DPO remains, which needs the SFT adapter already trained (method 1).
