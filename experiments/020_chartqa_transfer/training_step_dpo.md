# ChartQA Training Log: Suffix Step-DPO (method 3/7)

Kaggle kernel `qwen-vl-step-dpo-chartqa` (T4x2 requested). Same trainer,
hyperparameters (lr=1e-5, beta=0.1, 2 epochs, batch size 1), prefix masking, and
collapse guard (max_logp_drop=40 nats, aborting not warn-only) as CharXiv's
Step-DPO run -- only the dataset and image dir changed.

- Dataset: `experiments/020_chartqa_transfer/data/step_dpo_pairs.jsonl` (66 pairs,
  vs. CharXiv's 54).
- 132 training steps (66 pairs x 2 epochs).
- Loss: mean first 20 steps 0.678 -> mean last 20 steps 0.294.
- Preference accuracy (tail 50% of steps): 95.5%.
- Reward margin: mean first 20 steps 0.032 -> mean last 20 steps 1.627 (large jump
  -- consistent with Step-DPO's known behavior on this project: suffix training on
  short divergent fragments produces much larger implicit reward margins than
  full-trajectory DPO, see README's note on Step-DPO's "implicit reward margin
  explosion" in training dynamics).
- **Collapse guard: max chosen_logp drop was 38.84 nats -- close to the 40-nat abort
  threshold, did not trigger it.** No warning or abort appeared in the training log
  (only "DPO data guard OK" messages), so this was a genuine near-miss rather than a
  guard actually firing, but close enough to flag rather than gloss over. Worth
  watching in the holdout eval for the same structural collapse (Final Answer: /
  Step 1: rate dropping) the original CharXiv Step-DPO run showed.
- Adapter saved (`adapters/qwen_vl_step_dpo_chartqa_adapter/`, gitignored).

No holdout eval yet -- happens once all 7 methods are trained.
