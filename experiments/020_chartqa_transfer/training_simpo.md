# ChartQA Training Log: SimPO (method 5/7)

Kaggle kernel `qwen-vl-simpo-chartqa` (T4x2 requested). Same trainer/hyperparameters
as CharXiv's SimPO run (lr=1e-6, beta=2.0, gamma/beta=0.5, 1 epoch, batch size 1,
reference-free -- no collapse guard, since there's no reference model to drift
from). Trained on the exact same pairs file as ChartQA's Full DPO run
(`experiments/020_chartqa_transfer/data/dpo_pairs.jsonl`, 127 pairs), matching the
original project's SimPO methodology (isolate training method, not data).

- 127 training steps, 1 epoch.
- Loss: mean first 20 steps 1.609 -> mean last 20 steps 1.321.
- Reward margin: mean first 20 steps -0.329 -> mean last 20 steps 0.060 (crosses
  from negative to weakly positive).
- **Preference accuracy (tail 50% of steps): 51.6% -- barely above chance.**
  This closely reproduces CharXiv's own SimPO result: experiment 017's
  hyperparameter sweep found *all six* lr/beta configs converged to an identical
  tail preference accuracy of 0.515151... there too, and attributed it to 134
  pairs / 1 epoch not being enough signal to reliably flip preference outcomes at
  this training scale, not a bad hyperparameter choice. ChartQA's 0.516 on a
  differently-sized (127-pair) dataset landing at essentially the same number is
  evidence for that explanation: a structural property of SimPO's reference-free
  formulation at this data scale, not noise specific to either dataset.
- Adapter saved (`adapters/qwen_vl_simpo_chartqa_adapter/`, gitignored).

No holdout eval yet -- happens once all 7 methods are trained.
