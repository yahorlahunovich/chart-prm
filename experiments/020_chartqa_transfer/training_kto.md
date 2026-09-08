# ChartQA Training Log: KTO (method 4/7)

Kaggle kernel `qwen-vl-kto-chartqa` (T4x2 requested). Same v14 config as CharXiv's
KTO run: lr=2e-6, beta=0.1, 2 epochs, batch size 1, warn-only collapse guard
(max_logp_drop=55 nats), `--balance-kto --auto-desirable-weight` (hard-negative
filtering + ~1:3 desirable:undesirable subsampling). Only the dataset and image
dir changed.

- Dataset: `experiments/020_chartqa_transfer/data/kto_samples.jsonl` (147
  desirable / 465 undesirable before balancing, vs. CharXiv's 84 / 1190).
- 1176 training steps (2 epochs, post-balance).
- Loss: mean first 20% of steps 0.752 -> last 20% 0.657. 39.1% of all steps had
  loss < 0.01 (near-zero) -- the desirable class is small (147) and heavily
  reweighted, so the policy fits those samples very confidently.
- Reward margin: mean first 20% 0.481 -> last 20% 3.071.
- **Collapse guard fired 110 times** (of 1176 steps, ~9.4%), starting at step 290
  and continuing through step 1164 -- e.g. *"policy desirable_policy_logp=-290.3
  is 58.9 nats below desirable_ref_logp=-231.4; likely generative collapse"*.
  Warn-only mode meant training continued rather than aborting (same choice the
  original CharXiv KTO run made, for the same reason -- KTO needs this leniency).
  Max drop recorded: 91.6 nats, well past the 55-nat threshold.
- **This mirrors the original project's own documented finding**: CharXiv's KTO
  run showed the same generative-collapse tendency (README: KTO abandons the step
  template, 0% structural compliance in that run). Worth checking whether ChartQA's
  KTO adapter shows the same structural collapse in the holdout eval, or whether
  the smaller/differently-shaped dataset changes that outcome.
- Adapter saved (`adapters/qwen_vl_kto_chartqa_adapter/`, gitignored).

No holdout eval yet -- happens once all 7 methods are trained.
