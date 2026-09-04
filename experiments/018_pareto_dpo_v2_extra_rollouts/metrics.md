# Experiment 018: Pareto-DPO v2 (Full 0-9 Rollout Pool)

## What this is

Tests whether more candidate rollouts per question (not just a different pair-selection
method, already tested in experiment 014/v1) improves DPO training pairs. Combines the
original 0-4 rollouts with the newly-judged 5-9 rollouts (experiment 016) and rebuilds
Pareto-dominance pairs over the full pool: **298 pairs from 117/309 questions**, up from
v1's 154 pairs from 73/309 -- nearly double, confirming more candidates per question does
produce more Pareto-dominance pairs.

## First attempt: training collapsed

Training with the exact same hyperparameters as every prior DPO run in this project
(`lr=1e-5, beta=0.1`) hit the trainer's own collapse guard at **step 215/298**:
`policy chosen_logp=-148.8 is 42.0 nats below chosen_ref_logp=-106.8; likely generative
collapse`. The larger, more diverse v2 pair pool (which includes rollouts judged at a
lower 66.5% clean-parse rate than the original batch -- experiment 016) evidently doesn't
tolerate the same aggressive update size the smaller v1 set did.

## Hyperparameter sweep

Swept lr (lower = smaller steps) and beta (higher = tighter trust region around the
reference model) around the known-collapsing baseline, which was kept in the grid for a
documented before/after comparison (`scripts/train/sweep_dpo.py`):

| lr | beta | result |
| ---: | ---: | :--- |
| 1e-5 | 0.1 | **collapsed at step 215/298** (identical failure to the first attempt -- confirms it wasn't a fluke) |
| 1e-5 | 0.3 | **collapsed at step 259/298** (higher beta delayed but did not prevent collapse) |
| 5e-6 | 0.1 | completed, tail accuracy 64.9% |
| 5e-6 | 0.3 | completed, tail accuracy 64.9% |
| **2e-6** | **0.1** | **completed, tail accuracy 66.2% (winner)** |
| 2e-6 | 0.3 | completed, tail accuracy 64.9% |

Unlike the parallel SimPO sweep (experiment 019), this one produced genuine
differentiation between configs, not a tie: the 4 stable configs spread over 64.9-66.2%
tail accuracy, and the collapse was clearly reproducible at `lr=1e-5` regardless of beta,
confirming it's a real property of this pair set at that learning rate, not noise.

## Holdout result

| System | Official EM | Token match | Extracted-answer | Starts `Step 1:` | Structure score | Wrong committed | GT in full text |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pareto-DPO v2 (tuned, this experiment)** | **28%** | 31% | 100% | 85% | 93% | 37% | 63% |

For reference: Full DPO 29%, Pareto-DPO v1 30% (current project best), same scoring
convention throughout. **28/100, 100% extracted-answer rate.** Spot-checked: 100 unique
questions, real varied reasoning, no degenerate output.

## Bootstrapped comparisons

Paired bootstrap (10,000 resamples), same method used for every other comparison this
session:

- **Pareto-DPO v2 vs. Full DPO**: -1.0pp, 95% CI [-8.0%, +6.0%] -- crosses zero, not
  significant.
- **Pareto-DPO v2 vs. Pareto-DPO v1**: -2.0pp, 95% CI [-9.0%, +5.0%] -- crosses zero, not
  significant.

## Honest bottom line

More candidate rollouts per question produced more Pareto-dominance pairs (298 vs. 154)
and, after fixing the collapse, a real hyperparameter sweep with genuine differentiation
between configs -- but the final holdout number (28%) did not improve on v1 (30%) or Full
DPO (29%); both differences are within noise at n=100, and the point estimate actually
leans slightly below both. The learning rate that avoided collapse (2e-6) is 5x smaller
than the one v1 trained with (1e-5), which plausibly limited how far the larger pair set
could actually move the model within one epoch, even though more/better-separated
preference pairs were available. **More data alone, at a learning rate forced lower by
that data's own noise, did not translate into a better model here.** Whether a longer
training run (more epochs) at the safe learning rate would close the gap is untested.
