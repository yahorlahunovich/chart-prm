# Experiment 014: Pareto-Dominance-Filtered DPO Pairs (DG-PRM Phase 3)

## What this is

Completes the 3-phase DG-PRM adaptation. Instead of collapsing each rollout's dynamic
judge scores (experiment 011) into one scalar, each rollout is scored on a 9-dimensional
vector -- one score per reward-tree parent category (`chart_prm.pareto.per_parent_vector`).
A preference pair is kept only when the correct rollout Pareto-dominates the incorrect one:
at least as good on every category, strictly better on one (`pareto_dominates`). This
filters out pairs where the two rollouts trade off against each other on different failure
axes (e.g. one hallucinates less but misreads the axis more) -- ambiguous signal that a
single-scalar selection method can't distinguish from a clean win.

Built entirely from experiment 011's already-computed scores -- zero new API calls.
Trained with the exact same trainer, base model, and hyperparameters as the existing Full
DPO run (`scripts/train/train_dpo.py`, 1 epoch, lr 1e-5, beta 0.1), so pair-selection
method is the one isolated variable.

## Pair construction

- 154 pairs from 73/309 eligible training-pool questions (`scripts/data_prep/format_pareto_dpo.py`).
- 100% Final Answer rate, mean chosen length 428.8 chars (well clear of the data guard's
  fragment-detection thresholds).

## Training

Ran on Kaggle (`ertugrultaparci/qwen-vl-pareto-dpo`, 2xT4). Loss trended down over the run
(mid-run losses mostly 0.6-0.8, later steps regularly below 0.6, several under 0.5) and
reward margins grew more separated in the second half of training -- no sign of the
generative collapse this project has hit before on other runs.

## Holdout result

Same 100-question protected holdout, same exact-match scoring as every other system in
this project (`extract_final_answer` + `normalize_answer`):

| System | Accuracy |
| --- | ---: |
| Base | 26% |
| SFT | 23% |
| Step-DPO | 25% |
| KTO | 16-26% |
| SFT->DPO | 22% |
| ChartGemma (zero-shot specialist) | 27% |
| Full DPO (previous best) | 29% |
| **Pareto-Dominance DPO (this experiment)** | **30%** |

**30/100, 99% extracted-answer rate.** Spot-checked the generations: 100 unique questions,
real step-by-step reasoning with plausible mistakes, no degenerate/repeated output.

## Is 30% actually better than Full DPO's 29%? No -- not at this sample size.

Paired bootstrap against Full DPO's per-question results on the same 100 holdout questions
(`experiments/003_holdout_eval_full_traj/data/holdout_generations.jsonl`, 10,000 resamples):

- Both correct: 22 questions. Both wrong: 63. Pareto-DPO-only correct: 8. Full-DPO-only
  correct: 7 -- nearly balanced.
- Diff (Pareto-DPO - Full DPO): **+1.0pp, 95% CI [-7.0%, +9.0%]** -- crosses zero, not
  significant.

## Honest bottom line

Pareto-Dominance DPO produces the highest nominal accuracy of any system tried in this
project (30%), but the gap over Full DPO (29%) is well within noise at n=100 -- 8
questions flipped in Pareto-DPO's favor, 7 flipped the other way, and the confidence
interval comfortably includes zero. This is consistent with, not a reversal of, Phase 2's
finding that the dynamic multi-criteria judge doesn't demonstrate a *significant* selection
advantage over the original judge at this scale. What Phase 3 does show: a Pareto-filtered,
multi-dimensional pair-selection method is at least competitive with the existing
single-scalar approach used for Full DPO's pairs, built from criteria/data that were
already computed for other purposes at zero marginal API cost. It does not, on the evidence
here, prove DG-PRM's approach beats the baseline -- and shouldn't be reported as if it did.
