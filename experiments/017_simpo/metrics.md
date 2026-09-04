# Experiment 017: SimPO (Reference-Free Preference Optimization)

## What this is

Tests whether a different *training method* -- not a different judge or different
pair-selection, both already tried in Phase 2/3 -- beats DPO on this project's data.
SimPO (Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free
Reward", princeton-nlp/SimPO) drops the reference model entirely: the reward is the
policy's own length-normalized log-probability instead of a KL term against a frozen
reference. Reimplemented in `src/chart_prm/simpo/` from SimPO's actual trainer source
(`scripts/simpo_trainer.py`), not just the paper's notation.

Trained on the **exact same pairs file as Full DPO**
(`experiments/001_500_reasoning/data/dpo_pairs.jsonl`, 134 pairs) with the same base
model and 1 epoch, so training method is the one isolated variable -- same discipline
as Phase 3's Pareto-DPO comparison. Hyperparameters (`beta=2.0`, `gamma_beta_ratio=0.5`,
`lr=1e-6`) taken from the reference implementation's own tuning guidance, not swept
locally.

## Training

Ran on Kaggle (`ertugrultaparci/qwen-vl-simpo`, 2xT4), 134 steps. No collapse (loss
stayed finite throughout, no NaN/blowup), but noisier and less clearly convergent than
the DPO-family runs: preference accuracy averaged ~44.8% in the first half of training
vs. ~52.2% in the second half -- a real but modest improvement, not the cleaner
downward-trending loss seen in the Pareto-DPO run. Worth flagging as a genuine
difference in training dynamics, not just noting the final number.

## Holdout result

Same 100-question protected holdout, same exact-match scoring as every other system:

| System | Accuracy |
| --- | ---: |
| SFT->DPO | 22% |
| SFT | 23% |
| Step-DPO | 25% |
| **SimPO (this experiment)** | **26%** |
| Base | 26% |
| KTO | 16-26% |
| ChartGemma (zero-shot specialist) | 27% |
| Full DPO | 29% |
| Pareto-Dominance DPO | 30% (best) |

**26/100, 100% extracted-answer rate.** Spot-checked the generations: 100 unique
questions, real varied reasoning, no degenerate output.

## Bootstrapped against Full DPO

Paired bootstrap on the same 100 holdout questions (10,000 resamples):

- Both correct: 18. Both wrong: 63. SimPO-only correct: 8. DPO-only correct: 11.
- Diff (SimPO - Full DPO): **-3.0pp, 95% CI [-11.0%, +5.0%]** -- crosses zero, not
  significant, but the point estimate leans negative here, unlike Pareto-DPO's +1.0pp
  lean on the same kind of comparison.

## Honest bottom line

SimPO does not beat DPO on this project's data with these hyperparameters -- it lands
at parity with the untouched base model (26%) and below both DPO variants, though the
difference from Full DPO is not statistically significant at n=100. This is consistent
with the noisier training dynamics observed during the run: SimPO's much larger `beta`
and much smaller `lr` (both taken directly from the reference implementation's
guidance, not tuned for this ~134-pair, 1-epoch, batch-size-1 setting) may simply not
be well-matched to this much smaller scale than SimPO's own experiments were run at.
This is a legitimate negative result, reported as such rather than reframed --
"reference-free and length-normalized" was a real, principled hypothesis for improving
on DPO, and it didn't pan out here at default settings. A local learning-rate/beta
sweep, not attempted, would be the natural next step before concluding anything
stronger about the method itself.
