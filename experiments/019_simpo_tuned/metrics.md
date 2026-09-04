# Experiment 019: SimPO Hyperparameter Sweep

## What this is

Experiment 017's single SimPO run used the reference implementation's paper-scale
defaults (`beta=2.0`, `gamma_beta_ratio=0.5`, `lr=1e-6`) unvalidated for this project's
much smaller scale (134 pairs, 1 epoch, batch size 1, no warmup) and landed at 26%,
below both DPO variants, with a noisier training curve than the DPO runs. This sweeps
`lr` and `beta` locally -- the same discipline as experiment 009's reward-tree threshold
sweep -- instead of trusting the paper's defaults for a completely different scale.

Two real bugs were found and fixed while building this (see `scripts/train/sweep_simpo.py`'s
docstring and the `fix:` commits): reusing one base model across configs via PEFT's
add/set/delete-adapter produced a NaN loss on a config that trained cleanly standalone
(fixed by loading a fresh model per config), and the eval kernel's "find the one adapter"
logic broke once the sweep started saving every config's adapter for the full record
(fixed to target the winner's named directory).

## All 6 swept configs

| lr | beta | gamma_beta_ratio | final loss | mean loss (tail) | mean pref. accuracy (tail) | finite |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 5e-7 | 2.0 | 0.5 | 1.399 | 1.492 | 0.5152 | yes |
| 1e-6 | 2.0 | 0.5 | 1.403 | 1.491 | 0.5152 | yes |
| **1e-5** | **2.0** | **0.5** | **1.502** | **1.468** | **0.5152** | **yes (winner)** |
| 5e-7 | 5.0 | 0.5 | 2.852 | 3.081 | 0.5152 | yes |
| 1e-6 | 5.0 | 0.5 | 2.862 | 3.080 | 0.5152 | yes |
| 1e-5 | 5.0 | 0.5 | 3.155 | 3.015 | 0.5152 | yes |

**A real finding, not a footnote: every one of the 6 configs produced exactly the same
tail preference accuracy (0.515151...).** `beta` only rescales the loss/reward
magnitude, it cannot change the sign of `chosen_logp - rejected_logp` (which is what
`preference_accuracy` actually measures), so beta was mathematically never going to
differentiate configs on this metric -- that part is expected. But `lr` spanning 5e-7 to
1e-5 (a 20x range) *also* produced an identical tail accuracy across all three values,
which is not something `beta` explains. At this scale (134 pairs, single epoch,
batch size 1, LoRA r=16), 134 gradient steps evidently isn't enough to flip any of the
~33 held-late pairwise comparisons in the tail regardless of which of these learning
rates was used. **The training-time screening signal this sweep was designed to use did
not actually distinguish the 6 configs.** The "winner" (lr=1e-5, beta=2.0) was selected
by the tie-break rule (lowest tail loss among tied-accuracy configs) -- a real, principled
selection given the tie, but not a confident signal that this config trains meaningfully
better than the other 5.

## Holdout result

| System | Accuracy |
| --- | ---: |
| SFT->DPO | 22% |
| SFT | 23% |
| Step-DPO | 25% |
| SimPO (untuned, experiment 017) | 26% |
| Base | 26% |
| KTO | 16-26% |
| **SimPO (tuned, this experiment)** | **27%** |
| ChartGemma (zero-shot specialist) | 27% |
| Full DPO | 29% |
| Pareto-Dominance DPO | 30% (best) |

**27/100, 100% extracted-answer rate.** Spot-checked: 100 unique questions, real varied
reasoning, no degenerate output.

## Bootstrapped comparisons

Paired bootstrap (10,000 resamples) on the 100-question holdout, same method used for
every other comparison this session:

- **Tuned SimPO vs. Full DPO**: -2.0pp, 95% CI [-9.0%, +5.0%] -- crosses zero, not
  significant. (21 both-right, 65 both-wrong, 6 tuned-only, 8 DPO-only.)
- **Tuned SimPO vs. untuned SimPO**: +1.0pp, 95% CI [-7.0%, +9.0%] -- crosses zero, not
  significant. (18 both-right, 65 both-wrong, 9 tuned-only, 8 untuned-only.)

## Honest bottom line

The sweep found a config that nominally scores 1pp higher than the untuned run (27% vs.
26%), but that difference is well within noise -- the bootstrap CI is nearly symmetric
around zero and the underlying training-time signal that "selected" this config was
tied across the entire grid to begin with. Tuning did not find a SimPO configuration
that reliably beats Full DPO, and it did not find one that reliably beats the original
untuned SimPO run either. The honest conclusion is not "tuning failed to help" so much
as "134 pairs / 1 epoch / batch size 1 is too small a training budget for lr or beta,
across the ranges tested here, to produce a measurably different outcome" -- a genuine
limit of this project's scale, not evidence that SimPO as a method can't work here with
a larger preference dataset or more training steps, neither of which was tested.
