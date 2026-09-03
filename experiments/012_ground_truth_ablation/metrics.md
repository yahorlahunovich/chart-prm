# Experiment 012: Ground-Truth-Shown Ablation

## Question

Experiment 011's full-scale result (-2.6pp, not significant) left one confound unresolved: the old judge is shown the ground-truth answer while grading; the new judge deliberately is not. Is the new judge's *design* actually competitive, or does the blind-vs-sighted asymmetry alone account for the gap? This ablation levels the playing field: same tree, same prompt, same judge (`gemini-3.5-flash-lite`), but the new judge is now also shown the ground truth (`--show-ground-truth`), via `chart_prm.dynamic_scoring.build_dynamic_scoring_prompt`'s `ground_truth` parameter (an explicit, opt-in ablation switch -- the normal Phase 2 pipeline never sets it).

Scoped to the 100-question pilot set (not the full 309), to fit inside a single day's free-tier quota.

## Data

- 402 of ~403 attempted rollouts scored successfully (3 null, dropped).
- Spot-checked: the judge visibly uses the answer key now (e.g. one step's note reads *"the correct answer (ground truth) is subplot (c)"*), confirming the ablation prompt behaves as intended rather than silently ignoring the added information.

## Result: informed vs. informed (apples-to-apples)

| | Old (v0 binary, sees GT) | New (v1 dynamic, sees GT) |
| --- | ---: | ---: |
| PRM best-of-N accuracy | 21.2% | 22.2% |
| PRM accuracy \| oracle positive | 52.5% | 56.4% |

Bootstrapped (10,000 paired resamples, n=99 questions eligible in both): **+1.0pp, 95% CI [-3.0%, +6.1%]** -- crosses zero, not significant. Paired breakdown: 19 both-right, 3 new-only, 2 old-only, 75 both-wrong -- new wins slightly more often than it loses, but on too few discordant pairs to call.

## Reading this alongside experiment 011

| Comparison | n | Diff (new - old) | Significant? |
| --- | ---: | ---: | --- |
| Blind new vs. informed old, pilot (exp 010) | 100 | +1.0pp | No |
| Blind new vs. informed old, full scale (exp 011) | 309 | -2.6pp | No |
| **Informed new vs. informed old, pilot (this)** | 99 | **+1.0pp** | No |

None of these are statistically distinguishable from zero -- that has not changed. But the *pattern* across all three is informative: the one full-scale measurement (most statistical power, most trustworthy point estimate) is the blind comparison, and it's the one that leans negative. The two comparisons that lean flat-to-positive are both small-n subsets. This is consistent with -- not proof of -- the leading hypothesis from experiment 011: the blind design costs some real accuracy, and once that's controlled for for the *design itself* looks at least competitive, just still unproven at any scale tested so far due to sample size.

## Honest bottom line

This ablation does not settle the question. It rules out one alternative explanation less: the new judge is not obviously *worse in design* than the old one when both have equal information, since the point estimate goes the other way once the confound is controlled. But "not obviously worse in design" is a much weaker claim than "the DG-PRM approach improves selection accuracy," which remains unproven. Scaling this specific ablation (informed vs. informed) to the full 309-question pool would be the next step to actually resolve it, at the cost of another ~1,199-call run.
