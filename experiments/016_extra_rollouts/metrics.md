# Experiment 016: Extra Rollouts (indices 5-9) and the Extended Oracle Ceiling

## What this is

Experiment 013's free, local diagnostic found the best-of-K oracle ceiling still rising
steeply at K=5 (17.8% -> 53.4%), motivating a real Kaggle GPU job
(`ertugrultaparci/qwen-vl-extra-rollouts`) to generate 5 more rollouts per question
(indices 5-9) for all 309 eligible training-pool questions, using the exact same
generation code (`chart_prm.generator`) as the original 0-4 rollouts. Ran ~8.5 hours on
Kaggle (started 2026-09-03 20:21, completed same-session, well within Kaggle's 12h GPU
session limit).

## Data quality

- **1545 rows generated** (309 questions x 5 rollout indices, complete coverage -- no
  missing question/index combinations).
- **38 rows (2.5%) came back with a literally empty `model_output`** -- consistent with
  the null rate seen in earlier rollout-generation jobs this session.
- **Only 1027/1545 (66.5%) parsed successfully** under the exact same "Step 1:" +
  "Final Answer:" extraction convention `clean_dataset.py` used for the original 0-4
  rollouts (`extract_final_answer` in `scripts/evaluation/oracle_vs_rollout_count_extended.py`).
  This is a materially lower parse rate than the near-total success of the original
  rollouts. Spot-checked several failures and they're genuine generation problems, not a
  parsing bug: truncated single-phrase outputs (`"Number of parameters"`, 20 chars),
  garbled/looping text, a malformed `FinalAnswer:` (no space) that the strict convention
  correctly declines to match, and one response with stray HTML-like tags. Worth flagging
  as a real, if unexplained, data-quality gap between the original and extra rollout
  batches -- not something to paper over.

## Extended oracle-vs-K (K=1..10)

Restricted to questions with all 10 rollout indices (0-9) present *and* parseable, so
the K=1..10 comparison stays apples-to-apples within this run (mirroring experiment 013's
own "all 5 present" restriction, extended to 10):

- **52 / 309 questions qualify** (down from experiment 013's 118 questions that had all
  5 of 0-4 -- the added filter of needing all 5 *new* rollouts to also parse is a
  materially harder bar, given the 66.5% parse rate above).

| K | Oracle accuracy | Marginal gain |
| ---: | ---: | ---: |
| 1 | 21.2% | -- |
| 2 | 34.6% | +13.5% |
| 3 | 44.2% | +9.6% |
| 4 | 48.1% | +3.8% |
| 5 | 53.8% | +5.8% |
| 6 | 55.8% | +1.9% |
| 7 | 59.6% | +3.8% |
| 8 | 63.5% | +3.8% |
| 9 | 65.4% | +1.9% |
| 10 | **67.3%** | +1.9% |

(K=1-5 here are recomputed on this run's own n=52 subset, not copy-pasted from
experiment 013's n=118 -- close in shape, 53.8% vs. 53.4% at K=5, but not the identical
question set, so treat this table as internally consistent rather than a direct row-by-row
extension of 013's table.)

## Honest bottom line

**Experiment 013's prediction holds: the ceiling keeps rising well past K=5, reaching
67.3% at K=10, with no sign of flattening** -- marginal gains stay in the 1.9-3.8% range
all the way to K=10 rather than collapsing toward zero. This is consistent with the
earlier finding that generation attempts, not just alignment method, are a real lever on
this task. Two honest caveats: (1) this result is measured on a much smaller subset
(n=52) than experiment 013's n=118, because requiring all 5 new rollouts to also parse
cleanly is a harder bar than the original filter -- the curve shape is informative, but
the absolute n is small; (2) the 66.5% parse rate on the new rollouts is itself a data
point worth investigating before leaning on this batch for anything beyond this
diagnostic (e.g. before using indices 5-9 to build more DPO training pairs).
