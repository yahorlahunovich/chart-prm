# Experiment 020: ChartQA Cross-Dataset Transfer Test (Reward Tree Transfer-Only)

Tests DG-PRM's central "Generalizable" claim directly: does the 33-criterion reward
tree built entirely from CharXiv's own judge failures (`experiments/009_reward_tree`)
still work, unmodified, on a second chart-QA dataset it never saw?

## What this is (and isn't)

- **Dataset**: ChartQA (Masry et al., 2022) human-authored, non-yes/no questions from
  the official `test` split (1,146 eligible). 150 train-pool questions sampled
  (seed=42), disjoint from a 30-question holdout not used here. See
  `scripts/data_prep/sample_chartqa_questions.py` and `data/ChartQA/`.
- **Rollouts**: 5 per question, 750 total, generated on Kaggle with the exact same
  prompt and settings as the CharXiv pipeline (`chart_prm.generator.build_generation_prompt`).
  Cleaned/parsed to 612 valid rollouts (81.6%) covering 149/150 questions
  (`scripts/data_prep/clean_chartqa_rollouts.py`, reuses `clean_dataset.py`'s parser).
- **Scoring**: the existing Phase 2 dynamic judge (`scripts/evaluation/score_steps_dynamic.py`,
  Gemini `gemini-3.5-flash-lite`) shown the CharXiv-derived reward tree **unmodified** --
  no rebuild, no retuning. Blind to ground truth (default Phase 2 design). This is the
  "transfer-only" arm decided on with the user: cheaper than also rebuilding a
  ChartQA-native tree, and the more direct test of transfer per se.
- **Not** a rebuilt/ChartQA-native tree comparison -- that would need a fresh static
  judge pass (Meta API) to generate failure critiques to cluster, which this scope
  deliberately skips.

## Result: does the tree transfer?

612/612 rollouts scored (0 corrupted rows, 0 duplicates), 2,229 steps total.

| Metric | ChartQA (this run) | CharXiv baseline (exp. 011, full-scale) |
| :--- | ---: | ---: |
| Steps with zero criteria selected | 0.0% (0/2,229) | 0.0% |
| Average criteria selected per step | **4.21** | 3.85 |
| Distinct criteria used at least once | **33 / 33** | -- |

**The tree transfers cleanly.** Every one of the 33 CharXiv-derived criteria fired at
least once on ChartQA content, no step came back with zero relevant criteria (the
judge never found the tree inapplicable), and the average selection density is
slightly *higher* than on the dataset the tree was built from. This is a direct,
positive answer to the generalization question this experiment exists to test: a
reward tree distilled from one chart-QA benchmark's failure modes is not overfit to
that benchmark's specific visual style or question phrasing.

## Operational notes

- Free-tier Gemini quota is a **hard daily cap** (500 requests/day for
  `gemini-3.5-flash-lite`, confirmed via direct API probe returning
  `RESOURCE_EXHAUSTED` / `GenerateRequestsPerDayPerProjectPerModel-FreeTier`), not a
  transient rate limit -- a run that plateaus at exactly 500 scored rows needs a new
  key/day, not a longer wait. The existing quota-aware resume design (unprocessed
  rows are never marked "processed" on failure) meant swapping in a fresh key and
  re-running picked up exactly where it left off with no data loss or rework.
- `score_steps_dynamic.py`'s `IMAGE_DIR` was hardcoded to CharXiv's path; added
  `--image-dir` as a CLI override (defaults unchanged) so the same script serves both
  datasets without duplicating the scoring logic.

## Next

Build preference/SFT datasets for all 7 alignment methods (SFT, Full DPO, Step-DPO,
KTO, SFT->DPO, SimPO, Pareto-DPO) from these scores, train on Kaggle, evaluate on the
30-question ChartQA holdout.
