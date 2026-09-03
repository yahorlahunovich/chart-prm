# Experiment 015: ChartGemma Specialist Baseline on the Protected Holdout

## Question

Everything tried so far (SFT, DPO, Step-DPO, KTO, SFT->DPO) keeps the same general-purpose
Qwen2.5-VL-3B "student" and only changes how it's aligned. Is the ceiling actually an
alignment problem, or is it that a general-purpose vision-language model just isn't that
good at reading charts specifically? `ahmed-masry/chartgemma` (~3B) is a model pretrained
specifically for chart QA -- running it zero-shot on the same 100-question protected holdout
tests specialization as a different lever than alignment tricks on a fixed base model.

## Method

CharXiv's own vendored ChartGemma adapter (`data/CharXiv/src/generate_lib/chartgemma.py`)
and query-building code (`reasoning_utils.build_reasoning_queries`), completely unmodified,
run on the exact 100-question holdout every other system in this project was benchmarked on
(`data/splits/eval_reasoning_ids.json`). Scored with this project's own whole-token matcher
(`chart_prm.text_match.answers_match`), not CharXiv's official GPT-based reasoning scorer, so
the number is directly comparable to the exact-match numbers reported for the other six
systems. Zero training -- this is a pretrained-weights-only comparison.

## Result

| System | Accuracy |
| --- | ---: |
| Base (Qwen2.5-VL-3B, untouched) | 26% |
| SFT | 23% |
| Step-DPO | 25% |
| KTO | 16-26% |
| SFT->DPO | 22% |
| Full DPO | **29%** (best) |
| **ChartGemma (zero-shot specialist)** | **27%** |

## Reading

A zero-training chart-QA specialist (27%) beats the untouched general-purpose student (26%)
and 4 of the 5 trained alignment variants, landing just short of Full DPO (29%). This is
evidence -- not proof, n=100 is small and no significance test was run on this single
comparison -- that base model capability, not just alignment method, is a real lever for
this task. It does not show specialization beats alignment outright (Full DPO still wins),
but it does show alignment tricks on a fixed general-purpose base aren't the only place to
look for improvement.
