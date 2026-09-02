# Experiment 009: Reward Tree (DG-PRM Phase 1, v0)

Adapts the reward-tree component of Yin et al., "Dynamic and Generalizable Process Reward Modeling" (DG-PRM) to ChartPRM.

## What this is (and isn't)

- **Parents**: the 9 human-validated error categories from `scripts/evaluation/categorize_judge_errors.py`'s regex taxonomy over 2,920 real judge failure explanations. DG-PRM discovers parents from scratch; ChartPRM already had them, so they're reused as-is (`other_unspecified` excluded — it's a catch-all, not a criterion).
- **Children**: embedding sub-clusters within each parent (KMeans on the existing MiniLM `fail_analysis_embeddings.npy`, `k` scaled to category size via `choose_child_count`), described by TF-IDF top terms and 2 exemplar judge sentences, then deduplicated by merging near-identical sub-clusters (cosine distance <= 0.25, DG-PRM's `xi`, taken as-published pending retuning — see Observation below).
- **This is v0.** Child criteria are cluster descriptions, not the clean, reusable rubric statements ('axis values must be read from the correct gridline', etc.) DG-PRM's Phase 1 gets from an LLM judge examining contrastive pairs. That LLM-distillation pass is next, gated on getting a working (Gemini) judge API key — it's a text-only pass over these same clusters, no new rollout generation needed.
- **No new judge/API calls were made for this step.** Everything here is recomputed from data already in the repo (`fail_analyses_categorized.csv` + `fail_analysis_embeddings.npy`).

## Tree stats

- Parents: **9**
- Total child criteria: **17**
- Merge threshold (xi): 0.25
- Retrieval threshold (zeta, for Phase 2): 0.2
- Embedding model: all-MiniLM-L6-v2

## Observation: merge dominance at xi=0.25

Averaged across parents, **79%** of a parent's failures end up folded into its single largest child after merging at this threshold — worst case is **Arithmetic / calculation mistake** at 100%. Higher dominance means fewer, coarser children (less useful for Phase 2 retrieval); lower dominance keeps more distinct children but risks near-duplicates surviving as separate criteria. DG-PRM's published default is xi=0.25, tuned on their own domain/embedding model; their own ablation (Figure 10a in the paper) shows performance is sensitive to this constant, so it should be swept locally (`--merge-threshold`) rather than assumed to transfer as-is to short chart-critique sentences on this embedding model. Compare this run's dominance numbers against a run at a different threshold before picking one for Phase 2.

## Children per parent

| Parent | Source failures | Children | Largest child share |
| --- | ---: | ---: | ---: |
| Axis / layout / chart-structure misread | 702 | 2 | 76% |
| Wrong series / color / legend identity | 568 | 2 | 89% |
| Hallucinated entity / label not on chart | 445 | 2 | 61% |
| Wrong ranking / extremum (highest/lowest/second) | 319 | 3 | 86% |
| Logic inconsistency / false conclusion | 240 | 2 | 55% |
| Wrong numeric value read from chart | 181 | 1 | 100% |
| Bad comparison / threshold logic | 179 | 2 | 83% |
| Arithmetic / calculation mistake | 38 | 1 | 100% |
| Incomplete / truncated reasoning | 33 | 2 | 61% |

![Children per parent](figures/reward_tree_children_per_parent.png)

## Full tree

See [`data/reward_tree.json`](data/reward_tree.json).
