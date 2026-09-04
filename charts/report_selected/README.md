# Selected Publication-Quality Charts for Report

This directory contains the curated set of 10 primary and high-value figures selected from the full chart suite for inclusion in the research report and LaTeX paper (`report/acl_latex.tex`).

---

## 1. Primary Benchmark & Alignment Results (Sec 10)
- **`results_01_overall_model_comparison.png`**: Benchmark comparison across Base, SFT, Full DPO, Step-DPO, KTO, and SFT→DPO on the CharXiv reasoning holdout (N=100) across Exact Match, Token Match, Structure Correctness, and GT Recall.
- **`results_08_accuracy_vs_structure_tradeoff.png`**: 2D Pareto frontier mapping Official Exact-Match Accuracy vs. Structural Instruction-Following Score (bubble size = Latent Ground Truth Recall).
- **`results_03_error_mode_hallucination_breakdown.png`**: 100% stacked bar chart showing holdout error composition (Extracted Correct, Unextracted Correct in text, GT Mentioned but Wrong Commitment, and Hallucination Proxy).

---

## 2. PRM Judge Error Taxonomy & Critiques (Sec 8)
- **`judge_01_error_taxonomy.png`**: 9-category failure modes taxonomy classified from 2,920 Meta PRM Judge critiques (Axis/Layout misread 24.0%, Series/Color 19.5%, Hallucinated entity 15.2%, Arithmetic 1.3%).
- **`judge_02_error_by_step_depth.png`**: Evolution of error categories across reasoning step depth (Step 0 to Step 4+), showing early perception bottlenecks transitioning into deeper logical failures.
- **`judge_07_multilabel_cooccurrence.png`**: Multi-label co-occurrence matrix across diagnosed failure causes (N=2,920 critiques), quantitatively proving how early perceptual errors (axis/legend confusion) trigger compound downstream reasoning failures.
- **`judge_04_top_keywords_per_category.png`**: Faceted bar charts showing top TF-IDF diagnostic vocabulary for the 6 primary error categories.

---

## 3. Reasoning Trajectory Dynamics & PRM Motivation (Sec 6 & Sec 9)
- **`08_error_cascade.png`**: Conditional score distributions at Step $N+1$ given Step $N$, proving that an initial error ($Score=0$) cascades into downstream failure ~83% of the time.
- **`04_first_error_position.png`**: Distribution of the step index where rollouts make their first mistake (over 80% occur at Step 0 or Step 1).
- **`results_04_training_dynamics_loss_rewards.png`**: 2x2 grid of training loss curves and implicit reward margins across SFT, Step-DPO, and KTO under single-T4 GPU constraints.
