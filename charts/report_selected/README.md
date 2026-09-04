# Curated Publication-Quality Figures for ACL Report

This directory contains the curated suite of figures for the research paper:  
**"ChartPRM: Process Reward Modeling and Step-Level Preference Optimization for Complex Multimodal Chart Reasoning"** (`report/acl_latex.tex`).

Figures are organized into **Tier 1 (Main Body)**, **Tier 2 (Appendix Diagnostics)**, and **Replaced / Reference Figures** (which are superseded by high-density LaTeX tables).

---

## 1. Tier 1: Main Body Figures (Primary Paper)

These core figures directly substantiate the primary empirical claims in the main paper:

| Figure File | Paper Section | Core Claim & Empirical Finding |
| :--- | :--- | :--- |
| **`results_08_accuracy_vs_structure_tradeoff.png`** | \S 9 / \S 10 (*Results & Discussion*) | **Flagship Pareto Frontier:** Maps Exact-Match Accuracy vs. Structural Instruction-Following Score (bubble size = GT recall). Demonstrates that Full DPO sits on the optimal frontier (29% EM, 97.2% struct), SFT achieves 100% format compliance at the cost of accuracy (23%), and KTO achieves highest latent recall (66%) but undergoes structural collapse (21.2%). |
| **`judge_01_error_taxonomy.png`** | \S 7 (*Judge Error Analysis*) | **Perceptual Bottleneck:** 9-category taxonomy from 2,920 Meta PRM Judge critiques. Proves that VLM reasoning failures are overwhelmingly perceptual (Axis misreads 24.0% + Series/color 19.5% = 43.5%) rather than logical or arithmetic (1.3%). |
| **`judge_02_error_by_step_depth.png`** | \S 7 (*Judge Error Analysis*) | **Temporal Error Evolution:** Tracks failure composition across reasoning depth (Step 0 to Step 4+). Shows Step 0 is ~79% Axis/layout misreading, which subsequently cascades into logical inconsistency and ranking errors in later steps. *(Can be paired with `judge_01` as a 2-panel figure).* |
| **`08_error_cascade.png`** | \S 5 (*Reasoning Trajectory Dynamics*) | **The Error Cascade:** Quantifies conditional failure probability ($P(\text{Step}_{N+1}=0 \mid \text{Step}_N=0) = 82.7\%$). Proves that once visual grounding fails at Step $N$, the model almost never recovers in subsequent steps, providing foundational motivation for PRMs. |
| **`02_score_progression.png`** | \S 5 (*Reasoning Trajectory Dynamics*) | **The Reasoning Cliff:** Average accuracy by step index plummets sharply from 73% at Step 0 down to 37% at Step 1 and 26% at Step 3. *(Can be paired with `08_error_cascade` as a 2-panel figure).* |
| **`prm_best_of_n_accuracy.png`** | \S 4 / \S 6 (*LLM-as-a-Judge & Verification*) | **Inference-Time PRM Verifier:** Evaluates selection strategies across 309 multi-rollout questions. PRM Best-of-N reaches **27.5%**, significantly outperforming majority voting / self-consistency (**21.0%**, +6.5 pp) and random rollouts (**18.4%**, +9.1 pp). |
| **`results_03_error_mode_hallucination_breakdown.png`** | \S 9 / \S 10 (*Results & Discussion*) | **Holdout Error Composition:** 100% stacked bar chart partitioning predictions into Extracted Correct, Unextracted Correct in prose, Mentions GT but commits wrong, and Wrong Committed (hallucination proxy). |
| **`results_01_overall_model_comparison.png`** | \S 9 / \S 10 (*Results & Discussion*) | **Benchmark Summary Bar Chart:** Grouped 4-metric comparison across Base, SFT, Full DPO, Step-DPO, KTO, and SFT→DPO. (Optional visual counterpart to Table 1). |

---

## 2. Tier 2: Appendix Figures (Supplementary Diagnostics)

These figures provide valuable secondary analysis and optimization diagnostics for the Appendix:

| Figure File | Appendix Topic | Description |
| :--- | :--- | :--- |
| **`results_04_training_dynamics_loss_rewards.png`** | \S Training Dynamics | 4-panel training dynamics grid: SFT cross-entropy loss decay, DPO loss trajectories, Step-DPO implicit reward margin explosion ($\Delta r \to +10.0$), and KTO margin oscillations under single-T4 GPU constraints. |
| **`judge_07_multilabel_cooccurrence.png`** | \S Compound Error Co-occurrence | Multi-label co-occurrence matrix (N=2,920 critiques). Proves compound failure coupling: 460 co-occurrences between logic errors and series/color confusion, and 314 with axis misreads. |
| **`results_02_structure_instruction_following.png`** | \S Structural Compliance Details | Breakdown of individual instruction-following components (Overall Score, `Step 1:`, `Step 2:`, plain `Final Answer:`, Conversational Preamble Penalty). Details the mechanism behind KTO's formatting collapse. |
| **`judge_05_error_taxonomy_by_domain.png`** | \S Domain Failure Distributions | Heatmap of error distributions across 8 CharXiv academic disciplines (Physics and Econ suffer >34% axis misreads; Math and Stat suffer >22% hallucinations). |

---

## 3. Omitted / Superseded Figures (Replaced by Tables or Excluded)

The following figures have low informational density, high visual overhead, or lack actionable decision boundaries and are either replaced by concise LaTeX tables or excluded per scientific visualization standards:

* **`judge_08_umap_embeddings.png`**: (Critique text UMAP scatter). Excluded because the 2D projection of 2,920 short sentence embeddings forms an uninformative point cloud without distinct decision boundaries.
* **`judge_04_top_keywords_per_category.png`**: (6-panel TF-IDF bar chart). Replaced by **Table 2** (*9-Category PRM Judge Error Taxonomy*), which includes the top keywords alongside representative judge quotes in a compact table.
* **`04_first_error_position.png`**: (Single-distribution bar chart with step 13 outlier). Replaced by **Table 3** (*Reasoning Trajectory & Verification Statistics*), which reports that 79.7% of errors occur at Step $\le 1$.
* **`01_overall_accuracy.png` & `03_rollout_success.png`**: (2-bar charts showing step pass rate 41% and rollout success 12.1%). Replaced by **Table 3**.

---

## 4. LaTeX Integration Map for `report/acl_latex.tex`

```latex
% --- Section 5: Reasoning Trajectory Dynamics (2-Panel Composite) ---
\begin{figure*}[t]
  \centering
  \begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{../charts/report_selected/02_score_progression.png}
    \caption{Score Progression Cliff by Step Index}
  \end{subfigure}\hfill
  \begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{../charts/report_selected/08_error_cascade.png}
    \caption{Downstream Error Cascade ($P(\text{Step}_{N+1} \mid \text{Step}_N)$)}
  \end{subfigure}
  \caption{\textbf{Reasoning Trajectory Dynamics.} (a) Accuracy plummets from 73\% at Step 0 to 26\% at Step 3. (b) An initial error cascades into failure in 82.7\% of subsequent steps.}
  \label{fig:trajectory_dynamics}
\end{figure*}

% --- Section 6: PRM Best-of-N Verification ---
\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\columnwidth]{../charts/report_selected/prm_best_of_n_accuracy.png}
  \caption{\textbf{PRM Test-Time Verifier Performance.} Step-level process reward selection (27.5\%) outperforms majority voting (21.0\%) and random rollouts (18.4\%) on 309 multi-candidate questions.}
  \label{fig:prm_best_of_n}
\end{figure}

% --- Section 7: PRM Judge Error Taxonomy ---
\begin{figure*}[t]
  \centering
  \begin{subfigure}{0.52\textwidth}
    \includegraphics[width=\textwidth]{../charts/report_selected/judge_01_error_taxonomy.png}
    \caption{Distribution of 2,920 Refuted Reasoning Steps}
  \end{subfigure}\hfill
  \begin{subfigure}{0.46\textwidth}
    \includegraphics[width=\textwidth]{../charts/report_selected/judge_02_error_by_step_depth.png}
    \caption{Failure Mode Evolution Across Step Depth}
  \end{subfigure}
  \caption{\textbf{PRM Judge Error Taxonomy.} Failures are dominated by perceptual errors (axis and legend misreadings = 43.5\%) rather than arithmetic (1.3\%), with Step 0 forming the primary visual bottleneck.}
  \label{fig:error_taxonomy}
\end{figure*}

% --- Section 9: Alignment Pareto Frontier ---
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{../charts/report_selected/results_08_accuracy_vs_structure_tradeoff.png}
  \caption{\textbf{Accuracy vs. Structural Compliance Trade-Off.} Full DPO achieves the optimal Pareto frontier, SFT maximizes structure at the expense of accuracy, and KTO exhibits structural collapse.}
  \label{fig:pareto_frontier}
\end{figure}

% --- Section 10: Holdout Error Composition ---
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{../charts/report_selected/results_03_error_mode_hallucination_breakdown.png}
  \caption{\textbf{Holdout Error Mode Breakdown.} Partitioning of model outputs across extracted correctness, prose mention recall, and wrong commitment (hallucination proxy).}
  \label{fig:error_modes}
\end{figure}
```

