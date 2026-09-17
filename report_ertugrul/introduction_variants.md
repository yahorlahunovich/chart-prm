# Introduction Section Candidates for ChartPRM Report

This document records the three publication-ready Introduction section variants crafted to merge background literature (Process Reward Models, CharXiv benchmark, Qwen2.5-VL-3B, and alignment algorithms) directly into the Introduction without requiring a standalone Related Work section.

---

## Variant 1: Classical Top-Tier ML Style (Narrative, Rigorous & Comprehensive)

```latex
\section{Introduction}
\label{sec:introduction}

Interpreting scientific literature requires multimodal models to reason over complex visual data, including multi-panel figures, varied coordinate systems, and dense data distributions. While frontier Multimodal Large Language Models (MLLMs) demonstrate impressive perceptual capabilities, smaller, deployable models frequently fail when required to execute multi-step reasoning over charts. A primary cause of this failure lies in the training paradigm: standard Outcome Reward Models (ORMs) provide sparse, sequence-level supervision that credits or penalizes only the final answer \citep{uesato2022solving}. In complex reasoning chains, sparse feedback cannot identify the exact point of failure—frequently penalizing sound logical deduction following an early perceptual misread, or reinforcing flawed reasoning that arrived at the correct answer by chance.

To resolve this credit-assignment bottleneck, Process Reward Models (PRMs) provide dense, step-by-step verification across the reasoning trajectory \citep{lightman2023prm}. While process supervision has shown significant promise in pure text and mathematical domains, its mechanics in multimodal vision-language tasks remain under-explored. Chart reasoning introduces unique challenges: reasoning steps alternate between visual grounding (e.g., reading axis ticks, mapping legends to colors) and symbolic calculation (e.g., computing relative ratios, ranking extrema). 

In this work, we investigate step-level process supervision and post-training alignment for complex multimodal chart reasoning. We conduct our investigation on the challenging \textbf{CharXiv} benchmark \citep{wang2024charxiv}, which curates scientific charts from arXiv papers across diverse academic domains. CharXiv distinguishes between two task categories: simple \textit{descriptive} questions (surface-level perceptual lookups) and complex \textit{reasoning} questions requiring multi-step visual and mathematical deductions. To focus strictly on high-level cognitive tasks, we extract a balanced, stratified subset of \textbf{500 reasoning questions} across eight scientific disciplines (e.g., Computer Science, Physics, Economics, Quantitative Biology) and 62 distinct chart types.

Operating under strict single-GPU compute constraints (a single 16GB Nvidia T4), we evaluate the compact \texttt{Qwen2.5-VL-3B-Instruct} model \citep{bai2024qwen2vl}. We generate multi-step reasoning rollouts and score each intermediate step using a multimodal LLM-as-a-judge (\texttt{muse-spark-1.1}) to provide binary grounding verdicts and natural language critiques. Analyzing 2,920 refuted steps reveals an important empirical insight: model failures are overwhelmingly dominated by early visual perception errors (axis and layout misreads account for 24.0\%, legend/color confusion for 19.5\%, and hallucinated labels for 15.2\%), whereas intermediate arithmetic mistakes account for only 1.3\%. Furthermore, we find that error propagation is catastrophic: making an error at step $N$ leads to downstream failure in $\sim$83\% of subsequent steps.

Leveraging these step-level diagnostics, we train and benchmark five post-training alignment strategies:
\begin{enumerate}[noitemsep,topsep=0pt]
    \item \textbf{Supervised Fine-Tuning (SFT)}: Training on filtered, fully correct reasoning trajectories.
    \item \textbf{Direct Preference Optimization (DPO)}: Whole-sequence pairwise preference optimization \citep{rafailov2023dpo}.
    \item \textbf{Step-DPO}: Step-wise preference optimization utilizing divergence-masked loss to target the exact step of failure \citep{lai2024stepdpo}.
    \item \textbf{Kahneman-Tversky Optimization (KTO)}: Unpaired alignment optimizing per-step prospect utility without requiring paired rollouts \citep{ethayarajh2024kto}.
    \item \textbf{Sequential SFT$\rightarrow$DPO}: Initial SFT formatting warm-up followed by preference alignment.
\end{enumerate}

Our contributions are summarized as follows:
\begin{itemize}[noitemsep,topsep=0pt]
    \item We construct an empirical 9-category error taxonomy from 2,920 PRM judge critiques on CharXiv reasoning charts, proving that visual perception bottlenecks vastly outweigh chain-of-thought arithmetic errors.
    \item We quantify step-level error cascades, demonstrating that $>80\%$ of initial mistakes occur within the first two reasoning steps.
    \item We benchmark five alignment paradigms on a 100-question held-out test set, showing that Full DPO achieves the highest accuracy (29\% vs. 26\% base), while SFT secures 100\% formatting compliance but lowers reasoning accuracy (23\%).
    \item We map the Pareto trade-off between instruction-following fidelity and latent ground-truth recall, uncovering why unpaired methods (KTO) recall correct facts in 66\% of cases but suffer from structural output collapse.
\end{itemize}
```

---

## Variant 2: Diagnostic & Problem-First Style (Punchy, Data-Driven & Direct)

```latex
\section{Introduction}
\label{sec:introduction}

When a compact Vision-Language Model (VLM) fails to answer a complex question about a scientific chart, where does the error actually occur? Does the model fail to read the visual coordinates, misidentify the legend curves, or make an arithmetic error during intermediate deductions? Standard outcome-based evaluation (ORM) is blind to these distinctions: it delivers a single scalar reward at the end of the sequence, treating all erroneous trajectories identically \citep{uesato2022solving}.

To open this black box, Process Reward Models (PRMs) provide dense, per-step verification that pinpoints where a reasoning chain diverges from truth \citep{lightman2023prm}. While step supervision has achieved state-of-the-art results in mathematical reasoning, applying PRMs to multimodal chart reasoning presents distinct open challenges. In chart question answering, visual perception and symbolic reasoning are tightly coupled: an initial perceptual misread invalidates all downstream logical inferences.

In this paper, we conduct a systematic study of step-level process verification and preference optimization for complex chart reasoning. Our study is built upon the **CharXiv** benchmark \citep{wang2024charxiv}, a rigorous collection of real-world scientific figures from arXiv papers. While CharXiv contains both simple \textit{descriptive} queries (e.g., reading a direct legend label) and multi-step \textit{reasoning} queries (e.g., synthesizing trends across subplots), we deliberately isolate a balanced, domain-stratified subset of **500 reasoning questions** spanning 8 academic disciplines and 62 chart types. 

Focusing on the lightweight \texttt{Qwen2.5-VL-3B-Instruct} architecture \citep{bai2024qwen2vl} under strict single-GPU (Nvidia T4) compute constraints, we generate 1,287 multi-step rollouts ($N=4,947$ steps) and evaluate each step with a multimodal LLM-as-a-judge (\texttt{muse-spark-1.1}). Through semantic clustering and regex parsing of 2,920 failed step critiques, we discover that:
\begin{enumerate}[noitemsep,topsep=0pt]
    \item \textbf{Perception Dominates Logic}: Over 58\% of all step failures originate from visual grounding errors—primarily axis misreads (24.0\%), series/color confusion (19.5\%), and hallucinated labels (15.2\%). Pure calculation errors represent only 1.3\%.
    \item \textbf{Early Cascading Failure}: Initial errors occur early in the chain ($>80\%$ in Steps 0–1). Once an error occurs at step $N$, the likelihood of failure at step $N+1$ rises to $\sim$83\%.
\end{enumerate}

Building on these insights, we systematically evaluate post-training alignment techniques to determine how small VLMs can best learn from step feedback. We compare Supervised Fine-Tuning (SFT), Sequence-level DPO \citep{rafailov2023dpo}, Step-DPO with divergence masking \citep{lai2024stepdpo}, Kahneman-Tversky Optimization (KTO) \citep{ethayarajh2024kto}, and sequential SFT$\rightarrow$DPO. Evaluated on a 100-question holdout set, Full DPO delivers the top exact-match score (29\% vs. 26\% base), while SFT guarantees 100\% structure compliance but over-regularizes accuracy (23\%). Conversely, KTO recalls the correct answer in text 66\% of the time but fails to adhere to output formatting constraints.

In summary, our work provides:
\begin{itemize}[noitemsep,topsep=0pt]
    \item The first empirical step-level error taxonomy for multimodal chart reasoning across 2,920 natural language PRM critiques.
    \item A thorough analysis of reasoning trajectory degradation and error cascade probabilities in compact VLMs.
    \item A unified benchmark comparing five alignment strategies on CharXiv under accessible single-GPU hardware.
    \item Actionable insights on the fundamental trade-off between structural instruction following and latent reasoning recall in preference-tuned VLMs.
\end{itemize}
```

---

## Variant 3: Algorithmic & Systems Perspective (Focus on Alignment Mechanics & Efficiency)

```latex
\section{Introduction}
\label{sec:introduction}

Aligning small Vision-Language Models (VLMs) for complex scientific reasoning is a central goal for accessible and verifiable AI. However, multi-step chart question answering represents a notorious failure mode for compact models ($<7\text{B}$ parameters), where models must simultaneously parse non-standard visual encodings, extract coordinate values, and execute logical arithmetic. Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) typically optimize for sequence-level outcomes, ignoring the intermediate mechanics of multi-step failure \citep{uesato2022solving, rafailov2023dpo}.

Process Reward Modeling (PRM) offers a principled remedy by decomposing long reasoning chains into discrete, verifiable steps \citep{lightman2023prm}. By providing granular step-level feedback, PRMs enable fine-grained credit assignment. Recently, step-level preference alignment algorithms such as Step-DPO \citep{lai2024stepdpo} and prospect-theoretic methods like KTO \citep{ethayarajh2024kto} have emerged to optimize models directly from step preferences. Yet, how these alignment strategies translate to multimodal reasoning under constrained compute budgets remains an open question.

In this work, we establish an end-to-end framework for analyzing and aligning compact VLMs on scientific chart reasoning. We benchmark \texttt{Qwen2.5-VL-3B-Instruct} \citep{bai2024qwen2vl} using 4-bit quantization and LoRA adaptation \citep{hu2021lora, dettmers2023qlora}, strictly bounded by single-T4 GPU resources. We evaluate our methods on the \textbf{CharXiv} benchmark \citep{wang2024charxiv}. Unlike descriptive chart tasks that test simple OCR extraction, CharXiv features challenging reasoning queries requiring compositional deduction across complex plots. We sample a balanced, stratified benchmark of \textbf{500 reasoning questions} across eight academic domains (CS, Math, Physics, Economics, Finance, Biology, Engineering, Statistics).

Using an advanced multimodal LLM-as-a-judge (\texttt{muse-spark-1.1}) following the LLM-as-a-judge paradigm \citep{zheng2023judging}, we grade 1,287 reasoning rollouts (4,947 individual steps) with binary scores and natural language justifications. Our analysis yields key structural findings:
\begin{itemize}[noitemsep,topsep=0pt]
    \item \textbf{Visual Perception as the Core Bottleneck}: Categorizing 2,920 failed steps into a 9-class taxonomy shows that axis misreads (24.0\%), legend swapping (19.5\%), and hallucinated data entities (15.2\%) account for nearly 60\% of errors. In contrast, intermediate arithmetic errors represent only 1.3\%.
    \item \textbf{Error Propagation Dynamics}: We demonstrate that errors cascade rapidly: early reasoning steps (Steps 0–1) account for over 80\% of first failures, and an incorrect step propagates to subsequent steps with an 83\% failure rate.
\end{itemize}

We then benchmark five post-training alignment algorithms: SFT on verified positive rollouts, sequence-level DPO, Step-DPO with divergence suffix masking, KTO on unpaired step completions, and sequential SFT$\rightarrow$DPO. On a 100-question holdout evaluation, Full DPO achieves the highest accuracy at 29\% (surpassing the 26\% base model), while SFT guarantees perfect step formatting at 23\% accuracy. Furthermore, KTO demonstrates remarkable latent ground-truth recall (66\% of generations contain the correct answer) but suffers from formatting degeneration.

Our contributions are threefold:
\begin{enumerate}[noitemsep,topsep=0pt]
    \item We provide an empirical error diagnostic and cascading analysis of compact VLM reasoning across 500 CharXiv charts.
    \item We implement and benchmark five distinct preference alignment pipelines within a reproducible, single-GPU compute budget.
    \item We characterize the structural compliance versus reasoning accuracy trade-off across paired and unpaired alignment paradigms.
\end{enumerate}
```
