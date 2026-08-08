

1. Prepare everything for generating reasoning steps: length of tokens, format, final answer
We need to make sure that model generates real reasoning steps and formats final output well.
2. Parsing
3. Determine the evaluation process. 
- Which model to use as a judge?
- Which criteria to use? Or maybe generate the criteria?
- How to evaluate the steps? Binary or not? Criteria to use
- 
4. PRM Judge
5. Evaluation
6. Analysis of results. Metrics.

7. [x] Optional: fine-tuning (Proceeding with Step-DPO on Kaggle using QLoRA)


- Reproducibility: declare dependencies, not only notebooks
- 

---
  1. Data Efficiency in Small Regimes:
      • We only have 187 paired DPO samples because many prompts produced
      only incorrect rollouts or only correct rollouts.
      • KTO allows us to use all 1,287 evaluated rollouts (treating all 1-
      scored rollouts as positive and 0-scored rollouts as negative). Does
      KTO outperform DPO simply because it leverages 5x more data from the
      same inference budget?
  2. Negative Signal vs. Positive Imitation:
      • Does explicit preference optimization (DPO/KTO) prevent common chart
      reasoning traps (like reading the wrong line color or executing bad
      arithmetic) better than standard SFT imitation?
  3. Step-Level vs. Sequence-Level Alignment:
      • Step-DPO provides dense feedback at the exact step where reasoning
      diverged. Comparing Step-DPO against standard SFT/KTO shows whether
      fine-grained process rewards are superior to outcome rewards.
---
Ideas:
- Evaluate 3 fine-tuning methods on test dataset similiar to train dataset using API
- Use text analysis for every step to identify main problems in answers and compare with fine-tuned methods
- compare results based on different metrics: structure, correctenes, hallucinations
