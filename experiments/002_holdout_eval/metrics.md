# Holdout Eval Metrics (100 CharXiv reasoning questions)

Source kernel: `egorlagunovich/qwen-vl-holdout-eval` (COMPLETE)  
Protocol: greedy decoding (`do_sample=False`), shared reasoning prompt, exact-match on extracted `Final Answer:`.

| System | Exact match | Extracted `Final Answer:` rate | Notes |
|--------|-------------|----------------------------------|-------|
| Base   | **26 / 100 (26%)** | 100% | Proper `Step N:` + `Final Answer:` format |
| SFT    | 0 / 100 (0%) | 0% | Short free-form answers; no required format |
| DPO    | 0 / 100 (0%) | 0% | **Empty generations for all 100 IDs** |
| KTO    | 3 / 100 (3%) | 17% | Mostly unformatted / incomplete answers |

Artifacts: `data/holdout_generations.jsonl`, `data/holdout_accuracy.json`.
