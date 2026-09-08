# ChartQA Training Log: SFT->DPO (method 7/7, final)

Kaggle kernel `qwen-vl-sft-dpo-chartqa` (T4x2 requested). Full-trajectory DPO
initialized from the ChartQA SFT LoRA (method 1/7), frozen SFT as the DPO
reference -- same trainer, hyperparameters (lr=2e-6, beta=0.1, 1 epoch, batch
size 1, warn-only collapse guard at 70 nats) as CharXiv's SFT->DPO run.

The SFT adapter is gitignored, so it was uploaded as a private Kaggle Dataset
(`chartqa-sft-adapter`) after method 1/7 finished, and located inside the kernel
by globbing for `adapter_config.json` under `/kaggle/input` rather than a
hardcoded mount path (log confirms: `Using SFT init adapter:
/kaggle/input/datasets/ertugrultaparci/chartqa-sft-adapter`).

- Dataset: `experiments/020_chartqa_transfer/data/dpo_pairs.jsonl` (127 pairs,
  same file Full DPO trained on).
- 127 training steps, 1 epoch.
- Loss: mean first 20 steps 0.691 -> mean last 20 steps 0.677.
- **Preference accuracy (tail 50% of steps): 76.6% -- the highest of all five
  DPO-family methods trained on ChartQA** (Full DPO 65.6%, Step-DPO 95.5% but
  that's a different, easier suffix-only task, Pareto-DPO 60.9%, SFT->DPO 76.6%).
- Reward margin: mean first 20 steps 0.004 -> mean last 20 steps 0.035 (small,
  but consistent with a very stable run).
- Collapse guard: max chosen_logp drop only 1.34 nats, far below the 70-nat warn
  threshold -- the most stable full-trajectory DPO run of the five, plausibly
  because SFT-initialized weights already produce confident, well-formed
  completions close to what DPO is reinforcing.
- Adapter saved (`adapters/qwen_vl_sft_dpo_chartqa_adapter/`, gitignored).

**All 7 methods are now trained on ChartQA**: SFT, Full DPO, Step-DPO, KTO,
SimPO, Pareto-DPO, SFT->DPO. Next stage: holdout evaluation on the 30-question
ChartQA holdout, comparing all 7 + base Qwen, same metrics as the CharXiv
6-system table (exact-match, token match, structure, hallucination proxy).
