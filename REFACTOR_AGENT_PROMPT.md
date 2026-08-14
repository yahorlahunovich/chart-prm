# Autonomous Coding Agent Directive: ChartPRM Codebase Restructuring & Cleanup

## 1. Context & Objective
You are tasked with refactoring and cleaning the **ChartPRM** repository. The repository has evolved from an initial exploration of LLM-as-a-Judge for multimodal reasoning on CharXiv into a full alignment research framework containing **SFT**, **Full-Trajectory DPO**, **Suffix Step-DPO**, **KTO**, and **Sequential SFT→DPO**.

Your objective is to systematically reorganize the codebase, clean up root-level clutter, categorize operational scripts, update documentation to research-grade standards, ensure all unit tests pass, document your changes in `implementation_log.md`, and commit/push all modifications to GitHub.

---

## 2. Hard Constraints & Operating Rules
1. **Dependency Management**: NEVER manually edit `pyproject.toml` or `uv.lock`. If dependencies are needed, use `uv add <dependency>`.
2. **Compute Constraints**: All model loading, batch sizes, and scripts must respect memory limits of **1x NVIDIA T4 (16GB)** on Colab and **2x T4 / 1x P100** on Kaggle.
3. **Dataset Scope**: The project strictly uses a balanced subset of 500 **reasoning questions** from CharXiv (and 100 holdout questions). Descriptive questions are ignored.
4. **Test Suite Integrity**: You MUST run `uv run pytest` before and after all changes. All 49 existing tests must pass without regression.
5. **Logging**: Document all actions, rationale, and modifications in `implementation_log.md`.
6. **Git Workflow**: Commit your changes with descriptive messages and push directly to GitHub (`git push`).

---

## 3. Target Directory Layout

Reorganize the repository to match this clean structure:

```text
chart-prm/
├── adapters/                           # LoRA adapter checkpoint dirs (gitignored)
│   ├── qwen_vl_sft_adapter/
│   ├── qwen_vl_dpo_adapter/
│   ├── qwen_vl_step_dpo_adapter/
│   ├── qwen_vl_kto_adapter/
│   └── qwen_vl_sft_dpo_adapter/
│
├── data/                               # Dataset splits, raw JSON, and chart images
│   ├── CharXiv/
│   │   ├── CharXiv.json
│   │   └── images/
│   └── splits/
│       ├── main_reasoning_ids.json     # 500 balanced reasoning train IDs
│       └── eval_reasoning_ids.json     # 100 holdout reasoning eval IDs
│
├── experiments/                        # Frozen benchmark runs and artifacts
│   ├── 001_500_reasoning/
│   ├── 002_holdout_eval/
│   ├── 003_holdout_eval_full_traj/
│   ├── 004_holdout_eval_step_dpo_kto_v12/
│   ├── 005_holdout_eval_suffix_step_dpo/
│   ├── 006_sft_then_dpo/
│   └── 007_sft_dpo_holdout/
│
├── logs/                               # Execution and training logs (gitignored)
│   ├── qwen-vl-sft-custom.log
│   ├── qwen-vl-step-dpo-custom.log
│   └── qwen-vl-kto-custom.log
│
├── notebooks/                          # Interactive analysis & prototyping
│   ├── analyze_judge_errors.ipynb
│   ├── evaluate_dpo_model.ipynb
│   ├── evaluate_rollouts.ipynb
│   ├── model_inference.ipynb
│   ├── train_dpo.ipynb
│   ├── train_kto.ipynb
│   └── train_sft.ipynb
│
├── scripts/                            # Operational CLI scripts by pipeline stage
│   ├── data_prep/                      # Ingestion, filtering, and dataset formatting
│   │   ├── sample_questions.py
│   │   ├── download_images.py
│   │   ├── clean_dataset.py
│   │   ├── fix_jsonl_ids.py
│   │   ├── format_sft.py
│   │   ├── format_full_dpo.py
│   │   ├── format_step_dpo.py
│   │   └── format_kto.py
│   │
│   ├── train/                          # Core training entry points
│   │   ├── train_sft.py
│   │   ├── train_dpo.py
│   │   └── train_kto.py
│   │
│   ├── evaluation/                     # Scoring, PRM judge, quality metrics
│   │   ├── evaluate_rollouts_meta.py
│   │   ├── categorize_judge_errors.py
│   │   ├── analyze_holdout_quality.py
│   │   └── merge_sft_dpo_holdout.py
│   │
│   ├── tools/                          # Generator utilities and verification tools
│   │   ├── generate_style_example.py
│   │   ├── verify_sft_dpo.py
│   │   ├── create_notebook.py
│   │   └── create_kto_notebook.py
│   │
│   └── kaggle/                         # Remote Kaggle execution kernels & metadata
│       ├── kaggle_eval_holdout/
│       ├── kaggle_judge_errors/
│       ├── kaggle_train_dpo/
│       ├── kaggle_train_kto/
│       ├── kaggle_train_sft/
│       ├── kaggle_train_sft_dpo/
│       └── kaggle_train_step_dpo/
│
├── src/chart_prm/                      # Core reusable package
│   ├── __init__.py
│   ├── adapter_resolve.py              # Collision-free adapter loader
│   ├── data_guards.py                  # Training validation & logprob collapse guards
│   ├── evaluate.py                     # Exact-match and answer normalization
│   ├── generator.py                    # Qwen2.5-VL generation utilities
│   ├── holdout_merge.py                # Multi-adapter holdout merger
│   ├── holdout_metrics.py              # Structure, token match & hallucination proxy
│   ├── label_mask.py                   # Right-padded loss masking
│   ├── sft_dpo_init.py                 # Multi-adapter LoRA copy for SFT->DPO
│   ├── sft/                            # SFT trainer, loss, collator
│   ├── dpo/                            # DPO trainer, loss, prefix-masking collator
│   ├── kto/                            # KTO trainer, Kahneman-Tversky loss, collator
│   └── visualization/                  # Academic plotting styles & figures
│
├── tests/                              # Unit & integration test suite (49 tests)
├── .gitignore
├── pyproject.toml
├── uv.lock
├── implementation_log.md
└── README.md
```

---

## 4. Step-by-Step Execution Instructions

### Step 1: Create Directories & Update `.gitignore`
1. Create new directories:
   - `adapters/`
   - `logs/`
   - `scripts/data_prep/`
   - `scripts/train/`
   - `scripts/evaluation/`
   - `scripts/tools/`
   - `scripts/kaggle/`
2. Update `.gitignore` to ignore:
   - `adapters/`
   - `logs/`
   - `*.log`

### Step 2: Reorganize Files
1. **Move Root Log Files**:
   - Move `qwen-vl-*.log` into `logs/`.
2. **Move Root Adapter Directories**:
   - Move `qwen_vl_sft_adapter/`, `qwen_vl_dpo_adapter/`, `qwen_vl_step_dpo_adapter/`, `qwen_vl_kto_adapter/`, `qwen_vl_sft_dpo_adapter/` into `adapters/`.
3. **Move Training Scripts**:
   - Move `train_sft.py`, `train_dpo.py`, `train_kto.py` into `scripts/train/`.
   - *(Optional convenience)*: Create lightweight root shim/wrappers if desired or update CLI paths.
4. **Move Scripts to Subdirectories**:
   - `scripts/sample_questions.py` → `scripts/data_prep/sample_questions.py`
   - `scripts/download_images.py` → `scripts/data_prep/download_images.py`
   - `scripts/clean_dataset.py` → `scripts/data_prep/clean_dataset.py`
   - `scripts/fix_jsonl_ids.py` → `scripts/data_prep/fix_jsonl_ids.py`
   - `scripts/format_sft.py` → `scripts/data_prep/format_sft.py`
   - `scripts/format_full_dpo.py` → `scripts/data_prep/format_full_dpo.py`
   - `scripts/format_step_dpo.py` → `scripts/data_prep/format_step_dpo.py`
   - `scripts/format_kto.py` → `scripts/data_prep/format_kto.py`
   - `scripts/evaluate_rollouts_meta.py` → `scripts/evaluation/evaluate_rollouts_meta.py`
   - `scripts/categorize_judge_errors.py` → `scripts/evaluation/categorize_judge_errors.py`
   - `scripts/analyze_holdout_quality.py` → `scripts/evaluation/analyze_holdout_quality.py`
   - `scripts/merge_sft_dpo_holdout.py` → `scripts/evaluation/merge_sft_dpo_holdout.py`
   - `scripts/generate_style_example.py` → `scripts/tools/generate_style_example.py`
   - `scripts/verify_sft_dpo.py` → `scripts/tools/verify_sft_dpo.py`
   - `scripts/create_notebook.py` → `scripts/tools/create_notebook.py`
   - `scripts/create_kto_notebook.py` → `scripts/tools/create_kto_notebook.py`
   - Move all `scripts/kaggle_*` folders into `scripts/kaggle/`.
5. **Delete Obsolete Files**:
   - Remove `scripts/prepare_dpo.py` (deprecated exploration code superseded by `format_full_dpo.py`).

### Step 3: Check & Fix Internal Imports and File Paths
1. Verify if any scripts or tests reference moved files by relative paths (e.g. `../experiments/` or `data/`). Ensure default path arguments correctly resolve from repository root or current directory.
2. Verify all `tests/` import statements.
3. Run `uv run pytest` to ensure all 49 tests pass.

### Step 4: Update `README.md`
Overwrite `README.md` with the publication-grade research documentation including:
- Project title, badges, and abstract.
- Complete Benchmark Results Table comparing **Base, SFT, Full DPO, Suffix Step-DPO, KTO, and SFT→DPO** on 100 holdout reasoning questions.
- Key findings on format adherence vs. preference alignment vs. hallucination proxy.
- ASCII/Mermaid Pipeline Architecture diagram.
- Step-by-step reproduction instructions with the new `scripts/` paths.
- Hardware constraints and configuration details.

### Step 5: Test Suite Verification
Run the full test suite:
```bash
uv run pytest
```
Ensure 49 tests pass. If any test fails due to path assumptions, fix the test or utility path resolution.

### Step 6: Documentation & Git Commit / Push
1. Append an entry to `implementation_log.md` detailing:
   - **What**: Directory reorganization, script modularization, log/adapter cleanup, README overhaul.
   - **Why**: Transition codebase to a clean, professional research structure with clear separation of data prep, training, evaluation, and tooling.
2. Stage and commit all changes:
   ```bash
   git add -A
   git commit -m "refactor: restructure codebase, categorize scripts, and update research README"
   git push origin main
   ```

---

## 5. Success Criteria
- [ ] All root log files and adapter folders are properly relocated and gitignored.
- [ ] Scripts are cleanly partitioned in `scripts/{data_prep,train,evaluation,tools,kaggle}`.
- [ ] Obsolete script `scripts/prepare_dpo.py` is removed.
- [ ] `README.md` is comprehensive, accurate, and reflects the full 6-model research pipeline.
- [ ] `uv run pytest` passes 100% (49/49 tests).
- [ ] Changes are logged in `implementation_log.md`.
- [ ] Git commit and push completed cleanly.
