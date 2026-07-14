# Implementation Log

This file tracks the step-by-step implementation of the ChartPRM project. Every major step should be recorded here with details on *what* was implemented and *why*.

## Initial Setup
- **What**: Initialized project structure, created agent instructions (`agents/instructions.md`, `.cursorrules`, `.agents/AGENTS.md`), created `implementation_log.md`, and updated `README.md` with workflow details.
- **Why**: To establish a standardized workflow for AI agents (Cursor, Gemini, Claude, Antigravity) and ensure clean collaboration between team members.

## Project Structure Setup
- **What**: Created `src/chart_prm/`, `notebooks/`, and `scripts/` directories with placeholder files mapping directly to the project's pipeline (Generation -> Splitting -> Scoring -> Analysis). Created `pyproject.toml` base file and documented the structure in `README.md`.
- **Why**: To separate experimental code (`.ipynb`) from reusable pipeline code (`.py`) and ensure maintainable scaling, allowing modules to be imported cleanly in bulk processing scripts or interactive notebooks.

## Evaluation & Data Setup
- **What**: Removed unused blank files. Downloaded CharXiv dataset into `data/CharXiv` by cloning its repository. Modified the pre-existing `evaluate.py` to point `CHARXIV_PATH` correctly to this new data directory.
- **Why**: The user requested that we do not clutter the repository with blank files and only create files when needed. The CharXiv dataset is required for the evaluation pipeline to work, and `evaluate.py` needed an absolute path fix to dynamically resolve the CharXiv data directory from the project root.

## Agent Constraints Update
- **What**: Added strict compute constraints (T4 on Colab, 2xT4 on Kaggle) to all agent instruction files (`agents/instructions.md`, `.cursorrules`, `.agents/AGENTS.md`).
- **Why**: To ensure any generated model code, precision settings (like 4-bit quantization), and batch sizes accommodate the limited GPU VRAM.

## Reasoning Prompt Setup
- **What**: Added `build_generation_prompt` function to `src/chart_prm/generator.py` adapting the base chart QA prompt to enforce step-by-step reasoning.
- **Why**: The original baseline prompt forced models to output only the final answer ("Do not explain"). To build a Process Reward Model, we specifically need the intermediate reasoning steps clearly demarcated for future parsing (e.g. `Step 1: `, `Step 2: `, followed by `Final Answer: `).

## Inference Notebook for Kaggle/Colab
- **What**: Created `notebooks/02_model_inference.ipynb` containing the end-to-end generation script.
- **Why**: Allows execution on remote GPUs (T4). The notebook handles cloning the repo, installing `bitsandbytes`/`qwen-vl-utils`, downloading the CharXiv dataset directly via HuggingFace (which automatically fetches the images), loading Qwen2.5-VL-3B in 4-bit precision, generating the first 100 samples, and saving checkpoints every 10 iterations.

## Notebook Fixes
- Fixed the dataset split in `02_model_inference.ipynb` from `descriptive_val` to `validation`.
- Updated dataset column names from `question` and `answer` to `reasoning_q` and `reasoning_a` respectively to match the CharXiv schema.
- Verified that `BitsAndBytesConfig` is now correctly set up in the notebook to avoid `load_in_4bit` TypeError from older implementations.
- Modified `02_model_inference.ipynb` to be fully resumable. It now saves results using append mode (`"a"`) row-by-row and skips already processed indices on restart, preventing data loss.
- Updated the notebook to automatically load the `GITHUB` and `HF_TOKEN` Colab secrets using `google.colab.userdata` and embed the GitHub token directly into the `git clone` URL.
- Fixed the `git clone` command in the notebook by prepending `yahorlahunovich:` to the embedded token, as fine-grained PATs require the actual username rather than `oauth2`.

## Prompt Enhancement
- **What**: Updated `build_generation_prompt` in `src/chart_prm/generator.py` to strictly enforce that the final answer is only the exact short value or entity.
- **Why**: The model was generating full sentences for the final answer (e.g. "The value is 10" instead of "10"), which caused downstream string-matching evaluation to falsely mark correct reasoning paths as incorrect.

## Reasoning Prompt Enhancement for PRM Evaluation
- **What**: Updated `build_generation_prompt` in `src/chart_prm/generator.py` to explicitly force the model to extract intermediate values, perform comparisons, and show math calculations.
- **Why**: The previous generated steps (checked via `generated_reasoning_steps_10.jsonl`) were just high-level plans ("Step 1: Identify the models", "Step 2: Compare..."). To train/evaluate a PRM effectively, the model must show its actual execution steps, intermediate data reading, and explicit math operations.

## Few-Shot Prompt Addition
- **What**: Added a text-based "EXAMPLE FORMAT" into the prompt inside `src/chart_prm/generator.py`.
- **Why**: To further improve reasoning quality and consistency. While the previous prompt update helped, the 3B model occasionally fell back to high-level planning. Providing a concrete, in-context example of exactly what we expect (extracting points, comparing them, writing math) is the most robust way to anchor the output for smaller models without fine-tuning.

## Dataset Pivot & Stratified Sampling
- **What**: Shifted focus entirely to reasoning questions (ignoring descriptive ones). Wrote `scripts/00_sample_questions.py` to stratify sample 500 questions for the main pipeline and 100 questions for holdout evaluation, balanced across the 62 `chart_types`. Updated the `02_model_inference.ipynb` notebook to load these explicit IDs. Updated all documentation (`README.md`, `agents/instructions.md`, `.cursorrules`, `.agents/AGENTS.md`) to reflect this limitation.
- **Why**: Descriptive questions are purely retrieval and offer no reasoning paths for a PRM to evaluate. Scaling down to 500 well-distributed reasoning questions saves VRAM/Compute on Kaggle while maintaining diversity across different chart types.

## Verification Run Setup
- **What**: Updated `notebooks/02_model_inference.ipynb` to temporarily truncate the 500 selected dataset down to just 20 samples. Confirmed that the notebook correctly implements the `apply_chat_template` wrapping required by Qwen2.5-VL. Fixed a bug in the Hugging Face dataset filtering where `question_id` was missing from the HF dataset schema, by manually mapping the HF indices to the `reasoning_val.json` keys.
## Production Generation Setup (Kaggle)
- **What**: Refactored `02_model_inference.ipynb` for production-scale Step-DPO generation on Kaggle. Replaced Google Colab secret handling with Kaggle's `UserSecretsClient`. Removed the 20-sample validation limit. Introduced sequential generation of 5 rollouts (`NUM_ROLLOUTS = 5`) per sample to prevent VRAM OOM on the T4 GPU. Enabled `do_sample=True` with `temperature=0.7` for logic path exploration. Updated output dictionary schema to include `rollout_index`.
- **Why**: Training a PRM requires multiple distinct reasoning trajectories (rollouts) for the same prompt to score them (Step-DPO). Generating them sequentially rather than in a batch is essential because a 3B Vision model takes significant memory, and returning 5 sequences at once would cause an immediate OOM on Kaggle's T4 GPUs.

## Dataset Cleaning Script
- **What**: Created `scripts/clean_dataset.py` to filter structural and generation failures from raw model rollouts, and parse them into atomic steps and a final answer.
- **Why**: The generated reasoning steps often contain structural failures (missing `Step 1:` or `Final Answer:` markers), infinite repetitions from low quantization, or raw unstructured text. A clean structure is strictly required for the downstream PRM token-level or step-level evaluation pipelines.

## Experiment Tracking Setup
- **What**: Established a self-contained `experiments/` directory structure, starting with `experiments/001_qwen2.5_vl_3b_500_reasoning/`. Moved the generated jsonl data and evaluation script artifacts here, alongside a `run_config.json` and a detailed `metrics.md`.
- **Why**: To ensure perfect reproducibility and clean project organization. Grouping the exact inference snapshot, configuration, raw data, and cleaned results prevents data loss and makes comparing future experiments much easier.
