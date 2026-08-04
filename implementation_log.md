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
- **What**: Created `notebooks/model_inference.ipynb` containing the end-to-end generation script.
- **Why**: Allows execution on remote GPUs (T4). The notebook handles cloning the repo, installing `bitsandbytes`/`qwen-vl-utils`, downloading the CharXiv dataset directly via HuggingFace (which automatically fetches the images), loading Qwen2.5-VL-3B in 4-bit precision, generating the first 100 samples, and saving checkpoints every 10 iterations.

## Notebook Fixes
- Fixed the dataset split in `model_inference.ipynb` from `descriptive_val` to `validation`.
- Updated dataset column names from `question` and `answer` to `reasoning_q` and `reasoning_a` respectively to match the CharXiv schema.
- Verified that `BitsAndBytesConfig` is now correctly set up in the notebook to avoid `load_in_4bit` TypeError from older implementations.
- Modified `model_inference.ipynb` to be fully resumable. It now saves results using append mode (`"a"`) row-by-row and skips already processed indices on restart, preventing data loss.
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
- **What**: Shifted focus entirely to reasoning questions (ignoring descriptive ones). Wrote `scripts/sample_questions.py` to stratify sample 500 questions for the main pipeline and 100 questions for holdout evaluation, balanced across the 62 `chart_types`. Updated the `model_inference.ipynb` notebook to load these explicit IDs. Updated all documentation (`README.md`, `agents/instructions.md`, `.cursorrules`, `.agents/AGENTS.md`) to reflect this limitation.
- **Why**: Descriptive questions are purely retrieval and offer no reasoning paths for a PRM to evaluate. Scaling down to 500 well-distributed reasoning questions saves VRAM/Compute on Kaggle while maintaining diversity across different chart types.

## Verification Run Setup
- **What**: Updated `notebooks/model_inference.ipynb` to temporarily truncate the 500 selected dataset down to just 20 samples. Confirmed that the notebook correctly implements the `apply_chat_template` wrapping required by Qwen2.5-VL. Fixed a bug in the Hugging Face dataset filtering where `question_id` was missing from the HF dataset schema, by manually mapping the HF indices to the `reasoning_val.json` keys.
## Production Generation Setup (Kaggle)
- **What**: Refactored `model_inference.ipynb` for production-scale Step-DPO generation on Kaggle. Replaced Google Colab secret handling with Kaggle's `UserSecretsClient`. Removed the 20-sample validation limit. Introduced sequential generation of 5 rollouts (`NUM_ROLLOUTS = 5`) per sample to prevent VRAM OOM on the T4 GPU. Enabled `do_sample=True` with `temperature=0.7` for logic path exploration. Updated output dictionary schema to include `rollout_index`.
- **Why**: Training a PRM requires multiple distinct reasoning trajectories (rollouts) for the same prompt to score them (Step-DPO). Generating them sequentially rather than in a batch is essential because a 3B Vision model takes significant memory, and returning 5 sequences at once would cause an immediate OOM on Kaggle's T4 GPUs.

## Dataset Cleaning Script
- **What**: Created `scripts/clean_dataset.py` to filter structural and generation failures from raw model rollouts, and parse them into atomic steps and a final answer.
- **Why**: The generated reasoning steps often contain structural failures (missing `Step 1:` or `Final Answer:` markers), infinite repetitions from low quantization, or raw unstructured text. A clean structure is strictly required for the downstream PRM token-level or step-level evaluation pipelines.

## Experiment Tracking Setup
- **What**: Established a self-contained `experiments/` directory structure, starting with `experiments/001_500_reasoning/`. Moved the generated jsonl data and evaluation script artifacts here, alongside a `run_config.json` and a detailed `metrics.md`.
- **Why**: To ensure perfect reproducibility and clean project organization. Grouping the exact inference snapshot, configuration, raw data, and cleaned results prevents data loss and makes comparing future experiments much easier.

## Dataset Image Downloading
- **What**: Created `scripts/download_images.py` to download the CharXiv `images.zip` from Hugging Face and extract only the 500 images corresponding to our selected reasoning questions into `data/CharXiv/images/`.
- **Why**: The user requested a local copy of the dataset images for the cleaned 500-question reasoning subset. By fetching only the required images, we save disk space and bandwidth compared to extracting the full dataset.

## JSONL ID Alignment Fix
- **What**: Wrote and executed `scripts/fix_jsonl_ids.py` to map the `question_id`s in `experiments/001_500_reasoning/data/*.jsonl` from enumeration indices (0-499) back to their true `figure_id` keys from `main_reasoning_ids.json`.
- **Why**: The HuggingFace `load_dataset` `.filter()` step drops dataset keys, causing the generation script to fall back to the loop index (`sample_index`). Consequently, the generated `.jsonl` files had mismatching IDs with the downloaded images. The mapping restores alignment so that downstream training code can properly look up `images/{question_id}.jpg`.

## Automated PRM Evaluation via Meta API
- **What**: Built an async pipeline (`scripts/evaluate_steps_meta.py`) to grade every individual reasoning step in the dataset against the ground truth using the `muse-spark-1.1` vision-language model. Included unit tests (`tests/test_evaluate_steps_meta.py`) to verify prompt construction, JSON parsing, API mocking, and interrupt-resume capability.
- **Why**: The user wanted to evaluate thousands of model rollouts step-by-step for the PRM dataset. Using `aiohttp` allows for optimal concurrency within the 3,000 requests/minute limit, and intermediate `aiofiles` saving guarantees that progress is kept if the execution crashes.

## Cost Optimization: Rollout-Level Batching
- **What**: Created `scripts/evaluate_rollouts_meta.py` which passes an entire rollout (all steps) to the Meta API in a single prompt instead of iterating step-by-step. Also added image resizing via `Pillow` (max 512px) before base64 encoding. Included updated tests in `tests/test_evaluate_rollouts_meta.py`.
- **Why**: The step-by-step script consumed ~1,600 tokens per step (mostly from the model re-generating internal "reasoning tokens" about the image for every step). By sending the image and all steps simultaneously, we reduce the API calls from 7,600 down to 1,287, saving ~70-80% on API costs and bringing the entire run comfortably below the user's remaining budget.
- **Result**: Successfully evaluated the entire dataset of 1,287 rollouts (representing ~4,947 steps). The final pass rate was ~41.0% (Score 1) and fail rate was ~59.0% (Score 0). The data was saved to `experiments/001_500_reasoning/data/evaluated_rollouts.jsonl` and pushed to GitHub.

## Post-Evaluation Cleanup
- **What**: Cleaned up the codebase to prepare for the next phases.
    - Updated `README.md` to specify that `muse-spark-1.1` from the Meta API was used as the PRM Judge.
    - Deleted deprecated/test scripts (`evaluate_steps_meta.py`, `test_muse_api.py`, `test_muse_image.py`, `test_evaluate_steps_meta.py`) to reduce clutter.
    - Added comprehensive module-level docstrings to every `.py` file in `scripts/` and `tests/` detailing their purpose.

## Analysis Notebook
- **What**: Created `notebooks/evaluate_rollouts.ipynb` to analyze the PRM evaluations. Loaded the data into a pandas dataframe and set up a publication-ready scientific plotting theme using `seaborn` with a colorblind-friendly palette. Proposing 10 ways of analyzing the PRM performance.
- **Why**: To understand model reasoning patterns, calculate PRM accuracy, identify the most common steps of failure, and produce high-quality charts.

## DPO Sample Preparation Script
- **What**: Created `scripts/prepare_dpo.py` to process the evaluated rollouts into a DPO dataset. For each chart, it finds a "Chosen Path" (all steps passed and final answer matches ground truth) and a "Rejected Path" (at least one step failed).
- **Why**: To build a preference dataset (Direct Preference Optimization) for fine-tuning our reasoning models based on the PRM judge scores.

## Step-DPO Planning & Formatting
- **What**: Drafted an execution plan for a small-scale Step-DPO experiment (using Kaggle 2xT4). Wrote `scripts/format_step_dpo.py` to automatically find the exact divergent reasoning step between a chosen path and a rejected path.
- **Why**: Standard DPO evaluates the entire sequence. Step-DPO provides dense reward signals at the exact step where the model failed, giving maximum learning signal from our small 187-pair dataset.

## Step-DPO Training & Evaluation Notebooks
- **What**: Created `notebooks/train_dpo.ipynb` for distributed QLoRA training on 2xT4 GPUs using `trl.DPOTrainer` and `accelerate`. Created `notebooks/evaluate_dpo_model.ipynb` to run a direct side-by-side comparison of the base `Qwen2.5-VL` model versus the Step-DPO tuned model (via `disable_adapter()`) on holdout reasoning questions.
- **Why**: To execute the Step-DPO plan on Kaggle and empirically verify if the 187 pairs successfully taught the model to correct its reasoning structure without catastrophic forgetting.

## Step-DPO Kaggle Image Decoding & Git Track Fix
- **What**: Removed `images/*.jpg` from `data/CharXiv/.gitignore`, tracked and force-added all chart images to Git, and updated `notebooks/train_dpo.ipynb` with image existence checks, clean PIL RGB re-saving (`format="JPEG"`, `quality=95`), and an automatic fallback trigger to run `scripts/download_images.py` if images are missing.
- **Why**: When running `train_dpo.ipynb` on Kaggle via `git clone`, `data/CharXiv/images/*.jpg` was missing because it was ignored in `data/CharXiv/.gitignore`. `Image.open` failed silently under `except: pass`, passing invalid image paths to `DPOTrainer`'s `dataset.map()`. Hugging Face `transformers` passed those invalid paths/buffers to `torchvision.io.image.decode_image`, causing `RuntimeError: Unsupported image file. Only jpeg, png, webp and gif are currently supported.` Tracking images in Git and adding auto-download / clean PIL RGB re-encoding completely eliminates this runtime error.

## SFT Training Notebook Setup
- **What**: Created `notebooks/train_sft.ipynb` to run Supervised Fine-Tuning (SFT) on Qwen2.5-VL-3B using `trl.SFTTrainer` and QLoRA (4-bit NF4) on Kaggle 2xT4 / single T4 GPUs.
- **Why**: SFT serves as our baseline supervised alignment model. It filters `evaluated_rollouts.jsonl` and `001_500_reasoning_cleaned.jsonl` for positive rollouts where all reasoning steps scored `1` and final answers matched ground truth. Training on these verified correct trajectories teaches the model explicit step-by-step formatting (`Step 1: ...`, `Final Answer: ...`), solidifies chart reading logic, and reduces hallucinations, providing the exact SFT benchmark needed for our Base vs SFT vs DPO vs KTO comparison.


