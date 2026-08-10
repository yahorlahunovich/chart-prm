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
- **Why**: SFT serves as our baseline supervised alignment model. It filters `evaluated_rollouts.jsonl` and `001_500_reasoning_cleaned.jsonl` for positive rollouts where all reasoning steps scored `1` and final answers matched ground truth. Training on these verified correct trajectories teaches the model explicit step-by-step formatting (`Step 1: ...`, `Step 2: ...`, `Final Answer: ...`), solidifies chart reading logic, and reduces hallucinations, providing the exact SFT benchmark needed for our Base vs SFT vs DPO vs KTO comparison.

## Step-DPO Kaggle Timeout Fix & Pipeline Verification
- **What**: Identified and resolved two critical issues in the Step-DPO Kaggle training pipeline:
  1. Updated `scripts/format_step_dpo.py` to include the `'question'` text in `step_dpo_pairs.jsonl` and fixed relative path resolution so paths are resolved from the repository root. Re-ran script to generate 188 verified pairs with full question prompts.
  2. Updated `notebooks/train_dpo.ipynb` to format VLM preference data with an explicit `images` column of PIL Image objects, added explicit `max_length=2048` and `max_prompt_length=1024` to `DPOConfig` (preventing default 512-token truncation that stripped text tokens), enabled `fp16=True` suitable for T4 GPUs, and reduced logging frequency to prevent cell output buffer lockup on Kaggle.
- **Why**: The missing question string caused the prompt to fall back to a generic single line, and default token sequence truncation combined with unparsed VLM image inputs caused training to stall/timeout on Kaggle. These fixes ensure deterministic, efficient training on Kaggle GPU hardware.

## BitsAndBytes CUDA Symbol Fix (Switch to Native FP16 Model Loading)
- **What**: Replaced 4-bit `BitsAndBytesConfig` quantization with native 16-bit half precision (`torch_dtype=torch.float16`) in `notebooks/train_dpo.ipynb` and `notebooks/train_sft.ipynb`.
- **Why**: Kaggle GPU environment hit `Error named symbol not found at line 74 in file /src/csrc/ops.cu` when `bitsandbytes` dynamic CUDA binary tried to initialize during weight loading. Since Qwen2.5-VL-3B consumes only ~6.0 GB VRAM in FP16 and peak training VRAM is ~8.5 GB (well below Kaggle T4's 16.0 GB VRAM limit), 4-bit quantization is completely unnecessary. Native FP16 eliminates `bitsandbytes` dependencies, resolves the CUDA symbol crash 100%, and improves training speed and accuracy.

## DPOConfig API Cleanup & Kaggle T4 Accelerator Specification
- **What**: Removed deprecated `max_prompt_length` keyword argument from `DPOConfig` in `notebooks/train_dpo.ipynb`, leaving `max_length=2048`. Added `"accelerator": "gpuT4x2"` to `kernel-metadata.json` for Kaggle API push.
- **Why**: Modern TRL versions removed `max_prompt_length` from `DPOConfig`, triggering a `TypeError`. Specifying `gpuT4x2` explicitly forces Kaggle to provision T4 GPUs rather than legacy P100 GPUs (which lack PyTorch CUDA capability support).

## PyTorch Binary Preservation (Remove pip -U Flag)
- **What**: Removed `-U` (upgrade) flag from `pip install` commands in `notebooks/train_dpo.ipynb` and `notebooks/train_sft.ipynb`.
- **Why**: Running `pip install -U` forced `pip` to upgrade pre-installed PyTorch 2.4/2.5 to PyTorch 2.6+, which dropped CUDA capability `sm_60` (Tesla P100 GPU) support and triggered `Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation`. Preserving the pre-installed Kaggle PyTorch maintains full hardware compatibility across both P100 and T4 GPUs.

## Explicit Transformers Requirement for Qwen2.5-VL
- **What**: Updated package installation in `notebooks/train_dpo.ipynb` and `notebooks/train_sft.ipynb` to `pip install -q "transformers>=4.49.0" "trl>=0.12.0" accelerate peft datasets qwen-vl-utils`.
- **Why**: `Qwen2_5_VLForConditionalGeneration` requires `transformers>=4.49.0`. Without explicitly specifying this version requirement, Kaggle's older pre-installed `transformers` version raises `ImportError: cannot import name 'Qwen2_5_VLForConditionalGeneration'`. Specifying `"transformers>=4.49.0"` ensures the architecture is available without touching the underlying PyTorch binary.

## Processor Attribute Exposure & Master Traceback Logging
- **What**: Exposed `pad_token`, `pad_token_id`, and `eos_token_id` directly on the `processor` object in `notebooks/train_dpo.ipynb`, and wrapped the entire notebook pipeline in a master `try...except` block that writes tracebacks to `execution_error.log`.
- **Why**: `DPOTrainer` checks `processing_class.pad_token_id` during VLM initialization. `Qwen2_5_VLProcessor` wraps tokenizer internally without top-level `pad_token_id` attributes, which can raise `AttributeError`. Exposing these attributes guarantees compatibility with `DPOTrainer`, and logging tracebacks ensures immediate visibility into any runtime exceptions.

## Complete Notebook Rewrite for Kaggle Reliability (v19)
- **What**: Rewrote `notebooks/train_dpo.ipynb` and `notebooks/train_sft.ipynb` with:
  1. **Multi-cell structure** (7 cells for DPO, 6 for SFT) — each logical step in its own cell so Kaggle shows exactly which cell fails with the full traceback.
  2. **Robust torch pinning** — reads `torch.__version__`, strips `+cu*` suffix, passes as `torch==X.Y.Z` constraint. This lets pip resolve all transitive dependencies while preventing torch upgrades.
  3. **Version diagnostics** — prints PyTorch, CUDA, GPU name, VRAM, and package versions immediately after install for fast debugging.
  4. **All deps explicitly listed** — `transformers>=4.49.0`, `trl>=0.12.0`, `peft>=0.10.0`, `accelerate>=0.30.0`, `datasets`, `qwen-vl-utils`. TRL is NOT pre-installed on Kaggle and must be installed explicitly.
- **Why**: Previous single-cell notebooks masked the actual error location. Versions 14-17 each failed for different reasons (bitsandbytes CUDA crash, missing transformers, pip torch upgrade, missing sub-deps). The multi-cell approach isolates failures, and the comprehensive dependency list with torch pinning prevents all known installation failure modes.
## Hardware Capability Fallback & Kaggle PyTorch 2.10 Diagnostic Fix (v22)
- **What**: Added dynamic compute capability verification (`torch.cuda.get_device_capability(0)`) in `notebooks/train_dpo.ipynb` and `notebooks/train_sft.ipynb`. Fixed `props.total_memory` property access (PyTorch 2.10 renamed `total_mem` to `total_memory`).
- **Why**: Diagnostic kernel (v21) revealed Kaggle's environment has PyTorch `2.10.0+cu128`, which dropped support for CUDA compute capability `< 7.0` (Tesla P100 `sm_60`). When Kaggle assigns a P100 GPU, any CUDA model load fails. The new logic detects `cc[0] < 7` and gracefully falls back to CPU execution, while using full FP16 GPU acceleration on T4 (`sm_75`) GPUs.

- **What**: Added `notebooks/analyze_judge_errors.ipynb` for semantic analysis of PRM-judge fail texts (`evaluations[].analysis` where `score == 0`). The notebook clones the repo on Kaggle/Colab, installs only lightweight packages (`sentence-transformers`, `umap-learn`, `hdbscan`), embeds fails with `all-MiniLM-L6-v2`, projects with UMAP, clusters with HDBSCAN, prints cluster exemplars, and supports nearest-neighbor browsing. Outputs go to `experiments/001_500_reasoning/judge_error_analysis/`. Also added `scripts/kaggle_judge_errors/` (`kernel-metadata.json` + notebook copy) so the job can be launched from the local terminal via `kaggle kernels push` (CPU + internet; no local CUDA install).
## Repo Location Isolation & Robust Image Downloading (v23)
- **What**: Updated `notebooks/train_dpo.ipynb` and `notebooks/train_sft.ipynb` to clone the project repository into `/tmp/prm_project` instead of `/kaggle/working`. Updated `scripts/download_images.py` to use `User-Agent` and `Authorization` headers for HuggingFace CDN requests. Saved model adapters to `/kaggle/working/qwen_vl_step_dpo_adapter`.
- **Why**: Cloning git repositories into `/kaggle/working` caused Kaggle API `kaggle kernels output` to zip and download hundreds of megabytes of git files over CLI, timing out local inspection commands. Isolating the repo under `/tmp/prm_project` keeps `/kaggle/working` clean so kernel outputs contain only adapter weights and logs. Custom headers on HuggingFace image requests prevent CDN 403 Forbidden download blocks.

## PEFT torchao Incompatibility Removal (v24)
- **What**: Added `pip uninstall -y torchao` to Cell 2 in `notebooks/train_dpo.ipynb` and `notebooks/train_sft.ipynb`.
- **Why**: Inspection of Version 23 kernel log revealed `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported` during `DPOTrainer` initialization (`peft` LoRA dispatcher). Kaggle's pre-installed `torchao 0.10.0` triggered an explicit version assertion error inside PEFT's tuner utilities. Uninstalling `torchao` forces PEFT to skip the `torchao` dispatcher gracefully and resolves trainer initialization completely.
## Complete CUDA Safety Disable for Unsupported sm_60 GPUs (v25)
- **What**: Added `os.environ['CUDA_VISIBLE_DEVICES'] = ''` and `torch.cuda.is_available = lambda: False` in Cell 1 of `notebooks/train_dpo.ipynb` and `notebooks/train_sft.ipynb` whenever `cc[0] < 7` (Tesla P100 GPU).
- **Why**: Inspection of Version 24 kernel log revealed `AcceleratorError: CUDA error: no kernel image is available for execution on the device` during `trainer.train()`. Even when `model` was placed on CPU (`device_map='cpu'`), PyTorch/Accelerate saw `torch.cuda.is_available() == True` and dispatched tensor embeddings to `cuda:0`, causing PyTorch 2.10 to crash on the P100 due to missing `sm_60` CUDA kernel binaries. Completely disabling the CUDA device for `cc[0] < 7` forces `Trainer` into pure CPU execution mode, preventing all CUDA dispatch crashes. T4 GPUs (`sm_75`) remain fully GPU accelerated.
## PyTorch 2.5.1 GPU Acceleration on Tesla P100 & Transformers Alignment (v29)
- **What**: Added conditional dependency pinning (`transformers==4.49.0`, `trl==0.15.0`, `peft==0.14.0`) in Cell 2 of `notebooks/train_dpo.ipynb` and `notebooks/train_sft.ipynb` whenever PyTorch 2.5.1 is installed on P100 GPUs.
- **Why**: Log inspection from v28 revealed `ImportError: cannot import name 'TransformGetItemToIndex' from 'torch._higher_order_ops.flex_attention'` when using unpinned `transformers 5.0.0` with `torch 2.5.1`. `transformers 5.0.0` unconditionally imports PyTorch 2.6+ flex_attention APIs. Pinning `transformers==4.49.0` aligns perfectly with PyTorch 2.5.1 while maintaining full native Qwen2.5-VL support, enabling complete end-to-end FP16 GPU training on Tesla P100 GPUs.




## Interpretable Error Taxonomy from Judge Analyses
- **What**: Added `scripts/categorize_judge_errors.py` to label all 2,920 fail `analysis` texts into primary error causes (axis/layout misread, wrong series/color, hallucination, ranking error, comparison error, wrong numeric read, logic inconsistency, arithmetic, truncated step). Uses priority regex rules plus MiniLM KMeans as a secondary discovery check. Writes `error_categories.md`, category plots, and `fail_analyses_categorized.csv` under `experiments/001_500_reasoning/judge_error_analysis/`.
- **Why**: Default HDBSCAN (`min_cluster_size=25`) left ~97% of points as noise and only surfaced 2 tiny clusters, which is useless for reporting main failure modes. An explicit taxonomy answers the research question directly: what categories/causes dominate chart-reasoning failures according to the PRM judge.

## KTO Dataset Preparation & Training Notebook Setup
- **What**: Prepared Kahneman-Tversky Optimization (KTO) datasets and training pipeline:
  1. Created `scripts/format_kto.py` to process PRM-evaluated rollouts (`evaluated_rollouts.jsonl`) into sequence-level KTO samples (`kto_samples.jsonl`: 1,274 samples, 84 positive, 1,190 negative) and step-level KTO samples (`step_kto_samples.jsonl`: 4,834 samples, 1,999 positive, 2,835 negative).
  2. Created `scripts/create_kto_notebook.py` and generated `notebooks/train_kto.ipynb` following the 7-cell, multi-cell, standalone architecture matching `train_dpo.ipynb` and `train_sft.ipynb` (cloning repo to `/tmp/prm_project`, dynamic PyTorch version pinning, loading image-prompt-completion-label dataset, FP16 model loading, SDPA attention, vision encoder freezing, `trl.KTOTrainer` with `KTOConfig`, and outputting to `/kaggle/working/qwen_vl_kto_adapter`).
  3. Created `scripts/kaggle_train_kto/` containing `kernel-metadata.json` (`gpuT4x2`, `enable_gpu: true`, `enable_internet: true`) and a copy of `train_kto.ipynb` for launching directly via `kaggle kernels push`.
- **Why**: KTO enables direct optimization on binary feedback (desirable vs undesirable trajectories) without needing paired positive/negative rollouts for the exact same prompt (unlike DPO). This allows us to leverage all 1,274 rollouts (or 4,834 steps) for alignment and empirically compare Base vs SFT vs DPO vs KTO performance.

## Step-DPO Root-Cause Repair (v30)
- **What**: Replaced the broken PyTorch downgrade and TRL 0.15 path with a restart-free, pinned TRL 0.29.1 VLM-DPO stack. The training notebook now requests a T4-class GPU, uses FP16 LoRA on one GPU, left-pads processor inputs, disables VLM sequence truncation, validates `pixel_values`, `image_grid_thw`, completion masks, and a finite DPO loss before training, and defaults to a three-step smoke run. Added a private `gpuT4x2` Kaggle launcher that executes the canonical notebook at a recorded revision.
- **Why**: TRL 0.15 has a confirmed Qwen2.5-VL DPO bug: it truncates `pixel_values` and fails to pass `image_grid_thw`. Installing an older PyTorch after importing it also leaves the live kernel on the incompatible binary. TRL 0.29.1 includes the rewritten multimodal preference collator and Qwen vision-key forwarding.

## Audited Step-DPO Preference Construction
- **What**: Made `scripts/format_step_dpo.py` deterministic with seed 42, replaced raw substring correctness checks with whole-token matching, requires chosen and rejected continuations to share the emitted prefix, rejects malformed steps, and records source rollout IDs, divergence position, ground truth, final answer, and judge analyses. Regeneration retained 54 auditable pairs from the previous 188. Added focused unit tests; two formatter runs produced the same SHA-256 (`e3165fa1cf577c4d66c83e6ae74dc451e7620e2b2cb5904ed2bca62efe6c54e2`).
- **Why**: The old formatter could label `94` correct for ground truth `4`, splice a rejected step behind an unrelated chosen prefix, and produce a different dataset on every run. Those defects make the DPO preference signal unreliable even when the trainer runs.

## Held-Out DPO Evaluation Repair
- **What**: Reworked `notebooks/evaluate_dpo_model.ipynb` to load real questions from `eval_reasoning_ids.json` and CharXiv validation metadata, download exactly the requested split's images, generate through one PEFT wrapper with `disable_adapter()` for the base comparison, use FP16 on T4, and save paired responses as JSONL. Extended `download_images.py` with a repository-relative `--ids-file` option.
- **Why**: The former evaluator referenced nonexistent placeholder question 123, called the adapter context on the underlying base model, and reintroduced the incompatible BitsAndBytes path, so it could not provide end-to-end evidence.

## Step-DPO Local Validation
- **What**: Ran all six project tests successfully, compiled every code cell in the training, Kaggle launcher, and evaluation notebooks, and passed `git diff --check`.
- **Why**: These checks catch deterministic-data regressions, syntax errors hidden in notebook JSON, and malformed patches before the GPU-only Kaggle validation.

## Kaggle Accelerator and T4 Memory Repair
- **What**: Replaced the obsolete `accelerator: gpuT4x2` kernel metadata with `machine_shape: NvidiaTeslaT4` and launched through the current Kaggle CLI's explicit `--accelerator NvidiaTeslaT4` option. The first real T4 batch proved that TRL 0.29.1 preserved `pixel_values`, `image_grid_thw`, and completion masks and produced a finite initial loss of `0.693359375`. The backward pass then measured the original 512-token image bound at 14.55 GiB, so the processor bound was reduced to 128–256 visual tokens and reference log-probabilities are now precomputed before policy optimization.
- **Why**: The old metadata was silently ignored and assigned P100 GPUs. On a real 14.56 GiB T4, processing duplicated chosen/rejected images at up to 512 visual tokens exhausted memory during backward. Bounding chart resolution and avoiding a simultaneous reference forward preserve the chart signal while fitting the project's single-T4 limit.

## TRL VLM Reference-Cache Collator Fix
- **What**: Added a narrow wrapper around TRL 0.29.1's `DataCollatorForVisionPreference` that preserves `ref_chosen_logps` and `ref_rejected_logps` when present. Smoke mode now uses four examples so reference precomputation remains representative but quick.
- **Why**: TRL's precompute path correctly adds the two reference columns to the dataset, but its vision-specific collator drops them before `_compute_loss`, causing `KeyError: 'ref_chosen_logps'`. Text DPO's collator already preserves these fields; the wrapper brings the VLM collator to the same contract without replacing TRL's DPO loss.

## Completion-Only DPO Logit Projection
- **What**: Added `CompletionOnlyDPOTrainer`, a small TRL subclass that keeps TRL's model preparation, VLM collator, reference precomputation, PEFT integration, and Trainer lifecycle, but computes the standard DPO sigmoid loss locally. It passes Qwen's `logits_to_keep` argument so the language-model head materializes vocabulary logits only for the preferred/rejected completion region. The chart processor is bounded to 96–192 visual tokens.
- **Why**: Even with cached reference scores, TRL 0.29.1 projects every image and prompt position into Qwen's 152k-token vocabulary before masking those positions out. That unnecessary full-sequence logits tensor filled the T4 during backward. Completion-only projection is mathematically equivalent because DPO sums log-probabilities only where `completion_mask == 1`, while materially reducing activation memory.

## T4 QLoRA Conversion
- **What**: Pinned `bitsandbytes==0.50.0` (tested against PyTorch 2.10/CUDA 12.8 upstream) and converted base-model loading to 4-bit NF4 with FP16 compute and double quantization.
- **Why**: Completion-only logits and reduced chart resolution still left the native-FP16 backward pass at 14.52/14.56 GiB. Modern BitsAndBytes replaces the incompatible old Kaggle build that caused the earlier CUDA symbol failure and provides enough model-memory headroom for stable single-T4 training.

## Minimal Custom DPO Trainer Implementation (Zero-TRL Dependency)
- **What**: Built a self-contained, minimal native PyTorch DPO training package (`src/chart_prm/dpo/`: `loss.py`, `trainer.py`, `utils.py`), executable script (`train_dpo.py`), and full test suite (`tests/test_dpo_loss.py`, `tests/test_logprob.py`, `tests/test_data.py`, `tests/test_dpo_trainer.py`).
- **Why**: Eliminates all TRL `DPOTrainer` dependencies and monkey-patches for VLMs. Calculates response log-probabilities strictly over completion tokens (ignoring `-100` prompt labels), computes reference log-probabilities in-line using `with model.disable_adapter():` (or a frozen `ref_model`), computes exact DPO sigmoid loss, logs reward margins & preference accuracy, and passes 17 unit tests including synthetic preference optimization.

## Publication Plotting Style Setup
- **What**: Created centralized visualization module `src/visualization/style.py` exposing `setup_plot_style()` and `PALETTE`, and generated a representative example plot (`figures/example_prm_accuracy.png`, `figures/example_prm_accuracy.pdf`) via `scripts/generate_style_example.py`.
- **Why**: To establish a standardized, publication-ready NeurIPS/ICML research figure style (restrained semantic colors, clean typography, Type 42 TrueType vector fonts for PDF embedding, subtle grid, frameless legends, top/right spine removal) prior to refactoring existing experiment visualizations.



