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
- Fixed the `git clone` command in the notebook by prepending `oauth2:` to the embedded token, preventing interactive password prompts in headless environments like Colab.
