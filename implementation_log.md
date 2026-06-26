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
