# ChartPRM

> **Can Vision-Language Models Judge the Reasoning Process in Chart Question Answering?**

We are considering a project on PRM for chart question answering. Our idea is to use **CharXiv** as a benchmark and **Qwen2.5-VL-3B** as a base model. We would prompt the model to generate explicit step-by-step reasoning for chart questions, and then evaluate these reasoning steps with a PRM-style judge.

Our goal is not to train a large multimodal PRM from scratch (because we are not sure if it is possible with our compute), but rather to build a small evaluation setup and study whether process-level evaluation can reveal errors such as wrong chart reading, wrong comparison, or calculation mistakes.

---

## Resources

| | |
|---|---|
| **Dataset** | [CharXiv](https://charxiv.github.io/) · [Hugging Face](https://huggingface.co/datasets/princeton-nlp/CharXiv) <br> *(Note: We use a balanced subset of 500 **reasoning questions** for the main pipeline, and a holdout of 100 for evaluation. Descriptive questions are excluded.)* |
| **Model** | [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) (Generator) <br> [muse-spark-1.1](https://api.meta.ai/) (PRM Judge) |

---

## Pipeline

```text
CharXiv chart + question
        │
        ▼
Qwen2.5-VL-3B generates step-by-step reasoning
        │
        ▼
The answer is split into reasoning steps
        │
        ▼
A PRM-style judge scores each step
        │
        ▼
Compare process-level scores with final-answer correctness
        │
        ▼
Analyze failure cases
```

1. **Input** — CharXiv chart + question
2. **Generate** — Qwen2.5-VL-3B produces step-by-step reasoning
3. **Split** — The answer is divided into individual reasoning steps
4. **Score** — A PRM-style judge evaluates each step
5. **Compare** — Process-level scores are compared against final-answer correctness
6. **Analyze** — Failure cases are examined

---

## Current Progress

- [x] Sample 500 reasoning questions from CharXiv
- [x] Generate step-by-step reasoning using Qwen2.5-VL-3B
- [x] Clean and parse model outputs into discrete reasoning steps
- [x] Map dataset indices to original chart image IDs
- [x] Download chart images locally
- [x] Build and test automated PRM-judge pipeline using Meta API (`muse-spark-1.1`)
- [x] Evaluate all 1,287 reasoning rollouts (~4,947 steps)
- [x] Analyze failure cases and compare process-level scores with final-answer correctness
- [x] Build visualization plots to analyze PRM performance, metadata correlations, and error cascades

---

## Development Workflow & Environment

- **Environment Management**: We use `uv` for managing dependencies. **Do not** edit `pyproject.toml` or `uv.lock` directly. Always use `uv add <package>` or `uv remove <package>`.
- **Agents Setup**: We use multiple AI agents (Cursor, Gemini, Claude, Antigravity). Please see `agents/instructions.md` and `.cursorrules` for agent-specific rules.
- **Logging**: Every step, implementation detail, and its reasoning must be documented in `implementation_log.md`.
- **Version Control**: We work in a team of 2. After making any meaningful modifications, agents/developers should commit the changes with a clear message and push to GitHub.

---

## Project Structure

```text
project/
├── data/                       # Store raw and processed CharXiv data
├── notebooks/                  # (.ipynb) For exploration, visualization, and rapid prototyping
│   ├── data_exploration.ipynb    
│   ├── model_inference.ipynb     
│   ├── prm_judge_testing.ipynb   
│   └── failure_analysis.ipynb    
│
├── src/                        # (.py) Reusable, clean, and modular Python package
│   └── chart_prm/              
│       ├── __init__.py
│       ├── dataset.py          # Code to load and preprocess CharXiv
│       ├── generator.py        # Qwen2.5-VL-3B inference & step-by-step reasoning generation
│       ├── parser.py           # Logic to split the model's answer into discrete reasoning steps
│       ├── prm_judge.py        # The process-level evaluation logic and scoring
│       └── utils.py            # Helper functions (e.g., logging, saving/loading JSON lines)
│
├── scripts/                    # (.py) Entry points for running the pipeline in bulk
│   ├── sample_questions.py     # Samples 500 reasoning questions
│   ├── clean_dataset.py        # Cleans dataset and extracts reasoning steps
│   ├── fix_jsonl_ids.py        # Maps dataset indices to CharXiv figure_ids
│   ├── download_images.py      # Downloads the necessary charts from HuggingFace
│   └── evaluate_rollouts_meta.py # Batched PRM Judge script using Meta API
│
├── pyproject.toml              # Managed by uv for dependencies
├── uv.lock                     # Managed by uv
├── implementation_log.md       # Tracks implementation history
└── README.md                   # This file
```