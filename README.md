# ChartPRM

> **Can Vision-Language Models Judge the Reasoning Process in Chart Question Answering?**

We are considering a project on PRM for chart question answering. Our idea is to use **CharXiv** as a benchmark and **Qwen2.5-VL-3B** as a base model. We would prompt the model to generate explicit step-by-step reasoning for chart questions, and then evaluate these reasoning steps with a PRM-style judge.

Our goal is not to train a large multimodal PRM from scratch (because we are not sure if it is possible with our compute), but rather to build a small evaluation setup and study whether process-level evaluation can reveal errors such as wrong chart reading, wrong comparison, or calculation mistakes.

---

## Resources

| | |
|---|---|
| **Dataset** | [CharXiv](https://charxiv.github.io/) · [Hugging Face](https://huggingface.co/datasets/princeton-nlp/CharXiv) |
| **Model** | [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |

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
│   ├── 01_data_exploration.ipynb    
│   ├── 02_model_inference.ipynb     
│   ├── 03_prm_judge_testing.ipynb   
│   └── 04_failure_analysis.ipynb    
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
│   ├── 01_generate_reasoning.py   
│   ├── 02_score_steps.py          
│   └── 03_evaluate_pipeline.py    
│
├── pyproject.toml              # Managed by uv for dependencies
├── uv.lock                     # Managed by uv
├── implementation_log.md       # Tracks implementation history
└── README.md                   # This file
```
# ccs_project
