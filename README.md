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
