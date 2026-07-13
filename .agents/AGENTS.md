# Antigravity Workspace Rules

- **Clean Code**: Always write clean, maintainable code without redundancy.
- **Dependencies**: Use `uv add <dependency>` to add packages. Never manually edit `pyproject.toml` or `uv.lock`.
- **Logging**: Document all actions, decisions, and modifications in `implementation_log.md` (what and why).
- **Git Workflow**: You must `git commit` and `git push` all modifications directly to GitHub, as we work in a 2-person team.
- **Compute Constraints**: Keep in mind that our compute is strictly limited to a single T4 GPU on Colab and 2xT4 GPUs on Kaggle. Optimize model loading and batch sizes accordingly.
- **Dataset Focus**: We only evaluate on a balanced subset of 500 **reasoning questions** from CharXiv. Descriptive questions are ignored.
