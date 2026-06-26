# Agent Instructions

This project utilizes several AI agents including Cursor, Gemini, Claude, and Antigravity to assist in development for the "Process Reward Modeling in LLMs" course project.

## Core Rules for All Agents
1. **Clean Code**: Write clean, concise code. Avoid redundancy. Follow best practices.
2. **Dependency Management**: Add dependencies strictly using `uv add <dependency>`. Do not manually edit `pyproject.toml` or `uv.lock` files.
3. **Implementation Logging**: After completing a task, you must document the step in `implementation_log.md`. Clearly state *what* was implemented and *why*.
4. **Version Control**: After modifying files, commit the changes using `git commit` with descriptive commit messages, and push to GitHub using `git push`. (We work in a team of 2, so keeping the remote updated is critical).
5. **Project Context**: The primary objective is Process Reward Modeling (PRM) for chart question answering using Qwen2.5-VL-3B. Follow the pipeline in `README.md`.
6. **Compute Constraints**: We are strictly limited to a single T4 GPU on Colab, and 2xT4 GPUs on Kaggle. All model loading, batch sizes, and quantization strategies must be designed with these strict memory constraints in mind.
