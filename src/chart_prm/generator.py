def build_generation_prompt(question: str, extra_instruction: str = "") -> str:
    """
    Builds the prompt for Qwen2.5-VL-3B to generate step-by-step reasoning for chart QA.
    """
    prompt = (
        "You are a helpful and precise assistant for answering questions about scientific charts. "
        "Carefully inspect the axes, axis labels, legend, colors, markers, values, units, titles, ticks, colormaps, annotations, subplots and trends. "
        "Determine whether all of these parts are present in the chart. "
        "Do not confuse different attributes of a chart, such as x and y axis, legend and title, etc. "
        "Pay attention—some of the questions have a catch. "
        f"{extra_instruction} "
        "The axis LABEL is the single descriptive text alongside the full axis. "
        "Tick labels are individual values/words spaced along the axis. These are different things. "
        "If the question is not related to the image or there is not enough information in the image to answer the question, output 'Not Applicable'.\n\n"
        "You MUST break down your thought process into explicit, logical reasoning steps. "
        "Format your response such that each reasoning step is clearly labeled on a new line, starting with 'Step 1:', 'Step 2:', and so on. "
        "After all your reasoning steps, provide the final concise answer on a new line strictly starting with 'Final Answer:'.\n\n"
        f"Question: {question}"
    )
    return prompt
