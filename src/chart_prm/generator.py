def build_generation_prompt(question: str, extra_instruction: str = "") -> str:
    """
    Builds the prompt for Qwen2.5-VL-3B to generate step-by-step reasoning for chart QA.
    """
    prompt = (
        "You are a helpful and precise assistant for answering questions about scientific charts. "
        "Carefully inspect the axes, axis labels, legend, colors, markers, values, units, titles, ticks, colormaps, annotations, subplots and trends. "
        f"{extra_instruction} "
        "If the question is not related to the image or there is not enough information in the image to answer the question, output 'Not Applicable'.\n\n"
        "You MUST break down your thought process into explicit, logical reasoning steps. "
        "Crucially, your steps must NOT just be a high-level plan. You MUST explicitly state the intermediate values you read from the chart, "
        "perform explicit comparisons, and show any intermediate math calculations.\n"
        "Format your response such that each reasoning step is clearly labeled on a new line, starting with 'Step 1:', 'Step 2:', and so on. "
        "After all your reasoning steps, provide the final concise answer on a new line strictly starting with 'Final Answer:'. "
        "The final answer MUST be ONLY the exact short value or entity, with no additional words, filler, or explanation.\n\n"
        f"Question: {question}"
    )
    return prompt
