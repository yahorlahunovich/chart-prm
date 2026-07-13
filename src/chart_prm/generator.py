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
        "---\n"
        "EXAMPLE FORMAT:\n"
        "Question: Which model has the highest accuracy at Epoch 10?\n"
        "Step 1: Locate the x-axis representing 'Epochs' and find the tick mark for Epoch 10.\n"
        "Step 2: Trace vertically from Epoch 10 to find the intersection with the lines representing the models.\n"
        "Step 3: Extract the y-axis (Accuracy) values at these points: Model A (blue) = 72%, Model B (red) = 85%, Model C (green) = 60%.\n"
        "Step 4: Compare the extracted values: 85% > 72% > 60%.\n"
        "Step 5: Determine that Model B has the highest accuracy.\n"
        "Final Answer: Model B\n"
        "---\n\n"
        f"Question: {question}"
    )
    return prompt
