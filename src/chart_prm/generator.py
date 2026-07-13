def build_generation_prompt(question: str, extra_instruction: str = "") -> str:
    """
    Builds the prompt for Qwen2.5-VL-3B to generate step-by-step reasoning for chart QA.
    """
    prompt = (
        "You are an expert at extracting data and reasoning about scientific charts. "
        "Carefully inspect the axes, axis labels, legend, colors, markers, values, units, titles, ticks, and trends. "
        f"{extra_instruction} "
        "If the question is not related to the image or there is not enough information to answer, output 'Not Applicable'.\n\n"
        "RULES FOR REASONING:\n"
        "1. DO NOT write any introductory or conversational text. Begin immediately with 'Step 1:'.\n"
        "2. Break down your thought process into explicit, logical reasoning steps.\n"
        "3. DO NOT output a high-level plan (e.g., 'Find the x-axis'). You MUST explicitly state the concrete values, labels, and colors you read from the chart in each step.\n"
        "4. Perform and display explicit comparisons and intermediate math calculations.\n"
        "5. Label each step on a new line starting exactly with 'Step 1:', 'Step 2:', etc.\n"
        "6. Provide the final concise answer on a new line starting strictly with 'Final Answer:'. The final answer MUST be ONLY the exact short value or entity.\n\n"
        "---\n"
        "EXAMPLE FORMAT:\n"
        "Question: Which model has the highest accuracy at Epoch 10?\n"
        "Step 1: The x-axis represents 'Epochs'. I need to find the data points at the vertical line for Epoch 10.\n"
        "Step 2: At Epoch 10, Model A (blue line) has an accuracy of approximately 72%.\n"
        "Step 3: At Epoch 10, Model B (red line) has an accuracy of approximately 85%.\n"
        "Step 4: At Epoch 10, Model C (green line) has an accuracy of approximately 60%.\n"
        "Step 5: Comparing the extracted values: 85% > 72% > 60%.\n"
        "Final Answer: Model B\n"
        "---\n\n"
        f"Question: {question}"
    )
    return prompt
