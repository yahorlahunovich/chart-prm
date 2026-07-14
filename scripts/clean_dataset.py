import json
import re
import os
import sys

def parse_model_output(model_output):
    """
    Parses the model output into atomic steps and a final answer.
    """
    # Isolate Final Answer
    # We look for "Final Answer:" and capture everything after it.
    final_answer_match = re.search(r"Final Answer:(.*?)(?:$)", model_output, flags=re.IGNORECASE | re.DOTALL)
    if not final_answer_match:
        return None, None
        
    final_answer = final_answer_match.group(1).strip()
    
    # Text before the final answer is where the steps live
    text_before_final = model_output[:final_answer_match.start()]
    
    # Split into steps
    # We find all occurrences of "Step X:" and extract the text until the next "Step Y:" or end of string
    step_pattern = r"(Step\s+\d+:.*?)(?=Step\s+\d+:|$)"
    steps = re.findall(step_pattern, text_before_final, flags=re.IGNORECASE | re.DOTALL)
    
    cleaned_steps = [step.strip() for step in steps if step.strip()]
    
    return cleaned_steps, final_answer

def main():
    # Correctly resolve the path handling the user's possible typo
    input_path = 'data/reasoning-steps/500_reasoning_raw.jsonl'
    if not os.path.exists(input_path):
        # Fallback to the user's exact requested path just in case
        fallback_path = 'data/500_reasoning_raw.jsonl'
        if os.path.exists(fallback_path):
            input_path = fallback_path
        else:
            print(f"Error: Input file could not be found at {input_path} or {fallback_path}")
            sys.exit(1)
            
    # As requested exactly by user
    output_path = 'data/cleaned_reasoning.jsonl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_raw = 0
    discarded_missing_delimiters = 0
    discarded_too_long = 0
    total_clean = 0
    unique_questions = set()
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
         
        for line in f_in:
            if not line.strip():
                continue
                
            total_raw += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            model_output = data.get('model_output', '')
            
            # 2. Structural Regex Guard
            if "Step 1:" not in model_output or "Final Answer:" not in model_output:
                discarded_missing_delimiters += 1
                continue
                
            # 3. Infinite Loop / Repetition Filter
            if len(model_output) > 4000:
                discarded_too_long += 1
                continue
                
            # 4 & 5. Parse Into Atomic Steps & Isolate the Final Answer
            parsed_steps, final_answer = parse_model_output(model_output)
            
            # Strict validation: Ensure it actually parsed steps and found an answer
            if not parsed_steps or final_answer is None:
                discarded_missing_delimiters += 1
                continue
                
            q_id = str(data.get('question_id', data.get('sample_index', data.get('id', ''))))
            
            clean_record = {
                "question_id": q_id,
                "rollout_index": int(data.get('rollout_index', 0)),
                "question": data.get('question', ''),
                "ground_truth": str(data.get('ground_truth', '')),
                "parsed_steps": parsed_steps,
                "model_final_answer": final_answer
            }
            
            f_out.write(json.dumps(clean_record, ensure_ascii=False) + '\n')
            total_clean += 1
            unique_questions.add(q_id)
            
    # Reporting
    print("=" * 40)
    print("      DATA CLEANING REPORT")
    print("=" * 40)
    print(f"Total raw rollouts processed: {total_raw}")
    print(f"Total discarded (missing delimiters): {discarded_missing_delimiters}")
    print(f"Total discarded (>4000 chars loops): {discarded_too_long}")
    print(f"Total clean rollouts saved: {total_clean}")
    print(f"Total unique questions remaining: {len(unique_questions)}")
    print("=" * 40)

if __name__ == "__main__":
    main()
