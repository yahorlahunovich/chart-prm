import json
from pathlib import Path

def main():
    cleaned_path = Path('../experiments/001_500_reasoning/data/001_500_reasoning_cleaned.jsonl')
    evals_path = Path('../experiments/001_500_reasoning/data/evaluated_rollouts.jsonl')
    
    # Load cleaned data to get final answers and ground truth
    rollout_meta = {}
    with open(cleaned_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            qid = str(data['question_id'])
            ridx = data['rollout_index']
            gt = str(data['ground_truth']).strip().lower()
            ans = str(data['model_final_answer']).strip().lower()
            
            # Simple substring match for correctness
            # We also remove punctuation just to be safe, but 'in' is a good start.
            is_correct = gt in ans or ans in gt
            
            rollout_meta[(qid, ridx)] = {
                'is_correct': is_correct,
                'ground_truth': data['ground_truth'],
                'model_final_answer': data['model_final_answer'],
                'parsed_steps': data.get('parsed_steps', [])
            }
            
    # Load evaluations
    chosen_per_chart = {}
    rejected_per_chart = {}
    
    with open(evals_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            qid = str(data['question_id'])
            ridx = data['rollout_index']
            
            if qid not in chosen_per_chart:
                chosen_per_chart[qid] = []
                rejected_per_chart[qid] = []
                
            evals = data.get('evaluations', [])
            if not evals:
                continue
                
            # Check if any step has score 0
            has_fail = any(step.get('score') == 0 for step in evals)
            all_pass = all(step.get('score') == 1 for step in evals)
            
            meta = rollout_meta.get((qid, ridx), {})
            is_correct = meta.get('is_correct', False)
            
            if all_pass and is_correct:
                chosen_per_chart[qid].append((ridx, data, meta))
            elif has_fail:
                rejected_per_chart[qid].append((ridx, data, meta))
                
    # Count how many charts have at least 1 chosen and at least 1 rejected
    valid_charts = 0
    total_pairs = 0
    
    for qid in chosen_per_chart.keys():
        n_chosen = len(chosen_per_chart[qid])
        n_rejected = len(rejected_per_chart[qid])
        
        if n_chosen > 0 and n_rejected > 0:
            valid_charts += 1
            total_pairs += (n_chosen * n_rejected)
            
    print(f"Total charts processed: {len(chosen_per_chart)}")
    print(f"Charts with at least one Chosen and one Rejected path: {valid_charts}")
    print(f"Total possible (Chosen, Rejected) DPO pairs across these charts: {total_pairs}")

if __name__ == '__main__':
    main()
