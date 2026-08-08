import json
from pathlib import Path
import random

def main():
    base_dir = Path(__file__).resolve().parent.parent
    cleaned_path = base_dir / 'experiments/001_500_reasoning/data/001_500_reasoning_cleaned.jsonl'
    evals_path = base_dir / 'experiments/001_500_reasoning/data/evaluated_rollouts.jsonl'
    output_path = base_dir / 'experiments/001_500_reasoning/data/step_dpo_pairs.jsonl'
    images_dir = base_dir / 'data/CharXiv/images'
    
    # Load cleaned data for ground truth and question text
    rollout_meta = {}
    with open(cleaned_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            qid = str(data['question_id'])
            ridx = data['rollout_index']
            gt = str(data['ground_truth']).strip().lower()
            ans = str(data['model_final_answer']).strip().lower()
            
            is_correct = gt in ans or ans in gt
            
            rollout_meta[(qid, ridx)] = {
                'is_correct': is_correct,
                'question': data.get('question', ''), # if question is in cleaned
                'parsed_steps': data.get('parsed_steps', [])
            }
            
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
                
            has_fail = any(step.get('score') == 0 for step in evals)
            all_pass = all(step.get('score') == 1 for step in evals)
            
            meta = rollout_meta.get((qid, ridx), {})
            is_correct = meta.get('is_correct', False)
            
            # Step text is in evals
            if all_pass and is_correct:
                chosen_per_chart[qid].append((ridx, evals, meta))
            elif has_fail:
                rejected_per_chart[qid].append((ridx, evals, meta))
                
    pairs = []
    
    for qid in chosen_per_chart.keys():
        chosen_list = chosen_per_chart[qid]
        rejected_list = rejected_per_chart[qid]
        
        if not chosen_list or not rejected_list:
            continue
            
        combinations = [(c, r) for c in chosen_list for r in rejected_list]
        random.shuffle(combinations)
        
        for c, r in combinations[:5]: # Max 5 pairs per chart
            c_evals, c_meta = c[1], c[2]
            r_evals, r_meta = r[1], r[2]
            
            first_fail_idx = -1
            for i, step in enumerate(r_evals):
                if step.get('score') == 0:
                    first_fail_idx = i
                    break
                    
            if first_fail_idx == -1:
                continue
                
            c_steps = c_meta['parsed_steps']
            r_steps = r_meta['parsed_steps']
            
            if first_fail_idx >= len(c_steps) or first_fail_idx >= len(r_steps):
                continue
                
            prefix_steps = []
            for i in range(first_fail_idx):
                prefix_steps.append(c_steps[i])
                
            prefix = "\n".join(prefix_steps)
            if prefix:
                prefix += "\n"
                
            chosen_step = c_steps[first_fail_idx]
            rejected_step = r_steps[first_fail_idx]
            
            pairs.append({
                'question_id': qid,
                'image_path': f'data/CharXiv/images/{qid}.jpg',
                'question': c_meta.get('question', ''),
                'prefix': prefix,
                'chosen': chosen_step,
                'rejected': rejected_step
            })
            
    print(f"Generated {len(pairs)} Step-DPO pairs.")
    
    with open(output_path, 'w') as f:
        for p in pairs:
            f.write(json.dumps(p) + '\n')
            
if __name__ == '__main__':
    main()
