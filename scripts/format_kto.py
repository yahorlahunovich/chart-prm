"""
Format KTO (Kahneman-Tversky Optimization) Datasets from PRM Evaluated Rollouts.

This script processes evaluated model rollouts (from muse-spark-1.1 PRM judge) and cleaned rollout metadata to produce:
1. Sequence-level KTO samples (`kto_samples.jsonl`): Full reasoning trajectories labeled as desirable (all steps score=1 & correct answer) or undesirable (step failure or wrong answer).
2. Step-level KTO samples (`step_kto_samples.jsonl`): Individual reasoning steps given their prompt prefix, labeled as desirable (score=1) or undesirable (score=0).

Output files are saved to `experiments/001_500_reasoning/data/`.
"""

import json
from pathlib import Path
import random

def main():
    random.seed(42)
    
    base_dir = Path(__file__).resolve().parent.parent
    cleaned_path = base_dir / 'experiments/001_500_reasoning/data/001_500_reasoning_cleaned.jsonl'
    evals_path = base_dir / 'experiments/001_500_reasoning/data/evaluated_rollouts.jsonl'
    
    rollout_kto_path = base_dir / 'experiments/001_500_reasoning/data/kto_samples.jsonl'
    step_kto_path = base_dir / 'experiments/001_500_reasoning/data/step_kto_samples.jsonl'
    
    if not cleaned_path.exists() or not evals_path.exists():
        raise FileNotFoundError(f"Required input files not found at {cleaned_path} or {evals_path}")

    # Load cleaned rollout metadata
    rollout_meta = {}
    with open(cleaned_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            qid = str(data['question_id'])
            ridx = data['rollout_index']
            gt = str(data['ground_truth']).strip().lower()
            ans = str(data['model_final_answer']).strip().lower()
            is_correct = (gt in ans or ans in gt) and len(ans) > 0
            
            parsed_steps = data.get('parsed_steps', [])
            model_output = data.get('model_output', '')
            if not model_output and parsed_steps:
                model_output = '\n'.join(parsed_steps)
                if data.get('model_final_answer'):
                    model_output += f'\nFinal Answer: {data["model_final_answer"]}'
                    
            rollout_meta[(qid, ridx)] = {
                'is_correct': is_correct,
                'question': data.get('question', ''),
                'ground_truth': data.get('ground_truth', ''),
                'model_output': model_output,
                'model_final_answer': data.get('model_final_answer', ''),
                'parsed_steps': parsed_steps
            }

    rollout_kto_samples = []
    step_kto_samples = []

    with open(evals_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            qid = str(data['question_id'])
            ridx = data['rollout_index']
            meta = rollout_meta.get((qid, ridx))
            if not meta:
                continue
                
            evals = data.get('evaluations') or []
            if not evals:
                continue
                
            all_pass = all(s.get('score') == 1 for s in evals)
            is_correct = meta['is_correct']
            rollout_label = (all_pass and is_correct)
            
            # Sequence/Rollout-level sample
            rollout_kto_samples.append({
                'question_id': qid,
                'rollout_index': ridx,
                'image_path': f'data/CharXiv/images/{qid}.jpg',
                'question': meta['question'],
                'prefix': '',
                'completion': meta['model_output'],
                'label': rollout_label
            })
            
            # Step-level sample
            steps = meta['parsed_steps']
            for i, s in enumerate(evals):
                if i >= len(steps):
                    break
                step_score = s.get('score')
                if step_score is None:
                    continue
                step_label = (step_score == 1)
                prefix = '\n'.join(steps[:i])
                if prefix:
                    prefix += '\n'
                step_kto_samples.append({
                    'question_id': qid,
                    'rollout_index': ridx,
                    'step_index': i,
                    'image_path': f'data/CharXiv/images/{qid}.jpg',
                    'question': meta['question'],
                    'prefix': prefix,
                    'completion': steps[i],
                    'label': step_label
                })

    pos_rollouts = [s for s in rollout_kto_samples if s['label']]
    neg_rollouts = [s for s in rollout_kto_samples if not s['label']]
    pos_steps = [s for s in step_kto_samples if s['label']]
    neg_steps = [s for s in step_kto_samples if not s['label']]

    print(f"Generated Rollout-Level KTO Samples: Total={len(rollout_kto_samples)} (Pos={len(pos_rollouts)}, Neg={len(neg_rollouts)})")
    print(f"Generated Step-Level KTO Samples: Total={len(step_kto_samples)} (Pos={len(pos_steps)}, Neg={len(neg_steps)})")

    # Write rollout-level KTO samples
    with open(rollout_kto_path, 'w', encoding='utf-8') as f:
        for s in rollout_kto_samples:
            f.write(json.dumps(s) + '\n')
    print(f"Saved sequence-level KTO samples to: {rollout_kto_path}")

    # Write step-level KTO samples
    with open(step_kto_path, 'w', encoding='utf-8') as f:
        for s in step_kto_samples:
            f.write(json.dumps(s) + '\n')
    print(f"Saved step-level KTO samples to: {step_kto_path}")

if __name__ == '__main__':
    main()
