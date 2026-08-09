import json
from pathlib import Path
import random
import re
import unicodedata


SEED = 42
MAX_PAIRS_PER_CHART = 5


def normalize_text(value):
    """Normalize text for conservative token-aware comparisons."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("\\%", "%")
    return " ".join(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?|[%+\-]", text))


def answers_match(ground_truth, model_answer):
    """Match exact values or whole-token phrases, never raw substrings."""
    expected = normalize_text(ground_truth)
    actual = normalize_text(model_answer)
    if not expected or not actual:
        return False
    if expected == actual:
        return True

    expected_tokens = expected.split()
    actual_tokens = actual.split()
    width = len(expected_tokens)
    return any(
        actual_tokens[index:index + width] == expected_tokens
        for index in range(len(actual_tokens) - width + 1)
    )


def normalize_step(step):
    """Remove the presentation-only step label before prefix comparison."""
    text = unicodedata.normalize("NFKC", str(step)).strip()
    text = re.sub(r"^\s*step\s+\d+\s*:\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def valid_step(step):
    normalized = normalize_step(step)
    if not normalized or len(normalized) > 2000:
        return False
    return re.search(r"\bstep\s+(?:step|analysis|subs?)\b", normalized, re.IGNORECASE) is None


def shared_prefix_length(chosen_steps, rejected_steps):
    length = 0
    for chosen, rejected in zip(chosen_steps, rejected_steps):
        if normalize_text(normalize_step(chosen)) != normalize_text(normalize_step(rejected)):
            break
        length += 1
    return length

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
            is_correct = answers_match(data['ground_truth'], data['model_final_answer'])
            
            rollout_meta[(qid, ridx)] = {
                'is_correct': is_correct,
                'question': data.get('question', ''), # if question is in cleaned
                'parsed_steps': data.get('parsed_steps', []),
                'ground_truth': data.get('ground_truth', ''),
                'model_final_answer': data.get('model_final_answer', ''),
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
        random.Random(f"{SEED}:{qid}").shuffle(combinations)
        
        chart_pairs = 0
        for c, r in combinations:
            c_evals, c_meta = c[1], c[2]
            r_evals, r_meta = r[1], r[2]
            c_steps = c_meta['parsed_steps']
            r_steps = r_meta['parsed_steps']

            divergence_idx = shared_prefix_length(c_steps, r_steps)
            if divergence_idx >= len(c_steps) or divergence_idx >= len(r_steps):
                continue
            if divergence_idx >= len(c_evals) or divergence_idx >= len(r_evals):
                continue
            if c_evals[divergence_idx].get('score') != 1:
                continue
            if r_evals[divergence_idx].get('score') != 0:
                continue

            chosen_step = normalize_step(c_steps[divergence_idx])
            rejected_step = normalize_step(r_steps[divergence_idx])
            if not valid_step(chosen_step) or not valid_step(rejected_step):
                continue
            if normalize_text(chosen_step) == normalize_text(rejected_step):
                continue

            prefix_steps = [normalize_step(step) for step in c_steps[:divergence_idx]]
            prefix = "\n".join(prefix_steps)
            if prefix:
                prefix += "\n"
            
            pairs.append({
                'question_id': qid,
                'image_path': f'data/CharXiv/images/{qid}.jpg',
                'question': c_meta.get('question', ''),
                'prefix': prefix,
                'chosen': chosen_step,
                'rejected': rejected_step,
                'metadata': {
                    'chosen_rollout_index': c[0],
                    'rejected_rollout_index': r[0],
                    'divergence_step_index': divergence_idx,
                    'shared_prefix_steps': divergence_idx,
                    'ground_truth': c_meta['ground_truth'],
                    'chosen_final_answer': c_meta['model_final_answer'],
                    'chosen_judge_analysis': c_evals[divergence_idx].get('analysis', ''),
                    'rejected_judge_analysis': r_evals[divergence_idx].get('analysis', ''),
                },
            })
            chart_pairs += 1
            if chart_pairs >= MAX_PAIRS_PER_CHART:
                break
            
    print(f"Generated {len(pairs)} Step-DPO pairs.")
    
    with open(output_path, 'w') as f:
        for p in pairs:
            f.write(json.dumps(p) + '\n')
            
if __name__ == '__main__':
    main()
