import json
import random
from collections import defaultdict
import os

random.seed(42)

def main():
    # Load reasoning_val.json and chart_types_val.json
    val_reasoning_path = "data/CharXiv/data/reasoning_val.json"
    chart_types_path = "data/CharXiv/data/chart_types_val.json"
    
    with open(val_reasoning_path, "r") as f:
        reasoning_data = json.load(f)
        
    with open(chart_types_path, "r") as f:
        chart_types_data = json.load(f)
        
    # Group questions by chart type
    chart_type_to_qids = defaultdict(list)
    
    for qid, q_data in reasoning_data.items():
        # Get chart type
        # chart_types is a list, we'll just use the first one or a joined string
        c_types = chart_types_data.get(qid, {}).get("chart_types", ["Unknown"])
        primary_chart_type = c_types[0] if c_types else "Unknown"
        
        chart_type_to_qids[primary_chart_type].append(qid)
        
    print(f"Total reasoning questions: {len(reasoning_data)}")
    print("Distribution by chart type:")
    for ct, qids in sorted(chart_type_to_qids.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {ct}: {len(qids)}")
        
    # We want 600 total (500 main, 100 eval)
    target_total = 600
    selected_qids = []
    
    # Calculate target per chart type proportionally
    # Note: we shuffle the qids for random selection
    for ct in chart_type_to_qids:
        random.shuffle(chart_type_to_qids[ct])
        
    # Proportional sampling
    for ct, qids in chart_type_to_qids.items():
        proportion = len(qids) / len(reasoning_data)
        target_count = int(round(proportion * target_total))
        selected_qids.extend(qids[:target_count])
        
    # Adjust if we have slightly more or less than 600 due to rounding
    if len(selected_qids) > target_total:
        selected_qids = random.sample(selected_qids, target_total)
    elif len(selected_qids) < target_total:
        # Add a few more from the remaining pool
        remaining_pool = [qid for qid in reasoning_data.keys() if qid not in selected_qids]
        selected_qids.extend(random.sample(remaining_pool, target_total - len(selected_qids)))
        
    # Split into 500 main and 100 eval
    random.shuffle(selected_qids)
    eval_qids = selected_qids[:100]
    main_qids = selected_qids[100:]
    
    # Save the splits
    os.makedirs("data/splits", exist_ok=True)
    with open("data/splits/main_reasoning_ids.json", "w") as f:
        json.dump(main_qids, f, indent=4)
        
    with open("data/splits/eval_reasoning_ids.json", "w") as f:
        json.dump(eval_qids, f, indent=4)
        
    print(f"Saved {len(main_qids)} main IDs to data/splits/main_reasoning_ids.json")
    print(f"Saved {len(eval_qids)} eval IDs to data/splits/eval_reasoning_ids.json")

if __name__ == "__main__":
    main()
