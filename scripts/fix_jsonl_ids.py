"""
fix_jsonl_ids.py

This utility script corrects the `question_id` mismatch between the cleaned dataset and the 
downloaded CharXiv images. It reads the raw HuggingFace dataset to map the sequential dataset 
indices to the original `figure_id`s, ensuring that the PRM judge receives the correct image 
for each reasoning question.
"""
import json
import os

def main():
    # 1. Create the mapping from sample_index to figure_id
    with open("data/splits/main_reasoning_ids.json", "r") as f:
        target_ids = set(json.load(f))
    
    with open("data/CharXiv/data/reasoning_val.json", "r") as f:
        reasoning_data = json.load(f)
        all_keys = list(reasoning_data.keys())

    ordered_target_keys = [k for k in all_keys if k in target_ids]
    
    # Create mapping: sample_index (str or int) -> true figure_id
    index_to_figure_id = {str(i): k for i, k in enumerate(ordered_target_keys)}
    
    # 2. Fix the JSONL files
    files_to_fix = [
        "experiments/001_500_reasoning/data/001_500_reasoning_raw.jsonl",
        "experiments/001_500_reasoning/data/001_500_reasoning_cleaned.jsonl"
    ]
    
    for filepath in files_to_fix:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        print(f"Processing {filepath}...")
        temp_filepath = filepath + ".tmp"
        
        with open(filepath, "r", encoding="utf-8") as f_in, open(temp_filepath, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                # In raw.jsonl, the field is sample_index (int)
                # In cleaned.jsonl, the field might be question_id (str) or sample_index
                
                # Try to determine the index
                # In cleaned.jsonl we wrongly used the sample_index as the question_id
                old_id = None
                if 'sample_index' in data:
                    old_id = str(data['sample_index'])
                elif 'question_id' in data:
                    old_id = str(data['question_id'])
                
                if old_id and old_id in index_to_figure_id:
                    new_id = index_to_figure_id[old_id]
                    # Update question_id
                    data['question_id'] = new_id
                    
                    # Verify question text matches just to be safe
                    original_question = reasoning_data[new_id]['query']
                    if 'question' in data and data['question'].strip() != original_question.strip():
                        print(f"Warning: Question text mismatch for ID {new_id}!")
                
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                
        # Replace original file with fixed file
        os.replace(temp_filepath, filepath)
        print(f"Fixed {filepath}")
        
    # 3. Verify
    images_dir = "data/CharXiv/images"
    image_ids = set()
    for f in os.listdir(images_dir):
        if f.endswith(".jpg"):
            image_ids.add(f.replace(".jpg", ""))
            
    print(f"\nVerification:")
    print(f"Found {len(image_ids)} images in {images_dir}")
    
    for filepath in files_to_fix:
        if not os.path.exists(filepath):
            continue
        jsonl_ids = set()
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    jsonl_ids.add(str(data.get("question_id")))
        
        match_count = len(image_ids.intersection(jsonl_ids))
        print(f"{os.path.basename(filepath)} has {len(jsonl_ids)} unique IDs.")
        print(f"Matches with images: {match_count}")
        
if __name__ == "__main__":
    main()
