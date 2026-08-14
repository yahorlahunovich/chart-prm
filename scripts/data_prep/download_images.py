"""
download_images.py

This script downloads the chart images referenced in the CharXiv dataset from HuggingFace.
It fetches only the necessary images for our 500 reasoning questions subset, unzips them, 
and stores them in the local `data/CharXiv/images/` directory for use by the Meta API judge.
"""
import argparse
import json
import os
from pathlib import Path
import urllib.request
import zipfile
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ids-file",
        default="data/splits/main_reasoning_ids.json",
        help="JSON file containing the image IDs to extract.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    ids_path = Path(args.ids_file)
    if not ids_path.is_absolute():
        ids_path = base_dir / ids_path

    with ids_path.open("r", encoding="utf-8") as f:
        target_ids = set(json.load(f))
    
    print(f"Loaded {len(target_ids)} target IDs.")
    
    url = "https://huggingface.co/datasets/princeton-nlp/CharXiv/resolve/main/images.zip"
    output_dir = base_dir / "data/CharXiv/images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "images.zip")
        print(f"Downloading images.zip from {url}...")
        
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out_file:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                
        print("Download complete. Extracting required images...")
        
        extracted_count = 0
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                filename = file_info.filename
                basename = os.path.basename(filename)
                name, ext = os.path.splitext(basename)
                
                if name in target_ids:
                    file_info.filename = basename # remove folder structure
                    zip_ref.extract(file_info, str(output_dir))
                    extracted_count += 1
                    
        print(f"Extracted {extracted_count} images to {output_dir}")

if __name__ == "__main__":
    main()
