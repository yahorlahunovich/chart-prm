"""
Script to generate notebooks/train_kto.ipynb for Qwen2.5-VL KTO fine-tuning.
"""

import json
from pathlib import Path

def main():
    cells = []

    def add_md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [text + "\n"]})

    def add_code(text):
        lines = [line + "\n" for line in text.strip().split('\n')]
        # Remove trailing newline from last line for clean json formatting
        if lines:
            lines[-1] = lines[-1].rstrip('\n')
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines})

    add_md("""# Qwen2.5-VL KTO Fine-Tuning (Kaggle GPU)

This notebook trains a QLoRA adapter on PRM-evaluated rollouts using TRL's `KTOTrainer` (Kahneman-Tversky Optimization) to align step-by-step chart reasoning logic.""")

    add_code("""# === Cell 1: Environment Setup ===
import os, sys

# Load HF_TOKEN from Kaggle secrets if available
try:
    from kaggle_secrets import UserSecretsClient
    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
    print('HF_TOKEN loaded from Kaggle secrets.')
except Exception:
    print('Kaggle secrets unavailable, skipping HF_TOKEN.')

# Clone project repo to /tmp to keep /kaggle/working clean
repo_dir = '/tmp/prm_project'
if not os.path.exists(repo_dir):
    !git clone https://github.com/yahorlahunovich/prm_project.git {repo_dir}
else:
    !git -C {repo_dir} pull --ff-only

if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
os.chdir(repo_dir)

# Diagnostic environment print
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
    print(f'VRAM: {props.total_memory / 1e9:.1f} GB')""")

    add_code("""# === Cell 2: Install Dependencies ===
import torch as _t
_tv = _t.__version__
print(f'Pinning torch=={_tv}')

!pip install -q \\
    "torch=={_tv}" \\
    "transformers>=4.49.0" \\
    "trl>=0.12.0" \\
    "peft>=0.10.0" \\
    "accelerate>=0.30.0" \\
    "datasets" \\
    "qwen-vl-utils"

import transformers, trl, peft, accelerate
print(f'transformers: {transformers.__version__}')
print(f'trl: {trl.__version__}')
print(f'peft: {peft.__version__}')
print(f'accelerate: {accelerate.__version__}')""")

    add_code("""# === Cell 3: Load Dataset ===
import json, os, torch
from datasets import Dataset
from PIL import Image

# Ensure chart images are downloaded
images_dir = 'data/CharXiv/images'
if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
    print('Downloading chart images...')
    os.system('python scripts/data_prep/download_images.py')

# Ensure KTO samples are generated
data_path = 'experiments/001_500_reasoning/data/kto_samples.jsonl'
if not os.path.exists(data_path):
    print('Formatting KTO dataset...')
    os.system('python scripts/data_prep/format_kto.py')

with open(data_path) as f:
    raw_data = [json.loads(line) for line in f]

hf_data = {'prompt': [], 'completion': [], 'label': [], 'images': []}
skipped = 0

for item in raw_data:
    img_path = os.path.abspath(item['image_path'])
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        skipped += 1
        continue

    question = item.get('question', '').strip()
    prompt_text = f'Analyze this chart. Provide step-by-step reasoning and a final answer.\\n{question}'

    prompt_msg = [{
        'role': 'user',
        'content': [
            {'type': 'image'},
            {'type': 'text', 'text': prompt_text}
        ]
    }]

    prefix = item.get('prefix', '')
    completion_str = prefix + item['completion']
    completion_msg = [{'role': 'assistant', 'content': completion_str}]

    hf_data['prompt'].append(prompt_msg)
    hf_data['completion'].append(completion_msg)
    hf_data['label'].append(bool(item['label']))
    hf_data['images'].append([img])

dataset = Dataset.from_dict(hf_data)
pos_count = sum(hf_data['label'])
neg_count = len(hf_data['label']) - pos_count
print(f'Loaded {len(dataset)} KTO samples (Pos: {pos_count}, Neg: {neg_count}, skipped {skipped} due to missing images).')
print(dataset)""")

    add_code("""# === Cell 4: Load Model & Processor ===
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

model_id = 'Qwen/Qwen2.5-VL-3B-Instruct'

# Determine safe device placement based on CUDA capability
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    if cc[0] < 7:
        print(f'GPU {torch.cuda.get_device_name(0)} (cc {cc}) lacks PyTorch 2.10 sm_70+ support. Using CPU.')
        device_map = 'cpu'
    else:
        print(f'Using GPU {torch.cuda.get_device_name(0)} (cc {cc}).')
        device_map = {'': 0}
else:
    device_map = 'cpu'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device_map != 'cpu' else torch.float32,
    attn_implementation='sdpa' if device_map != 'cpu' else 'eager',
    device_map=device_map,
)
model.enable_input_require_grads()

# Freeze vision encoder — only train language model LoRA
if hasattr(model, 'visual'):
    model.visual.requires_grad_(False)

processor = AutoProcessor.from_pretrained(
    model_id,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,
)
processor.tokenizer.padding_side = 'right'
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

# Expose token properties directly on processor for KTOTrainer
processor.pad_token = processor.tokenizer.pad_token
processor.pad_token_id = processor.tokenizer.pad_token_id
processor.eos_token_id = processor.tokenizer.eos_token_id

print(f'Model loaded on device: {next(model.parameters()).device}')""")

    add_code("""# === Cell 5: Configure LoRA & KTO Trainer ===
from peft import LoraConfig
from trl import KTOTrainer, KTOConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    task_type='CAUSAL_LM',
)

is_gpu = next(model.parameters()).device.type == 'cuda'
out_dir = '/kaggle/working/kto_qwen_vl' if os.path.exists('/kaggle/working') else './kto_qwen_vl'

training_args = KTOConfig(
    output_dir=out_dir,
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=3,
    max_length=2048,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    gradient_checkpointing=is_gpu,
    dataset_num_proc=1,
    remove_unused_columns=False,
    report_to='none',
    fp16=is_gpu,
    bf16=False,
)

print('Initializing KTOTrainer...')
trainer = KTOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    processing_class=processor,
    peft_config=peft_config,
)
print('KTOTrainer initialized successfully.')""")

    add_code("""# === Cell 6: Train & Save ===
print('Starting KTO training...')
trainer.train()
save_path = '/kaggle/working/qwen_vl_kto_adapter' if os.path.exists('/kaggle/working') else 'qwen_vl_kto_adapter'
trainer.save_model(save_path)
print(f'\\nTraining complete! Adapter saved to {save_path}')""")

    add_code("""# === Cell 7: Check Output ===
import os
save_path = '/kaggle/working/qwen_vl_kto_adapter' if os.path.exists('/kaggle/working') else 'qwen_vl_kto_adapter'
if os.path.exists(save_path):
    print(f'Files in {save_path}:')
    for f in os.listdir(save_path):
        size = os.path.getsize(os.path.join(save_path, f)) / 1e6
        print(f' - {f}: {size:.2f} MB')
else:
    print(f'Save path {save_path} does not exist.')""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    base_dir = Path(__file__).resolve().parents[2]
    nb_path = base_dir / 'notebooks/train_kto.ipynb'

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

    print(f"Successfully generated {nb_path}")

if __name__ == '__main__':
    main()
