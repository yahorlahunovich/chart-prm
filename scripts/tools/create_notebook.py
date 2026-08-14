"""Script to generate notebooks/evaluate_rollouts.ipynb with publication-quality NeurIPS style plots."""

import json

cells = []


def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text + "\n"]})


def add_code(text):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")],
        }
    )


add_md(
    "# PRM Rollouts Evaluation Analysis\n\nThis notebook provides publication-quality analysis of evaluated rollouts from our Process Reward Model experiments."
)

add_code(
    """import json
import os
import sys
from pathlib import Path

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add repository root to path for visualization imports
sys.path.insert(0, os.path.abspath('..'))
from src.visualization.style import setup_plot_style, PALETTE

# Apply NeurIPS/ICML research plotting style
setup_plot_style()

# Ensure charts directory exists
os.makedirs('charts', exist_ok=True)"""
)

add_md(
    "## Data Loading\nLoading the evaluated rollouts and joining with metadata."
)

add_code(
    """# Load metadata
with open('../data/CharXiv/data/image_metadata_val.json', 'r') as f:
    meta = json.load(f)

with open('../data/CharXiv/data/chart_types_val.json', 'r') as f:
    chart_types = json.load(f)

# Load the dataset
data_path = Path('../experiments/001_500_reasoning/data/evaluated_rollouts.jsonl')
data = []
with open(data_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Flatten into step-level dataframe
rows = []
for item in data:
    q_id = str(item['question_id'])
    r_idx = item['rollout_index']
    cat = meta.get(q_id, {}).get('category', 'unknown')
    
    ctype_list = chart_types.get(q_id, {}).get('chart_types', ['unknown'])
    ctype = ctype_list[0] if len(ctype_list) > 0 else 'unknown'
    
    evals = item.get('evaluations') or []
    for i, step in enumerate(evals):
        rows.append({
            'question_id': q_id,
            'rollout_index': r_idx,
            'category': cat,
            'chart_type': ctype,
            'step_index': step.get('step_index', i),
            'score': step.get('score'),
            'analysis_len': len(step.get('analysis', '')),
            'total_steps': len(evals)
        })

df_steps = pd.DataFrame(rows)
display(df_steps.head())"""
)

# 1. Overall Step-Level Accuracy Distribution
add_md(
    "## 1. Overall Step-Level Accuracy Distribution\nThis plot shows the overall distribution of scores (0 and 1) across all individual reasoning steps."
)

add_code(
    """fig, ax = plt.subplots(figsize=(6, 4))
score_counts = df_steps['score'].value_counts().sort_index()
colors = [PALETTE["ours"], PALETTE["prm"]]
labels = ['0 (Incorrect)', '1 (Correct)']

bars = ax.bar(labels, score_counts.values, color=colors, width=0.45, edgecolor='none')
ax.set_title('Overall Step-Level Accuracy Distribution', loc='left', pad=10)
ax.set_xlabel('Step Score')
ax.set_ylabel('Count of Steps')
ax.set_ylim(0, max(score_counts.values) * 1.15)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{int(height):,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords='offset points',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.savefig('charts/01_overall_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()"""
)

# 2. Score Progression by Step Index
add_md(
    "## 2. Score Progression by Step Index\nHow does the average score change as we progress deeper into a rollout? Are later steps more likely to be wrong?"
)

add_code(
    """fig, ax = plt.subplots(figsize=(7, 4.2))
step_counts = df_steps['step_index'].value_counts()
valid_steps = step_counts[step_counts > 10].index
df_filtered = df_steps[df_steps['step_index'].isin(valid_steps)]

sns.lineplot(
    data=df_filtered,
    x='step_index',
    y='score',
    color=PALETTE["prm"],
    marker='o',
    markersize=6,
    linewidth=2.0,
    errorbar=('ci', 95),
    ax=ax
)

ax.set_title('Average Score Progression by Step Index', loc='left', pad=10)
ax.set_xlabel('Step Index')
ax.set_ylabel('Mean Accuracy')
ax.set_ylim(0.0, 1.0)
ax.set_xticks(sorted(valid_steps))

plt.savefig('charts/02_score_progression.png', dpi=300, bbox_inches='tight')
plt.show()"""
)

# 3. Rollout Success Rate
add_md(
    "## 3. Rollout Success Rate\nA rollout is entirely correct if all its steps have a score of 1. Here we visualize the proportion of perfect rollouts vs those with at least one error."
)

add_code(
    """rollout_success = df_steps.groupby(['question_id', 'rollout_index'])['score'].min().reset_index()
rollout_success['status'] = rollout_success['score'].map({1: 'Perfect (All 1s)', 0: 'Has Errors (Min 0)'})

fig, ax = plt.subplots(figsize=(6, 4))
status_counts = rollout_success['status'].value_counts()[['Perfect (All 1s)', 'Has Errors (Min 0)']]
colors = [PALETTE["prm"], PALETTE["ours"]]

bars = ax.bar(status_counts.index, status_counts.values, color=colors, width=0.45)
ax.set_title('Rollout Success Rate (Sequence Correctness)', loc='left', pad=10)
ax.set_xlabel('Rollout Status')
ax.set_ylabel('Count of Rollouts')
ax.set_ylim(0, max(status_counts.values) * 1.15)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{int(height):,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords='offset points',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.savefig('charts/03_rollout_success.png', dpi=300, bbox_inches='tight')
plt.show()"""
)

# 4. First Error Position Analysis
add_md(
    "## 4. First Error Position Analysis\nFor rollouts that have errors, at which step index does the model typically make its first mistake?"
)

add_code(
    """errors = df_steps[df_steps['score'] == 0]
first_errors = errors.groupby(['question_id', 'rollout_index'])['step_index'].min().reset_index()
err_counts = first_errors['step_index'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar(err_counts.index.astype(str), err_counts.values, color=PALETTE["dpo"], width=0.55)
ax.set_title('Position of First Error in Failed Rollouts', loc='left', pad=10)
ax.set_xlabel('Step Index of First Error')
ax.set_ylabel('Count of Rollouts')
ax.set_ylim(0, max(err_counts.values) * 1.15)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{int(height):,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords='offset points',
                ha='center', va='bottom', fontsize=9)

plt.savefig('charts/04_first_error_position.png', dpi=300, bbox_inches='tight')
plt.show()"""
)

# 5. Question Difficulty Ranking
add_md(
    "## 5. Question Difficulty Ranking\nDistribution of average rollout success rate across different questions. This shows if some questions are universally hard or easy."
)

add_code(
    """q_acc = df_steps.groupby('question_id')['score'].mean().reset_index()

fig, ax = plt.subplots(figsize=(7, 4.2))
sns.histplot(
    data=q_acc,
    x='score',
    bins=15,
    kde=True,
    color=PALETTE["sft"],
    edgecolor='white',
    linewidth=0.8,
    ax=ax
)
ax.set_title('Question Difficulty Distribution', loc='left', pad=10)
ax.set_xlabel('Mean Accuracy per Question')
ax.set_ylabel('Number of Questions')
ax.set_xlim(0.0, 1.0)

plt.savefig('charts/05_question_difficulty.png', dpi=300, bbox_inches='tight')
plt.show()"""
)

# 6. Rollout Length vs. Accuracy
add_md(
    "## 6. Rollout Length vs. Accuracy\nDoes the total number of steps in a rollout correlate with the average accuracy of its steps?"
)

add_code(
    """rollout_stats = df_steps.groupby(['question_id', 'rollout_index']).agg({
    'score': 'mean',
    'total_steps': 'first'
}).reset_index()

fig, ax = plt.subplots(figsize=(7, 4.2))
sns.boxplot(
    data=rollout_stats,
    x='total_steps',
    y='score',
    color=PALETTE["sft"],
    width=0.45,
    fliersize=3,
    linewidth=1.2,
    ax=ax
)
ax.set_title('Rollout Length vs. Mean Step Score', loc='left', pad=10)
ax.set_xlabel('Total Steps in Rollout')
ax.set_ylabel('Mean Step Score')
ax.set_ylim(-0.05, 1.05)

plt.savefig('charts/06_length_vs_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()"""
)

# 7. Score Variance per Question
add_md(
    "## 7. Score Variance per Question\nHow much variance is there in step scores for a given question? A high variance implies the model is inconsistent on that question."
)

add_code(
    """q_var = df_steps.groupby('question_id')['score'].var().fillna(0).reset_index()

fig, ax = plt.subplots(figsize=(7, 4.2))
sns.histplot(
    data=q_var,
    x='score',
    bins=15,
    color=PALETTE["dpo"],
    edgecolor='white',
    linewidth=0.8,
    ax=ax
)
ax.set_title('Step Score Variance per Question', loc='left', pad=10)
ax.set_xlabel('Score Variance')
ax.set_ylabel('Number of Questions')

plt.savefig('charts/07_score_variance.png', dpi=300, bbox_inches='tight')
plt.show()"""
)

# 8. Error Cascade Analysis
add_md(
    "## 8. Error Cascade Analysis\nIf step N is incorrect, what is the score distribution for step N+1? Does an error cascade into more errors?"
)

add_code(
    """df_steps_sorted = df_steps.sort_values(['question_id', 'rollout_index', 'step_index']).copy()
df_steps_sorted['prev_score'] = df_steps_sorted.groupby(['question_id', 'rollout_index'])['score'].shift(1)
cascade_data = df_steps_sorted.dropna(subset=['prev_score']).copy()

fig, ax = plt.subplots(figsize=(7, 4.2))
palette_cascade = {0: PALETTE["ours"], 1: PALETTE["prm"]}

sns.countplot(
    data=cascade_data,
    x='prev_score',
    hue='score',
    palette=palette_cascade,
    ax=ax
)
ax.set_title('Error Cascade Analysis (Step N+1 Score | Step N Score)', loc='left', pad=10)
ax.set_xlabel('Score at Step N')
ax.set_ylabel('Count of Steps N+1')
ax.set_xticks([0, 1])
ax.set_xticklabels(['0 (Incorrect)', '1 (Correct)'])

leg = ax.legend(title='Score at N+1', loc='upper right', frameon=False)
for text, label in zip(leg.get_texts(), ['0 (Incorrect)', '1 (Correct)']):
    text.set_text(label)

plt.savefig('charts/08_error_cascade.png', dpi=300, bbox_inches='tight')
plt.show()"""
)

# 9. Analysis Text Length Correlation
add_md(
    "## 9. Analysis Text Length Correlation\nDoes the length of the PRM's textual analysis correlate with the score? Do negative scores require longer explanations?"
)

add_code(
    """fig, ax = plt.subplots(figsize=(6, 4))
palette_box = {0: PALETTE["ours"], 1: PALETTE["prm"]}

sns.boxplot(
    data=df_steps,
    x='score',
    y='analysis_len',
    hue='score',
    palette=palette_box,
    legend=False,
    width=0.45,
    fliersize=3,
    linewidth=1.2,
    ax=ax
)
ax.set_title('PRM Judge Explanation Length by Step Score', loc='left', pad=10)
ax.set_xlabel('Step Score')
ax.set_ylabel('Analysis Length (characters)')
ax.set_xticks([0, 1])
ax.set_xticklabels(['0 (Incorrect)', '1 (Correct)'])

plt.savefig('charts/09_analysis_length.png', dpi=300, bbox_inches='tight')
plt.show()"""
)

# 10. Terminal State Analysis
add_md(
    "## 10. Terminal State Analysis\nThe correctness of the final step often dictates the final answer. What is the accuracy of exclusively the final steps?"
)

add_code(
    """terminal_steps = df_steps[df_steps['step_index'] == (df_steps['total_steps'] - 1)]
term_counts = terminal_steps['score'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(6, 4))
colors = [PALETTE["ours"], PALETTE["prm"]]
labels = ['0 (Incorrect)', '1 (Correct)']

bars = ax.bar(labels, term_counts.values, color=colors, width=0.45)
ax.set_title('Terminal State (Final Step) Accuracy', loc='left', pad=10)
ax.set_xlabel('Final Step Score')
ax.set_ylabel('Count of Final Steps')
ax.set_ylim(0, max(term_counts.values) * 1.15)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{int(height):,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords='offset points',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.savefig('charts/10_terminal_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()"""
)

# 11. Domain (Category) Analysis
add_md(
    "## 11. Domain (Category) Analysis\nAnalyzing the accuracy across different academic domains or categories from CharXiv metadata."
)

add_code(
    """cat_acc = df_steps.groupby('category')['score'].mean().sort_values(ascending=False).reset_index()

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(cat_acc['category'], cat_acc['score'], color=PALETTE["sft"], width=0.55)
ax.set_title('Average Reasoning Accuracy by Academic Category', loc='left', pad=10)
ax.set_xlabel('Academic Category')
ax.set_ylabel('Mean Step Score')
ax.set_ylim(0.0, 1.0)
plt.xticks(rotation=35, ha='right')

plt.savefig('charts/11_domain_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()"""
)

# 12. Chart Type Analysis
add_md(
    "## 12. Chart Type Analysis\nAnalyzing the accuracy broken down by the top 15 most frequent chart types."
)

add_code(
    """top_chart_types = df_steps['chart_type'].value_counts().nlargest(15).index
df_top_charts = df_steps[df_steps['chart_type'].isin(top_chart_types)]
chart_acc = df_top_charts.groupby('chart_type')['score'].mean().sort_values(ascending=False).reset_index()

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(chart_acc['chart_type'], chart_acc['score'], color=PALETTE["sft"], width=0.6)
ax.set_title('Average Reasoning Accuracy by Chart Type (Top 15)', loc='left', pad=10)
ax.set_xlabel('Chart Type')
ax.set_ylabel('Mean Step Score')
ax.set_ylim(0.0, 1.0)
plt.xticks(rotation=40, ha='right')

plt.savefig('charts/12_chart_type_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()"""
)

# 13. Hallucinated Correctness (Recovery from Error)
add_md(
    "## 13. Hallucinated Correctness (Recovery from Error)\nSpecifically investigating cases where Step N was scored 0, but Step N+1 was scored 1."
)

add_code(
    """recovery_cases = cascade_data[(cascade_data['prev_score'] == 0) & (cascade_data['score'] == 1)]
print(f"Total steps that recovered from 0 to 1: {len(recovery_cases)}")

fig, ax = plt.subplots(figsize=(7, 4.2))
if len(recovery_cases) > 0:
    rec_counts = recovery_cases['step_index'].value_counts().sort_index()
    bars = ax.bar(rec_counts.index.astype(str), rec_counts.values, color=PALETTE["dpo"], width=0.5)
    ax.set_title('Error Recovery Step Index (0 -> 1 Transition)', loc='left', pad=10)
    ax.set_xlabel('Step Index where Score became 1')
    ax.set_ylabel('Count of Occurrences')
    ax.set_ylim(0, max(rec_counts.values) * 1.15)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)
else:
    ax.text(0.5, 0.5, "No recovery cases found", ha='center', va='center')

plt.savefig('charts/13_hallucinated_correctness.png', dpi=300, bbox_inches='tight')
plt.show()"""
)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("notebooks/evaluate_rollouts.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print(
    "notebooks/evaluate_rollouts.ipynb generated successfully with publication style."
)
