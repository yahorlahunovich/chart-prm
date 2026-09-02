#!/usr/bin/env python3
"""
build_reward_tree.py

Phase 1 of a DG-PRM-style reward tree for ChartPRM (Yin et al., "Dynamic and
Generalizable Process Reward Modeling"). DG-PRM discovers coarse parent
criteria from scratch by clustering LLM judgments over positive/negative
pairs; ChartPRM already has 9 human-validated parent categories from
`categorize_judge_errors.py`'s regex taxonomy over 2,920 real judge failure
explanations, so this script only needs to build the missing child
(fine-grained) layer, reusing the existing per-failure MiniLM embeddings.

This is a v0 / bootstrap tree: children are embedding sub-clusters within
each parent, described by their TF-IDF top terms and exemplar judge
sentences (both computed with the existing `cluster_terms`/`exemplar_indices`
helpers from `categorize_judge_errors.py`) rather than LLM-distilled rubric
statements. No API key or new judge calls are needed for this step. The
next step (Phase 2: re-scoring rollouts against retrieved criteria) is what
actually needs a vision-capable judge, and is deliberately not done here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from categorize_judge_errors import CATEGORY_ORDER, DISPLAY, cluster_terms, exemplar_indices  # noqa: E402
from chart_prm.reward_tree import choose_child_count, merge_by_distance  # noqa: E402

EXCLUDED_PARENTS = {"other_unspecified"}  # catch-all bucket, not a real criterion
DEFAULT_MERGE_THRESHOLD = 0.25  # DG-PRM's xi, as published — see note in metrics.md about retuning
RETRIEVAL_THRESHOLD = 0.2  # DG-PRM's zeta, used at retrieval time (Phase 2), documented here
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_children_for_parent(
    parent_key: str, texts: pd.Series, embeddings: np.ndarray
) -> list[dict]:
    n = len(texts)
    k = choose_child_count(n)
    if k <= 1:
        centroid = embeddings.mean(axis=0)
        return [
            {
                "child_id": f"{parent_key}_0",
                "top_terms": [],
                "exemplars": texts.head(2).tolist(),
                "member_count": n,
                "embedding": centroid.tolist(),
            }
        ]

    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
    try:
        terms = cluster_terms(texts, labels, k)
    except ValueError:
        terms = {c: [] for c in range(k)}

    children = []
    for c in range(k):
        mask = labels == c
        member_count = int(mask.sum())
        if member_count == 0:
            continue
        centroid = embeddings[mask].mean(axis=0)
        exemplar_idx = exemplar_indices(embeddings, labels, c, n=2)
        children.append(
            {
                "child_id": f"{parent_key}_{c}",
                "top_terms": terms.get(c, []),
                "exemplars": [texts.iloc[i] for i in exemplar_idx],
                "member_count": member_count,
                "embedding": centroid.tolist(),
            }
        )
    return children


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/001_500_reasoning/judge_error_analysis"),
        help="Directory with fail_analyses_categorized.csv and fail_analysis_embeddings.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/009_reward_tree"),
        help="Where to write the reward tree JSON and metrics.md",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=DEFAULT_MERGE_THRESHOLD,
        help="Cosine-distance threshold (DG-PRM's xi) for folding near-duplicate child clusters together",
    )
    args = parser.parse_args()
    merge_threshold = args.merge_threshold

    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir
    data_dir = out_dir / "data"
    figures_dir = out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_dir / "fail_analyses_categorized.csv")
    embeddings = np.load(in_dir / "fail_analysis_embeddings.npy")
    if len(df) != len(embeddings):
        raise SystemExit(f"Row/embedding mismatch: {len(df)} vs {len(embeddings)}")

    parents: dict[str, dict] = {}
    for parent_key in CATEGORY_ORDER:
        if parent_key in EXCLUDED_PARENTS:
            continue
        mask = (df["error_category"] == parent_key).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        parent_texts = df.loc[mask, "analysis"].reset_index(drop=True)
        parent_embeddings = embeddings[mask]

        raw_children = build_children_for_parent(parent_key, parent_texts, parent_embeddings)
        merged_children = merge_by_distance(raw_children, threshold=merge_threshold)

        parents[parent_key] = {
            "label": DISPLAY[parent_key],
            "n_source_failures": n,
            "children": merged_children,
        }

    tree = {
        "merge_threshold_xi": merge_threshold,
        "retrieval_threshold_zeta": RETRIEVAL_THRESHOLD,
        "embedding_model": EMBEDDING_MODEL,
        "source": str(in_dir / "fail_analyses_categorized.csv"),
        "child_criteria_status": "v0: embedding sub-clusters described by TF-IDF terms + exemplars, "
        "not yet LLM-distilled into rubric statements (pending judge API access)",
        "parents": parents,
    }

    tree_path = data_dir / "reward_tree.json"
    tree_path.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")

    n_parents = len(parents)
    n_children = sum(len(p["children"]) for p in parents.values())
    child_counts = {k: len(p["children"]) for k, p in parents.items()}

    print(f"Parents: {n_parents}  Total children: {n_children}")
    for key, count in child_counts.items():
        print(f"  {DISPLAY[key]:45s} {count} children  ({parents[key]['n_source_failures']} source failures)")
    print(f"\nWrote {tree_path}")

    write_figure(parents, figures_dir)
    write_metrics_md(parents, n_parents, n_children, merge_threshold, out_dir)


def write_figure(parents: dict, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.visualization.style import setup_plot_style

    setup_plot_style()

    labels = [p["label"] for p in parents.values()]
    child_counts = [len(p["children"]) for p in parents.values()]
    source_counts = [p["n_source_failures"] for p in parents.values()]
    order = np.argsort(source_counts)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(order))
    ax.barh(y, [child_counts[i] for i in order], color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Number of child criteria")
    ax.set_title("Reward tree: child criteria per parent category", loc="left", pad=10)
    for i, idx in enumerate(order):
        ax.text(child_counts[idx] + 0.05, i, f"{child_counts[idx]}  (n={source_counts[idx]})", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "reward_tree_children_per_parent.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {figures_dir / 'reward_tree_children_per_parent.png'}")


def write_metrics_md(
    parents: dict, n_parents: int, n_children: int, merge_threshold: float, out_dir: Path
) -> None:
    # Diagnostic: for each parent, what fraction of its members ended up folded
    # into the single largest child — a direct measure of over-merging.
    dominance = {
        key: max(c["member_count"] for c in p["children"]) / p["n_source_failures"]
        for key, p in parents.items()
    }
    mean_dominance = sum(dominance.values()) / len(dominance)
    max_dominance_key = max(dominance, key=dominance.get)

    lines = [
        "# Experiment 009: Reward Tree (DG-PRM Phase 1, v0)\n",
        "\n",
        "Adapts the reward-tree component of Yin et al., \"Dynamic and Generalizable "
        "Process Reward Modeling\" (DG-PRM) to ChartPRM.\n",
        "\n",
        "## What this is (and isn't)\n",
        "\n",
        "- **Parents**: the 9 human-validated error categories from "
        "`scripts/evaluation/categorize_judge_errors.py`'s regex taxonomy over 2,920 real "
        "judge failure explanations. DG-PRM discovers parents from scratch; ChartPRM already "
        "had them, so they're reused as-is (`other_unspecified` excluded — it's a catch-all, "
        "not a criterion).\n",
        "- **Children**: embedding sub-clusters within each parent (KMeans on the existing "
        "MiniLM `fail_analysis_embeddings.npy`, `k` scaled to category size via "
        "`choose_child_count`), described by TF-IDF top terms and 2 exemplar judge sentences, "
        "then deduplicated by merging near-identical sub-clusters (cosine distance <= "
        f"{merge_threshold}, DG-PRM's `xi`, taken as-published pending retuning — see Observation "
        "below).\n",
        "- **This is v0.** Child criteria are cluster descriptions, not the clean, reusable "
        "rubric statements ('axis values must be read from the correct gridline', etc.) DG-PRM's "
        "Phase 1 gets from an LLM judge examining contrastive pairs. That LLM-distillation pass "
        "is next, gated on getting a working (Gemini) judge API key — it's a text-only pass "
        "over these same clusters, no new rollout generation needed.\n",
        "- **No new judge/API calls were made for this step.** Everything here is recomputed "
        "from data already in the repo (`fail_analyses_categorized.csv` + "
        "`fail_analysis_embeddings.npy`).\n",
        "\n",
        f"## Tree stats\n\n- Parents: **{n_parents}**\n- Total child criteria: **{n_children}**\n"
        f"- Merge threshold (xi): {merge_threshold}\n- Retrieval threshold (zeta, for Phase 2): "
        f"{RETRIEVAL_THRESHOLD}\n- Embedding model: {EMBEDDING_MODEL}\n",
        f"\n## Observation: merge dominance at xi={merge_threshold}\n\n"
        f"Averaged across parents, **{mean_dominance:.0%}** of a parent's failures end up folded "
        f"into its single largest child after merging at this threshold — worst case is "
        f"**{parents[max_dominance_key]['label']}** at {dominance[max_dominance_key]:.0%}. Higher "
        "dominance means fewer, coarser children (less useful for Phase 2 retrieval); lower "
        "dominance keeps more distinct children but risks near-duplicates surviving as separate "
        "criteria. DG-PRM's published default is xi=0.25, tuned on their own domain/embedding "
        "model; their own ablation (Figure 10a in the paper) shows performance is sensitive to "
        "this constant, so it should be swept locally (`--merge-threshold`) rather than assumed "
        "to transfer as-is to short chart-critique sentences on this embedding model. Compare "
        "this run's dominance numbers against a run at a different threshold before picking one "
        "for Phase 2.\n",
        "\n## Children per parent\n\n| Parent | Source failures | Children | Largest child share |\n"
        "| --- | ---: | ---: | ---: |\n",
    ]
    for key, parent in sorted(parents.items(), key=lambda kv: -kv[1]["n_source_failures"]):
        lines.append(
            f"| {parent['label']} | {parent['n_source_failures']} | {len(parent['children'])} | "
            f"{dominance[key]:.0%} |\n"
        )

    lines.append("\n![Children per parent](figures/reward_tree_children_per_parent.png)\n")
    lines.append("\n## Full tree\n\nSee [`data/reward_tree.json`](data/reward_tree.json).\n")

    metrics_path = out_dir / "metrics.md"
    metrics_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
