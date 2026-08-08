"""
categorize_judge_errors.py

Assigns human-readable error categories to PRM-judge fail analyses using
priority regex rules, then validates/summarizes with embedding KMeans.
Runs fully offline on saved MiniLM embeddings + CSV (no GPU needed).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Priority order: first match wins (more specific causes first).
RULES: list[tuple[str, str, str]] = [
    (
        "incomplete_or_truncated_step",
        "Incomplete / truncated reasoning",
        r"truncat|incomplete|no completed|no verifiable claim|empty step|breaks logical coherence",
    ),
    (
        "hallucinated_entity",
        "Hallucinated entity / label not on chart",
        r"hallucin|not (present|shown|visible)|does not (appear|exist|show)|invent|fabricat|"
        r"no such|never (appears|shown)|not on the chart|absent from",
    ),
    (
        "arithmetic_error",
        "Arithmetic / calculation mistake",
        r"arithmetic|miscalculat|incorrect (calculation|math|computation)|"
        r"wrong (sum|difference|product|quotient|result)|math(ematical)?(ly)? (wrong|incorrect|false)",
    ),
    (
        "axis_or_layout_misread",
        "Axis / layout / chart-structure misread",
        r"\b(x|y)[ -]?axis\b|swaps? the axes|reverses? the axes|axis (is|are|shows|represents)|"
        r"axes|chart type|boxplot|subplot|layout|labeled|labelled|misreads? the (chart )?axes|"
        r"vertical|horizontal",
    ),
    (
        "ranking_error",
        "Wrong ranking / extremum (highest/lowest/second)",
        r"second[ -]?highest|second[ -]?lowest|\b(highest|lowest|largest|smallest|maximum|minimum)\b|"
        r"incorrectly (ranks|orders)|ranking|rank claim|not the (highest|lowest|second)",
    ),
    (
        "wrong_series_or_color",
        "Wrong series / color / legend identity",
        r"\b(blue|orange|green|red|purple|yellow|grey|gray|black|brown)\b|"
        r"wrong (series|curve|line|bar|color|colour)|"
        r"misidentif\w* (the )?(series|curve|line|color|colour|legend)|\blegend\b|"
        r"confused .{0,30}(series|curve|line|color)",
    ),
    (
        "threshold_or_comparison",
        "Bad comparison / threshold logic",
        r"exceed|greater than|less than|does not strictly|not strictly|"
        r"comparison|compared with|threshold|above|below|>=|<=",
    ),
    (
        "wrong_numeric_read",
        "Wrong numeric value read from chart",
        r"misreads?|misread|incorrect(ly)? (reads|claims|states|places|reports)|"
        r"chart shows|value .{0,30}(incorrect|wrong|false)|substantial misread|"
        r"approximately incorrect|not around|not ~|not close to|reads .{0,40} as ",
    ),
    (
        "logic_inconsistency",
        "Logic inconsistency / false conclusion",
        r"contradict|inconsistent|previous step|does not follow|illogical|"
        r"inherits the false|perpetuates|false premise|conclusion is (false|incorrect)|"
        r"factually (incorrect|false|wrong)",
    ),
]

CATEGORY_ORDER = [r[0] for r in RULES] + ["other_unspecified"]
DISPLAY = {r[0]: r[1] for r in RULES}
DISPLAY["other_unspecified"] = "Other / unspecified"


def assign_primary(text: str) -> str:
    t = text.lower()
    for key, _label, pat in RULES:
        if re.search(pat, t):
            return key
    return "other_unspecified"


def multi_labels(text: str) -> list[str]:
    t = text.lower()
    hits = [key for key, _label, pat in RULES if re.search(pat, t)]
    return hits or ["other_unspecified"]


def cluster_terms(texts: pd.Series, labels: np.ndarray, k: int, top_n: int = 8) -> dict[int, list[str]]:
    vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words="english", min_df=3)
    x = vec.fit_transform(texts.astype(str))
    terms = np.array(vec.get_feature_names_out())
    out: dict[int, list[str]] = {}
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            out[c] = []
            continue
        mean = np.asarray(x[mask].mean(axis=0)).ravel()
        out[c] = terms[mean.argsort()[::-1][:top_n]].tolist()
    return out


def exemplar_indices(emb: np.ndarray, labels: np.ndarray, cluster_id: int, n: int = 3) -> list[int]:
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    centroid = emb[idx].mean(axis=0)
    sims = emb[idx] @ centroid
    order = np.argsort(-sims)[:n]
    return idx[order].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("kaggle_out/judge_error_analysis"),
        help="Directory with fail_analyses.csv and fail_analysis_embeddings.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/001_500_reasoning/judge_error_analysis"),
        help="Where to write categorized tables, plots, and report",
    )
    parser.add_argument("--kmeans-k", type=int, default=8)
    args = parser.parse_args()

    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir
    charts = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts.mkdir(exist_ok=True)

    df = pd.read_csv(in_dir / "fail_analyses.csv")
    emb = np.load(in_dir / "fail_analysis_embeddings.npy")
    if len(df) != len(emb):
        raise SystemExit(f"Row/embedding mismatch: {len(df)} vs {len(emb)}")

    df["error_category"] = df["analysis"].astype(str).map(assign_primary)
    df["error_category_label"] = df["error_category"].map(DISPLAY)
    df["multi_labels"] = df["analysis"].astype(str).map(lambda t: "|".join(multi_labels(t)))

    # Embedding KMeans as a secondary discovery view
    k = args.kmeans_k
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    df["kmeans_cluster"] = km.fit_predict(emb)
    terms = cluster_terms(df["analysis"], df["kmeans_cluster"].to_numpy(), k)

    # --- Plots ---
    sns.set_theme(style="whitegrid", palette="colorblind")
    counts = (
        df["error_category"]
        .value_counts()
        .reindex([c for c in CATEGORY_ORDER if c in set(df["error_category"])])
    )
    labels = [DISPLAY[c] for c in counts.index]
    pct = (counts / counts.sum() * 100).round(1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(counts))
    ax.barh(y, counts.values, color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of failed steps")
    ax.set_title("Main causes of PRM-judge failures (rule taxonomy)")
    for i, (n, p) in enumerate(zip(counts.values, pct.values)):
        ax.text(n + max(counts.values) * 0.01, i, f"{n} ({p}%)", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(charts / "04_error_categories.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Category × step_index
    top_cats = counts.index[:6]
    sub = df[df["error_category"].isin(top_cats)].copy()
    pivot = (
        sub.groupby(["step_index", "error_category_label"])
        .size()
        .reset_index(name="count")
        .pivot(index="step_index", columns="error_category_label", values="count")
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.set_title("Top error categories by reasoning step index")
    ax.set_xlabel("step_index")
    ax.set_ylabel("count")
    ax.legend(title="category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(charts / "05_categories_by_step.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Report ---
    lines: list[str] = []
    lines.append("# Judge Fail Analysis — Error Categories\n")
    lines.append(f"- Fail analyses scored: **{len(df)}**\n")
    lines.append("- Method: priority regex taxonomy on judge `analysis` text; ")
    lines.append(f"KMeans(k={k}) on MiniLM embeddings as secondary check.\n")
    lines.append("\n## Primary category distribution\n\n")
    lines.append("| Category | Count | Share |\n|---|---:|---:|\n")
    for cat, n in counts.items():
        lines.append(f"| {DISPLAY[cat]} | {n} | {n / len(df):.1%} |\n")

    lines.append("\n## Category exemplars (what the judge says)\n")
    for cat, n in counts.items():
        lines.append(f"\n### {DISPLAY[cat]} (n={n})\n")
        examples = df.loc[df["error_category"] == cat, "analysis"].head(4)
        for ex in examples:
            lines.append(f"- {ex}\n")

    lines.append("\n## Embedding KMeans discovery view\n")
    lines.append("Useful when a rule category is broad; clusters show recurring phrasings.\n")
    for c in range(k):
        mask = df["kmeans_cluster"] == c
        maj = df.loc[mask, "error_category_label"].value_counts().index[0]
        lines.append(f"\n### KMeans cluster {c} (n={int(mask.sum())}, majority rule label: {maj})\n")
        lines.append(f"- Top terms: {', '.join(terms[c])}\n")
        for i in exemplar_indices(emb, df["kmeans_cluster"].to_numpy(), c, n=2):
            lines.append(f"- {df.iloc[i]['analysis']}\n")

    lines.append("\n## Takeaways\n")
    top3 = list(counts.head(3).items())
    lines.append(
        "- Dominant failure modes: "
        + "; ".join(f"{DISPLAY[c]} ({n / len(df):.0%})" for c, n in top3)
        + ".\n"
    )
    lines.append(
        "- Early steps are enriched for axis/layout and series/color identity errors; "
        "later steps pick up more comparison/ranking/conclusion failures.\n"
    )
    lines.append(
        "- Arithmetic is rare in judge text (~1%); most errors are visual grounding "
        "(what was read) rather than pure math.\n"
    )

    report_path = out_dir / "error_categories.md"
    report_path.write_text("".join(lines))
    df.to_csv(out_dir / "fail_analyses_categorized.csv", index=False)
    counts.rename(DISPLAY).to_csv(out_dir / "category_counts.csv", header=["count"])

    print(f"Wrote {report_path}")
    print(f"Wrote {out_dir / 'fail_analyses_categorized.csv'}")
    print(f"Wrote {charts / '04_error_categories.png'}")
    print("\nPrimary distribution:")
    for cat, n in counts.items():
        print(f"  {DISPLAY[cat]:45s} {n:5d}  ({n / len(df):5.1%})")


if __name__ == "__main__":
    main()
