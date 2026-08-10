"""Centralized publication-quality plotting style configuration.

Designed to match NeurIPS/ICML research paper aesthetics:
- Clean, minimal, professional.
- Restrained color palette with fixed semantic mappings across experiments.
- Optimized for PDF vector embedding with Type 42 (TrueType) fonts.
"""

from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns

# Consistent semantic palette across all paper/report visualizations
PALETTE: Dict[str, str] = {
    "baseline": "#7A7A7A",
    "base": "#7A7A7A",
    "Base": "#7A7A7A",
    "Baseline": "#7A7A7A",
    "sft": "#4C78A8",
    "SFT": "#4C78A8",
    "dpo": "#F58518",
    "DPO": "#F58518",
    "prm": "#54A24B",
    "PRM": "#54A24B",
    "ours": "#E45756",
    "Ours": "#E45756",
}


def setup_plot_style() -> None:
    """Configures global Matplotlib and Seaborn parameters for publication-ready plots."""
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.1,
    )

    plt.rcParams.update(
        {
            "figure.figsize": (7, 4.2),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
