"""Centralized publication-quality plotting style configuration using SciencePlots.

Designed to match Nature/Science/ACL scientific paper aesthetics:
- Clean, minimal, professional typography and framing.
- Accessible, color-blind safe Paul Tol color palettes with fixed semantic mappings across experiments.
- Optimized for PDF vector embedding with Type 42 (TrueType) fonts and high-DPI rasterization.
"""

from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 - registers 'science', 'nature', 'ieee', etc.

# Consistent semantic palette across all models and baselines
PALETTE: Dict[str, str] = {
    "baseline": "#777777",
    "base": "#777777",
    "Base": "#777777",
    "Baseline": "#777777",
    "sft": "#0077BB",
    "SFT": "#0077BB",
    "dpo": "#EE7733",
    "DPO": "#EE7733",
    "full_dpo": "#EE7733",
    "Full DPO": "#EE7733",
    "step_dpo": "#AA4499",
    "Step-DPO": "#AA4499",
    "Step DPO": "#AA4499",
    "sft_dpo": "#CC3311",
    "SFT→DPO": "#CC3311",
    "SFT-DPO": "#CC3311",
    "kto": "#44AA99",
    "KTO": "#44AA99",
    "simpo": "#332288",
    "SimPO": "#332288",
    "pareto_dpo": "#009988",
    "Pareto-DPO": "#009988",
    "prm": "#117733",
    "PRM": "#117733",
    "ours": "#CC3311",
    "Ours": "#CC3311",
}

# Paul Tol color-blind safe binary evaluation palette (Muted Rose vs Muted Teal)
EVAL_PALETTE: Dict[Union[int, str], str] = {
    0: "#CC6677",
    1: "#44AA99",
    "0": "#CC6677",
    "1": "#44AA99",
    "Incorrect": "#CC6677",
    "Correct": "#44AA99",
    "0 (Incorrect)": "#CC6677",
    "1 (Correct)": "#44AA99",
    "Has Errors (Min 0)": "#CC6677",
    "Perfect (All 1s)": "#44AA99",
    "Fail": "#CC6677",
    "Pass": "#44AA99",
    "fail": "#CC6677",
    "pass": "#44AA99",
}

# Paul Tol Muted discrete cycle for the 9-category PRM judge taxonomy
TAXONOMY_PALETTE: Dict[str, str] = {
    "Axis / layout / chart-structure misread": "#332288",          # Indigo
    "Wrong series / color / legend identity": "#EE7733",           # Vibrant Orange
    "Hallucinated entity / label not on chart": "#CC6677",         # Muted Rose
    "Wrong ranking / extremum (highest/lowest/second)": "#44AA99", # Muted Teal
    "Logic inconsistency / false conclusion": "#AA4499",          # Muted Purple
    "Other / unspecified": "#BBBBBB",                              # Neutral Grey
    "Wrong numeric value read from chart": "#88CCEE",              # Muted Cyan
    "Bad comparison / threshold logic": "#999933",                 # Olive
    "Arithmetic / calculation mistake": "#117733",                 # Muted Green
    "Incomplete / truncated reasoning": "#DDCC77",                 # Sand
    "Other / Minor": "#BBBBBB",
    "Other / Minor Causes": "#BBBBBB",
}

# Benchmark comparison metrics palette
METRIC_PALETTE: Dict[str, str] = {
    "Official Exact-Match (%)": "#0077BB",
    "Robust Token Match (%)": "#88CCEE",
    "Structured + Correct (%)": "#44AA99",
    "GT Mentioned in Text (%)": "#EE7733",
}


def setup_plot_style(
    style: str = "science",
    palette: str = "vibrant",
    use_latex: bool = False,
    grid: bool = True,
) -> None:
    """Configures Matplotlib with SciencePlots and publication-ready parameters.

    Args:
        style: Base SciencePlots style (e.g. 'science', 'nature', 'ieee').
        palette: Color cycle style (e.g. 'vibrant', 'muted', 'bright').
        use_latex: Whether to enable LaTeX engine (defaults to False for Colab/Kaggle portability).
        grid: Whether to include subtle gridlines.
    """
    style_list: List[str] = [style]
    if not use_latex:
        style_list.append("no-latex")
    if grid:
        style_list.append("grid")
    if palette:
        style_list.append(palette)

    plt.style.use(style_list)

    # Harmonize figure and font parameters for publication
    plt.rcParams.update(
        {
            "figure.figsize": (7, 4.2),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def get_model_color(model_name: str, default: str = "#777777") -> str:
    """Returns the standardized palette color for a given model name."""
    return PALETTE.get(model_name, default)


def get_eval_color(label: Any, default: str = "#777777") -> str:
    """Returns the standardized binary evaluation color (Pass=Teal, Fail=Rose)."""
    return EVAL_PALETTE.get(label, default)


def get_taxonomy_color(category: str, default: str = "#BBBBBB") -> str:
    """Returns the standardized color for a judge error category."""
    return TAXONOMY_PALETTE.get(category, default)

