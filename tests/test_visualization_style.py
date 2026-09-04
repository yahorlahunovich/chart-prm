import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pytest

from src.visualization.style import (

    EVAL_PALETTE,
    METRIC_PALETTE,
    PALETTE,
    TAXONOMY_PALETTE,
    get_eval_color,
    get_model_color,
    get_taxonomy_color,
    setup_plot_style,
)


def test_setup_plot_style_applies_cleanly():
    """Verify setup_plot_style runs without error and sets expected params."""
    setup_plot_style()
    assert plt.rcParams["savefig.dpi"] == 300
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["text.usetex"] is False


def test_palette_keys_present():
    """Verify all models and backward-compatible keys exist in PALETTE."""
    expected_models = [
        "Base", "SFT", "Full DPO", "Step-DPO", "KTO", "SFT→DPO", "SimPO", "Pareto-DPO",
        "base", "sft", "dpo", "step_dpo", "kto", "prm", "ours"
    ]
    for model in expected_models:
        assert model in PALETTE, f"Missing {model} in PALETTE"
        assert PALETTE[model].startswith("#")


def test_eval_palette_and_helper():
    """Verify binary evaluation palette mapping."""
    assert EVAL_PALETTE[0] == "#CC6677"  # Muted Rose
    assert EVAL_PALETTE[1] == "#44AA99"  # Muted Teal
    assert get_eval_color(0) == "#CC6677"
    assert get_eval_color(1) == "#44AA99"
    assert get_eval_color("Unknown", default="#999999") == "#999999"


def test_taxonomy_palette_and_helper():
    """Verify judge error taxonomy colors."""
    assert "Axis / layout / chart-structure misread" in TAXONOMY_PALETTE
    assert "Hallucinated entity / label not on chart" in TAXONOMY_PALETTE
    assert get_taxonomy_color("Axis / layout / chart-structure misread") == "#332288"
    assert get_taxonomy_color("Nonexistent", default="#000000") == "#000000"


def test_get_model_color():
    """Verify get_model_color returns correct color and fallback."""
    assert get_model_color("Base") == PALETTE["Base"]
    assert get_model_color("SFT") == PALETTE["SFT"]
    assert get_model_color("Nonexistent", default="#123456") == "#123456"
