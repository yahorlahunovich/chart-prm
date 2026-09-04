"""
Unit tests for the pure sweep-summary and config-selection logic in
scripts/train/sweep_simpo.py -- no GPU/model needed for these.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "train"))

from sweep_simpo import pick_best_config, summarize_history


def _history(losses, accs, margins):
    return [
        {"step": i + 1, "loss": loss, "preference_accuracy": acc, "reward_margin": margin}
        for i, (loss, acc, margin) in enumerate(zip(losses, accs, margins))
    ]


def test_summarize_history_uses_last_quarter_of_steps():
    # 8 steps: first 6 bad, last 2 (the tail) good.
    losses = [2.0] * 6 + [0.5, 0.3]
    accs = [0.0] * 6 + [1.0, 1.0]
    margins = [-1.0] * 6 + [1.0, 1.0]
    history = _history(losses, accs, margins)

    summary = summarize_history(history)

    assert summary["n_steps"] == 8
    assert summary["final_loss"] == 0.3
    assert summary["mean_preference_accuracy_tail"] == 1.0
    assert summary["all_finite"] is True


def test_summarize_history_detects_nan_loss():
    history = _history([0.5, float("nan"), 0.4], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    summary = summarize_history(history)
    assert summary["all_finite"] is False


def test_summarize_history_empty():
    summary = summarize_history([])
    assert summary["n_steps"] == 0
    assert summary["all_finite"] is False


def test_pick_best_config_prefers_higher_tail_accuracy():
    results = [
        {
            "index": 0,
            "config": {"lr": 1e-6, "beta": 2.0},
            "summary": {"mean_preference_accuracy_tail": 0.5, "mean_loss_tail": 1.0, "all_finite": True},
        },
        {
            "index": 1,
            "config": {"lr": 1e-5, "beta": 2.0},
            "summary": {"mean_preference_accuracy_tail": 0.8, "mean_loss_tail": 1.5, "all_finite": True},
        },
    ]
    best = pick_best_config(results)
    assert best["index"] == 1


def test_pick_best_config_breaks_ties_by_lower_loss():
    results = [
        {
            "index": 0,
            "config": {"lr": 1e-6, "beta": 2.0},
            "summary": {"mean_preference_accuracy_tail": 0.7, "mean_loss_tail": 0.9, "all_finite": True},
        },
        {
            "index": 1,
            "config": {"lr": 1e-5, "beta": 2.0},
            "summary": {"mean_preference_accuracy_tail": 0.7, "mean_loss_tail": 0.4, "all_finite": True},
        },
    ]
    best = pick_best_config(results)
    assert best["index"] == 1


def test_pick_best_config_excludes_non_finite_runs_when_a_finite_one_exists():
    results = [
        {
            "index": 0,
            "config": {"lr": 1e-4, "beta": 10.0},
            "summary": {"mean_preference_accuracy_tail": 1.0, "mean_loss_tail": 0.0, "all_finite": False},
        },
        {
            "index": 1,
            "config": {"lr": 1e-6, "beta": 2.0},
            "summary": {"mean_preference_accuracy_tail": 0.5, "mean_loss_tail": 1.0, "all_finite": True},
        },
    ]
    best = pick_best_config(results)
    assert best["index"] == 1  # the only finite (non-collapsed) run, even though its accuracy is lower


def test_pick_best_config_falls_back_to_all_results_if_none_are_finite():
    results = [
        {
            "index": 0,
            "config": {"lr": 1e-4, "beta": 10.0},
            "summary": {"mean_preference_accuracy_tail": 0.9, "mean_loss_tail": 0.1, "all_finite": False},
        },
        {
            "index": 1,
            "config": {"lr": 1e-3, "beta": 10.0},
            "summary": {"mean_preference_accuracy_tail": 0.2, "mean_loss_tail": 0.1, "all_finite": False},
        },
    ]
    best = pick_best_config(results)
    assert best["index"] == 0  # still picks by the same ranking rule among the non-finite pool
