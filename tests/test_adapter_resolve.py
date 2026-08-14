"""Tests for unique Kaggle adapter path resolution."""

from pathlib import Path

import pytest

from chart_prm.adapter_resolve import resolve_adapter, resolve_all_adapters


def _write_adapter(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")


def test_resolve_uses_exact_adapter_dir_not_dpo_substring(tmp_path: Path):
    input_root = tmp_path / "input"
    _write_adapter(input_root / "qwen-vl-fragment-step-dpo" / "qwen_vl_step_dpo_adapter")
    _write_adapter(input_root / "notebooks" / "egorlagunovich" / "qwen-vl-step-dpo-custom" / "qwen_vl_dpo_adapter")
    _write_adapter(input_root / "notebooks" / "egorlagunovich" / "qwen-vl-sft-custom" / "qwen_vl_sft_adapter")
    _write_adapter(input_root / "notebooks" / "egorlagunovich" / "qwen-vl-kto-custom" / "qwen_vl_kto_adapter")
    _write_adapter(input_root / "qwen-vl-sft-dpo" / "qwen_vl_sft_dpo_adapter")

    paths = resolve_all_adapters(
        names=("sft", "dpo", "step_dpo", "kto", "sft_dpo"),
        input_root=input_root,
    )
    assert paths["dpo"].name == "qwen_vl_dpo_adapter"
    assert paths["step_dpo"].name == "qwen_vl_step_dpo_adapter"
    assert paths["sft"].name == "qwen_vl_sft_adapter"
    assert paths["kto"].name == "qwen_vl_kto_adapter"
    assert paths["sft_dpo"].name == "qwen_vl_sft_dpo_adapter"
    assert paths["dpo"] != paths["step_dpo"]
    assert paths["dpo"] != paths["sft_dpo"]


def test_resolve_refuses_adapter_path_collision(tmp_path: Path):
    input_root = tmp_path / "input"
    # Only the fragment adapter exists; substring matching used to assign it to both names.
    _write_adapter(input_root / "qwen-vl-fragment-step-dpo" / "qwen_vl_step_dpo_adapter")
    with pytest.raises(FileNotFoundError, match="qwen_vl_dpo_adapter"):
        resolve_adapter("dpo", input_root=input_root)
    step = resolve_adapter("step_dpo", input_root=input_root)
    assert step.name == "qwen_vl_step_dpo_adapter"
