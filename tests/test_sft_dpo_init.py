"""Tests for SFT→DPO adapter copy, reference switching, and path resolution."""

from pathlib import Path
from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn

from chart_prm.adapter_resolve import resolve_adapter
from chart_prm.dpo.trainer import compute_sequence_logprobs
from chart_prm.sft_dpo_init import (
    freeze_reference_unfreeze_policy,
    max_adapter_abs_diff,
    require_adapter_dir,
    save_policy_adapter,
)


def test_require_adapter_dir_needs_weights(tmp_path: Path):
    adapter = tmp_path / "qwen_vl_sft_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no weights"):
        require_adapter_dir(adapter, require_weights=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"not-empty")
    assert require_adapter_dir(adapter).name == "qwen_vl_sft_adapter"


class NestedAdapters(nn.Module):
    """Parameter names contain `.sft.` and `.dpo.` the same way PEFT does."""

    def __init__(self):
        super().__init__()
        self.lora_A = nn.ModuleDict(
            {
                "sft": nn.Linear(3, 3, bias=False),
                "dpo": nn.Linear(3, 3, bias=False),
            }
        )
        with torch.no_grad():
            self.lora_A["dpo"].weight.copy_(self.lora_A["sft"].weight)


def test_freeze_sft_unfreeze_dpo_and_copy_diff_is_zero():
    model = NestedAdapters()
    counts = freeze_reference_unfreeze_policy(model)
    assert counts["n_policy_params"] > 0
    assert counts["n_reference_params"] > 0
    assert max_adapter_abs_diff(model) == 0.0
    assert 7.63e-6 < 1e-4
    for name, param in model.named_parameters():
        if ".dpo." in name:
            assert param.requires_grad
        if ".sft." in name:
            assert not param.requires_grad


class DualAdapterToy(nn.Module):
    def __init__(self, vocab_size=10, hidden_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.sft_bias = nn.Parameter(torch.tensor(1.5))
        self.dpo_bias = nn.Parameter(torch.tensor(0.25))
        self._adapter = "dpo"
        self._disabled = False

    def set_adapter(self, name):
        self._adapter = name
        self._disabled = False

    @property
    def active_adapter(self):
        return None if self._disabled else self._adapter

    @contextmanager
    def disable_adapter(self):
        previous = self._adapter
        was_disabled = self._disabled
        self._disabled = True
        try:
            yield
        finally:
            self._disabled = was_disabled
            self._adapter = previous

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        if not self._disabled:
            if self._adapter == "sft":
                x = x + self.sft_bias
            elif self._adapter == "dpo":
                x = x + self.dpo_bias
        return self.linear(x)


def _batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
    }


def test_reference_adapter_uses_sft_and_restores_policy():
    torch.manual_seed(0)
    model = DualAdapterToy()
    batch = _batch()
    policy = compute_sequence_logprobs(model, batch, is_reference=False)
    ref_sft = compute_sequence_logprobs(
        model, batch, is_reference=True, reference_adapter="sft"
    )
    ref_base = compute_sequence_logprobs(model, batch, is_reference=True)
    assert model.active_adapter == "dpo"
    assert not torch.equal(policy, ref_sft)
    assert not torch.equal(ref_sft, ref_base)
    assert not torch.equal(policy, ref_base)


def test_save_policy_adapter_flattens_named_subdir(tmp_path: Path):
    class Dummy:
        def set_adapter(self, name):
            self.active = name

        def save_pretrained(self, path, selected_adapters=None):
            named = Path(path) / selected_adapters[0]
            named.mkdir(parents=True, exist_ok=True)
            (named / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
            (named / "adapter_model.safetensors").write_bytes(b"weights")

    out = tmp_path / "qwen_vl_sft_dpo_adapter"
    save_policy_adapter(Dummy(), out, adapter_name="dpo")
    assert (out / "adapter_config.json").is_file()
    assert (out / "adapter_model.safetensors").is_file()
    assert not (out / "dpo").exists()


def test_resolve_sft_dpo_uses_unique_dir(tmp_path: Path):
    input_root = tmp_path / "input"
    sft_dpo = input_root / "qwen-vl-sft-dpo" / "qwen_vl_sft_dpo_adapter"
    sft_dpo.mkdir(parents=True)
    (sft_dpo / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    dpo = input_root / "notebooks" / "egorlagunovich" / "qwen-vl-step-dpo-custom" / "qwen_vl_dpo_adapter"
    dpo.mkdir(parents=True)
    (dpo / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    assert resolve_adapter("sft_dpo", input_root=input_root).name == "qwen_vl_sft_dpo_adapter"
    assert resolve_adapter("dpo", input_root=input_root).name == "qwen_vl_dpo_adapter"
    assert resolve_adapter("sft_dpo", input_root=input_root) != resolve_adapter(
        "dpo", input_root=input_root
    )
