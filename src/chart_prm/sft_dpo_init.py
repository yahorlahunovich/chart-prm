"""Initialize a DPO LoRA as a copy of SFT, keeping SFT frozen as the DPO reference.

Merging SFT into a 4-bit backbone is unreliable, so both adapters live on the
quantized base: policy adapter `dpo` starts as a weight copy of `sft`, and
reference forwards switch to `sft` instead of `disable_adapter()` (Instruct).
Saving the trained `dpo` adapter is enough at eval — it already contains SFT+DPO.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


POLICY_ADAPTER = "dpo"
REFERENCE_ADAPTER = "sft"
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def adapter_weight_path(adapter_dir: Path) -> Optional[Path]:
    for name in WEIGHT_FILES:
        path = adapter_dir / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def require_adapter_dir(path: Path | str, *, require_weights: bool = True) -> Path:
    adapter_dir = Path(path)
    config = adapter_dir / "adapter_config.json"
    if not adapter_dir.is_dir() or not config.is_file():
        raise FileNotFoundError(
            f"Not a LoRA adapter directory (missing adapter_config.json): {adapter_dir}"
        )
    if require_weights and adapter_weight_path(adapter_dir) is None:
        raise FileNotFoundError(
            f"Adapter at {adapter_dir} has config but no weights "
            f"({', '.join(WEIGHT_FILES)}). On Kaggle, mount qwen-vl-sft-custom via kernel_sources."
        )
    return adapter_dir.resolve()


def read_adapter_config(adapter_dir: Path) -> Dict[str, Any]:
    return json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))


def resolve_sft_init_adapter(explicit: Optional[str] = None) -> Path:
    """Find the SFT adapter to copy into the DPO policy."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    try:
        from chart_prm.adapter_resolve import resolve_adapter

        candidates.append(resolve_adapter("sft"))
    except FileNotFoundError:
        pass
    candidates.extend(
        [
            Path("qwen_vl_sft_adapter"),
            Path("/kaggle/working/qwen_vl_sft_adapter"),
        ]
    )
    errors = []
    seen = set()
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else Path.cwd() / candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        try:
            return require_adapter_dir(resolved, require_weights=True)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    raise FileNotFoundError(
        "Could not resolve SFT adapter with weights. Tried:\n  " + "\n  ".join(errors)
    )


def adapter_param_pairs(model: Any, adapter_name: str) -> List[Tuple[str, Any]]:
    needle = f".{adapter_name}."
    pairs = [(name, param) for name, param in model.named_parameters() if needle in name]
    if pairs:
        return pairs
    suffix = f".{adapter_name}"
    return [
        (name, param)
        for name, param in model.named_parameters()
        if name.endswith(suffix) or f"{suffix}." in name
    ]


def freeze_reference_unfreeze_policy(
    model: Any,
    *,
    policy_adapter: str = POLICY_ADAPTER,
    reference_adapter: str = REFERENCE_ADAPTER,
) -> Dict[str, int]:
    n_policy = 0
    n_reference = 0
    for name, param in model.named_parameters():
        if f".{policy_adapter}." in name:
            param.requires_grad = True
            n_policy += param.numel()
        elif f".{reference_adapter}." in name:
            param.requires_grad = False
            n_reference += param.numel()
    if n_policy == 0:
        raise RuntimeError(
            f"No trainable parameters for policy adapter {policy_adapter!r}. "
            "SFT→DPO init failed to attach the DPO LoRA."
        )
    if n_reference == 0:
        raise RuntimeError(
            f"No frozen parameters for reference adapter {reference_adapter!r}."
        )
    return {"n_policy_params": n_policy, "n_reference_params": n_reference}


def max_adapter_abs_diff(
    model: Any,
    left_adapter: str = REFERENCE_ADAPTER,
    right_adapter: str = POLICY_ADAPTER,
) -> float:
    left = adapter_param_pairs(model, left_adapter)
    right = adapter_param_pairs(model, right_adapter)
    if not left or not right or len(left) != len(right):
        raise RuntimeError(
            f"Cannot compare adapters {left_adapter!r} vs {right_adapter!r}: "
            f"n_left={len(left)} n_right={len(right)}"
        )
    max_diff = 0.0
    for (_, left_param), (_, right_param) in zip(left, right):
        if left_param.shape != right_param.shape:
            raise RuntimeError(
                f"Adapter shape mismatch {tuple(left_param.shape)} vs {tuple(right_param.shape)}"
            )
        max_diff = max(max_diff, float((left_param.detach() - right_param.detach()).abs().max().item()))
    return max_diff


def init_peft_from_sft(
    model: Any,
    sft_path: Path | str,
    *,
    policy_adapter: str = POLICY_ADAPTER,
    reference_adapter: str = REFERENCE_ADAPTER,
    max_init_diff: float = 1e-4,
) -> Any:
    """Load SFT as frozen reference and a trainable DPO copy that starts identical.

    The copy check allows up to 1e-4 absolute error: loading the same LoRA twice
    on fp16/4-bit Qwen can differ at ~1e-5 without being a different adapter.
    """
    from peft import PeftModel

    sft_dir = require_adapter_dir(sft_path, require_weights=True)
    config = read_adapter_config(sft_dir)
    print(
        f"Initializing DPO policy from SFT adapter {sft_dir} "
        f"(r={config.get('r')}, alpha={config.get('lora_alpha')})"
    )
    peft_model = PeftModel.from_pretrained(
        model,
        str(sft_dir),
        adapter_name=reference_adapter,
        is_trainable=False,
    )
    try:
        peft_model.load_adapter(str(sft_dir), adapter_name=policy_adapter, is_trainable=True)
    except TypeError:
        peft_model.load_adapter(str(sft_dir), adapter_name=policy_adapter)
    peft_model.set_adapter(policy_adapter)
    counts = freeze_reference_unfreeze_policy(
        peft_model, policy_adapter=policy_adapter, reference_adapter=reference_adapter
    )
    init_diff = max_adapter_abs_diff(peft_model, reference_adapter, policy_adapter)
    print(
        f"SFT→DPO init OK: policy_params={counts['n_policy_params']} "
        f"frozen_sft_params={counts['n_reference_params']} max_copy_diff={init_diff:.2e}"
    )
    if init_diff > max_init_diff:
        raise RuntimeError(
            f"DPO adapter is not a copy of SFT (max abs diff {init_diff:.2e} > {max_init_diff:.2e}). "
            "Refusing to train: reference would not match the policy at step 0."
        )
    return peft_model


def save_policy_adapter(model: Any, output_dir: Path | str, adapter_name: str = POLICY_ADAPTER) -> Path:
    """Write only the trained policy adapter at the output root (not an SFT subfolder)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / ".peft_save_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        if hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
        try:
            model.save_pretrained(str(tmp_dir), selected_adapters=[adapter_name])
        except TypeError:
            model.save_pretrained(str(tmp_dir))
        source = _find_saved_adapter_dir(tmp_dir, adapter_name)
        for item in source.iterdir():
            dest = output_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    require_adapter_dir(output_dir, require_weights=True)
    return output_dir


def _find_saved_adapter_dir(root: Path, adapter_name: str) -> Path:
    named = root / adapter_name
    if (named / "adapter_config.json").is_file():
        return named
    if (root / "adapter_config.json").is_file():
        return root
    matches = [path.parent for path in root.rglob("adapter_config.json")]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find saved adapter files under {root} (adapter={adapter_name})"
    )


def active_adapter_name(model: Any) -> Optional[Any]:
    return getattr(model, "active_adapter", None)
