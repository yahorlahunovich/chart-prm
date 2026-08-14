"""
Resolve LoRA adapters from Kaggle kernel mounts without substring collisions.

Holdout eval v8 loaded the fragment Step-DPO adapter for both `dpo` and
`step_dpo` because `"dpo" in path` matches `qwen-vl-fragment-step-dpo`.
Matching is now by exact adapter directory name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


ADAPTER_DIR_NAMES: Dict[str, str] = {
    "sft": "qwen_vl_sft_adapter",
    "dpo": "qwen_vl_dpo_adapter",
    "step_dpo": "qwen_vl_step_dpo_adapter",
    "kto": "qwen_vl_kto_adapter",
    "sft_dpo": "qwen_vl_sft_dpo_adapter",
}

KERNEL_SLUGS: Dict[str, Sequence[str]] = {
    "sft": ("qwen-vl-sft-custom", "qwen-vl-sft-adapter"),
    "dpo": ("qwen-vl-step-dpo-custom", "qwen-vl-dpo-adapter"),
    "step_dpo": ("qwen-vl-fragment-step-dpo", "qwen-vl-step-dpo-adapter"),
    "kto": ("qwen-vl-kto-custom", "qwen-vl-kto-adapter"),
    "sft_dpo": ("qwen-vl-sft-dpo",),
}


def _candidate_roots(name: str, input_root: Path) -> List[Path]:
    dir_name = ADAPTER_DIR_NAMES[name]
    roots: List[Path] = []
    for slug in KERNEL_SLUGS[name]:
        roots.extend(
            [
                input_root / slug / dir_name,
                input_root / slug,
                input_root / "notebooks" / "egorlagunovich" / slug / dir_name,
                input_root / "notebooks" / "egorlagunovich" / slug,
            ]
        )
    return roots


def _is_adapter_dir(path: Path, dir_name: str) -> bool:
    return path.is_dir() and path.name == dir_name and (path / "adapter_config.json").exists()


def find_adapter_dir(name: str, search_roots: Iterable[Path]) -> Optional[Path]:
    """Return the unique adapter directory for `name`, or None if missing."""
    if name not in ADAPTER_DIR_NAMES:
        raise KeyError(f"Unknown adapter name {name!r}; expected one of {sorted(ADAPTER_DIR_NAMES)}")
    dir_name = ADAPTER_DIR_NAMES[name]
    seen = []
    seen_ids = set()
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        if _is_adapter_dir(root, dir_name):
            resolved = root.resolve()
            if resolved not in seen_ids:
                seen.append(resolved)
                seen_ids.add(resolved)
            continue
        for cfg in root.rglob("adapter_config.json"):
            parent = cfg.parent
            if _is_adapter_dir(parent, dir_name):
                resolved = parent.resolve()
                if resolved not in seen_ids:
                    seen.append(resolved)
                    seen_ids.add(resolved)
    if not seen:
        return None
    return seen[0]


def resolve_adapter(name: str, input_root: Path | str = "/kaggle/input") -> Path:
    input_root = Path(input_root)
    candidates = _candidate_roots(name, input_root)
    found = find_adapter_dir(name, candidates)
    if found is None and input_root.exists():
        found = find_adapter_dir(name, [input_root])
    if found is None:
        raise FileNotFoundError(
            f"Could not resolve {name} adapter directory {ADAPTER_DIR_NAMES[name]!r} under {input_root}"
        )
    return found


def resolve_all_adapters(
    names: Sequence[str] = ("sft", "dpo", "step_dpo", "kto", "sft_dpo"),
    input_root: Path | str = "/kaggle/input",
) -> Dict[str, Path]:
    """Resolve every requested adapter and fail if two names share a path."""
    input_root = Path(input_root)
    resolved: Dict[str, Path] = {}
    for name in names:
        resolved[name] = resolve_adapter(name, input_root=input_root)
    inverted: Dict[Path, List[str]] = {}
    for name, path in resolved.items():
        inverted.setdefault(path.resolve(), []).append(name)
    collisions = {str(path): names_ for path, names_ in inverted.items() if len(names_) > 1}
    if collisions:
        raise RuntimeError(
            "Adapter path collision — two eval systems loaded the same LoRA. "
            f"collisions={collisions}. Refusing to evaluate."
        )
    return resolved
