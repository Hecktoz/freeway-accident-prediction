"""Discover trained checkpoints and provide cached ForwardFn loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import streamlit as st

from .adapters import (
    timesfm_covariate_adapter,
    unet_feature_fusion_adapter,
    unet_matrix_adapter,
)

ADAPTERS: dict[str, Callable[[Path], object]] = {
    "unet_matrix": unet_matrix_adapter.load_checkpoint,
    "unet_feature_fusion": unet_feature_fusion_adapter.load_checkpoint,
    "timesfm_covariate": timesfm_covariate_adapter.load_checkpoint,
}

MODEL_DISPLAY = {
    "unet_matrix": "U-Net (matrix)",
    "unet_feature_fusion": "U-Net + 結構化特徵",
    "timesfm_covariate": "TimesFM + 共變量",
}


def discover_checkpoints(models_root: Path) -> dict[str, Path | None]:
    """For each registered architecture, return its newest best_model.pth."""
    out: dict[str, Path | None] = {}
    for name in ADAPTERS:
        arch_dir = models_root / name
        out[name] = _latest_best(arch_dir) if arch_dir.exists() else None
    return out


def _latest_best(arch_dir: Path) -> Path | None:
    """Pick the most recent run that contains best_model.pth (fall back to last)."""
    runs = sorted([p for p in arch_dir.iterdir() if p.is_dir()], reverse=True)
    for run in runs:
        ckpt = run / "checkpoints" / "best_model.pth"
        if ckpt.exists():
            return ckpt
        ckpt2 = run / "checkpoints" / "last_model.pth"
        if ckpt2.exists():
            return ckpt2
    return None


@st.cache_resource(show_spinner="載入模型 checkpoint…")
def get_forward_fn(name: str, ckpt_str: str):
    """Load and memoize a ForwardFn. Cache key is (name, ckpt path string)."""
    ckpt = Path(ckpt_str)
    return ADAPTERS[name](ckpt)


def training_window_for(name: str, ckpt: Path):
    """Return the SampleWindow the given checkpoint was trained on, when known."""
    if name == "unet_matrix":
        return unet_matrix_adapter.training_window(ckpt)
    return None
