"""Checkpoint discovery and cached forward-fn loaders (no Streamlit)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

from .adapters import (
    timesfm_covariate_adapter,
    unet_feature_fusion_adapter,
    unet_matrix_adapter,
)

ADAPTERS: dict[str, Callable[[Path], object]] = {
+    "unet_matrix": unet_matrix_adapter.load_checkpoint,
    "unet_feature_fusion": unet_feature_fusion_adapter.load_checkpoint,
    "timesfm_covariate": timesfm_covariate_adapter.load_checkpoint,
}

MODEL_DISPLAY = {
    "unet_matrix": "U-Net (matrix)",
    "unet_feature_fusion": "U-Net + 結構化特徵",
    "timesfm_covariate": "TimesFM + 共變量",
}


def discover_checkpoints(models_root: Path) -> dict[str, Path | None]:
    out: dict[str, Path | None] = {}
    for name in ADAPTERS:
        arch_dir = models_root / name
        out[name] = _latest_best(arch_dir) if arch_dir.exists() else None
    return out


def _latest_best(arch_dir: Path) -> Path | None:
    runs = sorted([p for p in arch_dir.iterdir() if p.is_dir()], reverse=True)
    for run in runs:
        # Flat layout: run_dir/checkpoints/best_model.pth
        for fname in ("best_model.pth", "last_model.pth"):
            ckpt = run / "checkpoints" / fname
            if ckpt.exists():
                return ckpt
        # Feature-fusion layout: run_dir/<experiment>/checkpoints/best_model.pth
        # Return the champion experiment (direct_concat preferred, else first found).
        experiment_dirs = sorted([p for p in run.iterdir() if p.is_dir()], reverse=False)
        for pref in ("direct_concat", "mid_inject", "attention"):
            exp_dir = run / pref
            for fname in ("best_model.pth", "last_model.pth"):
                ckpt = exp_dir / "checkpoints" / fname
                if ckpt.exists():
                    return ckpt
        for exp_dir in experiment_dirs:
            for fname in ("best_model.pth", "last_model.pth"):
                ckpt = exp_dir / "checkpoints" / fname
                if ckpt.exists():
                    return ckpt
    return None


@lru_cache(maxsize=8)
def get_forward_fn(name: str, ckpt_str: str):
    """Load and memoize a ForwardFn. Cache key is (name, checkpoint path string)."""
    ckpt = Path(ckpt_str)
    return ADAPTERS[name](ckpt)


def training_window_for(name: str, ckpt: Path):
    if name == "unet_matrix":
        return unet_matrix_adapter.training_window(ckpt)
    if name == "unet_feature_fusion":
        return _feature_fusion_window(ckpt)
    return None


def _feature_fusion_window(ckpt: Path):
    """Resolve the SampleWindow for a feature-fusion checkpoint.

    Checks run_config.json at the experiment and run_id levels, then falls
    back to the champion spec (all progressive variants share the same window).
    """
    import json
    from scripts.lib.windows import SampleWindow

    _WINDOW_KEYS = ("upstream_km", "downstream_km", "time_before_min",
                    "time_after_min", "spatial_resolution_km", "resolution_min")

    for cfg_parent in (ckpt.parent.parent, ckpt.parent.parent.parent):
        cfg_path = cfg_parent / "run_config.json"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as fh:
                run_cfg = json.load(fh)
            inner = run_cfg.get("config", run_cfg)
            if all(k in inner for k in _WINDOW_KEYS):
                return SampleWindow(**{k: int(inner[k]) for k in _WINDOW_KEYS})

    try:
        from pathlib import Path as _Path
        _PROJECT_ROOT = Path(__file__).resolve().parents[2]
        _spec_path = _PROJECT_ROOT / "reports" / "v1_champion.json"
        with open(_spec_path, encoding="utf-8") as fh:
            spec = json.load(fh)
        if "window" in spec:
            return SampleWindow(**{k: int(spec["window"][k]) for k in _WINDOW_KEYS})
    except Exception:
        pass

    return None
