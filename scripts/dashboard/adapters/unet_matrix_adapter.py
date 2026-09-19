"""Adapter for ``unet_matrix`` checkpoints via ``inference.load``.

Using the shared inference loader keeps dashboard support aligned with new
checkpoint metadata such as ``input_type=vd+hist`` and ``backbone=residual_attention``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from scripts.lib.inference import ModelInput
from scripts.lib.windows import SampleWindow

ForwardFn = Callable[[ModelInput], np.ndarray]


def _lazy_unet():
    """Defer importing scripts.models.unet_matrix (pulls torch/sklearn/tqdm)."""
    from scripts.models.unet_matrix import PAD_TARGET, SPEED_MAX, SimpleUNet, TrainConfig
    return PAD_TARGET, SPEED_MAX, SimpleUNet, TrainConfig


def _resolve_run_config(checkpoint: Path) -> Path:
    """run_config.json sits at <run_dir>/run_config.json, sibling to checkpoints/."""
    run_dir = checkpoint.parent.parent  # checkpoint.parent == checkpoints/
    cfg = run_dir / "run_config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"run_config.json missing for {checkpoint}: expected at {cfg}")
    return cfg


def _build_train_config(cfg_path: Path):
    _, _, _, TrainConfig = _lazy_unet()
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    fields = {f.name for f in TrainConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    cleaned = {k: v for k, v in raw.items() if k in fields}
    return TrainConfig(**cleaned)


def training_window(checkpoint: Path) -> SampleWindow:
    """Recover the SampleWindow the checkpoint was trained on."""
    cfg = _build_train_config(_resolve_run_config(checkpoint))
    return SampleWindow(
        upstream_km=cfg.upstream_km,
        downstream_km=cfg.downstream_km,
        time_before_min=cfg.time_before_min,
        time_after_min=cfg.time_after_min,
        spatial_resolution_km=cfg.spatial_resolution_km,
        resolution_min=cfg.resolution_min,
    )


def load_checkpoint(checkpoint: Path) -> ForwardFn:
    """Load weights and return a ForwardFn closure bound to the model + cfg."""
    from scripts.lib.inference import load

    run_id = checkpoint.parent.parent.name
    models_root = checkpoint.parent.parent.parent.parent
    model = load("unet_matrix", run_id=run_id, models_root=models_root)
    return model.predict_from_input


def _predict(model_input: ModelInput, model, cfg, device) -> np.ndarray:
    """Convert ModelInput → tensor, run model, denormalize, return (n_space, n_time)."""
    import torch
    import torch.nn.functional as F

    _, SPEED_MAX, _, _ = _lazy_unet()
    win = model_input.window

    # The orchestrator hands us a pre-only matrix shaped (n_space, n_time_pre).
    # The trained model expects a full-window (n_space, n_time) input where the
    # post-accident slice is zero (mask channel marks it invalid).
    pre = model_input.observed_pre.astype(np.float32)
    n_space, n_time = win.n_space, win.n_time
    full = np.full((n_space, n_time), np.nan, dtype=np.float32)
    pre_w = min(pre.shape[1], n_time)
    if pre_w > 0:
        full[:, :pre_w] = pre[:, :pre_w]

    speed_norm = np.where(np.isnan(full), np.nan,
                          np.clip(full, 0.0, SPEED_MAX) / SPEED_MAX)

    n_before_eff = pre_w  # cells available to the model's input channel
    ch0 = speed_norm.copy()
    ch0[:, n_before_eff:] = 0.0
    ch0 = np.nan_to_num(ch0, nan=0.0)

    ch1 = np.zeros((n_space, n_time), dtype=np.float32)
    if n_before_eff > 0:
        ch1[:, :n_before_eff] = (~np.isnan(speed_norm[:, :n_before_eff])).astype(np.float32)

    # Crop or pad spatial / temporal axes to match the trained model's
    # n_space / n_time. Crop is centered; if the trained window is larger, we
    # pad with reflect on the speed channel and zeros on the mask channel.
    ch0 = _resize_axis(ch0, axis=0, target=cfg.n_space, pad_mode="edge")
    ch1 = _resize_axis(ch1, axis=0, target=cfg.n_space, pad_mode="zeros")
    ch0 = _resize_axis(ch0, axis=1, target=cfg.n_time, pad_mode="edge")
    ch1 = _resize_axis(ch1, axis=1, target=cfg.n_time, pad_mode="zeros")

    x = torch.from_numpy(np.stack([ch0, ch1], axis=0)).float().unsqueeze(0)
    x = F.pad(x, (cfg.pad_w, cfg.pad_w, cfg.pad_h, cfg.pad_h), mode="reflect")
    x = x.to(device)

    with torch.no_grad():
        # Model returns (B, n_space, n_after) in [0, 1]; scale to km/h.
        post = model(x).squeeze(0).cpu().numpy() * SPEED_MAX

    # Reattach the input window (just the observed pre slice, denormalized) so
    # the dashboard heatmap shows speed values both pre- and post-accident.
    pre_kmh = np.full((cfg.n_space, cfg.n_before), np.nan, dtype=np.float32)
    pre_block = ch0[:, :cfg.n_before] * SPEED_MAX
    mask_block = ch1[:, :cfg.n_before] > 0
    pre_kmh[mask_block] = pre_block[mask_block]
    full_pred_train = np.concatenate([pre_kmh, post.astype(np.float32)], axis=1)
    # full_pred_train shape: (cfg.n_space, cfg.n_time)

    # Resize back to the orchestrator's window.
    out = _resize_axis(full_pred_train, axis=0, target=n_space, pad_mode="edge")
    out = _resize_axis(out, axis=1, target=n_time, pad_mode="edge")
    return out.astype(np.float32)


def _resize_axis(arr: np.ndarray, axis: int, target: int, pad_mode: str) -> np.ndarray:
    """Center-crop or center-pad along one axis to length `target`.

    pad_mode: 'edge' replicates boundary values; 'zeros' pads with 0.
    """
    cur = arr.shape[axis]
    if cur == target:
        return arr
    if cur > target:
        start = (cur - target) // 2
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(start, start + target)
        return arr[tuple(sl)]
    diff = target - cur
    pad_left = diff // 2
    pad_right = diff - pad_left
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (pad_left, pad_right)
    if pad_mode == "edge":
        return np.pad(arr, pad_widths, mode="edge")
    return np.pad(arr, pad_widths, mode="constant", constant_values=0.0)
