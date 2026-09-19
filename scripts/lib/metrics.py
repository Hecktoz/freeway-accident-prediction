"""Evaluation metrics for accident speed-matrix forecasting.

The MAE/RMSE/F1 functions ignore NaN cells. F1 is computed against a
congestion threshold; cells with speed below the threshold are positive.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _flatten_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[mask].astype(np.float64), y_pred[mask].astype(np.float64)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _flatten_pair(y_true, y_pred)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _flatten_pair(y_true, y_pred)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def f1_congestion(y_true: np.ndarray, y_pred: np.ndarray, threshold_kmh: float = 60.0) -> float:
    """Cell-wise F1 where positive means speed < threshold_kmh."""
    a, b = _flatten_pair(y_true, y_pred)
    if a.size == 0:
        return float("nan")
    pos_t = a < threshold_kmh
    pos_p = b < threshold_kmh
    tp = int(np.sum(pos_t & pos_p))
    fp = int(np.sum(~pos_t & pos_p))
    fn = int(np.sum(pos_t & ~pos_p))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def impact_time_min(
    observed: np.ndarray,
    baseline: np.ndarray,
    time_offsets_min: Iterable[int],
    *,
    speed_drop_kmh: float = 10.0,
    sustained_min: int = 10,
) -> float:
    """Estimate accident impact duration as the last sustained drop vs baseline.

    Returns the time offset (min) at which `observed` recovers to within
    `speed_drop_kmh` of `baseline` for at least `sustained_min` minutes,
    measured from t=0. Returns NaN if no recovery is detected.

    Both inputs are `(n_space, n_time)` matrices. The averaging is taken over
    space cells with valid data at each time bin.
    """
    if observed.shape != baseline.shape:
        raise ValueError("shape mismatch")
    time_offsets = np.asarray(list(time_offsets_min), dtype=np.int32)
    if observed.shape[1] != time_offsets.size:
        raise ValueError("time axis mismatch")

    diff = observed - baseline
    # Per-time-bin average difference over the spatial axis, ignoring NaNs.
    with np.errstate(invalid="ignore"):
        per_t = np.nanmean(diff, axis=0)
    # Find recovery: last time before which sustained drop ends.
    drop_mask = per_t < -speed_drop_kmh
    if not drop_mask.any():
        return 0.0

    resolution_min = int(time_offsets[1] - time_offsets[0]) if time_offsets.size > 1 else 5
    sustain_cells = max(1, sustained_min // resolution_min)

    # Walk forward from t=0 looking for the end of the drop region.
    t_zero_idx = int(np.argmin(np.abs(time_offsets)))
    n = drop_mask.size
    for i in range(t_zero_idx, n):
        window = drop_mask[i : i + sustain_cells]
        if window.size < sustain_cells:
            return float(time_offsets[-1])
        if not window.any():
            return float(time_offsets[i])
    return float(time_offsets[-1])
