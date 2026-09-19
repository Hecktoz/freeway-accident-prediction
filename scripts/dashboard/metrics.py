"""Per-accident metrics — pure computation + HTML render (no Streamlit)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _flatten_pair(observed: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if observed.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {observed.shape} vs {predicted.shape}")
    mask = ~np.isnan(observed) & ~np.isnan(predicted)
    return observed[mask].astype(np.float64), predicted[mask].astype(np.float64)


def per_accident_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    congestion_kmh: float = 60.0,
    mape_floor_kmh: float = 5.0,
) -> dict[str, float]:
    a, b = _flatten_pair(observed, predicted)
    if a.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"),
                "r2": float("nan"), "f1": float("nan"), "n_cells": 0}
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mape_mask = a >= mape_floor_kmh
    mape = (float(np.mean(np.abs(diff[mape_mask]) / a[mape_mask]) * 100.0)
            if mape_mask.any() else float("nan"))
    ss_res = float(np.sum(diff ** 2))
    mean_obs = float(np.mean(a))
    ss_tot = float(np.sum((a - mean_obs) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    pos_t = a < congestion_kmh
    pos_p = b < congestion_kmh
    tp = int(np.sum(pos_t & pos_p))
    fp = int(np.sum(~pos_t & pos_p))
    fn = int(np.sum(pos_t & ~pos_p))
    if tp == 0:
        f1 = 0.0
    else:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "f1": f1, "n_cells": int(a.size)}


def impact_time_offset(
    matrix: np.ndarray,
    space_idx: int,
    time_offsets_min: Sequence[int],
    *,
    free_flow_kmh: float = 80.0,
) -> float:
    if matrix.size == 0:
        return float("nan")
    offs = np.asarray(list(time_offsets_min), dtype=int)
    post_mask = offs >= 0
    if not post_mask.any():
        return float("nan")
    speeds = matrix[space_idx, post_mask]
    times = offs[post_mask]
    above = (~np.isnan(speeds)) & (speeds >= free_flow_kmh)
    if not above.any():
        return float("nan")
    for i in range(len(above)):
        if above[i:].all():
            return float(times[i])
    return float("nan")


def metrics_to_html(
    metrics: dict[str, float],
    *,
    impact_observed_min: float | None,
    impact_predicted_min: float | None,
) -> str:
    def fmt(v: float, suffix: str = "", digits: int = 2) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.{digits}f}{suffix}"

    if (impact_observed_min is None or impact_predicted_min is None
            or np.isnan(impact_observed_min) or np.isnan(impact_predicted_min)):
        impact_html = "—"
    else:
        err = impact_predicted_min - impact_observed_min
        impact_html = (f"{err:+.0f} 分 "
                       f"<small>(觀測 {impact_observed_min:.0f} / 預測 {impact_predicted_min:.0f})</small>")

    cells = [
        ("MAE", fmt(metrics.get("mae"), " km/h")),
        ("MAPE", fmt(metrics.get("mape"), " %", digits=1)),
        ("R²", fmt(metrics.get("r2"), digits=3)),
        ("壅塞 F1", fmt(metrics.get("f1"), digits=3)),
        ("衝擊時長誤差", impact_html),
    ]
    tds = "".join(
        f'<td style="padding:4px 12px;text-align:center">'
        f'<div style="font-size:11px;color:#64748b">{label}</div>'
        f'<div style="font-size:15px;font-weight:600">{value}</div>'
        f'</td>'
        for label, value in cells
    )
    return f'<table style="border-collapse:collapse;width:100%"><tr>{tds}</tr></table>'
