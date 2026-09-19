"""Accident-caused congestion mask (origin_cc) — reusable, dependency-light.

This module re-implements the `origin_cc` ground-truth congestion mask from
`scripts/analysis/measure_congestion_extent.py` as pure numpy/scipy functions
(no matplotlib / CJK-font imports), so the training/eval pipeline can build and
consume the same mask without dragging in the analysis script's plotting deps.

`origin_cc` is the eval_protocol_v1 (2026-05-29) ground-truth definition of
"accident-caused upstream congestion": the connected component(s) of the abs
drop-mask that touch the accident origin anchor zone. It is used both as the
metric mask for `upstream_congestion_MAE` and as the congested-cell selector for
the congestion-weighted loss. See:
  - reports/upstream_extent_20260528_summary_tw.md  (decision basis; p99≈14, max=26 km)
  - reports/planning_next_steps_20260529.md §0/§5    (decision)
  - reports/eval_protocol_v1.md §3.1/§4.1            (locked spec)

Matrix convention: `(n_space, n_time)` with `space_offsets_m` (negative = downstream,
0 = accident point, positive = upstream) and `time_offsets_min` axes.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# Locked origin_cc parameters (eval_protocol_v1 §4.1; measure_congestion_extent.py defaults).
DELTA_MIN_DEFAULT = 15.0     # speed drop (baseline - observed) threshold, km/h
CONG_KMH_DEFAULT = 60.0      # observed-speed congestion threshold, km/h
PERSIST_N_DEFAULT = 3        # sustained run length along time (cells), 3 = 15 min
ANCHOR_T_MIN_DEFAULT = 0     # anchor zone time lower bound (min)
ANCHOR_T_MAX_DEFAULT = 15    # anchor zone time upper bound (min)
ANCHOR_S_KM_MIN_DEFAULT = 0.0  # anchor zone upstream lower bound (km)
ANCHOR_S_KM_MAX_DEFAULT = 3.0  # anchor zone upstream upper bound (km)


def persistence_runs_along_time(mask: np.ndarray, n: int) -> np.ndarray:
    """Keep cells inside a length-`n` True run along the last (time) axis.

    Runs of length >= n are kept whole; shorter (transient) runs are dropped.
    """
    if n <= 1:
        return mask.copy()
    out = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        row = mask[i]
        if not row.any():
            continue
        padded = np.concatenate(([False], row, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]  # exclusive end
        for s, e in zip(starts, ends):
            if (e - s) >= n:
                out[i, s:e] = True
    return out


def base_abs_mask(
    obs: np.ndarray,
    base: np.ndarray,
    *,
    delta_min: float = DELTA_MIN_DEFAULT,
    cong_kmh: float = CONG_KMH_DEFAULT,
    persist_n: int = PERSIST_N_DEFAULT,
) -> np.ndarray:
    """Drop >= delta AND observed < cong AND finite, then keep sustained time-runs."""
    finite = np.isfinite(obs) & np.isfinite(base)
    drop = np.where(finite, base - obs, 0.0)
    cond = finite & (drop >= delta_min) & (obs < cong_kmh)
    return persistence_runs_along_time(cond, persist_n)


def origin_cc_mask(
    abs_mask: np.ndarray,
    space_offsets_m: np.ndarray,
    time_offsets_min: np.ndarray,
    *,
    anchor_t_min: int = ANCHOR_T_MIN_DEFAULT,
    anchor_t_max: int = ANCHOR_T_MAX_DEFAULT,
    anchor_s_km_min: float = ANCHOR_S_KM_MIN_DEFAULT,
    anchor_s_km_max: float = ANCHOR_S_KM_MAX_DEFAULT,
) -> np.ndarray:
    """8-connected component(s) of `abs_mask` that touch the accident origin anchor zone.

    Anchor zone: `t ∈ [anchor_t_min, anchor_t_max]` min AND
    `s ∈ [anchor_s_km_min, anchor_s_km_max]` km. Returns a boolean mask.
    """
    if not abs_mask.any():
        return np.zeros_like(abs_mask, dtype=bool)
    labels, n = ndi.label(abs_mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return np.zeros_like(abs_mask, dtype=bool)
    s_km = space_offsets_m.astype(np.float32) / 1000.0
    t_min = time_offsets_min.astype(np.float32)
    anchor = (
        (s_km[:, None] >= anchor_s_km_min)
        & (s_km[:, None] <= anchor_s_km_max)
        & (t_min[None, :] >= anchor_t_min)
        & (t_min[None, :] <= anchor_t_max)
    )
    touched = np.unique(labels[anchor & abs_mask])
    touched = touched[touched != 0]
    if touched.size == 0:
        return np.zeros_like(abs_mask, dtype=bool)
    return np.isin(labels, touched)


def origin_cc_congestion_mask(
    obs: np.ndarray,
    base: np.ndarray,
    space_offsets_m: np.ndarray,
    time_offsets_min: np.ndarray,
    *,
    delta_min: float = DELTA_MIN_DEFAULT,
    cong_kmh: float = CONG_KMH_DEFAULT,
    persist_n: int = PERSIST_N_DEFAULT,
    anchor_t_min: int = ANCHOR_T_MIN_DEFAULT,
    anchor_t_max: int = ANCHOR_T_MAX_DEFAULT,
    anchor_s_km_min: float = ANCHOR_S_KM_MIN_DEFAULT,
    anchor_s_km_max: float = ANCHOR_S_KM_MAX_DEFAULT,
) -> np.ndarray:
    """Convenience: observed + baseline -> origin_cc boolean congestion mask.

    Chains `base_abs_mask` then `origin_cc_mask`. Inputs are `(n_space, n_time)`
    matrices in km/h (NaN allowed) sharing the given offset axes.
    """
    abs_m = base_abs_mask(obs, base, delta_min=delta_min, cong_kmh=cong_kmh, persist_n=persist_n)
    return origin_cc_mask(
        abs_m, space_offsets_m, time_offsets_min,
        anchor_t_min=anchor_t_min, anchor_t_max=anchor_t_max,
        anchor_s_km_min=anchor_s_km_min, anchor_s_km_max=anchor_s_km_max,
    )
