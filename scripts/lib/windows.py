"""Sample window configuration and slicing helpers.

The canonical accident-sample window built by `scripts/data/build_accident_samples.py`
is the superset; downstream training and inference subset by slicing.

Spatial axis is normalized to upstream-distance (m): positive values are
upstream of the accident, negative values are downstream. This makes matrices
stackable across accidents and directions.

Time axis is offset-from-report (min): negative values are pre-accident.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SampleWindow:
    upstream_km: int = 50
    downstream_km: int = 10
    time_before_min: int = 60
    time_after_min: int = 240
    spatial_resolution_km: int = 1
    resolution_min: int = 5

    @property
    def n_space(self) -> int:
        return (self.upstream_km + self.downstream_km) // self.spatial_resolution_km + 1

    @property
    def n_time(self) -> int:
        return (self.time_before_min + self.time_after_min) // self.resolution_min + 1

    @property
    def space_offsets_m(self) -> np.ndarray:
        # Index 0 = farthest downstream. Last index = farthest upstream.
        start = -self.downstream_km * 1000
        step = self.spatial_resolution_km * 1000
        return np.arange(self.n_space, dtype=np.int32) * step + start

    @property
    def time_offsets_min(self) -> np.ndarray:
        start = -self.time_before_min
        return np.arange(self.n_time, dtype=np.int32) * self.resolution_min + start

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SampleWindow":
        return cls(**data)


DEFAULT_WINDOW = SampleWindow()


def absolute_mileage_axis_m(accident_mileage_m: int, direction: str, window: SampleWindow) -> np.ndarray:
    """Return absolute mileage values (m) for each spatial cell along the route.

    For S/E directions the upstream side has lower absolute mileage; for N/W
    the upstream side has higher absolute mileage. Output is in
    upstream-distance order to match `window.space_offsets_m`.
    """
    offsets = window.space_offsets_m.astype(np.int64)
    if direction in ("S", "E"):
        return (accident_mileage_m - offsets).astype(np.int64)
    return (accident_mileage_m + offsets).astype(np.int64)


def slice_station_df(
    station_df: pd.DataFrame,
    accident_mileage_m: int,
    direction: str,
    freeway: str,
    report_time: pd.Timestamp,
    window: SampleWindow,
) -> pd.DataFrame:
    """Filter a station-level dataframe to one accident window.

    Returns rows from `station_df` matching freeway/direction whose absolute
    mileage falls inside the spatial window and whose timestamp falls inside
    the time window. Adds two derived columns:
    - `space_offset_m` (upstream distance, int)
    - `time_offset_min` (signed minutes from report, int)
    """
    if station_df.empty:
        return station_df.iloc[0:0].copy()

    if direction in ("S", "E"):
        m_low = accident_mileage_m - window.upstream_km * 1000
        m_high = accident_mileage_m + window.downstream_km * 1000
    else:
        m_low = accident_mileage_m - window.downstream_km * 1000
        m_high = accident_mileage_m + window.upstream_km * 1000

    t_start = report_time - pd.Timedelta(minutes=window.time_before_min)
    t_end = report_time + pd.Timedelta(minutes=window.time_after_min)

    mask = (
        (station_df["freeway"] == freeway)
        & (station_df["direction"] == direction)
        & (station_df["mileage_m"] >= m_low)
        & (station_df["mileage_m"] <= m_high)
        & (station_df["timestamp"] >= t_start)
        & (station_df["timestamp"] <= t_end)
    )
    out = station_df.loc[mask].copy()
    if out.empty:
        return out

    if direction in ("S", "E"):
        out["space_offset_m"] = (accident_mileage_m - out["mileage_m"]).astype("int32")
    else:
        out["space_offset_m"] = (out["mileage_m"] - accident_mileage_m).astype("int32")
    out["time_offset_min"] = (
        (out["timestamp"] - report_time).dt.total_seconds() // 60
    ).astype("int32")
    return out


def rasterize_window(
    rows: pd.DataFrame,
    window: SampleWindow,
    value_col: str = "speed_x10",
    interp_along_space: bool = True,
) -> np.ndarray:
    """Place per-(VD, time) values into a (n_space, n_time) matrix.

    Inputs are rows already filtered by `slice_station_df`. Cells with no data
    are NaN. Optionally fills along the space axis at each time bin by
    1-D linear interpolation (matches legacy `spacetime.build_spacetime_matrix`).
    """
    matrix = np.full((window.n_space, window.n_time), np.nan, dtype=np.float32)
    if rows.empty:
        return matrix

    space_step_m = window.spatial_resolution_km * 1000
    space_origin_m = window.space_offsets_m[0]

    space_idx = ((rows["space_offset_m"].to_numpy() - space_origin_m) / space_step_m).round().astype(np.int64)
    time_idx = ((rows["time_offset_min"].to_numpy() + window.time_before_min) // window.resolution_min).astype(np.int64)
    values = rows[value_col].to_numpy(dtype=np.float32) / (10.0 if value_col.endswith("_x10") else 1.0)

    # Average duplicates by (space_idx, time_idx).
    n_s, n_t = window.n_space, window.n_time
    valid = (space_idx >= 0) & (space_idx < n_s) & (time_idx >= 0) & (time_idx < n_t)
    if not valid.any():
        return matrix
    space_idx = space_idx[valid]
    time_idx = time_idx[valid]
    values = values[valid]

    flat = space_idx * n_t + time_idx
    sums = np.bincount(flat, weights=values, minlength=n_s * n_t)
    counts = np.bincount(flat, minlength=n_s * n_t)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    matrix = means.reshape(n_s, n_t).astype(np.float32)

    if interp_along_space:
        for t in range(n_t):
            col = matrix[:, t]
            mask = ~np.isnan(col)
            if mask.sum() < 2:
                continue
            xs = np.where(mask)[0]
            ys = col[mask]
            full = np.arange(n_s)
            interp = np.interp(full, xs, ys, left=np.nan, right=np.nan)
            # Only fill cells inside the convex hull of observed VDs.
            interior = (full >= xs.min()) & (full <= xs.max())
            matrix[interior, t] = interp[interior]
    return matrix


def coverage(matrix: np.ndarray) -> dict:
    """Return cell-level coverage statistics for a (n_space, n_time) matrix."""
    total = int(matrix.size)
    filled = int(np.count_nonzero(~np.isnan(matrix)))
    return {
        "filled_cells": filled,
        "total_cells": total,
        "filled_ratio": (filled / total) if total else None,
    }


def subset(
    matrix: np.ndarray,
    window: SampleWindow,
    upstream_km: int | None = None,
    downstream_km: int | None = None,
    time_before_min: int | None = None,
    time_after_min: int | None = None,
) -> tuple[np.ndarray, SampleWindow]:
    """Return a smaller matrix and its corresponding `SampleWindow`."""
    new_window = SampleWindow(
        upstream_km=window.upstream_km if upstream_km is None else upstream_km,
        downstream_km=window.downstream_km if downstream_km is None else downstream_km,
        time_before_min=window.time_before_min if time_before_min is None else time_before_min,
        time_after_min=window.time_after_min if time_after_min is None else time_after_min,
        spatial_resolution_km=window.spatial_resolution_km,
        resolution_min=window.resolution_min,
    )
    if (
        new_window.upstream_km > window.upstream_km
        or new_window.downstream_km > window.downstream_km
        or new_window.time_before_min > window.time_before_min
        or new_window.time_after_min > window.time_after_min
    ):
        raise ValueError("subset window cannot exceed source window in any dimension")

    space_offsets = window.space_offsets_m
    time_offsets = window.time_offsets_min
    s_mask = (space_offsets >= -new_window.downstream_km * 1000) & (space_offsets <= new_window.upstream_km * 1000)
    t_mask = (time_offsets >= -new_window.time_before_min) & (time_offsets <= new_window.time_after_min)
    return matrix[s_mask][:, t_mask], new_window


def matrix_to_csv_dataframe(
    matrix: np.ndarray,
    accident_mileage_m: int,
    direction: str,
    window: SampleWindow,
) -> pd.DataFrame:
    """Build a debug-friendly wide DataFrame: rows=mileage_km, cols=time_offset_min."""
    abs_mileage_m = absolute_mileage_axis_m(accident_mileage_m, direction, window)
    df = pd.DataFrame(
        matrix,
        index=np.round(abs_mileage_m / 1000.0, 3),
        columns=window.time_offsets_min.tolist(),
    )
    df.index.name = "mileage_km"
    df.columns.name = "time_offset_min"
    return df.round(2)
