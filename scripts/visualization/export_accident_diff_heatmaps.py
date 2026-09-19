# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///
"""
For a selected set of accident events, export two clean PNGs each:
  (1) observed speed heatmap
  (2) speed-difference heatmap  (observed − historical baseline)

Output: output/slide_materials/accidents/

Usage:
  uv run scripts/visualization/export_accident_diff_heatmaps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import os as _os
if _os.environ.get("MPLBACKEND", "").startswith("module://"):
    _os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.lib.visualization import (
    SPEED_MIN, SPEED_MAX, DIFF_ABS_MAX,
    SPEED_CMAP, DIFF_CMAP,
    configure_cjk_font,
)
from scripts.lib.calendar import day_class as classify_day_class

configure_cjk_font()

DATA_ROOT = PROJECT_ROOT / "data"
OUT_DIR   = PROJECT_ROOT / "output" / "slide_materials" / "accidents"

DIR_LABEL = {"N": "北上", "S": "南下", "E": "東行", "W": "西行"}

# Accidents to render: pick a mix of morning peak / evening peak / off-peak
TARGET_IDS = [
    "2024040088",   # N1-N  198.4 km  07:09  morning peak
    "2024040350",   # N1-S  149.1 km  07:30  morning peak
    "2024040068",   # N1-N  147.2 km  19:01  evening peak
    "2024040049",   # N1-N  124.6 km  15:41  off-peak
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    arr  = np.load(str(DATA_ROOT / "accident_samples_5m.npz"), allow_pickle=False)
    meta = pq.read_table(str(DATA_ROOT / "accident_samples_5m_metadata.parquet")).to_pandas()
    meta["report_time"] = pd.to_datetime(meta["report_time"])
    return (
        arr["matrix"].astype(np.float32),
        arr["space_offsets_m"].astype(np.int32),
        arr["time_offsets_min"].astype(np.int32),
        meta,
    )


def load_hist_profile(freeway: str) -> pd.DataFrame:
    path = DATA_ROOT / f"historical_speed_profile_5m/freeway={freeway}/profile_0.parquet"
    df = pq.read_table(str(path)).to_pandas()
    df["speed_kmh"] = df["mean_speed_x10"].astype(float) / 10.0
    return df


# ── Historical baseline builder ───────────────────────────────────────────────

def build_hist_matrix(
    profile_df: pd.DataFrame,
    direction: str,
    day_cls: str,
    acc_mileage_m: int,
    space_offsets_m: np.ndarray,
    time_offsets_min: np.ndarray,
    report_time: pd.Timestamp,
) -> np.ndarray:
    """Return (n_space, n_time) historical baseline speed matrix (km/h)."""
    sub = profile_df[
        (profile_df["direction"] == direction) &
        (profile_df["day_class"] == day_cls)
    ]
    # Pivot to (mileage_m, minute_bucket) → speed_kmh
    pivot = sub.pivot_table(
        index="mileage_m",
        columns="minute_bucket",
        values="speed_kmh",
        aggfunc="mean",
    )
    profile_miles   = pivot.index.to_numpy(dtype=float)
    profile_buckets = pivot.columns.to_numpy(dtype=int)
    profile_mat     = pivot.to_numpy(dtype=float)

    n_space = len(space_offsets_m)
    n_time  = len(time_offsets_min)
    hist = np.full((n_space, n_time), np.nan, dtype=float)

    for t_idx, t_off in enumerate(time_offsets_min):
        abs_time = report_time + pd.Timedelta(minutes=int(t_off))
        bucket   = (abs_time.hour * 60 + abs_time.minute) // 5 * 5
        bucket   = min(max(bucket, 0), 1435)

        # Nearest bucket in case of any gap
        b_idx = int(np.argmin(np.abs(profile_buckets - bucket)))
        col   = profile_mat[:, b_idx]

        for s_idx, s_off in enumerate(space_offsets_m):
            # positive s_off = upstream; direction determines which way mileage grows
            if direction in ("S", "E"):
                abs_mile = float(acc_mileage_m) - float(s_off)
            else:
                abs_mile = float(acc_mileage_m) + float(s_off)

            # Interpolate; extrapolation → NaN
            valid = np.isfinite(col)
            if valid.sum() < 2:
                continue
            if abs_mile < profile_miles[valid][0] or abs_mile > profile_miles[valid][-1]:
                continue
            hist[s_idx, t_idx] = np.interp(abs_mile, profile_miles[valid], col[valid])

    return hist


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _imshow(ax, matrix, space_axis_km, time_axis_min, cmap, vmin, vmax):
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("lightgrey")
    return ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_obj,
        vmin=vmin, vmax=vmax,
        origin="lower",
        extent=[float(time_axis_min[0]), float(time_axis_min[-1]),
                float(space_axis_km[0]),  float(space_axis_km[-1])],
    )


def _decorate(ax, fig, im, report_time, space_axis_km, time_axis_min,
              arrived_min, completed_min, cbar_label):
    fs = 11

    # Reference lines
    ax.axvline(0, color="red", linestyle="--", linewidth=1.8, alpha=0.9, label="事故發送")
    ax.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.5)

    if arrived_min is not None:
        ax.axvline(arrived_min, color="#2563eb", linewidth=1.5, linestyle="-.", alpha=0.85, label="到達")
    if completed_min is not None:
        ax.axvline(completed_min, color="#16a34a", linewidth=1.5, linestyle="-.", alpha=0.85, label="處理完成")

    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    # X-axis: absolute time ticks every 15 min
    tick_step = 15
    first_tick = int(np.ceil(time_axis_min[0] / tick_step) * tick_step)
    x_ticks = list(range(first_tick, int(time_axis_min[-1]) + 1, tick_step))
    rt = pd.Timestamp(report_time)
    x_labels = [(rt + pd.Timedelta(minutes=t)).strftime("%H:%M") for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=fs - 1)
    ax.set_xlabel("時間（5 分鐘）", fontsize=fs, fontweight="bold")

    ax.set_ylabel("相對位置 (km，上游為正)", fontsize=fs, fontweight="bold")
    ax.tick_params(labelsize=fs - 1)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=fs - 1)
    cbar.ax.tick_params(labelsize=fs - 2)


def save_observed(
    matrix_ns: np.ndarray,
    space_axis_km: np.ndarray,
    time_axis_min: np.ndarray,
    row: pd.Series,
    out_path: Path,
) -> None:
    report_time = row["report_time"]
    arrived_min   = _offset(report_time, row.get("arrived_time"))
    completed_min = _offset(report_time, row.get("completed_time"))
    title = (
        f"觀測速度 — 國道{row['freeway']}  {DIR_LABEL.get(row['direction'], row['direction'])}"
        f"  {float(row['mileage_km']):.1f}K  |  {pd.Timestamp(report_time):%Y-%m-%d %H:%M}"
    )
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    im = _imshow(ax, matrix_ns, space_axis_km, time_axis_min,
                 SPEED_CMAP, SPEED_MIN, SPEED_MAX)
    _decorate(ax, fig, im, report_time, space_axis_km, time_axis_min,
              arrived_min, completed_min, "速度 (km/h)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def save_historical(
    hist_ns: np.ndarray,
    space_axis_km: np.ndarray,
    time_axis_min: np.ndarray,
    row: pd.Series,
    out_path: Path,
) -> None:
    report_time = row["report_time"]
    arrived_min   = _offset(report_time, row.get("arrived_time"))
    completed_min = _offset(report_time, row.get("completed_time"))
    title = (
        f"歷史基準速度（同時段平均） — 國道{row['freeway']}  {DIR_LABEL.get(row['direction'], row['direction'])}"
        f"  {float(row['mileage_km']):.1f}K  |  {pd.Timestamp(report_time):%Y-%m-%d %H:%M}"
    )
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    im = _imshow(ax, hist_ns, space_axis_km, time_axis_min,
                 SPEED_CMAP, SPEED_MIN, SPEED_MAX)
    _decorate(ax, fig, im, report_time, space_axis_km, time_axis_min,
              arrived_min, completed_min, "速度 (km/h)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def save_diff(
    diff_ns: np.ndarray,
    space_axis_km: np.ndarray,
    time_axis_min: np.ndarray,
    row: pd.Series,
    out_path: Path,
) -> None:
    report_time = row["report_time"]
    arrived_min   = _offset(report_time, row.get("arrived_time"))
    completed_min = _offset(report_time, row.get("completed_time"))
    title = (
        f"速度差值（觀測 - 歷史基準） — 國道{row['freeway']}  {DIR_LABEL.get(row['direction'], row['direction'])}"
        f"  {float(row['mileage_km']):.1f}K  |  {pd.Timestamp(report_time):%Y-%m-%d %H:%M}"
    )
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    im = _imshow(ax, diff_ns, space_axis_km, time_axis_min,
                 DIFF_CMAP, -DIFF_ABS_MAX, DIFF_ABS_MAX)
    _decorate(ax, fig, im, report_time, space_axis_km, time_axis_min,
              arrived_min, completed_min, "速度差 (km/h)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def _offset(report_time, ts) -> float | None:
    if ts is None or pd.isna(ts):
        return None
    diff = (pd.Timestamp(ts) - pd.Timestamp(report_time)).total_seconds() / 60.0
    return float(diff)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    matrix_all, space_offsets_m, time_offsets_min, meta = load_samples()
    space_axis_km = space_offsets_m.astype(float) / 1000.0
    time_axis_min = time_offsets_min.astype(float)

    hist_cache: dict[str, pd.DataFrame] = {}
    summary = []

    for acc_id in TARGET_IDS:
        idx_rows = meta[meta["accident_id"] == acc_id]
        if idx_rows.empty:
            print(f"[skip] {acc_id} not in metadata")
            continue
        idx = idx_rows.index[0]
        row = idx_rows.iloc[0]

        freeway   = str(row["freeway"])
        direction = str(row["direction"])
        report_time = pd.Timestamp(row["report_time"])
        acc_mileage_m = int(round(float(row["mileage_km"]) * 1000))
        day_cls = classify_day_class(report_time)

        matrix_ns = matrix_all[idx].astype(float)  # (n_space, n_time), speed km/h

        # Load historical profile
        if freeway not in hist_cache:
            hist_cache[freeway] = load_hist_profile(freeway)
        profile_df = hist_cache[freeway]

        hist_ns = build_hist_matrix(
            profile_df, direction, day_cls,
            acc_mileage_m, space_offsets_m, time_offsets_min, report_time,
        )
        diff_ns = matrix_ns - hist_ns

        out_obs  = OUT_DIR / f"{acc_id}_observed.png"
        out_hist = OUT_DIR / f"{acc_id}_historical.png"
        out_diff = OUT_DIR / f"{acc_id}_diff.png"

        save_observed(matrix_ns, space_axis_km, time_axis_min, row, out_obs)
        save_historical(hist_ns, space_axis_km, time_axis_min, row, out_hist)
        save_diff(diff_ns,     space_axis_km, time_axis_min, row, out_diff)

        obs_mean  = float(np.nanmean(matrix_ns))
        hist_mean = float(np.nanmean(hist_ns))
        diff_min  = float(np.nanmin(diff_ns))
        diff_max  = float(np.nanmax(diff_ns))
        diff_mean = float(np.nanmean(diff_ns))

        print(f"[{acc_id}] obs_mean={obs_mean:.1f}  hist_mean={hist_mean:.1f}  diff=[{diff_min:.1f}, {diff_max:.1f}]  diff_mean={diff_mean:.1f}")
        print(f"  → {out_obs.name}")
        print(f"  → {out_hist.name}")
        print(f"  → {out_diff.name}")

        summary.append({
            "事故編號": acc_id,
            "國道": freeway,
            "方向": DIR_LABEL.get(direction, direction),
            "里程 (km)": f"{float(row['mileage_km']):.1f}",
            "發送時間": f"{report_time:%Y-%m-%d %H:%M}",
            "日期類型": {"weekday": "平日", "weekend": "週末", "national_holiday": "國定假日"}.get(day_cls, day_cls),
            "觀測均速 (km/h)": f"{obs_mean:.1f}",
            "歷史基準均速 (km/h)": f"{hist_mean:.1f}",
            "差值最小 (km/h)": f"{diff_min:.1f}",
            "差值最大 (km/h)": f"{diff_max:.1f}",
            "差值均值 (km/h)": f"{diff_mean:.1f}",
        })

    # Markdown table
    print()
    print("## 事故樣本摘要")
    print()
    if summary:
        headers = list(summary[0].keys())
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in summary:
            print("| " + " | ".join(str(r[h]) for h in headers) + " |")


if __name__ == "__main__":
    main()
