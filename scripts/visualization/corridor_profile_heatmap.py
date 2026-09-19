# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb",
#     "matplotlib",
#     "numpy",
#     "pandas",
# ]
# ///
# ruff: noqa: E402
"""Render a corridor "typical-weekday" speed heatmap straight from the historical profile.

Unlike the accident-relative heatmaps in `scripts/lib/visualization.py`, this view is
*absolute*: y = absolute mileage (km) along a corridor, x = time of day (00:00-24:00).
Each cell is the historical mean speed for that (mileage, 5-min bucket) on a chosen
day_class. It makes recurring congestion (既定塞車地點) visible directly — a persistent
red band parked over the peak hours, with no accident involved.

Mileage rasterization mirrors `scripts/lib/windows.rasterize_window`: regular km grid,
per-time-column 1-D linear interpolation across observed mainline detectors (filled only
within the convex hull of observed mileages).

Source:
  data/historical_speed_profile_5m/freeway=<F>/profile_0.parquet

Usage (the central-Taiwan example used on slide Page 1):
  uv run scripts/visualization/corridor_profile_heatmap.py \
      --freeway N1 --direction S --mileage-min 150 --mileage-max 190 \
      --style slide --title "國道1號 南下 台中段（典型平日）" \
      --output output/slide_materials/corridors/N1_S_taichung_weekday.png

  # peek at other corridors / day types:
  uv run scripts/visualization/corridor_profile_heatmap.py --freeway N3 --direction N \
      --mileage-min 180 --mileage-max 215 --day-class weekday
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

import os as _os
if _os.environ.get("MPLBACKEND", "").startswith("module://"):
    _os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.visualization import (  # noqa: E402
    SPEED_CMAP,
    SPEED_MAX,
    SPEED_MIN,
    STYLE_MAP,
    configure_cjk_font,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_PROFILE = PROJECT_ROOT / "data" / "historical_speed_profile_5m"
# AM 07:00-09:00, PM 17:00-19:00, in minute-of-day
PEAK_BANDS = [(7 * 60, 9 * 60), (17 * 60, 19 * 60)]
DAY_CLASS_LABEL = {"weekday": "平日", "weekend": "週末", "national_holiday": "國定假日"}
DIR_LABEL = {"N": "北上", "S": "南下", "E": "東行", "W": "西行"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freeway", required=True, help="e.g. N1, N3")
    p.add_argument("--direction", required=True, choices=["N", "S", "E", "W"])
    p.add_argument("--mileage-min", type=float, required=True, help="corridor lower mileage (km)")
    p.add_argument("--mileage-max", type=float, required=True, help="corridor upper mileage (km)")
    p.add_argument("--day-class", default="weekday", choices=["weekday", "weekend", "national_holiday"])
    p.add_argument("--metric", default="mean", choices=["mean", "p50"],
                   help="speed statistic to plot (mean or median)")
    p.add_argument("--profile-dir", default=str(DEFAULT_PROFILE))
    p.add_argument("--spatial-resolution-km", type=float, default=1.0)
    p.add_argument("--min-coverage", type=float, default=0.5,
                   help="drop (vdid,bucket) cells observed on fewer than this fraction of days")
    p.add_argument("--mainline-only", dest="mainline_only", action="store_true", default=True)
    p.add_argument("--include-ramps", dest="mainline_only", action="store_false",
                   help="also include ramp/loop detectors (default: mainline only)")
    p.add_argument("--style", default="slide", choices=list(STYLE_MAP))
    p.add_argument("--title", default=None)
    p.add_argument("--output", default=None, help="PNG path; omit to only show stats")
    p.add_argument("--no-peak-bands", action="store_true", help="don't shade AM/PM peak windows")
    return p.parse_args()


def load_corridor(args) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Return (mileage_m, minute_bucket, speed_kmh) long arrays for the corridor."""
    fw_file = Path(args.profile_dir).resolve() / f"freeway={args.freeway}" / "profile_0.parquet"
    if not fw_file.exists():
        raise SystemExit(f"profile not found: {fw_file}")
    col = "mean_speed_x10" if args.metric == "mean" else "p50_speed_x10"
    mainline = "AND vdid LIKE '%-M-%'" if args.mainline_only else ""
    con = duckdb.connect()
    df = con.sql(
        f"""
        SELECT mileage_m, minute_bucket, {col} / 10.0 AS speed_kmh
        FROM read_parquet('{fw_file.as_posix()}')
        WHERE direction = '{args.direction}'
          AND day_class = '{args.day_class}'
          AND coverage_ratio >= {args.min_coverage}
          AND mileage_m BETWEEN {int(args.mileage_min * 1000)} AND {int(args.mileage_max * 1000)}
          {mainline}
        """
    ).df()
    if df.empty:
        raise SystemExit("no profile rows matched the corridor filter (check freeway/direction/mileage/coverage)")
    return (
        df["mileage_m"].to_numpy(),
        df["minute_bucket"].to_numpy().astype(int),
        df["speed_kmh"].to_numpy(dtype=np.float32),
    )


def rasterize_corridor(mileage_m, minute_bucket, speed, args) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Build a (n_space, n_time) matrix with per-time-column space interpolation."""
    step_m = args.spatial_resolution_km * 1000
    min_m = int(args.mileage_min * 1000)
    max_m = int(args.mileage_max * 1000)
    n_space = int(round((max_m - min_m) / step_m)) + 1
    mileage_axis_km = (min_m + np.arange(n_space) * step_m) / 1000.0

    buckets = np.arange(0, 1440, 5)  # 288 five-minute buckets, 00:00..23:55
    n_time = buckets.size
    bucket_to_col = {int(b): i for i, b in enumerate(buckets)}

    matrix = np.full((n_space, n_time), np.nan, dtype=np.float32)
    space_idx = np.round((mileage_m - min_m) / step_m).astype(np.int64)
    time_idx = np.array([bucket_to_col.get(int(b), -1) for b in minute_bucket])

    valid = (space_idx >= 0) & (space_idx < n_space) & (time_idx >= 0)
    space_idx, time_idx, speed = space_idx[valid], time_idx[valid], speed[valid]
    flat = space_idx * n_time + time_idx
    sums = np.bincount(flat, weights=speed, minlength=n_space * n_time)
    counts = np.bincount(flat, minlength=n_space * n_time)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    matrix = means.reshape(n_space, n_time).astype(np.float32)

    # Per-column 1-D interpolation across observed mileages (within convex hull).
    full = np.arange(n_space)
    for t in range(n_time):
        col = matrix[:, t]
        mask = ~np.isnan(col)
        if mask.sum() < 2:
            continue
        xs = np.where(mask)[0]
        interp = np.interp(full, xs, col[mask], left=np.nan, right=np.nan)
        interior = (full >= xs.min()) & (full <= xs.max())
        matrix[interior, t] = interp[interior]

    return matrix, mileage_axis_km, buckets


def render(matrix, mileage_axis_km, buckets, args) -> plt.Figure:
    configure_cjk_font()
    style = STYLE_MAP[args.style]
    fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

    cmap = plt.get_cmap(SPEED_CMAP).copy()
    cmap.set_bad("white")
    im = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=SPEED_MIN,
        vmax=SPEED_MAX,
        origin="lower",
        extent=[0.0, 24.0, float(mileage_axis_km[0]), float(mileage_axis_km[-1])],
    )

    if not args.no_peak_bands:
        for lo, hi in PEAK_BANDS:
            ax.axvspan(lo / 60.0, hi / 60.0, color="black", alpha=0.06, zorder=1)
            for x in (lo / 60.0, hi / 60.0):
                ax.axvline(x, color="black", linestyle=":", linewidth=0.8, alpha=0.45)

    fs = style.font_size
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], fontsize=fs, rotation=45, ha="right")
    ax.set_xlabel("一天時間（時:分）", fontsize=fs + 1, fontweight="bold")
    ax.set_ylabel("國道里程（km）", fontsize=fs + 1, fontweight="bold")
    ax.tick_params(labelsize=fs)

    if args.title:
        title = args.title
    else:
        title = (
            f"{args.freeway} {DIR_LABEL.get(args.direction, args.direction)} "
            f"{args.mileage_min:.0f}–{args.mileage_max:.0f}K｜"
            f"{DAY_CLASS_LABEL.get(args.day_class, args.day_class)}典型速度"
            f"（{'平均' if args.metric == 'mean' else '中位數'}）"
        )
    ax.set_title(title, fontsize=style.title_size, fontweight="bold", pad=8)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("速度 (km/h)", fontsize=fs)
    cbar.ax.tick_params(labelsize=max(fs - 1, 8))
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    mileage_m, minute_bucket, speed = load_corridor(args)
    matrix, mileage_axis_km, buckets = rasterize_corridor(mileage_m, minute_bucket, speed, args)

    filled = np.count_nonzero(~np.isnan(matrix))
    print(f"corridor {args.freeway} {args.direction} {args.mileage_min:.0f}-{args.mileage_max:.0f}K "
          f"| {args.day_class} | metric={args.metric}")
    print(f"matrix {matrix.shape} | filled {filled}/{matrix.size} ({filled / matrix.size:.0%})")
    # Quick PM-peak slowest-cell report so the chosen corridor can be sanity-checked from CLI.
    pm_cols = (buckets >= 17 * 60) & (buckets < 19 * 60)
    pm = matrix[:, pm_cols]
    if np.isfinite(pm).any():
        r, c = np.unravel_index(np.nanargmin(pm), pm.shape)
        bmin = buckets[pm_cols][c]
        print(f"slowest PM-peak cell: {mileage_axis_km[r]:.0f}K @ {bmin // 60:02d}:{bmin % 60:02d} "
              f"= {pm[r, c]:.0f} km/h")

    fig = render(matrix, mileage_axis_km, buckets, args)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"saved -> {out}")
    else:
        print("(no --output given; not saved)")


if __name__ == "__main__":
    main()
