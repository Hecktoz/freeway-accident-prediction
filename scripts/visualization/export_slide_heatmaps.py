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
Export clean historical-speed heatmaps for slide use.

Outputs (one PNG per scenario, no embedded tables):
  output/slide_materials/hist_speed_<freeway>_<dir>_<day_class>.png

Also prints a markdown summary table to stdout.

Usage:
  uv run scripts/visualization/export_slide_heatmaps.py
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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.visualization import configure_cjk_font
configure_cjk_font()
DATA_ROOT = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "output" / "slide_materials"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SPEED_MIN, SPEED_MAX = 0.0, 140.0
CMAP = "RdYlGn"

SCENARIOS = [
    ("N1", "N", "weekday"),
    ("N1", "S", "weekday"),
    ("N1", "N", "weekend"),
    ("N3", "N", "weekday"),
]

DAY_LABEL = {
    "weekday": "平日",
    "weekend": "週末",
    "national_holiday": "國定假日",
}
DIR_LABEL = {"N": "北上", "S": "南下", "E": "東行", "W": "西行"}


def load_profile(freeway: str) -> pd.DataFrame:
    path = DATA_ROOT / f"historical_speed_profile_5m/freeway={freeway}/profile_0.parquet"
    df = pq.read_table(str(path)).to_pandas()
    df["speed_kmh"] = df["mean_speed_x10"].astype(float) / 10.0
    df["mileage_km"] = df["mileage_m"].astype(float) / 1000.0
    return df


def build_matrix(df: pd.DataFrame, direction: str, day_class: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (matrix, mileage_km_axis, minute_axis).  matrix is (n_mile, n_time)."""
    sub = df[(df["direction"] == direction) & (df["day_class"] == day_class)].copy()
    pivot = sub.pivot_table(
        index="mileage_km",
        columns="minute_bucket",
        values="speed_kmh",
        aggfunc="mean",
    )
    mileage_axis = pivot.index.to_numpy(dtype=float)
    minute_axis = pivot.columns.to_numpy(dtype=int)
    matrix = pivot.to_numpy(dtype=float)
    return matrix, mileage_axis, minute_axis


def minute_to_hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def plot_heatmap(
    matrix: np.ndarray,
    mileage_axis: np.ndarray,
    minute_axis: np.ndarray,
    freeway: str,
    direction: str,
    day_class: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    im = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=plt.get_cmap(CMAP).copy(),
        vmin=SPEED_MIN,
        vmax=SPEED_MAX,
        origin="lower",
        extent=[float(minute_axis[0]), float(minute_axis[-1]),
                float(mileage_axis[0]), float(mileage_axis[-1])],
    )

    # X-axis: every 60 min, labelled HH:MM
    tick_minutes = list(range(0, 1441, 60))
    ax.set_xticks(tick_minutes)
    ax.set_xticklabels([minute_to_hhmm(m) for m in tick_minutes], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("時刻 (每 5 分鐘)", fontsize=11, fontweight="bold")

    ax.set_ylabel("里程 (km)", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=9)

    ax.set_title(
        f"歷史平均速度 — 國道 {freeway}  {DIR_LABEL.get(direction, direction)}  ({DAY_LABEL.get(day_class, day_class)})",
        fontsize=13, fontweight="bold", pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("平均速度 (km/h)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def summary_stats(matrix: np.ndarray, mileage_axis: np.ndarray, minute_axis: np.ndarray) -> dict:
    flat = matrix[np.isfinite(matrix)]
    return {
        "里程範圍 (km)": f"{mileage_axis.min():.0f} – {mileage_axis.max():.0f}",
        "VD 站數 (行數)": int(matrix.shape[0]),
        "時間步數 (5min)": int(matrix.shape[1]),
        "整體均速 (km/h)": f"{flat.mean():.1f}",
        "最低均速 (km/h)": f"{flat.min():.1f}",
        "最高均速 (km/h)": f"{flat.max():.1f}",
        "覆蓋格數比例": f"{np.isfinite(matrix).mean():.1%}",
    }


def main() -> None:
    cache: dict[str, pd.DataFrame] = {}
    rows = []

    for freeway, direction, day_class in SCENARIOS:
        if freeway not in cache:
            cache[freeway] = load_profile(freeway)
        df = cache[freeway]

        matrix, mileage_axis, minute_axis = build_matrix(df, direction, day_class)
        fname = f"hist_speed_{freeway}_{direction}_{day_class}.png"
        out_path = OUT_DIR / fname
        plot_heatmap(matrix, mileage_axis, minute_axis, freeway, direction, day_class, out_path)

        stats = summary_stats(matrix, mileage_axis, minute_axis)
        rows.append({
            "國道": freeway,
            "方向": DIR_LABEL.get(direction, direction),
            "日期類型": DAY_LABEL.get(day_class, day_class),
            **stats,
        })

    # Markdown table
    print()
    print("## 各情境摘要（貼到 PPT 說明用）")
    print()
    if rows:
        headers = list(rows[0].keys())
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            print("| " + " | ".join(str(r[h]) for h in headers) + " |")


if __name__ == "__main__":
    main()
