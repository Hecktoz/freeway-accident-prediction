# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///
# ruff: noqa: E402
"""
Render accident spacetime heatmaps from Drive station Parquet datasets.

Examples:
  uv run scripts/visualization/accident_heatmap.py --limit 5 --force
  uv run scripts/visualization/accident_heatmap.py --accident-id 2024040004 --force
  uv run scripts/visualization/accident_heatmap.py --start-date 2024-04-01 --end-date 2025-05-31
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import os as _os
if _os.environ.get("MPLBACKEND", "").startswith("module://"):
    _os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.visualization import (
    STYLE_MAP,
    STYLE_REPORT,
    HeatmapStyle,
    heatmap as _lib_heatmap,
    configure_cjk_font,
)
from dataclasses import replace as _dc_replace
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_ACCIDENT_CSV = DEFAULT_DATA_ROOT / "accident_data.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
RESEARCH_UPSTREAM_KM = 25.0
RESEARCH_DOWNSTREAM_KM = 10.0
RESEARCH_TIME_BEFORE_MIN = 30
# Endpoint convention: -30..+145 has 36 five-minute cells and covers a
# 150-minute post-incident prediction horizon.
RESEARCH_TIME_AFTER_MIN = 145

ACCIDENT_TIME_COLUMNS = ["發送", "到達", "處理完成"]
DIR_MAP = {"北": "N", "南": "S", "東": "E", "西": "W"}
STATION_COLUMNS = [
    "source_date",
    "timestamp",
    "vdid",
    "freeway",
    "direction",
    "mileage_m",
    "speed_x10",
    "n_lanes_valid",
]


@dataclass(frozen=True)
class HeatmapConfig:
    upstream_km: float = 50.0
    downstream_km: float = 10.0
    time_before_min: int = 60
    time_after_min: int = 240
    spatial_resolution_km: float = 1.0
    resolution_min: int = 5
    speed_min: float = 0
    speed_max: float = 140.0

    @property
    def n_time_steps(self) -> int:
        return int(math.floor((self.time_before_min + self.time_after_min) / self.resolution_min)) + 1

    @property
    def span_label(self) -> str:
        return f"[+{self.upstream_km:g}/-{self.downstream_km:g}km]"

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def station_dataset_name(resolution_min: int) -> str:
    if resolution_min not in (1, 5):
        raise ValueError("resolution_min must be 1 or 5")
    return f"vd_station_{resolution_min}m"


def station_path(data_root: Path, resolution_min: int, date_str: str) -> Path:
    ts = pd.Timestamp(date_str)
    return (
        data_root
        / station_dataset_name(resolution_min)
        / f"source_year={ts:%Y}"
        / f"source_month={ts:%m}"
        / f"vd_station_{ts:%Y%m%d}.parquet"
    )


def required_dates(report_time: pd.Timestamp, config: HeatmapConfig) -> tuple[str, ...]:
    start = report_time - pd.Timedelta(minutes=config.time_before_min)
    end = report_time + pd.Timedelta(minutes=config.time_after_min)
    return tuple(
        ts.strftime("%Y-%m-%d")
        for ts in pd.date_range(start.normalize(), end.normalize(), freq="D")
    )


def normalize_direction(raw: object) -> str:
    if pd.isna(raw):
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    first = text[0]
    if first in DIR_MAP:
        return DIR_MAP[first]
    upper = text.upper()
    if upper and upper[0] in {"N", "S", "E", "W"}:
        return upper[0]
    return DIR_MAP.get(text, text)


def load_accidents(
    csv_path: str | Path = DEFAULT_ACCIDENT_CSV,
    freeway: str | None = None,
) -> pd.DataFrame:
    path = _resolve_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Accident CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"任務編號": "string"})
    required = ["任務編號", "國道", "方向", "里程數", "發送", "到達", "處理完成"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Accident CSV missing required columns: {missing}")

    for col in ACCIDENT_TIME_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df[df["發送"].notna()].copy()
    df["accident_id"] = df["任務編號"].astype(str).str.strip()
    df["direction_code"] = df["方向"].map(normalize_direction)
    df["mileage_km"] = pd.to_numeric(df["里程數"], errors="coerce")
    df = df[df["mileage_km"].notna()].copy()

    if freeway:
        df = df[df["國道"] == freeway].copy()

    return df.sort_values(["發送", "accident_id"]).reset_index(drop=True)


def filter_accidents(
    acc_df: pd.DataFrame,
    accident_id: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    df = acc_df.copy()
    if accident_id:
        df = df[df["accident_id"] == str(accident_id)]
    if date:
        target = pd.Timestamp(date).normalize()
        df = df[df["發送"].dt.normalize() == target]
    if start_date:
        start = pd.Timestamp(start_date).normalize()
        df = df[df["發送"].dt.normalize() >= start]
    if end_date:
        end = pd.Timestamp(end_date).normalize()
        df = df[df["發送"].dt.normalize() <= end]
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def load_station_for_dates(
    data_root: Path,
    resolution_min: int,
    dates: Iterable[str],
) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    missing = []
    for date_str in dates:
        path = station_path(data_root, resolution_min, date_str)
        if not path.exists():
            missing.append(date_str)
            continue
        frames.append(pd.read_parquet(path, columns=STATION_COLUMNS))

    if not frames:
        return pd.DataFrame(columns=STATION_COLUMNS + ["speed_kmh", "mileage_km"]), missing

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["speed_kmh"] = df["speed_x10"].astype(float) / 10.0
    df["mileage_km"] = df["mileage_m"].astype(float) / 1000.0
    df = df[df["timestamp"].notna()].copy()
    return df, missing


def load_vd_bounds(data_root: Path) -> dict[tuple[str, str], dict[str, float]]:
    path = data_root / "metadata" / "vd_dim.parquet"
    if not path.exists():
        return {}

    dim = pd.read_parquet(path, columns=["freeway", "direction", "mileage_m"])
    dim["mileage_km"] = dim["mileage_m"].astype(float) / 1000.0
    grouped = dim.groupby(["freeway", "direction"], observed=True)["mileage_km"].agg(["min", "max"])
    return {
        (str(freeway), str(direction)): {"min": float(row["min"]), "max": float(row["max"])}
        for (freeway, direction), row in grouped.iterrows()
    }


def space_bounds(accident: dict, config: HeatmapConfig) -> tuple[float, float]:
    acc_km = float(accident["mileage_km"])
    direction = accident["direction"]
    if direction in ("S", "E"):
        return acc_km - config.upstream_km, acc_km + config.downstream_km
    return acc_km - config.downstream_km, acc_km + config.upstream_km


def build_space_axis(start_km: float, end_km: float, step_km: float) -> list[float]:
    n_steps = int(math.floor((end_km - start_km) / step_km + 1e-9)) + 1
    return [round(start_km + i * step_km, 6) for i in range(max(n_steps, 0))]


def build_time_offsets(config: HeatmapConfig) -> list[int]:
    return [
        -config.time_before_min + i * config.resolution_min
        for i in range(config.n_time_steps)
    ]


def _empty_coverage(matrix: np.ndarray, space_axis: list[float], route_bounds: dict | None, missing_dates: list[str]) -> dict:
    return {
        "vd_count": 0,
        "source_mileage_min": None,
        "source_mileage_max": None,
        "route_vd_mileage_min": route_bounds.get("min") if route_bounds else None,
        "route_vd_mileage_max": route_bounds.get("max") if route_bounds else None,
        "space_window_start": float(space_axis[0]) if space_axis else None,
        "space_window_end": float(space_axis[-1]) if space_axis else None,
        "filled_cells": int(np.count_nonzero(~np.isnan(matrix))),
        "total_cells": int(matrix.size),
        "filled_ratio": 0.0 if matrix.size else None,
        "missing_station_dates": missing_dates,
    }


def build_accident_heatmap_matrix(
    accident: dict,
    station_df: pd.DataFrame,
    config: HeatmapConfig,
    route_bounds: dict | None = None,
    missing_dates: list[str] | None = None,
) -> dict:
    report_time = pd.Timestamp(accident["report_time"])
    space_start, space_end = space_bounds(accident, config)
    time_start = report_time - pd.Timedelta(minutes=config.time_before_min)
    time_end = report_time + pd.Timedelta(minutes=config.time_after_min)
    time_offsets = build_time_offsets(config)
    time_axis = [report_time + pd.Timedelta(minutes=offset) for offset in time_offsets]
    space_axis = build_space_axis(space_start, space_end, config.spatial_resolution_km)
    matrix = np.full((len(time_offsets), len(space_axis)), np.nan, dtype=float)

    missing_dates = missing_dates or []
    if station_df.empty or not space_axis or not time_offsets:
        return {
            "matrix": matrix,
            "time_axis": time_axis,
            "space_axis": space_axis,
            "time_offsets_min": time_offsets,
            "accident": accident,
            "coverage": _empty_coverage(matrix, space_axis, route_bounds, missing_dates),
            "config": config.to_dict(),
        }

    route_min = route_bounds.get("min") if route_bounds else None
    route_max = route_bounds.get("max") if route_bounds else None

    min_allowed = max(0.0, float(route_min)) if route_min is not None else 0.0
    max_allowed = float(route_max) if route_max is not None else float("inf")

    interp_margin_km = config.spatial_resolution_km * 2
    mask = (
        (station_df["freeway"] == accident["freeway"])
        & (station_df["direction"] == accident["direction"])
        & (station_df["timestamp"] >= time_start)
        & (station_df["timestamp"] <= time_end)
        & (station_df["mileage_km"] >= min_allowed)
        & (station_df["mileage_km"] <= max_allowed)
        & (station_df["mileage_km"] >= space_start - interp_margin_km)
        & (station_df["mileage_km"] <= space_end + interp_margin_km)
    )
    window_df = station_df.loc[mask, ["timestamp", "vdid", "mileage_km", "speed_kmh"]].copy()

    if window_df.empty:
        return {
            "matrix": matrix,
            "time_axis": time_axis,
            "space_axis": space_axis,
            "time_offsets_min": time_offsets,
            "accident": accident,
            "coverage": _empty_coverage(matrix, space_axis, route_bounds, missing_dates),
            "config": config.to_dict(),
        }

    window_df["time_bin"] = (
        (window_df["timestamp"] - time_start).dt.total_seconds() / 60 / config.resolution_min
    ).astype(int)
    window_df = window_df[(window_df["time_bin"] >= 0) & (window_df["time_bin"] < len(time_offsets))]

    axis_arr = np.array(space_axis, dtype=float)
    valid_axis_mask = (axis_arr >= min_allowed) & (axis_arr <= max_allowed)

    for t_bin, group in window_df.groupby("time_bin", sort=True):
        speeds = group.groupby("mileage_km")["speed_kmh"].mean().sort_index()
        if speeds.empty:
            continue

        x = speeds.index.to_numpy(dtype=float)
        y = speeds.to_numpy(dtype=float)

        if len(speeds) == 1:
            idx = int(np.argmin(np.abs(axis_arr - x[0])))
            if valid_axis_mask[idx] and abs(axis_arr[idx] - x[0]) <= config.spatial_resolution_km / 2:
                matrix[int(t_bin), idx] = y[0]
            continue

        if np.any(valid_axis_mask):
            matrix[int(t_bin), valid_axis_mask] = np.interp(
                axis_arr[valid_axis_mask], x, y, left=y[0], right=y[-1]
            )

    filled = int(np.count_nonzero(~np.isnan(matrix)))
    total = int(matrix.size)
    coverage = {
        "vd_count": int(window_df["vdid"].nunique()),
        "source_mileage_min": float(window_df["mileage_km"].min()),
        "source_mileage_max": float(window_df["mileage_km"].max()),
        "route_vd_mileage_min": route_bounds.get("min") if route_bounds else None,
        "route_vd_mileage_max": route_bounds.get("max") if route_bounds else None,
        "space_window_start": float(space_axis[0]),
        "space_window_end": float(space_axis[-1]),
        "filled_cells": filled,
        "total_cells": total,
        "filled_ratio": (filled / total) if total else None,
        "missing_station_dates": missing_dates,
        "boundary_blank_space_cells": int(np.count_nonzero(~valid_axis_mask)) * len(time_offsets),
    }

    return {
        "matrix": matrix,
        "time_axis": time_axis,
        "space_axis": space_axis,
        "time_offsets_min": time_offsets,
        "accident": accident,
        "coverage": coverage,
        "config": config.to_dict(),
    }


def plot_accident_heatmap(
    result: dict,
    output_path: str | Path | None = None,
    font_path: str | Path | None = None,
    dpi: int = 150,
    style: HeatmapStyle | None = None,
) -> plt.Figure:
    """Render one accident heatmap, delegating to the canonical lib renderer.

    The internal matrix built by ``build_accident_heatmap_matrix`` uses
    ``(n_time, n_space)`` with an absolute-km spatial axis.  This function
    converts to the canonical ``(n_space, n_time)`` / upstream-positive-km
    convention expected by ``scripts.lib.visualization.heatmap``.
    """
    configure_cjk_font(font_path)
    style = style or _dc_replace(STYLE_REPORT, dpi=dpi)

    config = HeatmapConfig(**result["config"])
    matrix = np.asarray(result["matrix"], dtype=float)      # (n_time, n_space)
    space_axis_abs = np.asarray(result["space_axis"], dtype=float)
    time_offsets = np.asarray(result["time_offsets_min"], dtype=float)
    accident = result["accident"]

    # Transpose to (n_space, n_time) and convert to upstream-positive relative km.
    acc_km = float(accident["mileage_km"])
    direction = accident.get("direction", "N")
    matrix_ns = matrix.T  # (n_space, n_time)
    if direction in ("S", "E"):
        # upstream = smaller mileage; relative = acc_km − abs_km.
        # space_axis_abs[0] = acc_km − upstream_km → relative[0] = +upstream_km (upstream first).
        # Reverse so canonical: index 0 = downstream (negative km).
        relative_km: np.ndarray = (acc_km - space_axis_abs)[::-1]
        matrix_ns = matrix_ns[::-1, :]
    else:
        # upstream = larger mileage; relative = abs_km − acc_km.
        # space_axis_abs[0] = acc_km − downstream_km → relative[0] = −downstream_km ✓
        relative_km = space_axis_abs - acc_km

    accident_meta = {
        "accident_id":   accident.get("accident_id"),
        "freeway":       accident.get("freeway"),
        "direction":     direction,
        "mileage_km":    acc_km,
        "report_time":   accident.get("report_time"),
        "arrived_time":  accident.get("arrived_time"),
        "completed_time": accident.get("completed_time"),
    }

    return _lib_heatmap(
        matrix_ns, relative_km, time_offsets, accident_meta,
        style=style,
        resolution_min=config.resolution_min,
        output_path=_resolve_path(output_path) if output_path else None,
    )


def result_metadata(result: dict) -> dict:
    accident = result["accident"]

    def _ts(value):
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()

    return {
        "accident": {
            "accident_id": accident.get("accident_id"),
            "freeway": accident["freeway"],
            "direction": accident["direction"],
            "mileage_km": float(accident["mileage_km"]),
            "report_time": _ts(accident["report_time"]),
            "arrived_time": _ts(accident.get("arrived_time")),
            "completed_time": _ts(accident.get("completed_time")),
        },
        "config": result["config"],
        "coverage": result.get("coverage") or {},
        "space_axis_km": [float(km) for km in result["space_axis"]],
        "time_offsets_min": [int(offset) for offset in result["time_offsets_min"]],
    }


def accident_from_row(row: pd.Series) -> dict:
    return {
        "accident_id": str(row["accident_id"]),
        "freeway": str(row["國道"]),
        "direction": str(row["direction_code"]),
        "mileage_km": float(row["mileage_km"]),
        "report_time": pd.Timestamp(row["發送"]),
        "arrived_time": pd.Timestamp(row["到達"]) if pd.notna(row["到達"]) else None,
        "completed_time": pd.Timestamp(row["處理完成"]) if pd.notna(row["處理完成"]) else None,
    }


def output_dirs(output_root: str | Path, resolution_min: int) -> tuple[Path, Path]:
    root = _resolve_path(output_root) / f"accident_heatmaps_{resolution_min}m"
    metadata_dir = root / "metadata"
    root.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return root, metadata_dir


def render_heatmaps(
    accidents: pd.DataFrame,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    config: HeatmapConfig = HeatmapConfig(),
    force: bool = False,
    font_path: str | Path | None = None,
    dpi: int = 150,
) -> pd.DataFrame:
    data_root = _resolve_path(data_root)
    image_dir, metadata_dir = output_dirs(output_root, config.resolution_min)
    bounds = load_vd_bounds(data_root)
    station_cache: dict[tuple[str, ...], tuple[pd.DataFrame, list[str]]] = {}
    records = []

    for _, row in accidents.iterrows():
        accident = accident_from_row(row)
        acc_id = accident["accident_id"]
        image_path = image_dir / f"{acc_id}.png"
        metadata_path = metadata_dir / f"{acc_id}.json"
        if not force and image_path.exists() and metadata_path.exists():
            print(f"Skipping {acc_id}: output exists")
            continue

        dates = required_dates(pd.Timestamp(accident["report_time"]), config)
        if dates not in station_cache:
            station_cache[dates] = load_station_for_dates(data_root, config.resolution_min, dates)
        station_df, missing_dates = station_cache[dates]

        route_bounds = bounds.get((accident["freeway"], accident["direction"]))
        if route_bounds is None and not station_df.empty:
            route_df = station_df[
                (station_df["freeway"] == accident["freeway"])
                & (station_df["direction"] == accident["direction"])
            ]
            if not route_df.empty:
                route_bounds = {
                    "min": float(route_df["mileage_km"].min()),
                    "max": float(route_df["mileage_km"].max()),
                }

        result = build_accident_heatmap_matrix(
            accident=accident,
            station_df=station_df,
            config=config,
            route_bounds=route_bounds,
            missing_dates=missing_dates,
        )
        fig = plot_accident_heatmap(result, output_path=image_path, font_path=font_path, dpi=dpi)
        plt.close(fig)
        metadata_path.write_text(json.dumps(result_metadata(result), ensure_ascii=False, indent=2), encoding="utf-8")

        coverage = result.get("coverage") or {}
        records.append({
            "accident_id": acc_id,
            "freeway": accident["freeway"],
            "direction": accident["direction"],
            "mileage_km": accident["mileage_km"],
            "report_time": str(accident["report_time"]),
            "image_path": str(image_path),
            "metadata_path": str(metadata_path),
            "vd_count": coverage.get("vd_count"),
            "filled_cells": coverage.get("filled_cells"),
            "total_cells": coverage.get("total_cells"),
            "filled_ratio": coverage.get("filled_ratio"),
            "space_window_start": coverage.get("space_window_start"),
            "space_window_end": coverage.get("space_window_end"),
            "route_vd_mileage_min": coverage.get("route_vd_mileage_min"),
            "route_vd_mileage_max": coverage.get("route_vd_mileage_max"),
            "missing_station_dates": ",".join(coverage.get("missing_station_dates") or []),
        })
        print(f"Saved heatmap: {image_path}")

    report = pd.DataFrame(records)
    if not report.empty:
        report_path = image_dir / "coverage_report.csv"
        report.to_csv(report_path, index=False, encoding="utf-8-sig")
        print(f"Coverage report: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render accident heatmaps from station Parquet data.")
    parser.add_argument("--accident-id", default=None, help="Render one accident id")
    parser.add_argument("--date", default=None, help="Render accidents on one date (YYYY-MM-DD)")
    parser.add_argument("--start-date", default=None, help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=None, help="Render the first N accidents after filters")
    parser.add_argument("--freeway", default=None, help="Filter by freeway, e.g. N1 or N3")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Data root containing station datasets")
    parser.add_argument("--accident-csv", default=str(DEFAULT_ACCIDENT_CSV), help="Accident CSV path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory")
    parser.add_argument("--resolution", type=int, default=5, choices=[1, 5], help="Station data resolution in minutes")
    parser.add_argument("--upstream-km", type=float, default=50.0)
    parser.add_argument("--downstream-km", type=float, default=10.0)
    parser.add_argument("--time-before-min", type=int, default=60)
    parser.add_argument("--time-after-min", type=int, default=240)
    parser.add_argument("--spatial-resolution-km", type=float, default=1.0)
    parser.add_argument("--speed-min", type=float, default=0.0)
    parser.add_argument("--speed-max", type=float, default=140.0)
    parser.add_argument("--font-path", default=None, help="Optional CJK font file path")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--research-window",
        action="store_true",
        help=(
            "Use the canonical research/model window: -10..+25 km and "
            "-30..+145 min (36 five-minute cells; 150-minute post horizon)."
        ),
    )
    args = parser.parse_args()
    if args.research_window:
        args.upstream_km = RESEARCH_UPSTREAM_KM
        args.downstream_km = RESEARCH_DOWNSTREAM_KM
        args.time_before_min = RESEARCH_TIME_BEFORE_MIN
        args.time_after_min = RESEARCH_TIME_AFTER_MIN
    return args


def main() -> None:
    args = parse_args()
    config = HeatmapConfig(
        upstream_km=args.upstream_km,
        downstream_km=args.downstream_km,
        time_before_min=args.time_before_min,
        time_after_min=args.time_after_min,
        spatial_resolution_km=args.spatial_resolution_km,
        resolution_min=args.resolution,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
    )

    accidents = load_accidents(args.accident_csv, freeway=args.freeway)
    accidents = filter_accidents(
        accidents,
        accident_id=args.accident_id,
        date=args.date,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
    )
    if accidents.empty:
        raise SystemExit("No accidents matched the requested filters.")

    print(f"Rendering {len(accidents)} accident heatmap(s)")
    print(f"Config: {config}")
    render_heatmaps(
        accidents=accidents,
        data_root=args.data_root,
        output_root=args.output_dir,
        config=config,
        force=args.force,
        font_path=args.font_path,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
