# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "tqdm",
# ]
# ///
# ruff: noqa: E402
"""
Build per-accident spacetime sample matrices from station 5-minute data.

For each accident in `data/accident_data.csv`, slice the canonical
(50 km upstream + 10 km downstream, -60 to +240 min) window from
`vd_station_5m` and produce:

- one wide CSV per accident under
  `data/accident_samples_csv_5m/<accident_id>.csv`
  with rows = absolute mileage_km, columns = time_offset_min
- one combined NPZ at `data/accident_samples_5m.npz`
  with stacked (n_accidents, n_space, n_time) speed matrices on a
  direction-normalized upstream-distance axis
- a metadata parquet `data/accident_samples_5m_metadata.parquet`
  with one row per accident
- a build report `data/metadata/accident_samples_report.json`

Usage:
  uv run scripts/data/build_accident_samples.py --limit 5
  uv run scripts/data/build_accident_samples.py
  uv run scripts/data/build_accident_samples.py --start-date 2024-04-01 --end-date 2024-04-30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.data_loaders import read_accidents, read_station_dates
from scripts.lib.windows import (
    SampleWindow,
    coverage,
    matrix_to_csv_dataframe,
    rasterize_window,
    slice_station_df,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


METADATA_SCHEMA = pa.schema([
    ("accident_id", pa.string()),
    ("freeway", pa.string()),
    ("direction", pa.string()),
    ("mileage_km", pa.float32()),
    ("mileage_m", pa.int32()),
    ("report_time", pa.timestamp("ms")),
    ("arrived_time", pa.timestamp("ms")),
    ("completed_time", pa.timestamp("ms")),
    ("cleared_time", pa.timestamp("ms")),
    ("filled_cells", pa.int32()),
    ("total_cells", pa.int32()),
    ("filled_ratio", pa.float32()),
    ("status", pa.string()),
    ("note", pa.string()),
])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-accident sample matrices from station 5-minute data.")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--accident-csv", default=None, help="Default: <data-root>/accident_data.csv")
    p.add_argument("--resolution-min", type=int, default=5, choices=[1, 5])
    p.add_argument("--upstream-km", type=int, default=50)
    p.add_argument("--downstream-km", type=int, default=10)
    p.add_argument("--time-before-min", type=int, default=60)
    p.add_argument("--time-after-min", type=int, default=240)
    p.add_argument("--spatial-resolution-km", type=int, default=1)
    p.add_argument("--start-date", default=None, help="Optional inclusive accident-date filter")
    p.add_argument("--end-date", default=None, help="Optional inclusive accident-date filter")
    p.add_argument("--accident-id", default=None, help="Process a single accident id (overrides date filters)")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N matching accidents")
    p.add_argument("--freeway", default=None, help="Optional freeway filter, e.g. N1")
    p.add_argument("--no-csv", action="store_true", help="Skip per-accident CSV output")
    p.add_argument("--no-npz", action="store_true", help="Skip stacked NPZ output")
    p.add_argument("--force", action="store_true", help="Overwrite existing per-accident CSVs")
    p.add_argument("--metadata-dir", default=None)
    p.add_argument("--csv-dir-name", default=None)
    p.add_argument("--npz-name", default=None)
    p.add_argument("--tag", default="", help="Optional suffix (e.g. 'sms') applied to all output names so a second source does not overwrite the dispatch artifacts.")
    return p.parse_args()


def filter_accidents(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    accident_id: str | None,
    limit: int | None,
) -> pd.DataFrame:
    if accident_id:
        df = df[df["accident_id"] == accident_id].copy()
    else:
        report_dates = df["report_time"].dt.normalize()
        if start_date:
            df = df[report_dates >= pd.Timestamp(start_date)]
        if end_date:
            df = df[report_dates <= pd.Timestamp(end_date)]
        df = df.sort_values("report_time").reset_index(drop=True)
    if limit is not None:
        df = df.head(limit).copy()
    return df.reset_index(drop=True)


def required_dates(report_time: pd.Timestamp, window: SampleWindow) -> list[str]:
    t_start = report_time - pd.Timedelta(minutes=window.time_before_min)
    t_end = report_time + pd.Timedelta(minutes=window.time_after_min)
    return [
        ts.strftime("%Y-%m-%d")
        for ts in pd.date_range(t_start.normalize(), t_end.normalize(), freq="D")
    ]


class StationCache:
    """Single-day LRU-ish cache so adjacent accidents share loaded parquet days."""

    def __init__(self, data_root: Path, resolution_min: int, max_days: int = 4):
        self.data_root = data_root
        self.resolution_min = resolution_min
        self.max_days = max_days
        self._days: dict[str, pd.DataFrame] = {}
        self._order: list[str] = []
        self.columns = [
            "timestamp", "vdid", "freeway", "direction", "mileage_m", "speed_x10",
        ]

    def get(self, dates: list[str]) -> pd.DataFrame:
        for d in dates:
            if d not in self._days:
                try:
                    self._days[d] = read_station_dates(
                        [d],
                        resolution_min=self.resolution_min,
                        data_root=self.data_root,
                        columns=self.columns,
                    )
                except FileNotFoundError:
                    self._days[d] = pd.DataFrame(columns=self.columns)
                self._order.append(d)
                while len(self._order) > self.max_days:
                    drop = self._order.pop(0)
                    self._days.pop(drop, None)
        parts = [self._days[d] for d in dates if d in self._days and not self._days[d].empty]
        if not parts:
            return pd.DataFrame(columns=self.columns)
        return pd.concat(parts, ignore_index=True)


def _safe_ts(value) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).to_pydatetime()


def write_csv_bundle(
    csv_path: Path,
    matrix: np.ndarray,
    accident_row: pd.Series,
    window: SampleWindow,
) -> None:
    df = matrix_to_csv_dataframe(
        matrix,
        accident_mileage_m=int(accident_row["mileage_m"]),
        direction=accident_row["direction"],
        window=window,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, encoding="utf-8-sig", float_format="%.1f", na_rep="")
    meta = {
        "accident_id": accident_row["accident_id"],
        "freeway": accident_row["freeway"],
        "direction": accident_row["direction"],
        "mileage_km": float(accident_row["mileage_km"]),
        "mileage_m": int(accident_row["mileage_m"]),
        "report_time": pd.Timestamp(accident_row["report_time"]).isoformat() if pd.notna(accident_row["report_time"]) else None,
        "arrived_time": pd.Timestamp(accident_row["arrived_time"]).isoformat() if pd.notna(accident_row.get("arrived_time")) else None,
        "completed_time": pd.Timestamp(accident_row["completed_time"]).isoformat() if pd.notna(accident_row.get("completed_time")) else None,
        "window": window.to_dict(),
        "space_offsets_m": window.space_offsets_m.tolist(),
        "time_offsets_min": window.time_offsets_min.tolist(),
        "coverage": coverage(matrix),
    }
    csv_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else data_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.tag}" if args.tag else ""
    csv_dir_name = args.csv_dir_name or f"accident_samples_csv_{args.resolution_min}m{suffix}"
    npz_name = args.npz_name or f"accident_samples_{args.resolution_min}m{suffix}.npz"
    csv_dir = data_root / csv_dir_name
    npz_path = data_root / npz_name
    metadata_parquet = data_root / f"accident_samples_{args.resolution_min}m{suffix}_metadata.parquet"
    report_path = metadata_dir / f"accident_samples_report{suffix}.json"

    window = SampleWindow(
        upstream_km=args.upstream_km,
        downstream_km=args.downstream_km,
        time_before_min=args.time_before_min,
        time_after_min=args.time_after_min,
        spatial_resolution_km=args.spatial_resolution_km,
        resolution_min=args.resolution_min,
    )

    accidents = read_accidents(
        data_root=data_root,
        accident_csv=args.accident_csv,
        freeway=args.freeway,
    )
    valid = accidents.dropna(subset=["mileage_m", "report_time", "direction", "freeway"]).copy()
    valid = valid[valid["direction"].isin({"N", "S", "E", "W"})]
    valid["mileage_m"] = valid["mileage_m"].astype(int)
    accidents = filter_accidents(valid, args.start_date, args.end_date, args.accident_id, args.limit)

    if accidents.empty:
        raise SystemExit("No accidents matched the filter criteria.")

    print(f"Window: up={window.upstream_km}km, down={window.downstream_km}km, time={-window.time_before_min}~+{window.time_after_min}min, dx={window.spatial_resolution_km}km, dt={window.resolution_min}min")
    print(f"Matrix shape per accident: ({window.n_space}, {window.n_time})")
    print(f"Accidents to process: {len(accidents)}")

    cache = StationCache(data_root=data_root, resolution_min=args.resolution_min, max_days=4)

    matrices: list[np.ndarray] = []
    metadata_rows: list[dict] = []
    success = 0
    skipped_csv = 0
    failures = 0
    t0 = time.time()

    for row in tqdm(accidents.itertuples(index=False), total=len(accidents), unit="acc"):
        record = row._asdict()
        accident_id = str(record["accident_id"])
        report_time = pd.Timestamp(record["report_time"])
        try:
            dates = required_dates(report_time, window)
            station_df = cache.get(dates)
            sliced = slice_station_df(
                station_df,
                accident_mileage_m=int(record["mileage_m"]),
                direction=record["direction"],
                freeway=record["freeway"],
                report_time=report_time,
                window=window,
            )
            matrix = rasterize_window(sliced, window, value_col="speed_x10")
            cov = coverage(matrix)
            row_series = pd.Series(record)

            if not args.no_csv:
                csv_path = csv_dir / f"{accident_id}.csv"
                if args.force or not csv_path.exists():
                    write_csv_bundle(csv_path, matrix, row_series, window)
                else:
                    skipped_csv += 1

            matrices.append(matrix)
            metadata_rows.append({
                "accident_id": accident_id,
                "freeway": record["freeway"],
                "direction": record["direction"],
                "mileage_km": float(record["mileage_km"]),
                "mileage_m": int(record["mileage_m"]),
                "report_time": _safe_ts(record["report_time"]),
                "arrived_time": _safe_ts(record.get("arrived_time")),
                "completed_time": _safe_ts(record.get("completed_time")),
                "cleared_time": _safe_ts(record.get("cleared_time")),
                "filled_cells": int(cov["filled_cells"]),
                "total_cells": int(cov["total_cells"]),
                "filled_ratio": float(cov["filled_ratio"] or 0.0),
                "status": "ok" if cov["filled_cells"] > 0 else "empty",
                "note": "",
            })
            success += 1
        except Exception as exc:
            failures += 1
            empty = np.full((window.n_space, window.n_time), np.nan, dtype=np.float32)
            matrices.append(empty)
            metadata_rows.append({
                "accident_id": accident_id,
                "freeway": record.get("freeway"),
                "direction": record.get("direction"),
                "mileage_km": float(record.get("mileage_km") or 0.0),
                "mileage_m": int(record.get("mileage_m") or 0),
                "report_time": _safe_ts(record.get("report_time")),
                "arrived_time": _safe_ts(record.get("arrived_time")),
                "completed_time": _safe_ts(record.get("completed_time")),
                "cleared_time": _safe_ts(record.get("cleared_time")),
                "filled_cells": 0,
                "total_cells": int(window.n_space * window.n_time),
                "filled_ratio": 0.0,
                "status": "failed",
                "note": f"{type(exc).__name__}: {exc}",
            })

    metadata_df = pd.DataFrame(metadata_rows)

    if not args.no_npz:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        stacked = np.stack(matrices, axis=0).astype(np.float32)
        np.savez_compressed(
            str(npz_path),
            matrix=stacked,
            accident_ids=metadata_df["accident_id"].to_numpy().astype(str),
            space_offsets_m=window.space_offsets_m.astype(np.int32),
            time_offsets_min=window.time_offsets_min.astype(np.int32),
            window_upstream_km=np.int32(window.upstream_km),
            window_downstream_km=np.int32(window.downstream_km),
            window_time_before_min=np.int32(window.time_before_min),
            window_time_after_min=np.int32(window.time_after_min),
            window_spatial_resolution_km=np.int32(window.spatial_resolution_km),
            window_resolution_min=np.int32(window.resolution_min),
        )
    else:
        npz_path = None

    metadata_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(metadata_df, schema=METADATA_SCHEMA, preserve_index=False),
        str(metadata_parquet),
        compression="zstd",
        version="2.6",
    )

    elapsed = round(time.time() - t0, 1)
    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "window": window.to_dict(),
        "n_accidents_total": int(len(accidents)),
        "n_success": int(success),
        "n_failed": int(failures),
        "n_csv_skipped_existing": int(skipped_csv),
        "csv_dir": str(csv_dir) if not args.no_csv else None,
        "npz_path": str(npz_path) if npz_path else None,
        "metadata_parquet": str(metadata_parquet),
        "elapsed_sec": elapsed,
        "matrix_shape": [int(window.n_space), int(window.n_time)],
        "filter": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "accident_id": args.accident_id,
            "limit": args.limit,
            "freeway": args.freeway,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Accidents processed: {success + failures}/{len(accidents)} (failed={failures})")
    if not args.no_csv:
        print(f"CSV dir: {csv_dir} (skipped existing: {skipped_csv})")
    if npz_path:
        print(f"NPZ: {npz_path}")
    print(f"Metadata parquet: {metadata_parquet}")
    print(f"Report: {report_path}")
    print("=" * 72)

    if failures > 0 and success == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
