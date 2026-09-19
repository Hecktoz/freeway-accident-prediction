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
Build station-level VD Parquet datasets from Drive raw lane-level VD data.

This is the first downstream step after `download_vd_drive.py`.

Inputs:
  data/vd_raw_lane_1m_full/source_year=YYYY/source_month=MM/vd_YYYYMMDD.parquet

Outputs:
  data/vd_station_1m/source_year=YYYY/source_month=MM/vd_station_YYYYMMDD.parquet
  data/vd_station_5m/source_year=YYYY/source_month=MM/vd_station_YYYYMMDD.parquet
  data/metadata/vd_dim.parquet
  data/metadata/station_build_report.json

Example:
  uv run scripts/data/build_station_vd_drive.py --start-date 2024-04-01 --end-date 2024-04-01 --force
  uv run scripts/data/build_station_vd_drive.py --start-date 2024-04-01 --end-date 2025-05-31
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
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
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

DEFAULT_RAW_DATASET = "vd_raw_lane_1m_full"
DEFAULT_STATION_1M_DATASET = "vd_station_1m"
DEFAULT_STATION_5M_DATASET = "vd_station_5m"
DEFAULT_ROW_GROUP_SIZE = 131_072
DEFAULT_SPEED_MAX_KMH = 140.0

RAW_COLUMNS = [
    "source_date",
    "timestamp",
    "vdid",
    "freeway",
    "direction",
    "mileage_m",
    "speed",
    "occupancy",
    "vol_s",
    "vol_l",
    "vol_t",
]

STATION_COLUMNS = [
    "source_date",
    "timestamp",
    "vdid",
    "freeway",
    "direction",
    "mileage_m",
    "speed_x10",
    "occupancy_x10",
    "vol_s",
    "vol_l",
    "vol_t",
    "n_lanes_valid",
]

STATION_SCHEMA = pa.schema([
    ("source_date", pa.date32()),
    ("timestamp", pa.timestamp("ms")),
    ("vdid", pa.string()),
    ("freeway", pa.string()),
    ("direction", pa.string()),
    ("mileage_m", pa.int32()),
    ("speed_x10", pa.int16()),
    ("occupancy_x10", pa.int16()),
    ("vol_s", pa.int32()),
    ("vol_l", pa.int32()),
    ("vol_t", pa.int32()),
    ("n_lanes_valid", pa.int16()),
])

VD_DIM_SCHEMA = pa.schema([
    ("vdid", pa.string()),
    ("freeway", pa.string()),
    ("direction", pa.string()),
    ("mileage_m", pa.int32()),
    ("first_source_date", pa.date32()),
    ("last_source_date", pa.date32()),
    ("n_source_days", pa.int32()),
    ("n_rows_station_1m", pa.int64()),
])


@dataclass
class DayStats:
    date: str
    status: str
    raw_path: str | None = None
    station_1m_path: str | None = None
    station_5m_path: str | None = None
    raw_rows: int = 0
    valid_lane_rows: int = 0
    station_1m_rows: int = 0
    station_5m_rows: int = 0
    unique_vds: int = 0
    station_1m_size_mb: float = 0.0
    station_5m_size_mb: float = 0.0
    elapsed_sec: float = 0.0
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build station-level VD datasets from Drive raw lane Parquet.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help=f"Data root (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--raw-dataset", default=DEFAULT_RAW_DATASET, help=f"Raw dataset folder (default: {DEFAULT_RAW_DATASET})")
    parser.add_argument("--station-1m-dataset", default=DEFAULT_STATION_1M_DATASET, help=f"1m station output folder (default: {DEFAULT_STATION_1M_DATASET})")
    parser.add_argument("--station-5m-dataset", default=DEFAULT_STATION_5M_DATASET, help=f"5m station output folder (default: {DEFAULT_STATION_5M_DATASET})")
    parser.add_argument("--metadata-dir", default=None, help="Metadata output directory (default: DATA_ROOT/metadata)")
    parser.add_argument("--speed-max-kmh", type=float, default=DEFAULT_SPEED_MAX_KMH, help="Maximum valid raw speed (default: 140)")
    parser.add_argument("--row-group-size", type=int, default=DEFAULT_ROW_GROUP_SIZE, help=f"Parquet row group size (default: {DEFAULT_ROW_GROUP_SIZE})")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even when both station files already exist")
    parser.add_argument("--no-5m", action="store_true", help="Only build station 1m output")
    parser.add_argument("--dry-run", action="store_true", help="Print planned paths without reading/writing data")
    return parser.parse_args()


def date_range(start: str, end: str) -> list[str]:
    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    if dates.empty:
        raise ValueError("--end-date must be greater than or equal to --start-date")
    return [ts.strftime("%Y-%m-%d") for ts in dates]


def partition_dir(root: Path, dataset: str, date_str: str) -> Path:
    ts = pd.Timestamp(date_str)
    return root / dataset / f"source_year={ts:%Y}" / f"source_month={ts:%m}"


def raw_path(root: Path, dataset: str, date_str: str) -> Path:
    ts = pd.Timestamp(date_str)
    return partition_dir(root, dataset, date_str) / f"vd_{ts:%Y%m%d}.parquet"


def station_path(root: Path, dataset: str, date_str: str) -> Path:
    ts = pd.Timestamp(date_str)
    return partition_dir(root, dataset, date_str) / f"vd_station_{ts:%Y%m%d}.parquet"


def _round_to_nullable_int16(values: pd.Series) -> pd.Series:
    rounded = np.rint(values.astype("float64"))
    return pd.Series(pd.array(rounded, dtype="Int16"), index=values.index)


def _write_parquet(df: pd.DataFrame, path: Path, schema: pa.Schema, row_group_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(
        table,
        str(path),
        version="2.6",
        compression="zstd",
        row_group_size=row_group_size,
        use_dictionary=True,
        write_statistics=True,
    )


def read_raw_day(path: Path) -> pd.DataFrame:
    table = pq.read_table(str(path), columns=RAW_COLUMNS)
    return table.to_pandas()


def build_station_1m(raw_df: pd.DataFrame, speed_max_kmh: float) -> tuple[pd.DataFrame, int]:
    valid_mask = (raw_df["speed"] > 0) & (raw_df["speed"] <= speed_max_kmh)
    df = raw_df.loc[valid_mask].copy()
    valid_lane_rows = len(df)

    if df.empty:
        return pd.DataFrame(columns=STATION_COLUMNS), valid_lane_rows

    for col in ["vol_s", "vol_l", "vol_t"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).astype("int32")

    occupancy = pd.to_numeric(df["occupancy"], errors="coerce")
    df["occupancy_for_mean"] = occupancy.where((occupancy >= 0) & (occupancy <= 100))

    keys = ["source_date", "timestamp", "vdid", "freeway", "direction", "mileage_m"]
    station = (
        df.groupby(keys, observed=True)
        .agg(
            speed=("speed", "mean"),
            occupancy=("occupancy_for_mean", "mean"),
            vol_s=("vol_s", "sum"),
            vol_l=("vol_l", "sum"),
            vol_t=("vol_t", "sum"),
            n_lanes_valid=("speed", "size"),
        )
        .reset_index()
    )

    station["speed_x10"] = np.rint(station["speed"] * 10).clip(0, speed_max_kmh * 10).astype("int16")
    station["occupancy_x10"] = _round_to_nullable_int16(station["occupancy"] * 10)
    station["vol_s"] = station["vol_s"].astype("int32")
    station["vol_l"] = station["vol_l"].astype("int32")
    station["vol_t"] = station["vol_t"].astype("int32")
    station["n_lanes_valid"] = station["n_lanes_valid"].clip(0, np.iinfo(np.int16).max).astype("int16")

    station = (
        station[STATION_COLUMNS]
        .sort_values(["source_date", "freeway", "direction", "mileage_m", "vdid", "timestamp"])
        .reset_index(drop=True)
    )
    return station, valid_lane_rows


def build_station_5m(station_1m: pd.DataFrame, speed_max_kmh: float) -> pd.DataFrame:
    if station_1m.empty:
        return pd.DataFrame(columns=STATION_COLUMNS)

    work = station_1m.copy()
    work["timestamp_5m"] = pd.to_datetime(work["timestamp"]).dt.floor("5min")

    keys = ["source_date", "timestamp_5m", "vdid", "freeway", "direction", "mileage_m"]
    station_5m = (
        work.groupby(keys, observed=True)
        .agg(
            speed_x10=("speed_x10", "mean"),
            occupancy_x10=("occupancy_x10", "mean"),
            vol_s=("vol_s", "sum"),
            vol_l=("vol_l", "sum"),
            vol_t=("vol_t", "sum"),
            n_lanes_valid=("n_lanes_valid", "sum"),
        )
        .reset_index()
        .rename(columns={"timestamp_5m": "timestamp"})
    )

    station_5m["speed_x10"] = np.rint(station_5m["speed_x10"]).clip(0, speed_max_kmh * 10).astype("int16")
    station_5m["occupancy_x10"] = _round_to_nullable_int16(station_5m["occupancy_x10"])
    station_5m["vol_s"] = station_5m["vol_s"].astype("int32")
    station_5m["vol_l"] = station_5m["vol_l"].astype("int32")
    station_5m["vol_t"] = station_5m["vol_t"].astype("int32")
    station_5m["n_lanes_valid"] = station_5m["n_lanes_valid"].clip(0, np.iinfo(np.int16).max).astype("int16")

    return (
        station_5m[STATION_COLUMNS]
        .sort_values(["source_date", "freeway", "direction", "mileage_m", "vdid", "timestamp"])
        .reset_index(drop=True)
    )


def vd_dim_from_station(station_1m: pd.DataFrame) -> pd.DataFrame:
    if station_1m.empty:
        return pd.DataFrame(columns=[field.name for field in VD_DIM_SCHEMA])

    return (
        station_1m.groupby(["vdid", "freeway", "direction", "mileage_m"], observed=True)
        .agg(
            first_source_date=("source_date", "min"),
            last_source_date=("source_date", "max"),
            n_source_days=("source_date", "nunique"),
            n_rows_station_1m=("timestamp", "size"),
        )
        .reset_index()
    )


def merge_vd_dims(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=[field.name for field in VD_DIM_SCHEMA])
    combined = pd.concat(parts, ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=[field.name for field in VD_DIM_SCHEMA])
    out = (
        combined.groupby(["vdid", "freeway", "direction", "mileage_m"], observed=True)
        .agg(
            first_source_date=("first_source_date", "min"),
            last_source_date=("last_source_date", "max"),
            n_source_days=("n_source_days", "sum"),
            n_rows_station_1m=("n_rows_station_1m", "sum"),
        )
        .reset_index()
        .sort_values(["freeway", "direction", "mileage_m", "vdid"])
        .reset_index(drop=True)
    )
    out["n_source_days"] = out["n_source_days"].astype("int32")
    out["n_rows_station_1m"] = out["n_rows_station_1m"].astype("int64")
    return out


def process_day(
    date_str: str,
    data_root: Path,
    raw_dataset: str,
    station_1m_dataset: str,
    station_5m_dataset: str,
    speed_max_kmh: float,
    row_group_size: int,
    force: bool,
    build_5m: bool,
    dry_run: bool,
) -> tuple[DayStats, pd.DataFrame | None]:
    t0 = time.time()
    in_path = raw_path(data_root, raw_dataset, date_str)
    out_1m = station_path(data_root, station_1m_dataset, date_str)
    out_5m = station_path(data_root, station_5m_dataset, date_str)

    stats = DayStats(
        date=date_str,
        status="pending",
        raw_path=str(in_path),
        station_1m_path=str(out_1m),
        station_5m_path=str(out_5m) if build_5m else None,
    )

    if dry_run:
        stats.status = "planned"
        return stats, None

    if not in_path.exists():
        stats.status = "missing_raw"
        stats.error = f"raw parquet not found: {in_path}"
        return stats, None

    if not force and out_1m.exists() and (out_5m.exists() or not build_5m):
        stats.status = "skipped_existing"
        stats.station_1m_rows = pq.ParquetFile(out_1m).metadata.num_rows
        stats.station_1m_size_mb = out_1m.stat().st_size / (1024 * 1024)
        if build_5m and out_5m.exists():
            stats.station_5m_rows = pq.ParquetFile(out_5m).metadata.num_rows
            stats.station_5m_size_mb = out_5m.stat().st_size / (1024 * 1024)
        dim_table = pq.read_table(
            str(out_1m),
            columns=["source_date", "timestamp", "vdid", "freeway", "direction", "mileage_m"],
        )
        dim_part = vd_dim_from_station(dim_table.to_pandas())
        stats.unique_vds = int(dim_part["vdid"].nunique()) if not dim_part.empty else 0
        stats.elapsed_sec = round(time.time() - t0, 3)
        return stats, dim_part

    try:
        raw_df = read_raw_day(in_path)
        stats.raw_rows = len(raw_df)

        station_1m, valid_lane_rows = build_station_1m(raw_df, speed_max_kmh)
        del raw_df
        stats.valid_lane_rows = valid_lane_rows
        stats.station_1m_rows = len(station_1m)
        stats.unique_vds = int(station_1m["vdid"].nunique()) if not station_1m.empty else 0

        _write_parquet(station_1m, out_1m, STATION_SCHEMA, row_group_size)
        stats.station_1m_size_mb = out_1m.stat().st_size / (1024 * 1024)

        if build_5m:
            station_5m = build_station_5m(station_1m, speed_max_kmh)
            stats.station_5m_rows = len(station_5m)
            _write_parquet(station_5m, out_5m, STATION_SCHEMA, row_group_size)
            stats.station_5m_size_mb = out_5m.stat().st_size / (1024 * 1024)
            del station_5m

        dim_part = vd_dim_from_station(station_1m)
        del station_1m

        stats.status = "built"
        stats.elapsed_sec = round(time.time() - t0, 3)
        return stats, dim_part
    except Exception as exc:
        stats.status = "failed"
        stats.error = f"{type(exc).__name__}: {exc}"
        stats.elapsed_sec = round(time.time() - t0, 3)
        return stats, None


def write_report(metadata_dir: Path, args: argparse.Namespace, days: list[DayStats], elapsed_sec: float) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, int] = {}
    for day in days:
        statuses[day.status] = statuses.get(day.status, 0) + 1

    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "range": {"start": args.start_date, "end": args.end_date, "days": len(days)},
        "data_root": str(Path(args.data_root)),
        "raw_dataset": args.raw_dataset,
        "station_1m_dataset": args.station_1m_dataset,
        "station_5m_dataset": None if args.no_5m else args.station_5m_dataset,
        "speed_max_kmh": args.speed_max_kmh,
        "row_group_size": args.row_group_size,
        "force": args.force,
        "elapsed_sec": round(elapsed_sec, 1),
        "statuses": statuses,
        "totals": {
            "raw_rows": sum(day.raw_rows for day in days),
            "valid_lane_rows": sum(day.valid_lane_rows for day in days),
            "station_1m_rows": sum(day.station_1m_rows for day in days),
            "station_5m_rows": sum(day.station_5m_rows for day in days),
            "station_1m_size_mb": round(sum(day.station_1m_size_mb for day in days), 3),
            "station_5m_size_mb": round(sum(day.station_5m_size_mb for day in days), 3),
        },
        "days": [asdict(day) for day in days],
    }

    report_name = "station_build_report_dry_run.json" if args.dry_run else "station_build_report.json"
    path = metadata_dir / report_name
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else data_root / "metadata"
    dates = date_range(args.start_date, args.end_date)
    build_5m = not args.no_5m

    print(f"Date range: {args.start_date} -> {args.end_date} ({len(dates)} day(s))")
    print(f"Data root: {data_root}")
    print(f"Raw dataset: {args.raw_dataset}")
    print(f"Station 1m dataset: {args.station_1m_dataset}")
    print(f"Station 5m dataset: {args.station_5m_dataset if build_5m else '(disabled)'}")
    print(f"Metadata dir: {metadata_dir}")
    print(f"Speed max: {args.speed_max_kmh:g} km/h")
    print(f"Force rebuild: {args.force}")

    t0 = time.time()
    day_stats: list[DayStats] = []
    dim_parts: list[pd.DataFrame] = []

    for date_str in tqdm(dates, desc="Building station data", unit="day"):
        stats, dim_part = process_day(
            date_str=date_str,
            data_root=data_root,
            raw_dataset=args.raw_dataset,
            station_1m_dataset=args.station_1m_dataset,
            station_5m_dataset=args.station_5m_dataset,
            speed_max_kmh=args.speed_max_kmh,
            row_group_size=args.row_group_size,
            force=args.force,
            build_5m=build_5m,
            dry_run=args.dry_run,
        )
        day_stats.append(stats)
        if dim_part is not None and not dim_part.empty:
            dim_parts.append(dim_part)

        if stats.status == "failed":
            print(f"  [{date_str}] FAILED: {stats.error}", flush=True)
        elif stats.status == "missing_raw":
            print(f"  [{date_str}] missing raw parquet", flush=True)

    if not args.dry_run and dim_parts:
        vd_dim = merge_vd_dims(dim_parts)
        vd_dim_path = metadata_dir / "vd_dim.parquet"
        _write_parquet(vd_dim, vd_dim_path, VD_DIM_SCHEMA, args.row_group_size)
        print(f"VD dimension: {vd_dim_path} ({len(vd_dim):,} rows)")

    report_path = write_report(metadata_dir, args, day_stats, time.time() - t0)

    statuses: dict[str, int] = {}
    for day in day_stats:
        statuses[day.status] = statuses.get(day.status, 0) + 1

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Statuses: {statuses}")
    print(f"Station 1m rows reported: {sum(day.station_1m_rows for day in day_stats):,}")
    print(f"Station 5m rows reported: {sum(day.station_5m_rows for day in day_stats):,}")
    print(f"Station 1m size reported: {sum(day.station_1m_size_mb for day in day_stats):.1f} MB")
    print(f"Station 5m size reported: {sum(day.station_5m_size_mb for day in day_stats):.1f} MB")
    print(f"Report: {report_path}")
    print("=" * 72)

    if any(day.status == "failed" for day in day_stats):
        sys.exit(1)


if __name__ == "__main__":
    main()
