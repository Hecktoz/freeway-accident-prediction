# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb",
#     "pandas",
# ]
# ///
# ruff: noqa: E402
"""
Build per-VD historical speed profile from station 5-minute data.

DuckDB is used for the heavy aggregation: vectorized `quantile_cont` over many
groups is orders of magnitude faster than pandas groupby with percentile
lambdas (~1.5M groups for 1300+ VDs x day_class x peak_class x minute_bucket).
See `docs/COLAB_MIGRATION.md` -> Tooling Conventions and PITFALLS.md entry 11.

Processed one freeway at a time so the working set fits in Colab's 12 GB RAM.
The classify temp table for the full 5-year dataset is ~30 GB; per-freeway it
is ~3-4 GB. Each iteration drops its temp tables before the next CREATE so the
allocator has a clean slate.

For each (vdid, freeway, direction, mileage_m) and each
(day_class, peak_class, minute_bucket) combination, compute:
- mean_speed_x10, std_speed_x10, p10_speed_x10, p50_speed_x10, p90_speed_x10
- n_obs, coverage_ratio (= n_obs / n_days_in_class)

Inputs:
  data/vd_station_5m/source_year=YYYY/source_month=MM/vd_station_YYYYMMDD.parquet

Outputs:
  data/historical_speed_profile_5m/freeway=<F>/profile_0.parquet
  data/metadata/historical_profile_report.json

Usage:
  uv run scripts/data/build_historical_profile.py --start-date 2024-04-01 --end-date 2024-04-02 --force
  uv run scripts/data/build_historical_profile.py --start-date 2021-06-01 --end-date 2025-05-31 --force
  uv run scripts/data/build_historical_profile.py --start-date 2021-06-01 --end-date 2025-05-31 --freeway N1 --force
  uv run scripts/data/build_historical_profile.py --start-date 2021-06-29 --end-date 2024-03-31 --freeways N1 N3 --profile-dataset paper_trb_2026/historical_speed_profile_5m_pre20240401_n1n3 --report-path paper_trb_2026/historical_profile_report_pre20240401_n1n3.json --force
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.calendar import TAIWAN_MAKEUP_WORKDAYS, TAIWAN_PUBLIC_HOLIDAYS

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_STATION_DATASET = "vd_station_5m"
DEFAULT_PROFILE_DATASET = "historical_speed_profile_5m"
DEFAULT_RESOLUTION_MIN = 5
DEFAULT_DUCKDB_TEMP_DIR = Path(tempfile.gettempdir()) / "duckdb_tmp"
PEAK_WINDOW_AM = (7 * 60, 9 * 60)   # [start_min, end_min)
PEAK_WINDOW_PM = (17 * 60, 19 * 60)


def _to_date(d):
    return d if isinstance(d, date) else date.fromisoformat(str(d))


HOLIDAYS_DATE = frozenset(_to_date(d) for d in TAIWAN_PUBLIC_HOLIDAYS)
MAKEUP_DATE = frozenset(_to_date(d) for d in TAIWAN_MAKEUP_WORKDAYS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-VD historical speed profile from station 5-minute data using DuckDB.")
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--station-dataset", default=DEFAULT_STATION_DATASET)
    p.add_argument("--profile-dataset", default=DEFAULT_PROFILE_DATASET)
    p.add_argument("--resolution-min", type=int, default=DEFAULT_RESOLUTION_MIN, choices=[1, 5])
    p.add_argument("--metadata-dir", default=None)
    p.add_argument("--memory-limit", default="8GB", help="DuckDB memory limit (e.g. 4GB, 8GB, 16GB)")
    p.add_argument("--threads", type=int, default=None, help="DuckDB threads (default: auto)")
    p.add_argument("--temp-dir", default=str(DEFAULT_DUCKDB_TEMP_DIR), help="DuckDB spill-to-disk directory (must be local, not Drive)")
    p.add_argument("--max-temp-size", default="100GiB", help="DuckDB max_temp_directory_size (e.g. 50GiB, 100GiB)")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--freeway", default=None, help="Process only this freeway (e.g. 'N1'). If omitted, processes all freeways one at a time.")
    scope.add_argument(
        "--freeways",
        nargs="+",
        default=None,
        help="Build a clean dataset containing only these freeways (e.g. N1 N3). With --force, replaces the entire named profile dataset.",
    )
    p.add_argument(
        "--report-path",
        default=None,
        help="Build-report JSON path. Relative paths resolve under --data-root. Default: metadata/historical_profile_report.json.",
    )
    p.add_argument("--vdid-chunk-size", type=int, default=20, help="Number of vdids to aggregate per chunk (default 20). Reduce if still OOM.")
    p.add_argument("--force", action="store_true", help="Overwrite existing profile dataset (full or per-freeway)")
    return p.parse_args()


def classify_day(d: date) -> str:
    if d in MAKEUP_DATE:
        return "weekday"
    if d in HOLIDAYS_DATE:
        return "national_holiday"
    if d.isoweekday() >= 6:
        return "weekend"
    return "weekday"


def discover_available_dates(station_root: Path, start: date, end: date) -> set[date]:
    """Read available source dates straight from filenames — no parquet scan needed."""
    available: set[date] = set()
    for p in station_root.rglob("vd_station_*.parquet"):
        stem = p.stem  # vd_station_YYYYMMDD
        try:
            d = datetime.strptime(stem.split("_")[-1], "%Y%m%d").date()
        except ValueError:
            continue
        if start <= d <= end:
            available.add(d)
    return available


def discover_freeways(
    con: duckdb.DuckDBPyConnection,
    station_root: Path,
    available_dates: set[date],
    n_sample: int = 3,
) -> list[str]:
    """Read distinct freeways from a few sample files instead of the full glob.

    The freeway set is stable across days, so a few sample reads suffice.
    Full-glob scan over 1759 Drive parquet files takes ~30 min on Colab FUSE
    (footer fetch is ~1 s per file); sampling brings it down to seconds.
    """
    candidates: list[Path] = []
    for p in sorted(station_root.rglob("vd_station_*.parquet"), reverse=True):
        try:
            d = datetime.strptime(p.stem.split("_")[-1], "%Y%m%d").date()
        except ValueError:
            continue
        if d in available_dates:
            candidates.append(p)
        if len(candidates) >= n_sample:
            break
    if not candidates:
        return []
    file_list_sql = "[" + ", ".join(f"'{p.as_posix()}'" for p in candidates) + "]"
    rows = con.sql(
        f"SELECT DISTINCT freeway FROM read_parquet({file_list_sql}) ORDER BY freeway"
    ).fetchall()
    return [r[0] for r in rows if r[0] is not None]


def duckdb_mem_mb(con: duckdb.DuckDBPyConnection) -> float:
    try:
        row = con.sql("SELECT COALESCE(SUM(memory_usage_bytes), 0) FROM duckdb_memory()").fetchone()
        return (row[0] or 0) / (1024 * 1024)
    except Exception:
        return -1.0


def drop_iter_tables(con: duckdb.DuckDBPyConnection) -> None:
    for t in ("profile_tmp", "classified"):
        con.execute(f"DROP TABLE IF EXISTS {t}")


def get_vdids_for_freeway(
    con: duckdb.DuckDBPyConnection,
    station_glob: str,
    freeway: str,
    start: str,
    end: str,
    data_root: Path,
) -> list[str]:
    """Return sorted distinct vdids for a freeway.

    Reads vd_dim.parquet (fast, metadata-only) when available; falls back to
    sampling two 2-week windows from the parquet glob to catch sensors added
    or removed mid-period.
    """
    vd_dim = data_root / "metadata" / "vd_dim.parquet"
    if vd_dim.exists():
        try:
            rows = con.execute(
                f"SELECT DISTINCT vdid FROM read_parquet('{vd_dim.as_posix()}') WHERE freeway = ?",
                [freeway],
            ).fetchall()
            vdids = sorted(r[0] for r in rows if r[0])
            if vdids:
                return vdids
        except Exception:
            pass
    # Fallback: sample start and end of date range (two 2-week windows).
    start_d = _to_date(start)
    end_d = _to_date(end)
    sample_end = min(start_d + timedelta(days=13), end_d).isoformat()
    sample_start2 = max(end_d - timedelta(days=13), start_d).isoformat()
    rows = con.execute(
        f"""
        SELECT DISTINCT vdid
        FROM read_parquet('{station_glob}', hive_partitioning=true)
        WHERE freeway = ?
          AND (CAST(timestamp AS DATE) BETWEEN DATE '{start}' AND DATE '{sample_end}'
               OR  CAST(timestamp AS DATE) BETWEEN DATE '{sample_start2}' AND DATE '{end}')
        ORDER BY vdid
        """,
        [freeway],
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def build_one_freeway(
    con: duckdb.DuckDBPyConnection,
    freeway: str,
    station_glob: str,
    start: str,
    end: str,
    res: int,
    am_lo: int,
    am_hi: int,
    pm_lo: int,
    pm_hi: int,
    profile_base: Path,
    vdid_chunk_size: int,
    data_root: Path,
    chunk_dir: Path,
) -> dict:
    drop_iter_tables(con)

    # Discover vdids so we can chunk the aggregation.
    # approx_quantile uses ~10 KB per group (T-Digest); 1.5M groups for a large
    # freeway = ~15 GB regardless of memory_limit.  Chunks of 20 vdids keep the
    # group count to ~30k (≈300 MB) — well inside any reasonable memory limit.
    print(f"  discovering vdids for freeway={freeway}...", flush=True)
    vdids = get_vdids_for_freeway(con, station_glob, freeway, start, end, data_root)
    if not vdids:
        print(f"  no vdids found for freeway={freeway}, skipping")
        return {
            "freeway": freeway, "rows": 0, "size_mb": 0.0, "files": [],
            "aggregate_sec": 0.0, "write_sec": 0.0,
        }

    n_chunks = math.ceil(len(vdids) / vdid_chunk_size)
    print(f"  {len(vdids)} vdids → {n_chunks} chunk(s) of ≤{vdid_chunk_size}", flush=True)

    chunk_paths: list[Path] = []
    overall_t0 = time.time()

    for chunk_i in range(n_chunks):
        batch = vdids[chunk_i * vdid_chunk_size : (chunk_i + 1) * vdid_chunk_size]
        vdid_in = ", ".join(f"'{v}'" for v in batch)
        chunk_path = chunk_dir / f"_profile_{freeway}_c{chunk_i}.parquet"
        # A prior interrupted run may have left this local scratch file behind.
        # It is never a durable artifact, so replace it deterministically.
        chunk_path.unlink(missing_ok=True)

        print(f"  chunk {chunk_i + 1}/{n_chunks}: {len(batch)} vdids...", end=" ", flush=True)
        t_c = time.time()
        con.execute(
            f"""
            COPY (
                WITH src AS (
                    SELECT
                        vdid, freeway, direction, mileage_m,
                        CAST(timestamp AS DATE) AS observation_date,
                        timestamp,
                      CAST(ROUND(AVG(speed_x10)) AS INTEGER) AS speed_x10
                    FROM read_parquet('{station_glob}', hive_partitioning=true)
                    WHERE CAST(timestamp AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'
                      AND freeway = ?
                      AND vdid IN ({vdid_in})
                    GROUP BY vdid, freeway, direction, mileage_m, observation_date, timestamp
                ),
                classified AS (
                    SELECT
                        vdid, freeway, direction, mileage_m, speed_x10,
                        CASE
                            WHEN observation_date IN (SELECT d FROM makeup_workdays) THEN 'weekday'
                            WHEN observation_date IN (SELECT d FROM holidays) THEN 'national_holiday'
                            WHEN EXTRACT(isodow FROM observation_date) >= 6 THEN 'weekend'
                            ELSE 'weekday'
                        END AS day_class,
                        CASE
                            WHEN (EXTRACT(hour FROM timestamp) * 60 + EXTRACT(minute FROM timestamp)) >= {am_lo}
                             AND (EXTRACT(hour FROM timestamp) * 60 + EXTRACT(minute FROM timestamp)) <  {am_hi}
                              THEN 'peak'
                            WHEN (EXTRACT(hour FROM timestamp) * 60 + EXTRACT(minute FROM timestamp)) >= {pm_lo}
                             AND (EXTRACT(hour FROM timestamp) * 60 + EXTRACT(minute FROM timestamp)) <  {pm_hi}
                              THEN 'peak'
                            ELSE 'nonpeak'
                        END AS peak_class,
                        CAST(FLOOR((EXTRACT(hour FROM timestamp) * 60 + EXTRACT(minute FROM timestamp)) / {res}) AS INTEGER) * {res}
                        AS minute_bucket
                    FROM src
                )
                SELECT
                    c.vdid, c.freeway, c.direction, c.mileage_m,
                    CAST({res} AS SMALLINT) AS resolution_min,
                    c.day_class, c.peak_class,
                    CAST(c.minute_bucket AS SMALLINT) AS minute_bucket,
                    CAST(COUNT(*) AS INTEGER) AS n_obs,
                    CAST(LEAST(GREATEST(ROUND(AVG(c.speed_x10)), 0), 32767) AS SMALLINT) AS mean_speed_x10,
                    CAST(LEAST(GREATEST(ROUND(COALESCE(stddev_pop(c.speed_x10), 0)), 0), 32767) AS SMALLINT) AS std_speed_x10,
                    CAST(LEAST(GREATEST(ROUND(approx_quantile(c.speed_x10, 0.10)), 0), 32767) AS SMALLINT) AS p10_speed_x10,
                    CAST(LEAST(GREATEST(ROUND(approx_quantile(c.speed_x10, 0.50)), 0), 32767) AS SMALLINT) AS p50_speed_x10,
                    CAST(LEAST(GREATEST(ROUND(approx_quantile(c.speed_x10, 0.90)), 0), 32767) AS SMALLINT) AS p90_speed_x10,
                    CAST(
                        CAST(COUNT(*) AS DOUBLE) / NULLIF(MAX(d.n_days), 0)
                        AS REAL
                    ) AS coverage_ratio
                FROM classified c
                JOIN day_class_counts d USING (day_class)
                GROUP BY c.vdid, c.freeway, c.direction, c.mileage_m, c.day_class, c.peak_class, c.minute_bucket
            ) TO '{chunk_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [freeway],
        )
        print(f"{round(time.time() - t_c, 1)}s | duckdb mem={duckdb_mem_mb(con):.0f} MB", flush=True)
        chunk_paths.append(chunk_path)

    # Merge chunks into the final per-freeway parquet.
    fw_dir = profile_base / f"freeway={freeway}"
    fw_dir.mkdir(parents=True, exist_ok=True)
    out_path = fw_dir / "profile_0.parquet"

    t_w = time.time()
    if len(chunk_paths) == 1:
        chunk_paths[0].rename(out_path)
    else:
        print(f"  merging {len(chunk_paths)} chunks...", flush=True)
        chunk_list = "[" + ", ".join(f"'{p.as_posix()}'" for p in chunk_paths) + "]"
        con.execute(
            f"""
            COPY (
                SELECT vdid, direction, mileage_m, resolution_min,
                       day_class, peak_class, minute_bucket,
                       n_obs, mean_speed_x10, std_speed_x10,
                       p10_speed_x10, p50_speed_x10, p90_speed_x10, coverage_ratio
                FROM read_parquet({chunk_list})
                ORDER BY vdid, day_class, peak_class, minute_bucket
            ) TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        for p in chunk_paths:
            p.unlink(missing_ok=True)

    write_sec = round(time.time() - t_w, 1)
    aggregate_sec = round(time.time() - overall_t0 - write_sec, 1)
    size_mb = round(out_path.stat().st_size / (1024 * 1024), 3)
    n_profile = con.sql(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    print(f"  {n_profile:,} rows, {size_mb:.1f} MB | aggregate={aggregate_sec}s write={write_sec}s", flush=True)

    return {
        "freeway": freeway,
        "rows": int(n_profile),
        "size_mb": size_mb,
        "files": [str(out_path)],
        "aggregate_sec": aggregate_sec,
        "write_sec": write_sec,
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    metadata_dir = Path(args.metadata_dir).resolve() if args.metadata_dir else data_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    station_root = (data_root / args.station_dataset).resolve()
    station_glob = station_root.as_posix() + "/**/*.parquet"
    profile_base = (data_root / args.profile_dataset).resolve()

    requested_freeways = [str(fw) for fw in args.freeways] if args.freeways else None
    if requested_freeways and len(requested_freeways) != len(set(requested_freeways)):
        raise SystemExit(f"duplicate values in --freeways: {requested_freeways}")

    if requested_freeways:
        # A multi-freeway scope describes a complete named dataset, rather than
        # a resumable single partition. Replacing the full target prevents stale
        # partitions (for example N4/N6) from surviving a paper-scope rebuild.
        if profile_base.exists():
            if not args.force:
                raise SystemExit(f"profile dataset already exists at {profile_base} (use --force to rebuild)")
        shutil.rmtree(profile_base)
    elif args.freeway:
        fw_dir = profile_base / f"freeway={args.freeway}"
        if fw_dir.exists():
            if not args.force:
                raise SystemExit(f"freeway={args.freeway} already exists at {fw_dir} (use --force to rebuild)")
        shutil.rmtree(fw_dir)
    else:
        if profile_base.exists():
            if not args.force:
                raise SystemExit(f"profile dataset already exists at {profile_base} (use --force to rebuild)")
        shutil.rmtree(profile_base)
    profile_base.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA temp_directory='{temp_dir.as_posix()}'")
    con.execute(f"PRAGMA max_temp_directory_size='{args.max_temp_size}'")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute("PRAGMA preserve_insertion_order=false")
    if args.threads:
        con.execute(f"PRAGMA threads={int(args.threads)}")

    # Calendar tables — built once and reused across all freeway iterations.
    con.execute("CREATE TEMP TABLE holidays(d DATE)")
    con.executemany("INSERT INTO holidays VALUES (?)", [(d,) for d in sorted(TAIWAN_PUBLIC_HOLIDAYS)])
    con.execute("CREATE TEMP TABLE makeup_workdays(d DATE)")
    con.executemany("INSERT INTO makeup_workdays VALUES (?)", [(d,) for d in sorted(TAIWAN_MAKEUP_WORKDAYS)])

    print(f"Date range: {args.start_date} -> {args.end_date}")
    print(f"Station glob: {station_glob}")
    print(f"Profile dataset: {profile_base}")
    print(f"Resolution: {args.resolution_min} min | memory_limit={args.memory_limit}")

    res = int(args.resolution_min)
    am_lo, am_hi = PEAK_WINDOW_AM
    pm_lo, pm_hi = PEAK_WINDOW_PM

    # Global day_class counts derived from filename dates + calendar — no parquet scan needed.
    start_d = _to_date(args.start_date)
    end_d = _to_date(args.end_date)
    if start_d > end_d:
        raise SystemExit(f"start-date {start_d} is after end-date {end_d}")
    available_dates = discover_available_dates(station_root, start_d, end_d)
    if not available_dates:
        raise SystemExit(f"No vd_station_*.parquet files found under {station_root} in date range.")
    day_counts: dict[str, int] = {"weekday": 0, "weekend": 0, "national_holiday": 0}
    for d in available_dates:
        day_counts[classify_day(d)] += 1
    print(
        f"Available dates: {len(available_dates)} files | "
        + " | ".join(f"{k}: {day_counts.get(k, 0)}" for k in ("weekday", "weekend", "national_holiday"))
    )

    # Shared day_class_counts table (joined by every freeway aggregation).
    con.execute(
        """
        CREATE TEMP TABLE day_class_counts (day_class VARCHAR, n_days INTEGER)
        """
    )
    con.executemany(
        "INSERT INTO day_class_counts VALUES (?, ?)",
        [(k, int(v)) for k, v in day_counts.items()],
    )

    # Discover freeways (or use the explicit one).
    if requested_freeways:
        freeways = requested_freeways
    elif args.freeway:
        freeways = [args.freeway]
    else:
        print("Discovering freeways (sampling 3 files)...", flush=True)
        t_disc = time.time()
        freeways = discover_freeways(con, station_root, available_dates)
        print(f"  found {len(freeways)} freeway(s) in {round(time.time() - t_disc, 1)}s: {freeways}")
    if not freeways:
        raise SystemExit("No freeways found in the requested date range.")

    rows_per_freeway: list[dict] = []
    total_profile_rows = 0
    aggregate_sec_total = 0.0
    write_sec_total = 0.0

    overall_t0 = time.time()
    for i, fw in enumerate(freeways, 1):
        print(f"\n[{i}/{len(freeways)}] freeway={fw}  (elapsed so far: {round(time.time() - overall_t0, 1)}s)")
        stats = build_one_freeway(
            con, fw, station_glob, args.start_date, args.end_date,
            res, am_lo, am_hi, pm_lo, pm_hi, profile_base,
            args.vdid_chunk_size, data_root, temp_dir,
        )
        rows_per_freeway.append(stats)
        total_profile_rows += stats["rows"]
        aggregate_sec_total += stats["aggregate_sec"]
        write_sec_total += stats["write_sec"]
        print(f"  duckdb mem after drop: {duckdb_mem_mb(con):.0f} MB")

    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": "duckdb",
        "duckdb_version": duckdb.__version__,
        "range": {"start": args.start_date, "end": args.end_date},
        "resolution_min": res,
        "data_root": str(data_root),
        "station_dataset": args.station_dataset,
        "profile_dataset": args.profile_dataset,
        "rows_total": int(total_profile_rows),
        "day_class_days": day_counts,
        "n_weekday_days": int(day_counts.get("weekday", 0)),
        "n_weekend_days": int(day_counts.get("weekend", 0)),
        "n_national_holiday_days": int(day_counts.get("national_holiday", 0)),
        "aggregate_sec": round(aggregate_sec_total, 1),
        "write_sec": round(write_sec_total, 1),
        "rows_per_freeway": rows_per_freeway,
        "memory_limit": args.memory_limit,
        "threads": args.threads,
        "freeway_filter": args.freeway,
        "freeway_scope": freeways,
        "available_dates": {
            "min": min(available_dates).isoformat(),
            "max": max(available_dates).isoformat(),
            "count": len(available_dates),
        },
    }
    if args.report_path:
        report_path = Path(args.report_path)
        if not report_path.is_absolute():
            report_path = data_root / report_path
        report_path = report_path.resolve()
    else:
        report_path = metadata_dir / "historical_profile_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Per-freeway runs merge into the existing report instead of overwriting.
    if args.freeway and report_path.exists():
        try:
            prior = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            prior = None
        if isinstance(prior, dict) and isinstance(prior.get("rows_per_freeway"), list):
            kept = [r for r in prior["rows_per_freeway"] if r.get("freeway") != args.freeway]
            report["rows_per_freeway"] = kept + rows_per_freeway
            report["rows_total"] = sum(int(r.get("rows", 0)) for r in report["rows_per_freeway"])
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Profile rows: {total_profile_rows:,}")
    for s in rows_per_freeway:
        print(f"  freeway={s['freeway']}: {s['rows']:,} rows, {s['size_mb']} MB")
    print(f"Aggregate: {round(aggregate_sec_total, 1)}s | Write: {round(write_sec_total, 1)}s")
    print(f"Total elapsed: {round(time.time() - overall_t0, 1)}s")
    print(f"Report: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
