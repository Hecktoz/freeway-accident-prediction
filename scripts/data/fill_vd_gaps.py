# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx[http2]",
#     "lxml",
#     "pyarrow",
#     "tqdm",
# ]
# ///
"""
Audit VD download manifests for gaps and fill missing/errored minutes.

Audit mode (default):
  Scans the date range, classifies every day, and prints a gap summary.

Gap-fill mode (--fill):
  For each day that has an existing parquet but still has http_error,
  parse_error, or unrecorded minutes, retries those minutes, merges the
  new rows into the existing daily parquet, and updates the manifest.

VD-expand mode (--fill --expand-vd):
  For each day with an existing parquet, re-downloads ALL non-server_404
  minutes using --filter-mode all (or a specified wider filter), then
  merges ONLY the genuinely new data:
    - rows for VD IDs not already in the parquet (previously filtered out)
    - rows for minutes that were previously http_error/parse_error/unrecorded
      (for any VD, since those minutes have no data in the parquet at all)
  This avoids duplicating rows that are already correct in the parquet.

Combine both:
  --fill --expand-vd handles gap-fill and VD-expansion in a single pass.

Example:
  uv run scripts/data/fill_vd_gaps.py --start-date 2024-04-01 --end-date 2025-05-31
  uv run scripts/data/fill_vd_gaps.py --start-date 2024-04-01 --end-date 2025-05-31 --fill
  uv run scripts/data/fill_vd_gaps.py --start-date 2024-04-01 --end-date 2025-05-31 --fill --expand-vd
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import importlib.util
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Load shared utilities from download_vd_drive without duplicating code
# ---------------------------------------------------------------------------

def _load_dvd():
    spec = importlib.util.spec_from_file_location(
        "download_vd_drive",
        Path(__file__).resolve().parent / "download_vd_drive.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["download_vd_drive"] = mod  # must register before exec so @dataclass __module__ resolves
    spec.loader.exec_module(mod)
    return mod


dvd = _load_dvd()

DEFAULT_DATA_ROOT = dvd.DEFAULT_DATA_ROOT
DEFAULT_RAW_DATASET = dvd.DEFAULT_RAW_DATASET
DEFAULT_MANIFEST_DATASET = dvd.DEFAULT_MANIFEST_DATASET
DEFAULT_TMP_DIR = dvd.DEFAULT_TMP_DIR
DEFAULT_VD_LIST = dvd.DEFAULT_VD_LIST
DEFAULT_ROW_GROUP_SIZE = dvd.DEFAULT_ROW_GROUP_SIZE
TOTAL_MINUTES = dvd.TOTAL_MINUTES

STATUS_DOWNLOADED = dvd.STATUS_DOWNLOADED
STATUS_404 = dvd.STATUS_404
STATUS_HTTP_ERROR = dvd.STATUS_HTTP_ERROR
STATUS_PARSE_ERROR = dvd.STATUS_PARSE_ERROR

SORT_KEYS = [
    ("freeway", "ascending"),
    ("direction", "ascending"),
    ("mileage_m", "ascending"),
    ("vdid", "ascending"),
    ("timestamp", "ascending"),
    ("lane_id", "ascending"),
]


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

@dataclass
class DayStatus:
    date: str
    has_manifest: bool
    has_parquet: bool          # parquet exists on Drive
    has_local: bool            # parquet exists in local tmp
    complete: bool             # manifest marks all 1440 minutes done
    n_downloaded: int = 0
    n_404: int = 0
    n_http_error: int = 0
    n_parse_error: int = 0
    n_unrecorded: int = 0      # minutes absent from manifest["minutes"]

    @property
    def n_gaps(self) -> int:
        return self.n_http_error + self.n_parse_error + self.n_unrecorded

    @property
    def has_any_parquet(self) -> bool:
        return self.has_parquet or self.has_local

    @property
    def is_patchable(self) -> bool:
        """Has existing data AND has gap minutes to fill."""
        return self.has_any_parquet and self.n_gaps > 0 and not self.complete

    @property
    def is_expandable(self) -> bool:
        """Has existing parquet (candidate for VD expansion, regardless of completeness)."""
        return self.has_any_parquet


def audit_day(
    date_str: str,
    data_root: Path,
    tmp_dir: Path,
    raw_dataset: str,
    manifest_dataset: str,
) -> DayStatus:
    final_path = dvd.final_parquet_path(data_root, raw_dataset, date_str)
    local_path = dvd.tmp_parquet_path(tmp_dir, date_str)
    mf_path = dvd.manifest_path(data_root, manifest_dataset, date_str)

    has_manifest = mf_path.exists()
    has_parquet = final_path.exists()
    has_local = local_path.exists()

    if not has_manifest:
        return DayStatus(
            date=date_str,
            has_manifest=False,
            has_parquet=has_parquet,
            has_local=has_local,
            complete=False,
        )

    manifest = dvd.load_manifest(mf_path)
    counts = dvd.manifest_counts(manifest)
    complete = dvd.manifest_is_complete(manifest)

    all_minutes_set = set(dvd._all_minutes())
    recorded = set(manifest.get("minutes", {}).keys())
    n_unrecorded = len(all_minutes_set - recorded)

    n_http_error = sum(
        1 for v in manifest.get("minutes", {}).values() if v == STATUS_HTTP_ERROR
    )
    n_parse_error = sum(
        1 for v in manifest.get("minutes", {}).values() if v == STATUS_PARSE_ERROR
    )

    return DayStatus(
        date=date_str,
        has_manifest=True,
        has_parquet=has_parquet,
        has_local=has_local,
        complete=complete,
        n_downloaded=counts.get(STATUS_DOWNLOADED, 0),
        n_404=counts.get(STATUS_404, 0),
        n_http_error=n_http_error,
        n_parse_error=n_parse_error,
        n_unrecorded=n_unrecorded,
    )


# ---------------------------------------------------------------------------
# Shared fetch + parse pipeline (used by both fill and expand modes)
# ---------------------------------------------------------------------------

async def _fetch_and_parse(
    date_str: str,
    minutes_to_fetch: list[str],
    vd_filter,
    concurrency: int,
    retries: int,
    executor: ProcessPoolExecutor,
) -> tuple[list, dict[str, str], dict[str, str], dict[str, str]]:
    """
    Fetch and parse a list of minute files.

    Returns:
        (new_rows, fetch_status, fetch_errors, parse_errors)
        fetch_status: hhmm → "ok" | "404" | "error"
    """
    import httpx

    yyyymmdd = date_str.replace("-", "")
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    loop = asyncio.get_event_loop()

    parse_futures: dict[str, asyncio.Future] = {}
    fetch_status: dict[str, str] = {}
    fetch_errors: dict[str, str] = {}
    parse_errors: dict[str, str] = {}

    pbar = tqdm(total=len(minutes_to_fetch), desc=f"  {date_str}", unit="file", leave=False)
    ok_count = missing_count = error_count = 0

    async with httpx.AsyncClient(verify=False, http2=True, limits=limits) as client:
        async def fetch_one(hhmm: str) -> None:
            nonlocal ok_count, missing_count, error_count
            url = dvd.BASE_URL.format(date=yyyymmdd, hhmm=hhmm)
            tag, payload = await dvd._get(client, url, sem, retries)
            fetch_status[hhmm] = tag
            pbar.update(1)
            if tag == "ok":
                ok_count += 1
                parse_futures[hhmm] = loop.run_in_executor(
                    executor,
                    dvd._decompress_and_parse,
                    payload,
                    vd_filter,
                    date_str,
                    hhmm,
                )
            elif tag == "404":
                missing_count += 1
            else:
                error_count += 1
                fetch_errors[hhmm] = str(payload)[:300]
            pbar.set_postfix(ok=ok_count, miss=missing_count, err=error_count)

        await asyncio.gather(*[fetch_one(hhmm) for hhmm in minutes_to_fetch])
    pbar.close()

    new_rows = []
    if parse_futures:
        slots = list(parse_futures.keys())
        print(f"  [{date_str}] Parsing {len(slots)} file(s)...", flush=True)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*parse_futures.values()), timeout=900
            )
        except asyncio.TimeoutError:
            results = []
            for slot, future in parse_futures.items():
                if future.done():
                    try:
                        results.append(future.result())
                    except Exception as e:
                        results.append(([], f"{type(e).__name__}: {e}"))
                else:
                    future.cancel()
                    results.append(([], "parse_timeout"))

        for slot, (rows, err) in zip(slots, results):
            if err and not rows:
                parse_errors[slot] = err
            else:
                new_rows.extend(rows)

    return new_rows, fetch_status, fetch_errors, parse_errors


def _update_manifest_from_fetch(
    manifest: dict,
    fetched_hhmm: list[str],
    fetch_status: dict[str, str],
    fetch_errors: dict[str, str],
    parse_errors: dict[str, str],
) -> None:
    for hhmm in fetched_hhmm:
        tag = fetch_status.get(hhmm)
        if tag == "ok":
            if hhmm in parse_errors:
                manifest["minutes"][hhmm] = STATUS_PARSE_ERROR
                manifest["errors"][hhmm] = f"parse: {parse_errors[hhmm]}"
            else:
                manifest["minutes"][hhmm] = STATUS_DOWNLOADED
                manifest["errors"].pop(hhmm, None)
        elif tag == "404":
            manifest["minutes"][hhmm] = STATUS_404
            manifest["errors"].pop(hhmm, None)
        else:
            manifest["minutes"][hhmm] = STATUS_HTTP_ERROR
            manifest["errors"][hhmm] = f"http: {fetch_errors.get(hhmm, 'unknown')}"


async def _finalize_and_write(
    date_str: str,
    new_rows: list,
    local_path: Path,
    final_path: Path,
    mf_path: Path,
    manifest: dict,
    row_group_size: int,
    release_cache: bool,
    executor: ProcessPoolExecutor,
    merge_fn,  # callable(old_table, new_table) → merged_table
) -> dict:
    """Build parquet from new_rows, apply merge_fn with existing data, write back."""
    loop = asyncio.get_event_loop()

    if new_rows:
        new_table = dvd._build_table(new_rows)

        if local_path.exists():
            old_table = pq.read_table(str(local_path))
            # Coerce legacy timestamp[ms] (old parquets) to timestamp[s] (canonical schema)
            if old_table.schema != dvd.SCHEMA:
                old_table = old_table.cast(dvd.SCHEMA)
            merged = merge_fn(old_table, new_table)
        else:
            merged = new_table

        local_path.parent.mkdir(parents=True, exist_ok=True)
        write_start = time.time()
        await loop.run_in_executor(
            None,
            functools.partial(dvd._write_table, merged, local_path, row_group_size),
        )
        rows = merged.num_rows
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(
            f"  [{date_str}] Wrote parquet ({rows:,} rows, {size_mb:.1f} MB) "
            f"in {time.time() - write_start:.1f}s",
            flush=True,
        )
    elif local_path.exists():
        rows = pq.ParquetFile(local_path).metadata.num_rows
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  [{date_str}] No new rows; parquet unchanged.", flush=True)
    else:
        rows = 0
        size_mb = 0.0
        print(f"  [{date_str}] No rows to write.", flush=True)

    counts = dvd.manifest_counts(manifest)
    complete = dvd.manifest_is_complete(manifest)
    manifest["complete"] = complete
    manifest["counts"] = dict(counts)
    manifest["last_run_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dvd.save_manifest(mf_path, manifest)

    if local_path.exists():
        dvd._copy_to_drive_and_release(local_path, final_path, mf_path, release_cache)
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"  [{date_str}] → Drive: {final_path.name} ({size_mb:.1f} MB)", flush=True)

    print(
        f"  [{date_str}] dl={counts.get(STATUS_DOWNLOADED, 0)} "
        f"404={counts.get(STATUS_404, 0)} "
        f"http_err={counts.get(STATUS_HTTP_ERROR, 0)} "
        f"parse_err={counts.get(STATUS_PARSE_ERROR, 0)} "
        f"{'[COMPLETE]' if complete else '[INCOMPLETE]'}",
        flush=True,
    )

    return {
        "date": date_str,
        "status": "complete" if complete else "patched_incomplete",
        "counts": dict(counts),
        "rows": rows,
        "size_mb": size_mb,
    }


def _copy_drive_to_local(date_str: str, local_path: Path, final_path: Path) -> None:
    """Bring the Drive parquet into local tmp so we can merge safely."""
    if not local_path.exists() and final_path.exists():
        print(f"  [{date_str}] Copying Drive parquet to local tmp...", flush=True)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_copy = local_path.with_suffix(".copy.tmp")
        shutil.copy2(final_path, tmp_copy)
        tmp_copy.rename(local_path)


# ---------------------------------------------------------------------------
# Gap-fill mode: retry http_error / parse_error / unrecorded minutes only
# ---------------------------------------------------------------------------

async def fill_day(
    date_str: str,
    data_root: Path,
    tmp_dir: Path,
    raw_dataset: str,
    manifest_dataset: str,
    vd_filter,
    concurrency: int,
    retries: int,
    executor: ProcessPoolExecutor,
    row_group_size: int,
    release_cache: bool,
) -> dict:
    local_path = dvd.tmp_parquet_path(tmp_dir, date_str)
    final_path = dvd.final_parquet_path(data_root, raw_dataset, date_str)
    mf_path = dvd.manifest_path(data_root, manifest_dataset, date_str)

    manifest = dvd.load_manifest(mf_path)
    manifest["date"] = date_str
    manifest["schema_version"] = 1
    manifest["raw_dataset"] = raw_dataset

    to_fetch = dvd.minutes_needing_fetch(manifest)
    if not to_fetch:
        return {"date": date_str, "status": "already_complete"}

    # Bring existing Drive parquet to local tmp so merge keeps all old rows.
    _copy_drive_to_local(date_str, local_path, final_path)

    print(f"  [{date_str}] Gap-fill: {len(to_fetch)}/{TOTAL_MINUTES} minute(s).", flush=True)

    new_rows, fetch_status, fetch_errors, parse_errors = await _fetch_and_parse(
        date_str, to_fetch, vd_filter, concurrency, retries, executor
    )
    _update_manifest_from_fetch(manifest, to_fetch, fetch_status, fetch_errors, parse_errors)

    def _simple_merge(old_table: pa.Table, new_table: pa.Table) -> pa.Table:
        return pa.concat_tables([old_table, new_table]).sort_by(SORT_KEYS)

    return await _finalize_and_write(
        date_str, new_rows, local_path, final_path, mf_path,
        manifest, row_group_size, release_cache, executor, _simple_merge,
    )


# ---------------------------------------------------------------------------
# VD-expand mode: re-download all non-404 minutes, add new VDs + fix gaps
# ---------------------------------------------------------------------------

def _minutes_for_expand(manifest: dict) -> list[str]:
    """All minutes except confirmed server_404 — these need re-download for VD expansion."""
    minutes = manifest.get("minutes", {})
    return [hhmm for hhmm in dvd._all_minutes() if minutes.get(hhmm) != STATUS_404]


def _gap_hhmm_int_set(manifest: dict) -> set[int]:
    """source_hhmm values (int) for minutes that have NO rows in the existing parquet."""
    all_hhmm = {int(h) for h in dvd._all_minutes()}
    recorded = {h: v for h, v in manifest.get("minutes", {}).items()}
    # Minutes with error status have been attempted but produced no parquet rows
    error_hhmm = {int(h) for h, v in recorded.items() if v in {STATUS_HTTP_ERROR, STATUS_PARSE_ERROR}}
    # Minutes never recorded at all were never attempted
    unrecorded_hhmm = all_hhmm - {int(h) for h in recorded}
    return error_hhmm | unrecorded_hhmm


async def expand_vd_day(
    date_str: str,
    data_root: Path,
    tmp_dir: Path,
    raw_dataset: str,
    manifest_dataset: str,
    vd_filter,
    concurrency: int,
    retries: int,
    executor: ProcessPoolExecutor,
    row_group_size: int,
    release_cache: bool,
) -> dict:
    """
    Re-download all non-404 minutes with an expanded VD filter, then merge
    only genuinely new data into the existing parquet:

      - Rows for VD IDs not already present in the parquet  (new VDs)
      - Rows for minutes that were previously error/unrecorded (any VD,
        since those minutes have zero rows in the existing parquet)

    Already-correct rows (existing VD in previously-downloaded minute) are
    not duplicated.
    """
    local_path = dvd.tmp_parquet_path(tmp_dir, date_str)
    final_path = dvd.final_parquet_path(data_root, raw_dataset, date_str)
    mf_path = dvd.manifest_path(data_root, manifest_dataset, date_str)

    manifest = dvd.load_manifest(mf_path)
    manifest["date"] = date_str
    manifest["schema_version"] = 1
    manifest["raw_dataset"] = raw_dataset

    to_fetch = _minutes_for_expand(manifest)
    gap_hhmm = _gap_hhmm_int_set(manifest)

    _copy_drive_to_local(date_str, local_path, final_path)

    print(
        f"  [{date_str}] Expand-VD: {len(to_fetch)}/{TOTAL_MINUTES} minute(s) "
        f"({len(gap_hhmm)} gap min, skip {TOTAL_MINUTES - len(to_fetch)} known-404).",
        flush=True,
    )

    new_rows, fetch_status, fetch_errors, parse_errors = await _fetch_and_parse(
        date_str, to_fetch, vd_filter, concurrency, retries, executor
    )
    _update_manifest_from_fetch(manifest, to_fetch, fetch_status, fetch_errors, parse_errors)

    def _smart_merge(old_table: pa.Table, new_table: pa.Table) -> pa.Table:
        """
        From new_table keep only:
          (a) rows with a VD ID not already in old_table
          (b) rows for minutes that previously had no data (gap minutes)
        Then concat with old_table and sort.
        """
        old_vdid_set = set(old_table.column("vdid").to_pylist())

        is_new_vd = pc.invert(
            pc.is_in(new_table.column("vdid"), value_set=pa.array(list(old_vdid_set)))
        )
        if gap_hhmm:
            is_gap_min = pc.is_in(
                new_table.column("source_hhmm"),
                value_set=pa.array(list(gap_hhmm), type=pa.int16()),
            )
            keep = pc.or_(is_new_vd, is_gap_min)
        else:
            keep = is_new_vd

        rows_to_add = new_table.filter(keep)
        n_new_vd = pc.sum(is_new_vd.cast(pa.int32())).as_py()
        n_added = rows_to_add.num_rows
        print(
            f"  [{date_str}] Merge: +{n_added:,} rows "
            f"({n_new_vd:,} from new VDs, {n_added - n_new_vd:,} from gap minutes)",
            flush=True,
        )

        if rows_to_add.num_rows == 0:
            return old_table
        return pa.concat_tables([old_table, rows_to_add]).sort_by(SORT_KEYS)

    return await _finalize_and_write(
        date_str, new_rows, local_path, final_path, mf_path,
        manifest, row_group_size, release_cache, executor, _smart_merge,
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    tmp_dir = Path(args.tmp_dir)

    if not data_root.parent.exists():
        print(f"[FATAL] Drive data parent unavailable: {data_root.parent}", file=sys.stderr)
        print("        Check that Google Drive for Desktop is running and mounted.", file=sys.stderr)
        sys.exit(2)

    dates = dvd.date_range(args.start_date, args.end_date)
    print(f"Scanning {len(dates)} day(s): {args.start_date} → {args.end_date}")
    print(f"Data root : {data_root}")
    print(f"Manifest  : {args.manifest_dataset}")
    print(f"Raw data  : {args.raw_dataset}")
    print(f"Local tmp : {tmp_dir}")
    if args.expand_vd:
        print(f"Mode      : VD-expand + gap-fill  (filter-mode={args.filter_mode})")
    else:
        print(f"Mode      : gap-fill only  (filter-mode={args.filter_mode})")

    # --- Audit phase ---
    statuses = [
        audit_day(d, data_root, tmp_dir, args.raw_dataset, args.manifest_dataset)
        for d in tqdm(dates, desc="Auditing", unit="day", leave=False)
    ]

    complete_ok      = [s for s in statuses if s.complete and s.has_parquet]
    complete_no_pq   = [s for s in statuses if s.complete and not s.has_parquet]
    patchable        = [s for s in statuses if s.is_patchable]
    expandable       = [s for s in statuses if s.is_expandable]
    incomplete_no_pq = [s for s in statuses if not s.complete and not s.has_any_parquet and s.has_manifest]
    undownloaded     = [s for s in statuses if not s.has_manifest and not s.has_any_parquet]

    print(f"\n{'='*64}")
    print("AUDIT SUMMARY")
    print(f"{'='*64}")
    print(f"  Complete + parquet OK            : {len(complete_ok)}")
    print(f"  Patchable (parquet + gap minutes): {len(patchable)}")
    print(f"  Expandable (has parquet, any)    : {len(expandable)}")
    print(f"  Complete manifest, parquet LOST  : {len(complete_no_pq)}")
    print(f"  Incomplete, no parquet anywhere  : {len(incomplete_no_pq)}")
    print(f"  Never downloaded                 : {len(undownloaded)}")

    if patchable and not args.expand_vd:
        print(f"\nPatchable days ({len(patchable)}):")
        for s in patchable:
            parts = []
            if s.n_http_error:
                parts.append(f"http_err={s.n_http_error}")
            if s.n_parse_error:
                parts.append(f"parse_err={s.n_parse_error}")
            if s.n_unrecorded:
                parts.append(f"unrecorded={s.n_unrecorded}")
            src = "Drive" if s.has_parquet else "local-tmp"
            print(f"  {s.date}  [{src}]  {' '.join(parts)}")

    if complete_no_pq:
        print(f"\nComplete manifest but parquet missing — re-run original downloader ({len(complete_no_pq)}):")
        for s in complete_no_pq[:10]:
            print(f"  {s.date}")
        if len(complete_no_pq) > 10:
            print(f"  ... and {len(complete_no_pq) - 10} more")

    if incomplete_no_pq or undownloaded:
        all_missing = incomplete_no_pq + undownloaded
        print(f"\nNo parquet + not complete — use original downloader ({len(all_missing)}):")
        for s in all_missing[:10]:
            tag = "no-manifest" if not s.has_manifest else "incomplete-no-pq"
            print(f"  {s.date}  [{tag}]")
        if len(all_missing) > 10:
            print(f"  ... and {len(all_missing) - 10} more")

    if not args.fill:
        if args.expand_vd:
            print(
                f"\n[AUDIT ONLY] Pass --fill --expand-vd to expand VDs for "
                f"{len(expandable)} day(s) with existing parquets."
            )
        else:
            print(f"\n[AUDIT ONLY] Pass --fill to patch {len(patchable)} patchable day(s).")
        return

    # --- Fill / expand phase ---
    work_days: list[DayStatus]
    if args.expand_vd:
        work_days = expandable
        if not work_days:
            print("\nNo days with existing parquets found. Nothing to expand.")
            return
        # Default filter to "all" for VD expansion unless user explicitly overrode it
        effective_filter_mode = args.filter_mode
        print(
            f"\nExpanding VD coverage for {len(work_days)} day(s) "
            f"(filter={effective_filter_mode})...",
            flush=True,
        )
    else:
        work_days = patchable
        if not work_days:
            print("\nNo patchable days found. Nothing to do.")
            return
        effective_filter_mode = args.filter_mode
        print(f"\nGap-filling {len(work_days)} day(s)...", flush=True)

    vd_list_path = Path(args.vd_list)
    vd_filter = dvd.build_vd_filter(vd_list_path, effective_filter_mode, args.edge_buffer_km)

    data_root.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n_parse_workers = max(1, os.cpu_count() or 4)
    executor = ProcessPoolExecutor(max_workers=n_parse_workers)

    print(
        f"Parallelism: day_batch={args.day_batch}, "
        f"http_workers/day={args.workers}, "
        f"parse_processes={n_parse_workers}",
        flush=True,
    )

    report_rows = []
    t0 = time.time()
    day_pbar = tqdm(total=len(work_days), desc="Days", unit="day")

    _day_fn = expand_vd_day if args.expand_vd else fill_day

    try:
        for batch_start in range(0, len(work_days), args.day_batch):
            batch = work_days[batch_start : batch_start + args.day_batch]
            tasks = [
                _day_fn(
                    date_str=s.date,
                    data_root=data_root,
                    tmp_dir=tmp_dir,
                    raw_dataset=args.raw_dataset,
                    manifest_dataset=args.manifest_dataset,
                    vd_filter=vd_filter,
                    concurrency=args.workers,
                    retries=args.retries,
                    executor=executor,
                    row_group_size=args.row_group_size,
                    release_cache=args.release_cache,
                )
                for s in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for s, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"  [{s.date}] FAILED: {type(result).__name__}: {result}", flush=True)
                    report_rows.append({"date": s.date, "status": "failed", "error": str(result)})
                else:
                    report_rows.append(result)
                day_pbar.update(1)
    finally:
        day_pbar.close()
        executor.shutdown(wait=False)

    n_complete = sum(1 for r in report_rows if r.get("status") == "complete")
    n_failed = sum(1 for r in report_rows if r.get("status") == "failed")

    print(f"\n{'='*64}")
    print("SUMMARY")
    print(f"{'='*64}")
    print(f"  Processed        : {len(report_rows)}")
    print(f"  Now complete     : {n_complete}")
    print(f"  Still incomplete : {len(report_rows) - n_complete - n_failed}")
    print(f"  Failed           : {n_failed}")
    print(f"  Elapsed          : {time.time() - t0:.0f}s")

    if n_failed:
        print("\nFailed days:")
        for r in report_rows:
            if r.get("status") == "failed":
                print(f"  {r['date']}  {r.get('error', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit VD download gaps and fill missing/errored minutes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--raw-dataset", default=DEFAULT_RAW_DATASET)
    parser.add_argument("--manifest-dataset", default=DEFAULT_MANIFEST_DATASET)
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DIR)
    parser.add_argument("--vd-list", default=DEFAULT_VD_LIST)
    parser.add_argument(
        "--filter-mode",
        choices=["buffered-vd-list", "exact-list", "all"],
        default="all",
        help=(
            "VD retention rule for downloaded data. "
            "Defaults to 'all' (keep every VD in the XML). "
            "Use 'buffered-vd-list' or 'exact-list' to restrict."
        ),
    )
    parser.add_argument("--edge-buffer-km", type=float, default=70.0)
    parser.add_argument(
        "--workers", type=int, default=64, help="Concurrent HTTP connections per day"
    )
    parser.add_argument("--day-batch", type=int, default=3, help="Days processed concurrently")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--row-group-size", type=int, default=DEFAULT_ROW_GROUP_SIZE)
    parser.add_argument("--release-cache", action="store_true", default=True)
    parser.add_argument("--keep-cache", dest="release_cache", action="store_false")
    parser.add_argument(
        "--fill",
        action="store_true",
        help="Download and patch (default: audit only)",
    )
    parser.add_argument(
        "--expand-vd",
        action="store_true",
        help=(
            "Re-download all non-404 minutes with --filter-mode to add VDs "
            "that were filtered out during the original download. "
            "Only rows for genuinely new VDs (or previously-error minutes) "
            "are added — existing rows are not duplicated. "
            "Requires --fill. Processes ALL days with existing parquets."
        ),
    )
    args = parser.parse_args()

    if args.expand_vd and not args.fill:
        parser.error("--expand-vd requires --fill")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    main()
