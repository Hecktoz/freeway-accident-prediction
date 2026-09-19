# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx[http2]",
#     "lxml",
#     "pyarrow",
#     "tqdm",
# ]
# ///
# ruff: noqa: E402
"""
Download Taiwan freeway VD data into a Google Drive + Parquet layout.

This is the reset-friendly downloader for the Colab workflow:

  * Download and parse locally with async HTTP + process-pool XML parsing.
  * Filter VD rows by an expanded VD universe so future upstream/downstream
    windows have room at the edges.
  * Write one source-date parquet per file, partitioned by source year/month.
  * Keep raw lane-level columns plus parsed freeway/direction/mileage fields.
  * Copy the finished daily parquet to Google Drive, delete local scratch, and
    ask Drive for Desktop to make the synced file online-only.

Default output root:
  G:\\My Drive\\陽交大高速公路計畫\\main\\data

Example:
  uv run scripts/data/download_vd_drive.py --start-date 2024-04-01 --end-date 2024-04-30
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import lxml.etree as ET
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


BASE_URL = "https://tisvcloud.freeway.gov.tw/history/motc20/VD/{date}/VDLive_{hhmm}.xml.gz"

DEFAULT_DATA_ROOT = r"G:\My Drive\陹交大高速公路計畫\main\data"
DEFAULT_RAW_DATASET = "vd_raw_lane_1m_full"
DEFAULT_MANIFEST_DATASET = "_vd_download_manifests"
DEFAULT_TMP_DIR = str(Path(tempfile.gettempdir()) / "vd_drive_download")
DEFAULT_VD_LIST = str(PROJECT_ROOT / "central_vd_list.txt")

TOTAL_MINUTES = 1440
DEFAULT_ROW_GROUP_SIZE = 131_072
VDID_PATTERN = re.compile(r"^VD-(N\d+)-([NSEW])-([\d.]+)")

STATUS_DOWNLOADED = "downloaded"
STATUS_404 = "server_404"
STATUS_HTTP_ERROR = "http_error"
STATUS_PARSE_ERROR = "parse_error"


SCHEMA = pa.schema([
    ("source_date", pa.date32()),
    ("source_hhmm", pa.int16()),
    ("timestamp", pa.timestamp("s")),
    ("vdid", pa.string()),
    ("freeway", pa.string()),
    ("direction", pa.string()),
    ("mileage_m", pa.int32()),
    ("link_id", pa.string()),
    ("lane_id", pa.int8()),
    ("lane_type", pa.int8()),
    ("speed", pa.int16()),
    ("occupancy", pa.int16()),
    ("status", pa.int8()),
    ("vol_s", pa.int16()),
    ("spd_s", pa.int16()),
    ("vol_l", pa.int16()),
    ("spd_l", pa.int16()),
    ("vol_t", pa.int16()),
    ("spd_t", pa.int16()),
])


@dataclass(frozen=True)
class MileageRange:
    freeway: str
    direction: str
    min_km: float
    max_km: float


@dataclass(frozen=True)
class VDFilter:
    mode: str
    exact_vds: frozenset[str]
    ranges: tuple[MileageRange, ...]

    def keep(self, vdid: str) -> tuple[bool, str | None, str | None, int | None]:
        parsed = parse_vdid(vdid)
        if parsed is None:
            return (self.mode == "all"), None, None, None

        freeway, direction, mileage_km = parsed
        mileage_m = int(round(mileage_km * 1000))

        if self.mode == "all":
            return True, freeway, direction, mileage_m
        if self.mode == "exact-list":
            return vdid in self.exact_vds, freeway, direction, mileage_m

        for item in self.ranges:
            if (
                item.freeway == freeway
                and item.direction == direction
                and item.min_km <= mileage_km <= item.max_km
            ):
                return True, freeway, direction, mileage_m
        return False, freeway, direction, mileage_m


def parse_vdid(vdid: str) -> tuple[str, str, float] | None:
    match = VDID_PATTERN.match(vdid)
    if not match:
        return None
    return match.group(1), match.group(2), float(match.group(3))


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------


def _tag(element) -> str:
    return element.tag.split("}")[-1] if "}" in element.tag else element.tag


def _children(element, tag_name: str):
    return [child for child in element if _tag(child) == tag_name]


def _first_child(element, tag_name: str):
    for child in element:
        if _tag(child) == tag_name:
            return child
    return None


def _child_text(element, tag_name: str, default: str | None = None) -> str | None:
    child = _first_child(element, tag_name)
    if child is None:
        return default
    return child.text if child.text is not None else default


def _parse_int_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _vehicle_map(vehicles_elem) -> dict[str, tuple[str | None, str | None]]:
    values: dict[str, tuple[str | None, str | None]] = {}
    if vehicles_elem is None:
        return values

    for vehicle in _children(vehicles_elem, "Vehicle"):
        vehicle_type = _child_text(vehicle, "VehicleType")
        if vehicle_type not in {"S", "L", "T"}:
            continue
        values[vehicle_type] = (
            _parse_int_text(_child_text(vehicle, "Volume")),
            _parse_int_text(_child_text(vehicle, "Speed")),
        )
    return values


def _parse_xml_raw(
    xml_bytes: bytes,
    vd_filter: VDFilter,
    source_date: str,
    source_hhmm: str,
) -> tuple[list, str | None]:
    cleaned = xml_bytes.replace(b" xmlns=", b" xmlns_disabled=")
    try:
        root = ET.fromstring(cleaned)
    except Exception as e:
        return [], f"xml_root: {type(e).__name__}: {e}"

    vdlives = _first_child(root, "VDLives")
    if vdlives is None:
        return [], "no_VDLives_element"

    rows = []
    row_errors = 0
    for vd_elem in _children(vdlives, "VDLive"):
        try:
            vdid = _child_text(vd_elem, "VDID")
            if not vdid:
                continue

            keep, freeway, direction, mileage_m = vd_filter.keep(vdid)
            if not keep:
                continue

            ts = _child_text(vd_elem, "DataCollectTime")
            if not ts:
                continue

            status = _parse_int_text(_child_text(vd_elem, "Status"))
            link_flows = _first_child(vd_elem, "LinkFlows")
            if link_flows is None:
                continue

            for link_flow in _children(link_flows, "LinkFlow"):
                link_id = _child_text(link_flow, "LinkID")
                lanes = _first_child(link_flow, "Lanes")
                if not link_id or lanes is None:
                    continue

                for lane in _children(lanes, "Lane"):
                    lane_id = _parse_int_text(_child_text(lane, "LaneID"))
                    lane_type = _parse_int_text(_child_text(lane, "LaneType"))
                    speed = _parse_int_text(_child_text(lane, "Speed"))
                    occupancy = _parse_int_text(_child_text(lane, "Occupancy"))
                    if lane_id is None or lane_type is None or speed is None or occupancy is None:
                        continue
                    vehicles = _vehicle_map(_first_child(lane, "Vehicles"))
                    vol_s, spd_s = vehicles.get("S", (None, None))
                    vol_l, spd_l = vehicles.get("L", (None, None))
                    vol_t, spd_t = vehicles.get("T", (None, None))
                    rows.append((
                        source_date,
                        source_hhmm,
                        ts,
                        vdid,
                        freeway,
                        direction,
                        mileage_m,
                        link_id,
                        lane_id,
                        lane_type,
                        speed,
                        occupancy,
                        status,
                        vol_s,
                        spd_s,
                        vol_l,
                        spd_l,
                        vol_t,
                        spd_t,
                    ))
        except (IndexError, AttributeError, ValueError):
            row_errors += 1
            continue

    err = f"{row_errors}_row_errors" if row_errors and not rows else None
    return rows, err


def _decompress_and_parse(
    gz_bytes: bytes,
    vd_filter: VDFilter,
    source_date: str,
    source_hhmm: str,
) -> tuple[list, str | None]:
    try:
        xml_bytes = gzip.decompress(gz_bytes)
    except Exception as e:
        return [], f"gunzip: {type(e).__name__}: {e}"
    return _parse_xml_raw(xml_bytes, vd_filter, source_date, source_hhmm)


# ---------------------------------------------------------------------------
# Arrow / Parquet
# ---------------------------------------------------------------------------


def _timestamp_array(raw_values) -> pa.Array:
    raw = pa.array(raw_values, type=pa.string())
    # Upstream strings are like 2024-04-01T00:00:00+08:00.
    # Store local Taiwan wall time as timestamp[s] to match accident CSV times.
    local_text = pc.replace_substring(pc.utf8_slice_codeunits(raw, 0, 19), "T", " ")
    parsed = pc.strptime(local_text, format="%Y-%m-%d %H:%M:%S", unit="s")
    return parsed.cast(pa.timestamp("s"))


def _source_date_array(raw_values) -> pa.Array:
    raw = pa.array(raw_values, type=pa.string())
    parsed = pc.strptime(raw, format="%Y-%m-%d", unit="s")
    return parsed.cast(pa.date32())


def _build_table(rows: list) -> pa.Table:
    columns = list(zip(*rows))
    arrays = []
    for i, field in enumerate(SCHEMA):
        if field.name == "timestamp":
            arrays.append(_timestamp_array(columns[i]))
        elif field.name == "source_date":
            arrays.append(_source_date_array(columns[i]))
        elif pa.types.is_string(field.type):
            arrays.append(pa.array(columns[i], type=pa.string()))
        else:
            arrays.append(pa.array(columns[i]).cast(field.type))

    table = pa.table({field.name: arr for field, arr in zip(SCHEMA, arrays)}, schema=SCHEMA)
    return table.sort_by([
        ("freeway", "ascending"),
        ("direction", "ascending"),
        ("mileage_m", "ascending"),
        ("vdid", "ascending"),
        ("timestamp", "ascending"),
        ("lane_id", "ascending"),
    ])


def _write_table(table: pa.Table, path: Path, row_group_size: int) -> None:
    pq.write_table(
        table,
        str(path),
        version="2.6",
        compression="zstd",
        row_group_size=row_group_size,
        use_dictionary=True,
        write_statistics=True,
    )


# ---------------------------------------------------------------------------
# Paths / manifests / disk handling
# ---------------------------------------------------------------------------


def _all_minutes() -> list[str]:
    return [f"{h:02d}{m:02d}" for h in range(24) for m in range(60)]


def date_range(start: str, end: str) -> list[str]:
    start_ts = datetime.strptime(start, "%Y-%m-%d")
    end_ts = datetime.strptime(end, "%Y-%m-%d")
    if end_ts < start_ts:
        raise ValueError("--end-date must be greater than or equal to --start-date")
    out = []
    cur = start_ts
    while cur <= end_ts:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def partition_dir(root: Path, date_str: str, dataset: str) -> Path:
    year, month, _ = date_str.split("-")
    return root / dataset / f"source_year={year}" / f"source_month={month}"


def final_parquet_path(root: Path, dataset: str, date_str: str) -> Path:
    yyyymmdd = date_str.replace("-", "")
    return partition_dir(root, date_str, dataset) / f"vd_{yyyymmdd}.parquet"


def manifest_path(root: Path, dataset: str, date_str: str) -> Path:
    yyyymmdd = date_str.replace("-", "")
    return partition_dir(root, date_str, dataset) / f"vd_{yyyymmdd}.manifest.json"


def tmp_parquet_path(tmp_dir: Path, date_str: str) -> Path:
    return tmp_dir / f"vd_{date_str.replace('-', '')}.parquet"


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"minutes": {}, "errors": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("minutes", {})
        data.setdefault("errors", {})
        return data
    except Exception:
        return {"minutes": {}, "errors": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def manifest_counts(manifest: dict) -> Counter:
    return Counter(manifest.get("minutes", {}).values())


def manifest_is_complete(manifest: dict) -> bool:
    counts = manifest_counts(manifest)
    return counts.get(STATUS_DOWNLOADED, 0) + counts.get(STATUS_404, 0) == TOTAL_MINUTES


def minutes_needing_fetch(manifest: dict) -> list[str]:
    minutes = manifest.get("minutes", {})
    return [
        hhmm
        for hhmm in _all_minutes()
        if minutes.get(hhmm) not in {STATUS_DOWNLOADED, STATUS_404}
    ]


def infer_minutes_from_parquet(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        table = pq.read_table(str(path), columns=["source_hhmm"])
    except Exception:
        try:
            table = pq.read_table(str(path), columns=["timestamp"])
        except Exception:
            return set()

        out = set()
        for value in table.column("timestamp").to_pylist():
            if value is None:
                continue
            if hasattr(value, "hour"):
                out.add(f"{value.hour:02d}{value.minute:02d}")
        return out

    out = set()
    for value in table.column("source_hhmm").to_pylist():
        if value is None:
            continue
        out.add(f"{int(value):04d}")
    return out


def _free_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    _, _, free = shutil.disk_usage(str(probe))
    return free / (1024**3)


def _mark_online_only(path: Path) -> None:
    if os.name != "nt" or not path.exists():
        return
    try:
        subprocess.run(["attrib", "-P", "+U", str(path)], check=False, capture_output=True, timeout=10)
    except Exception:
        pass


def _wait_for_space(paths: list[Path], min_free_gb: float, label: str) -> None:
    if min_free_gb <= 0:
        return
    low = [(path, _free_gb(path)) for path in paths if _free_gb(path) < min_free_gb]
    if not low:
        return
    details = ", ".join(f"{path.anchor or path}: {free:.1f} GB" for path, free in low)
    print(f"  [DISK] Free space below {min_free_gb:.1f} GB before {label}: {details}", flush=True)
    print("  [DISK] Waiting; Drive for Desktop may need time to upload/evict cached files.", flush=True)
    while True:
        time.sleep(30)
        low = [(path, _free_gb(path)) for path in paths if _free_gb(path) < min_free_gb]
        if not low:
            recovered = ", ".join(f"{path.anchor or path}: {_free_gb(path):.1f} GB" for path in paths)
            print(f"  [DISK] Recovered: {recovered}", flush=True)
            return


def _copy_to_drive_and_release(
    local_path: Path,
    final_path: Path,
    manifest_file: Path,
    release_cache: bool,
) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_drive_path = final_path.with_name(final_path.name + ".tmp")
    if tmp_drive_path.exists():
        tmp_drive_path.unlink()
    shutil.copy2(local_path, tmp_drive_path)
    tmp_drive_path.replace(final_path)
    local_path.unlink()

    if release_cache:
        _mark_online_only(final_path)
        _mark_online_only(manifest_file)


# ---------------------------------------------------------------------------
# Filter construction
# ---------------------------------------------------------------------------


def _load_vd_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_vd_filter(vd_list_path: Path, mode: str, edge_buffer_km: float) -> VDFilter:
    exact_vds = frozenset(_load_vd_list(vd_list_path))
    if mode == "all":
        return VDFilter(mode=mode, exact_vds=frozenset(), ranges=())
    if mode == "exact-list":
        return VDFilter(mode=mode, exact_vds=exact_vds, ranges=())

    grouped: dict[tuple[str, str], list[float]] = {}
    for vdid in exact_vds:
        parsed = parse_vdid(vdid)
        if parsed is None:
            continue
        freeway, direction, mileage_km = parsed
        grouped.setdefault((freeway, direction), []).append(mileage_km)

    ranges = tuple(
        MileageRange(
            freeway=freeway,
            direction=direction,
            min_km=max(0.0, min(values) - edge_buffer_km),
            max_km=max(values) + edge_buffer_km,
        )
        for (freeway, direction), values in sorted(grouped.items())
        if values
    )
    return VDFilter(mode=mode, exact_vds=exact_vds, ranges=ranges)


def write_filter_metadata(root: Path, vd_filter: VDFilter, vd_list_path: Path, edge_buffer_km: float) -> None:
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": vd_filter.mode,
        "vd_list_path": str(vd_list_path),
        "exact_vd_count": len(vd_filter.exact_vds),
        "edge_buffer_km": edge_buffer_km,
        "ranges": [
            {
                "freeway": item.freeway,
                "direction": item.direction,
                "min_km": item.min_km,
                "max_km": item.max_km,
            }
            for item in vd_filter.ranges
        ],
    }
    (metadata_dir / "vd_download_filter.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# HTTP and per-day pipeline
# ---------------------------------------------------------------------------


async def _get(client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore, retries: int) -> tuple[str, bytes | str | None]:
    async with sem:
        last_error = None
        for attempt in range(retries):
            try:
                response = await client.get(url, timeout=30)
                if response.status_code == 404:
                    return "404", None
                response.raise_for_status()
                return "ok", response.content
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
        return "error", str(last_error) if last_error is not None else "unknown error"


async def process_day(
    date_str: str,
    data_root: Path,
    tmp_dir: Path,
    raw_dataset: str,
    manifest_dataset: str,
    vd_filter: VDFilter,
    concurrency: int,
    retries: int,
    executor: ProcessPoolExecutor,
    row_group_size: int,
    release_cache: bool,
) -> dict:
    yyyymmdd = date_str.replace("-", "")
    local_path = tmp_parquet_path(tmp_dir, date_str)
    final_path = final_parquet_path(data_root, raw_dataset, date_str)
    mf_path = manifest_path(data_root, manifest_dataset, date_str)

    manifest = load_manifest(mf_path)
    manifest["date"] = date_str
    manifest["schema_version"] = 1
    manifest["raw_dataset"] = raw_dataset

    if manifest_is_complete(manifest) and final_path.exists():
        counts = manifest_counts(manifest)
        print(f"  [{date_str}] Already complete on Drive. Skipping.", flush=True)
        return {"date": date_str, "status": "already_complete", "counts": dict(counts), "rows": None, "size_mb": None}

    for existing_path in (local_path, final_path):
        inferred = infer_minutes_from_parquet(existing_path)
        for hhmm in inferred:
            manifest["minutes"][hhmm] = STATUS_DOWNLOADED
            manifest["errors"].pop(hhmm, None)

    to_fetch = minutes_needing_fetch(manifest)
    if not to_fetch and final_path.exists():
        manifest["complete"] = True
        manifest["counts"] = dict(manifest_counts(manifest))
        manifest["last_run_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(mf_path, manifest)
        if release_cache:
            _mark_online_only(final_path)
            _mark_online_only(mf_path)
        return {
            "date": date_str,
            "status": "complete",
            "counts": manifest["counts"],
            "rows": None,
            "size_mb": final_path.stat().st_size / (1024 * 1024),
        }

    print(f"  [{date_str}] Need {len(to_fetch)}/{TOTAL_MINUTES} minute(s).", flush=True)

    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    loop = asyncio.get_event_loop()

    parse_futures: dict[str, asyncio.Future] = {}
    fetch_status: dict[str, str] = {}
    fetch_errors: dict[str, str] = {}
    parse_errors: dict[str, str] = {}

    pbar = tqdm(total=len(to_fetch), desc=f"  {date_str}", unit="file", leave=False)
    ok_count = missing_count = error_count = 0

    async with httpx.AsyncClient(verify=False, http2=True, limits=limits) as client:
        async def fetch_one(hhmm: str) -> None:
            nonlocal ok_count, missing_count, error_count
            url = BASE_URL.format(date=yyyymmdd, hhmm=hhmm)
            tag, payload = await _get(client, url, sem, retries)
            fetch_status[hhmm] = tag
            pbar.update(1)
            if tag == "ok":
                ok_count += 1
                parse_futures[hhmm] = loop.run_in_executor(
                    executor,
                    _decompress_and_parse,
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
            pbar.set_postfix(ok=ok_count, missing=missing_count, err=error_count)

        await asyncio.gather(*[fetch_one(hhmm) for hhmm in to_fetch])
    pbar.close()

    new_rows = []
    if parse_futures:
        slots = list(parse_futures.keys())
        print(f"  [{date_str}] Parsing {len(slots)} downloaded files...", flush=True)
        try:
            results = await asyncio.wait_for(asyncio.gather(*parse_futures.values()), timeout=900)
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

    for hhmm in to_fetch:
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

    if new_rows:
        table = _build_table(new_rows)
        del new_rows
        if local_path.exists():
            old_table = pq.read_table(str(local_path))
            table = pa.concat_tables([old_table, table]).sort_by([
                ("freeway", "ascending"),
                ("direction", "ascending"),
                ("mileage_m", "ascending"),
                ("vdid", "ascending"),
                ("timestamp", "ascending"),
                ("lane_id", "ascending"),
            ])

        local_path.parent.mkdir(parents=True, exist_ok=True)
        write_start = time.time()
        await loop.run_in_executor(
            None,
            functools.partial(_write_table, table, local_path, row_group_size),
        )
        rows = table.num_rows
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(
            f"  [{date_str}] Wrote local parquet ({rows:,} rows, {size_mb:.1f} MB) "
            f"in {time.time() - write_start:.1f}s",
            flush=True,
        )
    else:
        rows = pq.ParquetFile(local_path).metadata.num_rows if local_path.exists() else 0
        size_mb = local_path.stat().st_size / (1024 * 1024) if local_path.exists() else 0.0
        print(f"  [{date_str}] No new rows parsed.", flush=True)

    counts = manifest_counts(manifest)
    complete = manifest_is_complete(manifest)
    manifest["complete"] = complete
    manifest["counts"] = dict(counts)
    manifest["last_run_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_manifest(mf_path, manifest)

    if complete and local_path.exists():
        _copy_to_drive_and_release(local_path, final_path, mf_path, release_cache)
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"  [{date_str}] Copied to Drive -> {final_path.name} ({size_mb:.1f} MB)", flush=True)
    elif complete and final_path.exists():
        size_mb = final_path.stat().st_size / (1024 * 1024)

    print(
        f"  [{date_str}] downloaded={counts.get(STATUS_DOWNLOADED, 0)} "
        f"server_404={counts.get(STATUS_404, 0)} "
        f"http_error={counts.get(STATUS_HTTP_ERROR, 0)} "
        f"parse_error={counts.get(STATUS_PARSE_ERROR, 0)} "
        f"{'[COMPLETE]' if complete else '[INCOMPLETE]'}",
        flush=True,
    )

    return {
        "date": date_str,
        "status": "complete" if complete else "incomplete",
        "counts": dict(counts),
        "rows": rows,
        "size_mb": size_mb,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    tmp_dir = Path(args.tmp_dir)
    vd_list_path = Path(args.vd_list)

    if not data_root.parent.exists():
        print(f"[FATAL] Drive data parent is not available: {data_root.parent}", file=sys.stderr)
        print("        Check that Google Drive for Desktop is running and mounted.", file=sys.stderr)
        sys.exit(2)

    vd_filter = build_vd_filter(vd_list_path, args.filter_mode, args.edge_buffer_km)
    dates = date_range(args.start_date, args.end_date)

    print(f"Date range: {args.start_date} -> {args.end_date} ({len(dates)} source day(s))")
    print(f"Drive data root: {data_root}")
    print(f"Raw dataset: {args.raw_dataset}")
    print(f"Manifest dataset: {args.manifest_dataset}")
    print(f"Local tmp: {tmp_dir}")
    print(f"Filter mode: {vd_filter.mode}")
    if vd_filter.mode == "buffered-vd-list":
        print(f"VD list: {vd_list_path} ({len(vd_filter.exact_vds)} exact VD IDs)")
        print(f"Edge buffer: {args.edge_buffer_km:g} km")
        print(f"Buffered route/direction ranges: {len(vd_filter.ranges)}")
        for item in vd_filter.ranges[:12]:
            print(f"  {item.freeway}-{item.direction}: {item.min_km:.3f} -> {item.max_km:.3f} km")
        if len(vd_filter.ranges) > 12:
            print(f"  ... {len(vd_filter.ranges) - 12} more")
    elif vd_filter.mode == "exact-list":
        print(f"VD list: {vd_list_path} ({len(vd_filter.exact_vds)} exact VD IDs)")
    print(f"Row group size: {args.row_group_size:,}")
    print(f"Release Drive cache after copy: {args.release_cache}")

    if args.dry_run:
        print("Dry run only; no downloads started.")
        return

    data_root.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    write_filter_metadata(data_root, vd_filter, vd_list_path, args.edge_buffer_km)

    n_parse_workers = max(1, os.cpu_count() or 4)
    executor = ProcessPoolExecutor(max_workers=n_parse_workers)
    print(
        f"Parallelism: day_batch={args.day_batch}, http_workers/day={args.workers}, "
        f"parse_processes={n_parse_workers}",
        flush=True,
    )

    report_rows = []
    completed = 0
    t0 = time.time()
    day_pbar = tqdm(total=len(dates), desc="Days", unit="day")

    try:
        for batch_start in range(0, len(dates), args.day_batch):
            batch = dates[batch_start : batch_start + args.day_batch]
            _wait_for_space([tmp_dir, data_root], args.min_free_gb, f"batch {batch[0]}")
            tasks = [
                process_day(
                    date_str=date_str,
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
                for date_str in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for date_str, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"  [{date_str}] FAILED: {type(result).__name__}: {result}", flush=True)
                    report_rows.append({
                        "date": date_str,
                        "status": "failed",
                        "error": f"{type(result).__name__}: {result}",
                    })
                else:
                    report_rows.append(result)
                    if result["status"] in {"complete", "already_complete"}:
                        completed += 1
                day_pbar.update(1)
                day_pbar.set_postfix(done=completed, tmp_gb=f"{_free_gb(tmp_dir):.1f}")
    finally:
        day_pbar.close()
        executor.shutdown(wait=False)

    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "range": {"start": args.start_date, "end": args.end_date, "days": len(dates)},
        "data_root": str(data_root),
        "raw_dataset": args.raw_dataset,
        "manifest_dataset": args.manifest_dataset,
        "filter_mode": vd_filter.mode,
        "edge_buffer_km": args.edge_buffer_km,
        "elapsed_sec": round(time.time() - t0, 1),
        "completed_days": completed,
        "days": report_rows,
    }
    report_path = data_root / args.manifest_dataset / f"_run_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Complete days: {completed} / {len(dates)}")
    print(f"Elapsed: {report['elapsed_sec']:.0f}s")
    print(f"Run report: {report_path}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VD raw lane data into Google Drive Parquet.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help=f"Drive data root (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--raw-dataset", default=DEFAULT_RAW_DATASET, help=f"Raw dataset folder name (default: {DEFAULT_RAW_DATASET})")
    parser.add_argument(
        "--manifest-dataset",
        default=DEFAULT_MANIFEST_DATASET,
        help=f"Manifest folder name under data root (default: {DEFAULT_MANIFEST_DATASET})",
    )
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DIR, help=f"Local scratch dir (default: {DEFAULT_TMP_DIR})")
    parser.add_argument("--vd-list", default=DEFAULT_VD_LIST, help=f"VD list used for filtering (default: {DEFAULT_VD_LIST})")
    parser.add_argument(
        "--filter-mode",
        choices=["buffered-vd-list", "exact-list", "all"],
        default="buffered-vd-list",
        help="VD retention rule. buffered-vd-list expands each route/direction mileage range by --edge-buffer-km.",
    )
    parser.add_argument(
        "--edge-buffer-km",
        type=float,
        default=70.0,
        help="Extra km kept before/after each VD-list route range in buffered-vd-list mode (default: 70).",
    )
    parser.add_argument("--workers", type=int, default=64, help="Concurrent HTTP connections per source day (default: 64)")
    parser.add_argument("--day-batch", type=int, default=3, help="Number of source days processed concurrently (default: 2)")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retries per minute file (default: 3)")
    parser.add_argument("--row-group-size", type=int, default=DEFAULT_ROW_GROUP_SIZE, help=f"Parquet row group size (default: {DEFAULT_ROW_GROUP_SIZE})")
    parser.add_argument("--min-free-gb", type=float, default=20.0, help="Pause if tmp/Drive cache drive has less free space (default: 20)")
    parser.add_argument("--release-cache", action="store_true", default=True, help="Mark Drive parquet online-only after copy (default: on)")
    parser.add_argument("--keep-cache", dest="release_cache", action="store_false", help="Keep local Drive cache copy")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and filter ranges without downloading")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    main()
