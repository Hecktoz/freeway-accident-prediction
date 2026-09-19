"""Live MOTC VD fetch for in-memory inference.

Wraps the async fetch + XML parse from `scripts/data/download_vd_drive.py` but
tailored for the Phase 5 GUI:

  * Accepts an accident location (freeway / direction / mileage_km) and only
    fetches/parses VDs inside the spatial window for that accident. The
    nationwide MOTC XML covers ~2,000+ VDs per minute; we typically need
    ~30-80 VDs along the accident route, so spatial filtering at parse time
    is the largest win.
  * No disk writes — returns DataFrames in memory, suitable for handoff to
    `predict_accident_impact`.
  * Synchronous entry point so the GUI handler doesn't need to manage an
    event loop.

This is a pure library module (no inline deps). The runnable smoke test lives
at `scripts/demo/live_vd_smoke.py` so `scripts/lib/calendar.py` doesn't shadow
the stdlib `calendar` module when Python adds the script's directory to
`sys.path`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd

from scripts.data.download_vd_drive import (
    BASE_URL,
    SCHEMA,
    MileageRange,
    VDFilter,
    _decompress_and_parse,
)

DEFAULT_CONCURRENCY = 16
DEFAULT_RETRIES = 2
DEFAULT_SPEED_MAX_KMH = 140.0

STATION_5M_COLUMNS = [
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


@dataclass(frozen=True)
class FetchStatus:
    """Outcome of a live fetch — surfaced to the GUI coverage panel."""

    minutes_requested: int
    minutes_downloaded: int
    minutes_404: int
    minutes_error: int
    parse_errors: int
    lane_rows: int
    station_5m_rows: int
    unique_vds: int

    def to_dict(self) -> dict:
        return {
            "minutes_requested": self.minutes_requested,
            "minutes_downloaded": self.minutes_downloaded,
            "minutes_404": self.minutes_404,
            "minutes_error": self.minutes_error,
            "parse_errors": self.parse_errors,
            "lane_rows": self.lane_rows,
            "station_5m_rows": self.station_5m_rows,
            "unique_vds": self.unique_vds,
        }


def make_accident_vd_filter(
    freeway: str,
    direction: str,
    mileage_m: int,
    upstream_km: int = 50,
    downstream_km: int = 10,
) -> VDFilter:
    """Build a tight VDFilter that keeps only VDs in this accident's window.

    Spatial bounds follow the same convention as `windows.slice_station_df`:
    for S/E directions, upstream is the lower-mileage side; for N/W, upstream
    is the higher-mileage side.
    """
    mileage_km = mileage_m / 1000.0
    if direction in ("S", "E"):
        min_km = max(0.0, mileage_km - upstream_km)
        max_km = mileage_km + downstream_km
    else:
        min_km = max(0.0, mileage_km - downstream_km)
        max_km = mileage_km + upstream_km

    ranges = (
        MileageRange(
            freeway=freeway,
            direction=direction,
            min_km=min_km,
            max_km=max_km,
        ),
    )
    return VDFilter(mode="ranges", exact_vds=frozenset(), ranges=ranges)


def _minutes_between(start: datetime, end: datetime) -> list[tuple[str, str]]:
    """Yield (date_str, hhmm) for each minute in [start, end] inclusive."""
    cur = start.replace(second=0, microsecond=0)
    end_floor = end.replace(second=0, microsecond=0)
    out: list[tuple[str, str]] = []
    while cur <= end_floor:
        out.append((cur.strftime("%Y-%m-%d"), cur.strftime("%H%M")))
        cur += timedelta(minutes=1)
    return out


async def _get_one(
    client: httpx.AsyncClient,
    url: str,
    sem: asyncio.Semaphore,
    retries: int,
) -> tuple[str, bytes | None]:
    async with sem:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = await client.get(url, timeout=15)
                if response.status_code == 404:
                    return "404", None
                response.raise_for_status()
                return "ok", response.content
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
        return "error", None


async def fetch_lane_rows(
    start: datetime,
    end: datetime,
    vd_filter: VDFilter,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    retries: int = DEFAULT_RETRIES,
) -> tuple[list[tuple], dict[str, int]]:
    """Async fetch + parse minute XML files for [start, end].

    Parse runs in-process — the per-call volume is small (typically ~75 files)
    and the spatial filter inside `_decompress_and_parse` keeps each parse
    fast, so a process pool is not worth the overhead.

    Returns (rows, fetch_counts) where rows are the lane-level tuples
    matching `download_vd_drive.SCHEMA` and fetch_counts has download status
    aggregates.
    """
    minutes = _minutes_between(start, end)
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )

    payloads: dict[tuple[str, str], bytes] = {}
    counts = {"ok": 0, "404": 0, "error": 0, "parse_error": 0}

    async with httpx.AsyncClient(verify=False, http2=True, limits=limits) as client:
        async def fetch_one(date_str: str, hhmm: str) -> None:
            yyyymmdd = date_str.replace("-", "")
            url = BASE_URL.format(date=yyyymmdd, hhmm=hhmm)
            tag, payload = await _get_one(client, url, sem, retries)
            if tag == "ok" and payload is not None:
                counts["ok"] += 1
                payloads[(date_str, hhmm)] = payload
            elif tag == "404":
                counts["404"] += 1
            else:
                counts["error"] += 1

        await asyncio.gather(*[fetch_one(d, h) for d, h in minutes])

    rows: list[tuple] = []
    for (date_str, hhmm), gz_bytes in payloads.items():
        parsed, err = _decompress_and_parse(gz_bytes, vd_filter, date_str, hhmm)
        if err:
            counts["parse_error"] += 1
        rows.extend(parsed)

    counts["minutes_requested"] = len(minutes)
    return rows, counts


def lane_rows_to_dataframe(rows: list[tuple]) -> pd.DataFrame:
    """Convert tuple rows from `_decompress_and_parse` to a typed DataFrame."""
    field_names = [field.name for field in SCHEMA]
    if not rows:
        return pd.DataFrame(columns=field_names)

    cols = list(zip(*rows))
    df = pd.DataFrame({name: list(col) for name, col in zip(field_names, cols)})

    df["source_date"] = pd.to_datetime(df["source_date"], errors="coerce").dt.date
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(str).str.slice(0, 19),
        errors="coerce",
    )
    for col in ("source_hhmm", "mileage_m", "lane_id", "lane_type", "status"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("speed", "occupancy", "vol_s", "spd_s", "vol_l", "spd_l", "vol_t", "spd_t"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int16")
    df["mileage_m"] = df["mileage_m"].fillna(0).astype("int32")
    df["lane_id"] = df["lane_id"].fillna(0).astype("int8")
    df["lane_type"] = df["lane_type"].fillna(0).astype("int8")
    df["status"] = df["status"].fillna(0).astype("int8")
    df["source_hhmm"] = df["source_hhmm"].fillna(0).astype("int16")
    return df


def lane_to_station_5m(
    lane_df: pd.DataFrame,
    speed_max_kmh: float = DEFAULT_SPEED_MAX_KMH,
) -> pd.DataFrame:
    """Aggregate lane rows to station 5-minute rows.

    Mirrors `build_station_vd_drive.build_station_1m` + `build_station_5m`
    cleaning rules so the live data has the same schema as the training cache:
      * keep lane rows with `0 < speed <= speed_max_kmh`
      * occupancy only contributes to the mean when in `[0, 100]`
      * vol_* clipped to >= 0 before summing
    """
    if lane_df.empty:
        return pd.DataFrame(columns=STATION_5M_COLUMNS)

    valid = (lane_df["speed"] > 0) & (lane_df["speed"] <= speed_max_kmh)
    df = lane_df.loc[valid].copy()
    if df.empty:
        return pd.DataFrame(columns=STATION_5M_COLUMNS)

    for col in ("vol_s", "vol_l", "vol_t"):
        df[col] = df[col].clip(lower=0).astype("int32")
    occupancy = pd.to_numeric(df["occupancy"], errors="coerce")
    df["occupancy_for_mean"] = occupancy.where((occupancy >= 0) & (occupancy <= 100))

    keys_1m = ["source_date", "timestamp", "vdid", "freeway", "direction", "mileage_m"]
    s1m = (
        df.groupby(keys_1m, observed=True)
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
    s1m["speed_x10"] = np.rint(s1m["speed"] * 10).clip(0, speed_max_kmh * 10).astype("int16")
    s1m["occupancy_x10"] = np.rint(s1m["occupancy"].fillna(0) * 10).clip(0, 1000).astype("int16")
    s1m["timestamp_5m"] = pd.to_datetime(s1m["timestamp"]).dt.floor("5min")

    keys_5m = ["source_date", "timestamp_5m", "vdid", "freeway", "direction", "mileage_m"]
    s5m = (
        s1m.groupby(keys_5m, observed=True)
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
    s5m["speed_x10"] = np.rint(s5m["speed_x10"]).clip(0, speed_max_kmh * 10).astype("int16")
    s5m["occupancy_x10"] = np.rint(s5m["occupancy_x10"]).clip(0, 1000).astype("int16")
    s5m["vol_s"] = s5m["vol_s"].astype("int32")
    s5m["vol_l"] = s5m["vol_l"].astype("int32")
    s5m["vol_t"] = s5m["vol_t"].astype("int32")
    s5m["n_lanes_valid"] = s5m["n_lanes_valid"].clip(0, np.iinfo(np.int16).max).astype("int16")

    return (
        s5m[STATION_5M_COLUMNS]
        .sort_values(["source_date", "freeway", "direction", "mileage_m", "vdid", "timestamp"])
        .reset_index(drop=True)
    )


def fetch_live_station_5m(
    start: datetime,
    end: datetime,
    freeway: str,
    direction: str,
    mileage_m: int,
    *,
    upstream_km: int = 50,
    downstream_km: int = 10,
    concurrency: int = DEFAULT_CONCURRENCY,
    retries: int = DEFAULT_RETRIES,
) -> tuple[pd.DataFrame, FetchStatus]:
    """Top-level synchronous entry point used by the GUI / inference layer.

    Args mirror the accident location so this function can be used directly
    as the `fetch_live_vd` callable in `predict_accident_impact`. Returns the
    station-5m DataFrame ready to feed into the model assembly step plus a
    `FetchStatus` for the coverage panel.
    """
    vd_filter = make_accident_vd_filter(
        freeway=freeway,
        direction=direction,
        mileage_m=mileage_m,
        upstream_km=upstream_km,
        downstream_km=downstream_km,
    )
    rows, counts = asyncio.run(
        fetch_lane_rows(start, end, vd_filter, concurrency=concurrency, retries=retries)
    )
    lane_df = lane_rows_to_dataframe(rows)
    station_5m = lane_to_station_5m(lane_df)

    status = FetchStatus(
        minutes_requested=counts.get("minutes_requested", 0),
        minutes_downloaded=counts.get("ok", 0),
        minutes_404=counts.get("404", 0),
        minutes_error=counts.get("error", 0),
        parse_errors=counts.get("parse_error", 0),
        lane_rows=int(len(lane_df)),
        station_5m_rows=int(len(station_5m)),
        unique_vds=int(lane_df["vdid"].nunique()) if not lane_df.empty else 0,
    )
    return station_5m, status

