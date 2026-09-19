# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "pandas>=2.0",
#     "pyarrow>=15",
# ]
# ///
"""Build per-accident weather candidate rows from the CWA hourly corridor archive.

Produces the timestamped weather artifact that `build_incident_features.py --weather-csv`
requires for causal matching: several candidate observation rows per accident, each with
the observation timestamp, so the builder can pick the latest observation *before* the
report time within its staleness limit (default 90 min).

Per accident:
1. locate the accident on the corridor: nearest mainline VD on the same freeway/direction
   by mileage -> lat/lon (from `data/metadata/vd_static.xml`),
2. rank CWA stations by distance; for each variable take the nearest station (within
   --max-station-km) that has a non-null value in the lookback window, so a rain-only
   automatic station can supply precipitation while a manned station supplies temperature,
3. emit one row per hourly observation in [report - lookback, report + 1h] (the builder
   ignores post-report rows; they are kept only to audit `has_future_candidate`).

Inputs:  data/weather/cwa_hourly_corridor.parquet, data/weather/cwa_stations_corridor.csv,
         an accident CSV with 任務編號/國道/方向/里程數/發送 (dispatch schema; SMS output is compatible)
Output:  data/weather/weather_per_task_observations.csv  (+ _coverage.json)

Usage:
  uv run scripts/features/build_weather_candidates.py
  uv run scripts/features/build_weather_candidates.py --accident-csv data/accident_data_sms.csv --output data/weather/weather_per_task_observations_sms.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEATHER_DIR = PROJECT_ROOT / "data" / "weather"
VD_STATIC_XML = PROJECT_ROOT / "data" / "metadata" / "vd_static.xml"
DIRECTION_ZH = {"北": "N", "南": "S", "東": "E", "西": "W", "N": "N", "S": "S", "E": "E", "W": "W"}
MAINLINE_RE = re.compile(r"^VD-(N\d+)-([NSEW])-([\d.]+)-(M|N)-")
VARS = {"temp_c": "temp_c", "rh_pct": "humidity_pct", "wind_mps": "wind_speed_mps", "precip_mm": "rain_mm", "visibility_km": "visibility_km"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accident-csv", default=str(PROJECT_ROOT / "data" / "accident_data.csv"))
    p.add_argument("--hourly", default=str(WEATHER_DIR / "cwa_hourly_corridor.parquet"))
    p.add_argument("--stations", default=str(WEATHER_DIR / "cwa_stations_corridor.csv"))
    p.add_argument("--vd-static", default=str(VD_STATIC_XML))
    p.add_argument("--output", default=str(WEATHER_DIR / "weather_per_task_observations.csv"))
    p.add_argument("--lookback-hours", type=int, default=3)
    p.add_argument("--max-station-km", type=float, default=20.0)
    p.add_argument("--max-station-km-visibility", type=float, default=40.0, help="Visibility is only measured at manned stations; allow a wider search")
    return p.parse_args()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def load_vd_coords(xml_path: Path) -> pd.DataFrame:
    ns = {"m": "http://traffic.transportdata.tw/standard/traffic/schema/"}
    root = ET.parse(xml_path).getroot()
    rows = []
    for vd in root.iterfind(".//m:VD", ns):
        vdid = vd.findtext("m:VDID", namespaces=ns) or ""
        m = MAINLINE_RE.match(vdid)
        if not m:
            continue
        lat = float(vd.findtext("m:PositionLat", namespaces=ns) or 0)
        lon = float(vd.findtext("m:PositionLon", namespaces=ns) or 0)
        if lat and lon:
            rows.append({"vdid": vdid, "freeway": m.group(1), "direction": m.group(2), "km": float(m.group(3)), "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def load_accidents(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df = pd.DataFrame(
        {
            "task_id": raw["任務編號"].str.strip(),
            "freeway": raw["國道"].str.strip(),
            "direction": raw["方向"].str.strip().map(DIRECTION_ZH),
            "km": pd.to_numeric(raw["里程數"], errors="coerce"),
            "report_time": pd.to_datetime(raw["發送"], errors="coerce"),
        }
    )
    return df.dropna(subset=["task_id", "freeway", "direction", "km", "report_time"]).drop_duplicates("task_id")


def main() -> None:
    args = parse_args()
    acc = load_accidents(Path(args.accident_csv))
    vds = load_vd_coords(Path(args.vd_static))
    stations = pd.read_csv(args.stations, dtype={"station_id": str})
    hourly = pd.read_parquet(args.hourly)
    hourly["time"] = pd.to_datetime(hourly["time"])
    print(f"[in] accidents {len(acc)}, VDs {len(vds)}, stations {len(stations)}, hourly rows {len(hourly)}")

    # accident -> coordinates via nearest mainline VD on the same freeway/direction
    lat = np.full(len(acc), np.nan)
    lon = np.full(len(acc), np.nan)
    vd_gap = np.full(len(acc), np.nan)
    for (fw, dr), grp in acc.groupby(["freeway", "direction"]):
        cand = vds[(vds.freeway == fw) & (vds.direction == dr)]
        if cand.empty:
            continue
        ck = cand["km"].to_numpy()
        for i, km in zip(grp.index, grp["km"].to_numpy()):
            j = int(np.argmin(np.abs(ck - km)))
            pos = acc.index.get_loc(i)
            lat[pos], lon[pos], vd_gap[pos] = cand.iloc[j]["lat"], cand.iloc[j]["lon"], abs(ck[j] - km)
    acc["lat"], acc["lon"], acc["vd_gap_km"] = lat, lon, vd_gap
    located = acc.dropna(subset=["lat"]).copy()
    print(f"[loc] located {len(located)}/{len(acc)} accidents (median VD gap {np.nanmedian(vd_gap):.2f} km)")

    # per-station hourly index for fast slicing
    by_station = {sid: g.set_index("time").sort_index() for sid, g in hourly.groupby("station_id")}
    st_lat, st_lon, st_ids = stations["lat"].to_numpy(), stations["lon"].to_numpy(), stations["station_id"].to_numpy()

    rows = []
    coverage = {v: 0 for v in VARS}
    n_any = 0
    lookback = pd.Timedelta(hours=args.lookback_hours)
    for a in located.itertuples(index=False):
        d = haversine_km(a.lat, a.lon, st_lat, st_lon)
        order = np.argsort(d)
        t0, t1 = a.report_time - lookback, a.report_time + pd.Timedelta(hours=1)
        chosen: dict[str, tuple[str, float, pd.Series]] = {}
        for var in VARS:
            limit = args.max_station_km_visibility if var == "visibility_km" else args.max_station_km
            for j in order:
                if d[j] > limit:
                    break
                g = by_station.get(st_ids[j])
                if g is None:
                    continue
                win = g.loc[t0:t1, var].dropna()
                if not win.empty and (win.index < a.report_time).any():
                    chosen[var] = (st_ids[j], float(d[j]), win)
                    break
        if not chosen:
            continue
        n_any += 1
        for var in chosen:
            coverage[var] += 1
        # union of observation times across chosen stations; one row per time with each var from its own station
        times = sorted({t for _, _, w in chosen.values() for t in w.index})
        for t in times:
            row = {"task_id": a.task_id, "observation_time": t.isoformat(), "report_time": a.report_time.isoformat()}
            for var, out_name in VARS.items():
                if var in chosen:
                    sid, dist, w = chosen[var]
                    row[out_name] = float(w.get(t, np.nan))
                    row[f"{out_name}_station"] = sid
                    row[f"{out_name}_station_km"] = round(dist, 2)
                else:
                    row[out_name] = np.nan
            rows.append(row)

    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accident_csv": str(args.accident_csv),
        "n_accidents": int(len(acc)),
        "n_located": int(len(located)),
        "n_with_any_weather_candidate": n_any,
        "coverage_by_variable": {VARS[v]: {"n": c, "share": c / max(len(acc), 1)} for v, c in coverage.items()},
        "lookback_hours": args.lookback_hours,
        "max_station_km": args.max_station_km,
        "rows": int(len(out)),
        "station_km": {
            out_name: {"median": float(out[f"{out_name}_station_km"].median()), "p90": float(out[f"{out_name}_station_km"].quantile(0.9))}
            for out_name in VARS.values()
            if f"{out_name}_station_km" in out.columns
        },
    }
    Path(str(out_path).replace(".csv", "_coverage.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
