# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "pandas>=2.0",
#     "pyarrow>=15",
# ]
# ///
"""Fetch hourly CWA station observations for the study corridor from the public CODiS archive.

Source: the community-maintained mirror of CWA CODiS station data
(https://github.com/Raingel/historical_weather), one CSV per station-year at
  https://raw.githubusercontent.com/Raingel/historical_weather/main/data/<station>/<station>_<year>.csv
plus the CODiS station list (https://codis.cwa.gov.tw/api/station_list) for coordinates.
No API key is needed.  Both manned (46xxxx) and automatic (C0xxxx) stations are present.

Steps
1. Load VD coordinates from `data/metadata/vd_static.xml`, keep mainline VDs inside the paper
   corridor (N1 100.8-251.1 km, N3 110.9-269.0 km).
2. Select CWA stations whose distance to the nearest corridor VD is <= --radius-km and that were
   active during --start-year..--end-year.
3. Download <station>_<year>.csv for each selected station-year (skips files already present).
4. Build a tidy parquet: station_id, time, temp_c, rh_pct, wind_mps, wind_dir_deg, precip_mm,
   visibility_km, pressure_hpa (CODiS sentinels like -99.x -> NaN).
5. Write a station table (with nearest-VD distance) and a coverage report.

Outputs (defaults):
  --raw-dir     data/weather/cwa_hourly_raw/<station>/<station>_<year>.csv
  --output-dir  data/weather/  -> cwa_hourly_corridor.parquet, cwa_stations_corridor.csv,
                                  cwa_fetch_report.json, CWA_FETCH_SUMMARY.md

Usage:
  uv run scripts/data/fetch_cwa_hourly_archive.py --start-year 2023 --end-year 2026
  uv run scripts/data/fetch_cwa_hourly_archive.py --radius-km 20 --raw-dir D:/vd_audit/weather_raw
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import urllib.error
import urllib.request

import numpy as np
import pandas as pd


class _Resp:
    def __init__(self, status: int, content: bytes):
        self.status_code = status
        self.content = content


def http_get(url: str, timeout: int = 120) -> _Resp:
    """Minimal stdlib GET returning (status, bytes); 404 is returned, not raised."""
    req = urllib.request.Request(url, headers={"User-Agent": "highway-accident-audit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _Resp(r.status, r.read())
    except urllib.error.HTTPError as exc:
        return _Resp(exc.code, b"")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import xml.etree.ElementTree as ET  # noqa: E402

VD_STATIC_XML = PROJECT_ROOT / "data" / "metadata" / "vd_static.xml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "weather" / "cwa_hourly_raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "weather"
STATION_LIST_URL = "https://codis.cwa.gov.tw/api/station_list"
ARCHIVE_URL = "https://raw.githubusercontent.com/Raingel/historical_weather/main/data/{sid}/{sid}_{year}.csv"
CORRIDOR = {"N1": (100.8, 251.1), "N3": (110.9, 269.0)}
MAINLINE_RE = re.compile(r"^VD-N\d+-[NSEW]-[\d.]+-(M|N)-")

# CODiS column -> tidy name (legacy names first, then the new-style names used in recent years)
COLUMN_MAP = {
    "Tx": "temp_c",
    "AirTemperature.Instantaneous": "temp_c",
    "RH": "rh_pct",
    "RelativeHumidity.Instantaneous": "rh_pct",
    "WS": "wind_mps",
    "WindSpeed.Mean": "wind_mps",
    "WD": "wind_dir_deg",
    "WindDirection.Mean": "wind_dir_deg",
    "Precp": "precip_mm",
    "Precipitation.Accumulation": "precip_mm",
    "Visb": "visibility_km",
    "Visibility.Instantaneous": "visibility_km",
    "Visibility.AutoMean": "visibility_km",
    "StnPres": "pressure_hpa",
    "StationPressure.Instantaneous": "pressure_hpa",
}
TIDY_COLS = ["temp_c", "rh_pct", "wind_mps", "wind_dir_deg", "precip_mm", "visibility_km", "pressure_hpa"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, default=2023)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--radius-km", type=float, default=15.0, help="Keep stations within this distance of any corridor mainline VD")
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--vd-static", default=str(VD_STATIC_XML))
    p.add_argument("--sleep", type=float, default=0.3, help="Seconds between downloads")
    p.add_argument("--force", action="store_true", help="Re-download existing raw files")
    return p.parse_args()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def parse_vd_xml(xml_path: Path) -> list[dict]:
    """Minimal stdlib parser for the MOTC VD static list (vdid, lat, lon)."""
    ns = {"m": "http://traffic.transportdata.tw/standard/traffic/schema/"}
    root = ET.parse(xml_path).getroot()
    out = []
    for vd in root.iterfind(".//m:VD", ns):
        out.append(
            {
                "vdid": vd.findtext("m:VDID", namespaces=ns) or "",
                "lat": float(vd.findtext("m:PositionLat", namespaces=ns) or 0),
                "lon": float(vd.findtext("m:PositionLon", namespaces=ns) or 0),
            }
        )
    return out


def corridor_vds(xml_path: Path) -> pd.DataFrame:
    recs = pd.DataFrame(parse_vd_xml(xml_path))
    recs = recs[(recs.lat != 0) & (recs.lon != 0)]
    m = recs["vdid"].str.extract(r"^VD-(N\d+)-([NSEW])-([\d.]+)-")
    recs["freeway"], recs["direction"], recs["km"] = m[0], m[1], pd.to_numeric(m[2], errors="coerce")
    keep = pd.Series(False, index=recs.index)
    for fw, (lo, hi) in CORRIDOR.items():
        keep |= (recs.freeway == fw) & (recs.km >= lo) & (recs.km <= hi)
    keep &= recs["vdid"].str.match(MAINLINE_RE)
    return recs[keep][["vdid", "freeway", "direction", "km", "lat", "lon"]].reset_index(drop=True)


def load_station_list() -> pd.DataFrame:
    r = http_get(STATION_LIST_URL, timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"station list fetch failed: http {r.status_code}")
    js = json.loads(r.content.decode("utf-8"))
    rows = []
    for group in js["data"]:
        attr = group.get("stationAttribute")
        for it in group["item"]:
            rows.append(
                {
                    "station_id": it["stationID"],
                    "station_name": it.get("stationName"),
                    "attribute": attr,
                    "lat": it.get("latitude"),
                    "lon": it.get("longitude"),
                    "altitude_m": it.get("altitude"),
                    "county": it.get("countryName"),
                    "start_date": it.get("stationStartDate") or None,
                    "end_date": it.get("stationEndDate") or None,
                }
            )
    df = pd.DataFrame(rows)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["lat", "lon"])


def select_stations(stations: pd.DataFrame, vds: pd.DataFrame, radius_km: float, start_year: int, end_year: int) -> pd.DataFrame:
    active = stations[(stations.end_date.isna()) | (pd.to_datetime(stations.end_date, errors="coerce") >= pd.Timestamp(f"{start_year}-01-01"))]
    active = active[(active.start_date.isna()) | (pd.to_datetime(active.start_date, errors="coerce") <= pd.Timestamp(f"{end_year}-12-31"))]
    vlat, vlon = vds["lat"].to_numpy(), vds["lon"].to_numpy()
    out = []
    for s in active.itertuples(index=False):
        d = haversine_km(s.lat, s.lon, vlat, vlon)
        j = int(np.argmin(d))
        if d[j] <= radius_km:
            out.append({**s._asdict(), "nearest_vd": vds.iloc[j]["vdid"], "nearest_vd_km": float(d[j])})
    return pd.DataFrame(out).sort_values("nearest_vd_km").reset_index(drop=True)


def read_codis_csv(path: Path, sid: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    time_col = df.columns[0]
    out = pd.DataFrame({"station_id": sid, "time": pd.to_datetime(df[time_col], errors="coerce")})
    for src, dst in COLUMN_MAP.items():
        if src in df.columns:
            vals = pd.to_numeric(df[src], errors="coerce")
            if dst in out.columns:
                out[dst] = out[dst].fillna(vals)
            else:
                out[dst] = vals
    for c in TIDY_COLS:
        if c not in out.columns:
            out[c] = np.nan
    # CODiS sentinels: -9.x, -99.x, -999 -> missing; trace precipitation 'T' becomes NaN via to_numeric -> treat as 0.05
    for c in TIDY_COLS:
        out.loc[out[c] < -1, c] = np.nan
    out["precip_mm"] = out["precip_mm"].clip(lower=0)
    return out.dropna(subset=["time"])[["station_id", "time", *TIDY_COLS]]


def main() -> None:
    args = parse_args()
    t0 = time.time()
    raw_dir, out_dir = Path(args.raw_dir), Path(args.output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(args.start_year, args.end_year + 1))

    vds = corridor_vds(Path(args.vd_static))
    print(f"[vd] {len(vds)} corridor mainline VDs with coordinates")
    stations = load_station_list()
    print(f"[cwa] station list: {len(stations)} stations")
    sel = select_stations(stations, vds, args.radius_km, args.start_year, args.end_year)
    print(f"[cwa] {len(sel)} stations within {args.radius_km} km of the corridor")

    fetched, missing, failed = [], [], []
    for s in sel.itertuples(index=False):
        for y in years:
            dest = raw_dir / s.station_id / f"{s.station_id}_{y}.csv"
            if dest.exists() and not args.force:
                fetched.append((s.station_id, y))
                continue
            url = ARCHIVE_URL.format(sid=s.station_id, year=y)
            try:
                r = http_get(url, timeout=120)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                failed.append((s.station_id, y, repr(exc)))
                continue
            if r.status_code == 404:
                missing.append((s.station_id, y))
                continue
            if r.status_code != 200:
                failed.append((s.station_id, y, f"http {r.status_code}"))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            fetched.append((s.station_id, y))
            time.sleep(args.sleep)
        print(f"[fetch] {s.station_id} {s.station_name}: {sum(1 for f in fetched if f[0]==s.station_id)}/{len(years)} years")

    frames = []
    for sid, y in fetched:
        p = raw_dir / sid / f"{sid}_{y}.csv"
        try:
            frames.append(read_codis_csv(p, sid))
        except Exception as exc:  # noqa: BLE001
            failed.append((sid, y, f"parse: {exc!r}"))
    tidy = pd.concat(frames, ignore_index=True).sort_values(["station_id", "time"]) if frames else pd.DataFrame(columns=["station_id", "time", *TIDY_COLS])
    tidy = tidy.drop_duplicates(["station_id", "time"])
    tidy.to_parquet(out_dir / "cwa_hourly_corridor.parquet", index=False)

    # coverage per station-year
    tidy["year"] = tidy["time"].dt.year
    cov = tidy.groupby(["station_id", "year"]).agg(n_hours=("time", "size"), **{f"{c}_fill": (c, lambda s: float(s.notna().mean())) for c in TIDY_COLS}).reset_index()
    cov.to_csv(out_dir / "cwa_hourly_coverage.csv", index=False)
    stn = sel.merge(cov.groupby("station_id").agg(n_years=("year", "nunique"), temp_fill=("temp_c_fill", "mean"), precip_fill=("precip_mm_fill", "mean"), vis_fill=("visibility_km_fill", "mean")).reset_index(), on="station_id", how="left")
    stn.to_csv(out_dir / "cwa_stations_corridor.csv", index=False)

    # nearest-station distance per corridor VD (using stations that actually have data)
    good = stn[stn.n_years.fillna(0) > 0]
    d = haversine_km(vds["lat"].to_numpy()[:, None], vds["lon"].to_numpy()[:, None], good["lat"].to_numpy()[None, :], good["lon"].to_numpy()[None, :])
    vds["nearest_station_km"] = d.min(axis=1)
    vds["nearest_station"] = good["station_id"].to_numpy()[d.argmin(axis=1)]
    vds.to_csv(out_dir / "corridor_vd_nearest_station.csv", index=False)

    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "years": years,
        "radius_km": args.radius_km,
        "n_corridor_vds": int(len(vds)),
        "n_stations_selected": int(len(sel)),
        "n_stations_with_data": int(len(good)),
        "station_years_fetched": len(fetched),
        "station_years_missing_404": len(missing),
        "failures": failed[:20],
        "tidy_rows": int(len(tidy)),
        "time_span": [str(tidy.time.min()), str(tidy.time.max())] if len(tidy) else None,
        "vd_nearest_station_km": {"median": float(vds.nearest_station_km.median()), "p90": float(vds.nearest_station_km.quantile(0.9)), "max": float(vds.nearest_station_km.max())},
        "fill_rates_overall": {c: float(tidy[c].notna().mean()) for c in TIDY_COLS} if len(tidy) else None,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "cwa_fetch_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = [
        "# CWA hourly archive fetch — corridor stations",
        "",
        f"Run: {report['run_utc']}  ·  years {years[0]}–{years[-1]}  ·  radius {args.radius_km} km",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Corridor mainline VDs with coordinates | {report['n_corridor_vds']} |",
        f"| Stations selected / with data | {report['n_stations_selected']} / {report['n_stations_with_data']} |",
        f"| Station-years fetched / 404 | {report['station_years_fetched']} / {report['station_years_missing_404']} |",
        f"| Tidy rows | {report['tidy_rows']} |",
        f"| Time span | {report['time_span']} |",
        f"| VD → nearest station km (median / P90 / max) | {report['vd_nearest_station_km']['median']:.1f} / {report['vd_nearest_station_km']['p90']:.1f} / {report['vd_nearest_station_km']['max']:.1f} |",
        "",
        "## Fill rates (all selected station-hours)",
        "",
        "| Variable | Fill |",
        "| --- | ---: |",
    ]
    if report["fill_rates_overall"]:
        lines += [f"| {c} | {v:.3f} |" for c, v in report["fill_rates_overall"].items()]
    lines += ["", "## Stations", "", "| Station | Name | Type | Nearest VD | km | Years | Temp fill | Precip fill | Vis fill |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for s in stn.itertuples(index=False):
        lines.append(f"| {s.station_id} | {s.station_name} | {s.attribute} | {s.nearest_vd} | {s.nearest_vd_km:.1f} | {int(s.n_years) if pd.notna(s.n_years) else 0} | {s.temp_fill if pd.notna(s.temp_fill) else float('nan'):.2f} | {s.precip_fill if pd.notna(s.precip_fill) else float('nan'):.2f} | {s.vis_fill if pd.notna(s.vis_fill) else float('nan'):.2f} |")
    (out_dir / "CWA_FETCH_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[: lines.index("## Stations")]))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
