# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "openpyxl",
#     "pandas",
#     "pyarrow",
# ]
# ///
# ruff: noqa: E402
"""
Process the 交控事故簡訊 (SMS) incident records into a dispatch-compatible CSV.

The SMS source (`data/accident_交控事故簡訊紀錄表/*.xlsx`) covers the national
freeway network and overlaps the dispatch range (2024-04-01 .. 2025-05-31).
Per the project decision, SMS is only used for the period OUTSIDE dispatch
coverage (default: report date < 2024-04-01); inside the overlap the dispatch
records are authoritative and SMS is ignored.

Output is written with the SAME column names that `accident_data.csv` uses, so
`scripts/data/build_accident_samples.py` and `scripts/features/build_incident_features.py`
can consume it unchanged (point them at `--accident-csv`).

Key field mapping (SMS -> dispatch schema):
  任務編號        <- synthetic "SMS"+YYYYMMDD+seq
  國道            <- 國道名稱 (國道1號 -> N1)
  方向            <- 方向 (北向 -> N / 南向 -> S / 東向 -> E / 西向 -> W)
  里程數          <- 里程 (km)
  發送 (report)   <- date(年/月/日) + 事件發生 (HH:MM:SS)  [occurrence time]
  處理完成        <- date + 事件排除 (HH:MM:SS), cross-midnight aware
  受傷人數        <- 受傷    (blank stays blank -> avail=0 downstream)
  死亡人數        <- 死亡    (blank stays blank -> avail=0 downstream)
  事故類別        <- 事故類型 (A1/A2/A3 or blank -> avail=0 for blanks)
  主線車道數      <- geometry 車道數 lookup by (freeway, direction, mileage)
  剩餘車道數      <- num_lanes - blocked main lanes (內/中內/中/中外/外車道)
  位置            <- 簡訊內容 (scanned by the feature builder for special-accident keywords)
  性質            <- "交通事故"

Usage:
  uv run scripts/data/process_sms_data.py
  uv run scripts/data/process_sms_data.py --end-date-exclusive 2024-04-01 --no-central-only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.features.build_incident_features import (
    load_geometry_records,
    select_geometry_record,
)

DEFAULT_SMS_DIR = PROJECT_ROOT / "data" / "accident_交控事故簡訊紀錄表"
DEFAULT_GEOMETRY_DIR = PROJECT_ROOT / "data" / "geometry"
DEFAULT_VD_DIM = PROJECT_ROOT / "data" / "metadata" / "vd_dim.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "accident_data_sms.csv"

# Dispatch range; SMS inside this window is ignored (dispatch is authoritative).
DISPATCH_START = "2024-04-01"

FREEWAY_RE = re.compile(r"(\d+)")
DIRECTION_MAP = {"北向": "N", "南向": "S", "東向": "E", "西向": "W",
                 "北": "N", "南": "S", "東": "E", "西": "W"}
MAIN_LANE_COLS = ["內車道", "中內車道", "中車道", "中外車道", "外車道"]

# SMS has structured special-accident flag columns. Map each set flag to a
# keyword token recognised by build_incident_features.SPECIAL_INCIDENT_PATTERNS,
# so special_accident reflects SMS's structured signal rather than only the text.
FLAG_TOKEN_MAP = {
    "翻覆事故註記": "翻覆",
    "車輛起火註記": "起火",
    "危險物品車輛註記": "危險物",
}

# Output columns (dispatch-compatible subset that downstream scripts read).
OUTPUT_COLUMNS = [
    "任務編號", "國道", "方向", "里程數", "路段", "位置",
    "主線車道數", "輔助車道數", "剩餘車道數", "性質", "事故類別", "車種",
    "發送", "到達", "處理完成", "開始清理", "受傷人數", "死亡人數",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standardize SMS incident records into a dispatch-compatible CSV.")
    p.add_argument("--sms-dir", default=str(DEFAULT_SMS_DIR))
    p.add_argument("--geometry-dir", default=str(DEFAULT_GEOMETRY_DIR))
    p.add_argument("--vd-dim", default=str(DEFAULT_VD_DIM))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--end-date-exclusive", default=DISPATCH_START,
                   help="Keep SMS with report date strictly before this date (default: dispatch start).")
    p.add_argument("--start-date", default=None, help="Optional inclusive lower bound on report date.")
    p.add_argument("--central-only", dest="central_only", action="store_true", default=True,
                   help="Keep only SMS within observed VD mileage coverage per freeway+direction (default on).")
    p.add_argument("--no-central-only", dest="central_only", action="store_false")
    p.add_argument("--coverage-buffer-km", type=float, default=5.0,
                   help="Mileage buffer (km) added to the VD coverage range for the central filter.")
    return p.parse_args()


def standardize_freeway(raw: object) -> str | None:
    text = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw)
    m = FREEWAY_RE.search(text)
    return f"N{m.group(1)}" if m else None


_MILEAGE_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_mileage_km(raw: object) -> float:
    """SMS 里程 may be plain ('154.6') or unit-suffixed ('158.5公里', '132公里')."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    m = _MILEAGE_RE.search(str(raw))
    return float(m.group()) if m else np.nan


def roc_or_ad_year(value: object) -> float:
    """SMS '年' mixes ROC (112/113/114) and AD (2024/2025). Normalize to AD."""
    y = pd.to_numeric(value, errors="coerce")
    if pd.isna(y):
        return np.nan
    return float(y + 1911) if y < 1911 else float(y)


def _combine_date_time(date: pd.Timestamp, time_val: object) -> pd.Timestamp:
    if pd.isna(date) or time_val is None or (isinstance(time_val, float) and np.isnan(time_val)):
        return pd.NaT
    t = pd.to_datetime(str(time_val), errors="coerce")
    if pd.isna(t):
        return pd.NaT
    return pd.Timestamp(year=date.year, month=date.month, day=date.day,
                        hour=t.hour, minute=t.minute, second=t.second)


def load_sms_frames(sms_dir: Path) -> pd.DataFrame:
    files = sorted(sms_dir.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"No SMS xlsx files found in {sms_dir}")
    frames = []
    for f in files:
        df = pd.read_excel(f, engine="openpyxl")
        df.columns = [str(c).replace("\n", "").strip() for c in df.columns]
        df["__file"] = f.name
        frames.append(df)
        print(f"  {f.name}: {len(df)} rows")
    return pd.concat(frames, ignore_index=True)


def coverage_ranges(vd_dim_path: Path, buffer_km: float) -> dict[tuple[str, str], tuple[int, int]]:
    if not vd_dim_path.exists():
        print(f"[WARN] vd_dim not found at {vd_dim_path}; central filter disabled")
        return {}
    dim = pd.read_parquet(vd_dim_path, columns=["freeway", "direction", "mileage_m"])
    buf = int(buffer_km * 1000)
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for (fwy, direction), grp in dim.groupby(["freeway", "direction"]):
        out[(str(fwy), str(direction))] = (int(grp["mileage_m"].min()) - buf, int(grp["mileage_m"].max()) + buf)
    return out


def main() -> None:
    args = parse_args()
    sms_dir = Path(args.sms_dir)
    geometry_dir = Path(args.geometry_dir)
    output_path = Path(args.output)

    print(f"Reading SMS files from {sms_dir}")
    raw = load_sms_frames(sms_dir)
    print(f"Total raw SMS rows: {len(raw)}")

    df = pd.DataFrame(index=raw.index)
    df["freeway"] = raw["國道名稱"].map(standardize_freeway)
    df["direction"] = raw["方向"].map(lambda x: DIRECTION_MAP.get(str(x).strip(), None) if pd.notna(x) else None)
    df["mileage_km"] = raw["里程"].map(parse_mileage_km)

    year = raw["年"].map(roc_or_ad_year)
    date = pd.to_datetime(dict(
        year=year,
        month=pd.to_numeric(raw["月"], errors="coerce"),
        day=pd.to_numeric(raw["日"], errors="coerce"),
    ), errors="coerce")
    df["report_time"] = [_combine_date_time(d, t) for d, t in zip(date, raw["事件發生"])]
    cleared = [_combine_date_time(d, t) for d, t in zip(date, raw["事件排除"])]
    # Cross-midnight: cleared earlier than report -> rolled past midnight.
    df["cleared_time"] = [
        (c + pd.Timedelta(days=1)) if (pd.notna(c) and pd.notna(r) and c < r) else c
        for c, r in zip(cleared, df["report_time"])
    ]

    df["injuries"] = pd.to_numeric(raw["受傷"], errors="coerce")
    df["fatalities"] = pd.to_numeric(raw["死亡"], errors="coerce")
    df["accident_class"] = raw["事故類型"].astype(str).str.strip().where(raw["事故類型"].notna(), other=np.nan)
    sms_text = raw["簡訊內容"].astype(str).where(raw["簡訊內容"].notna(), other="")
    flag_tokens = pd.Series("", index=raw.index)
    for col, token in FLAG_TOKEN_MAP.items():
        if col in raw.columns:
            is_set = pd.to_numeric(raw[col], errors="coerce").fillna(0) > 0
            flag_tokens = flag_tokens.where(~is_set, flag_tokens + " " + token)
    df["sms_text"] = (sms_text + " " + flag_tokens).str.strip()

    blocked_main = sum(
        pd.to_numeric(raw[c], errors="coerce").fillna(0).clip(upper=1).astype(int)
        for c in MAIN_LANE_COLS if c in raw.columns
    )
    df["blocked_main"] = blocked_main

    # --- filter: drop rows without the essentials for VD-window alignment ---
    before = len(df)
    df = df[df["freeway"].notna() & df["direction"].notna()
            & df["mileage_km"].notna() & df["report_time"].notna()].copy()
    print(f"Dropped {before - len(df)} rows missing freeway/direction/mileage/report_time")

    # --- time filter: only the period OUTSIDE dispatch coverage ---
    end_excl = pd.Timestamp(args.end_date_exclusive)
    mask = df["report_time"] < end_excl
    if args.start_date:
        mask &= df["report_time"] >= pd.Timestamp(args.start_date)
    df = df[mask].copy()
    print(f"After non-dispatch time filter (< {end_excl.date()}"
          f"{', >= ' + args.start_date if args.start_date else ''}): {len(df)} rows")

    # --- central VD coverage filter ---
    if args.central_only:
        ranges = coverage_ranges(Path(args.vd_dim), args.coverage_buffer_km)
        if ranges:
            mileage_m = (df["mileage_km"] * 1000).round().astype(int)
            keep = []
            for fwy, direction, mm in zip(df["freeway"], df["direction"], mileage_m):
                rng = ranges.get((fwy, direction))
                keep.append(rng is not None and rng[0] <= mm <= rng[1])
            kept = int(np.sum(keep))
            print(f"Central VD coverage filter: kept {kept} / {len(df)}")
            df = df[pd.Series(keep, index=df.index)].copy()

    if df.empty:
        raise SystemExit("No SMS rows left after filtering.")

    df = df.sort_values("report_time").reset_index(drop=True)

    # --- geometry lane-count lookup -> 主線車道數 / 剩餘車道數 ---
    geom = load_geometry_records(geometry_dir) if geometry_dir.exists() else {}
    num_lanes_list: list[float] = []
    remain_list: list[float] = []
    for fwy, direction, km, blk in zip(df["freeway"], df["direction"], df["mileage_km"], df["blocked_main"]):
        hwy_digit = fwy[1:] if fwy.startswith("N") else fwy
        records = geom.get(hwy_digit, {}).get(direction, [])
        rec, _ = select_geometry_record(records, float(km))
        nl = int(rec.num_lanes) if (rec is not None and rec.num_lanes > 0) else 0
        if nl > 0:
            num_lanes_list.append(nl)
            remain_list.append(max(nl - int(blk), 0))
        else:
            num_lanes_list.append(np.nan)   # unknown -> avail_lane_blockage_rate=0 downstream
            remain_list.append(np.nan)

    # --- synthetic accident ids: SMS + YYYYMMDD + per-day sequence ---
    day_key = df["report_time"].dt.strftime("%Y%m%d")
    seq = day_key.groupby(day_key).cumcount() + 1
    accident_id = "SMS" + day_key + seq.map(lambda s: f"{s:03d}")

    out = pd.DataFrame()
    out["任務編號"] = accident_id
    out["國道"] = df["freeway"]
    out["方向"] = df["direction"]
    out["里程數"] = df["mileage_km"]
    out["路段"] = ""
    out["位置"] = df["sms_text"]            # special-accident keyword source
    out["主線車道數"] = num_lanes_list
    out["輔助車道數"] = ""
    out["剩餘車道數"] = remain_list
    out["性質"] = "交通事故"
    out["事故類別"] = df["accident_class"]
    out["車種"] = ""
    out["發送"] = df["report_time"]
    out["到達"] = pd.NaT
    out["處理完成"] = df["cleared_time"]
    out["開始清理"] = pd.NaT
    out["受傷人數"] = df["injuries"]
    out["死亡人數"] = df["fatalities"]

    out = out[OUTPUT_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("SMS STANDARDIZATION SUMMARY")
    print("=" * 72)
    print(f"Output: {output_path}")
    print(f"Rows: {len(out)}")
    print(f"Report-time range: {df['report_time'].min()} -> {df['report_time'].max()}")
    print(f"Freeway counts: {df['freeway'].value_counts().to_dict()}")
    print(f"主線車道數 known: {int(pd.to_numeric(out['主線車道數'], errors='coerce').notna().sum())} / {len(out)}")
    print(f"受傷人數 reported: {int(pd.to_numeric(out['受傷人數'], errors='coerce').notna().sum())} / {len(out)}")
    print(f"事故類別 reported: {int(out['事故類別'].notna().sum())} / {len(out)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
