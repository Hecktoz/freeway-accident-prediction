# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
# ]
# ///
# ruff: noqa: E402
"""
Build the incident feature table and descriptive statistics report.

Inputs:
  - data/accident_data.csv (produced by process_accident_data.py)
  - data/geometry/國*_道路*變數.csv  (geometry segment attributes)
  - Optional per-task weather-candidate CSV (--weather-csv). A paper-valid
    weather source includes the actual observation timestamp. The builder
    selects the latest observation at or before incident dispatch.

Outputs:
  - <output-name>.csv
  - <stem>_statistics.csv
  - <stem>_summary.md
  - <stem>_weather_matches.csv
  - <stem>_manifest.json (configuration, schema, provenance, SHA-256 hashes)

Usage:
  uv run scripts/features/build_incident_features.py
  uv run scripts/features/build_incident_features.py \
    --weather-csv data/weather/weather_per_task_observations.csv \
    --weather-match-direction backward --weather-max-staleness-minutes 90 \
    --output-name incident_features_paper_trb_2026.csv --immutable

If no timestamped weather artifact is available, omit --weather-csv. This
produces weather value/availability zeros and remains causally valid. The
--allow-unverified-weather escape hatch is for legacy diagnostics only and is
marked paper_valid=false in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACCIDENT_CSV = PROJECT_ROOT / "data" / "accident_data.csv"
DEFAULT_GEOMETRY_DIR = PROJECT_ROOT / "data" / "geometry"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "feature_engineering"
DEFAULT_OUTPUT_NAME = "incident_features_44.csv"
DEFAULT_WEATHER_MATCH_DIRECTION = "backward"
DEFAULT_WEATHER_MAX_STALENESS_MINUTES = 90.0

WEATHER_LOW_TEMP_C = 18.0
WEATHER_HIGH_HUMIDITY_PCT = 85.0
WEATHER_STRONG_WIND_MPS = 3.4
WEATHER_RAIN_MM_MIN = 0.0
WEATHER_LOW_VISIBILITY_KM = 1.0
SHARP_CURVE_RADIUS_M = 3000.0
STEEP_SLOPE_PCT = 3.0
LANE_BLOCKAGE_MIN = 0.0
LANE_BLOCKAGE_MAX = 1.0


# ---------------------------------------------------------------------------
# Parsers (inlined from old feature_parsers.py)
# ---------------------------------------------------------------------------

DIRECTION_CODE_MAP = {
    "北": "N", "北上": "N", "N": "N",
    "南": "S", "南下": "S", "S": "S",
    "東": "E", "東行": "E", "E": "E",
    "西": "W", "西行": "W", "W": "W",
}
KM_STAKE_RE = re.compile(r"(?P<k>\d+(?:\.\d+)?)\s*[kK]\s*\+\s*(?P<m>\d+(?:\.\d+)?)")
RADIUS_RE = re.compile(r"[Rr]\s*(\d+(?:\.\d+)?)")
PCT_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")
HIGHWAY_CODE_RE = re.compile(r"(\d+)")


def _as_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def parse_km_stake(raw: object) -> float:
    text = _as_str(raw)
    if not text:
        return float("nan")
    m = KM_STAKE_RE.search(text)
    if m:
        return float(m.group("k")) + float(m.group("m")) / 1000.0
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_radius(raw: object) -> float:
    text = _as_str(raw)
    if not text or text == "-":
        return float("nan")
    m = RADIUS_RE.search(text)
    if m:
        return float(m.group(1))
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_percent(raw: object) -> float:
    text = _as_str(raw)
    if not text:
        return float("nan")
    m = PCT_RE.search(text)
    if m:
        return float(m.group(1))
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_highway_number(raw: object) -> str:
    text = _as_str(raw)
    if not text:
        return ""
    if text.startswith("N") and text[1:].isdigit():
        return text[1:]
    m = HIGHWAY_CODE_RE.search(text)
    return m.group(1) if m else ""


def parse_direction_code(raw: object) -> str:
    return DIRECTION_CODE_MAP.get(_as_str(raw), "")


def parse_bool(raw: object) -> int:
    text = _as_str(raw)
    if not text:
        return 0
    true_tokens = {"是", "有", "1", "true", "True", "隧道"}
    false_tokens = {"否", "無", "0", "false", "False", "-"}
    if text in true_tokens:
        return 1
    if text in false_tokens:
        return 0
    return int(any(token in text for token in ["是", "有", "隧道"]))


def parse_curve_flags(raw: object) -> tuple[int, int, int]:
    text = _as_str(raw)
    if "左" in text:
        return 1, 0, 0
    if "右" in text:
        return 0, 1, 0
    if "直" in text:
        return 0, 0, 1
    return 0, 0, 0


def parse_pavement_flags(raw: object) -> tuple[int, int]:
    text = _as_str(raw)
    return int("柔" in text), int("剛" in text)


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "task_id",
    "mileage", "injuries", "fatalities", "num_lanes", "lane_blockage_rate",
    "low_temp", "high_humidity", "strong_wind", "raining", "low_visibility",
    "special_accident",
    "direction_south", "direction_north", "direction_east", "direction_west",
    "highway_1", "highway_3", "highway_4", "highway_6",
    "accident_A1", "accident_A2", "accident_A3",
    "tunnel", "bridge", "lit", "shoulder",
    "curve_left", "curve_right", "curve_straight",
    "pavement_flexible", "pavement_rigid",
    "sharp_curve", "steep_slope",
]
NUMERIC_COLUMNS = ["mileage", "injuries", "fatalities", "num_lanes", "lane_blockage_rate"]

# Columns whose value can be genuinely missing ("unknown") rather than a real 0.
# Each gets a paired ``avail_<col>`` indicator (1 = source provided the value,
# 0 = unknown). The t0/t5/t15 stage masking in the trainer further zeroes both
# value and indicator for stage-unavailable columns, so the model sees
# effective_avail = source_avail AND stage_avail. See reports/feature_time_grouping.md.
MISSING_PRONE_COLUMNS = [
    "injuries", "fatalities", "lane_blockage_rate", "special_accident",
    "accident_A1", "accident_A2", "accident_A3",
    "low_temp", "high_humidity", "strong_wind", "raining", "low_visibility",
]
AVAIL_COLUMNS = [f"avail_{c}" for c in MISSING_PRONE_COLUMNS]
WEATHER_FEATURE_COLUMNS = [
    "low_temp", "high_humidity", "strong_wind", "raining", "low_visibility",
]
# Standalone data-quality flag (not paired with a value column): 1 = genuine mainline
# accident (主線車道數 >= 2), 0 = ramp/shoulder/special section (num_lanes < 2). Added
# 2026-05-29: 83% of full-blockage cases are num_lanes=1 pseudo full-blockage that are not
# true mainline closures (reports/upstream_extent_20260528_summary_tw.md §4; planning §8).
# Downstream congestion/lane-blockage analysis should filter on avail_mainline == 1.
QC_AVAIL_COLUMNS = ["avail_mainline"]
OUTPUT_COLUMNS = FEATURE_COLUMNS + AVAIL_COLUMNS + QC_AVAIL_COLUMNS
SPECIAL_INCIDENT_PATTERNS = [r"起火", r"火燒", r"燃燒", r"翻覆", r"翻車", r"爆炸", r"危險物", r"化學"]


@dataclass
class GeometryRecord:
    start_km: float
    end_km: float
    tunnel: int
    bridge: int
    lit: int
    shoulder: int
    curve_left: int
    curve_right: int
    curve_straight: int
    pavement_flexible: int
    pavement_rigid: int
    sharp_curve: int
    steep_slope: int
    num_lanes: int = 0  # mainline lane count from the geometry "車道數" column


def load_geometry_records(geometry_dir: Path) -> dict[str, dict[str, list[GeometryRecord]]]:
    grouped: dict[str, dict[str, list[GeometryRecord]]] = {}
    files = sorted(geometry_dir.glob("國*_道路*變數.csv"))

    for path in files:
        df = pd.read_csv(path, dtype=str)
        if df.empty:
            continue

        for _, row in df.iterrows():
            highway_num = parse_highway_number(row.get("國道編碼", ""))
            direction_code = parse_direction_code(row.get("方向", ""))
            if not highway_num or not direction_code:
                continue

            start_km = parse_km_stake(row.get("起里程樁", ""))
            end_km = parse_km_stake(row.get("迄里程樁", ""))
            if np.isnan(start_km) or np.isnan(end_km):
                continue
            lo, hi = sorted([start_km, end_km])

            curve_left, curve_right, curve_straight = parse_curve_flags(row.get("左彎右彎", ""))
            pavement_flexible, pavement_rigid = parse_pavement_flags(row.get("路面鋪裝", ""))
            radius = parse_radius(row.get("彎道曲率半徑 R值", ""))
            slope = parse_percent(row.get("縱向坡度", ""))
            lane_match = re.search(r"\d+", _as_str(row.get("車道數", "")))
            lane_count = int(lane_match.group()) if lane_match else 0

            rec = GeometryRecord(
                start_km=lo,
                end_km=hi,
                tunnel=parse_bool(row.get("是否為隧道", "")),
                bridge=parse_bool(row.get("是否為橋樑", "")),
                lit=parse_bool(row.get("是否為照明路段", "")),
                shoulder=parse_bool(row.get("路肩(有 沒有3m)", row.get("路肩(有沒有3m)", ""))),
                curve_left=curve_left,
                curve_right=curve_right,
                curve_straight=curve_straight,
                pavement_flexible=pavement_flexible,
                pavement_rigid=pavement_rigid,
                sharp_curve=int(not np.isnan(radius) and radius < SHARP_CURVE_RADIUS_M),
                steep_slope=int(not np.isnan(slope) and abs(slope) > STEEP_SLOPE_PCT),
                num_lanes=lane_count,
            )

            grouped.setdefault(highway_num, {}).setdefault(direction_code, []).append(rec)

    for highway in grouped.values():
        for direction, records in highway.items():
            highway[direction] = sorted(records, key=lambda r: (r.start_km, r.end_km))
    return grouped


def select_geometry_record(records: list[GeometryRecord], mileage_km: float) -> tuple[GeometryRecord | None, bool]:
    if not records or np.isnan(mileage_km):
        return None, False
    contained = [r for r in records if r.start_km <= mileage_km <= r.end_km]
    if contained:
        best = min(contained, key=lambda r: (r.end_km - r.start_km, abs((r.start_km + r.end_km) / 2 - mileage_km)))
        return best, False
    nearest = min(records, key=lambda r: abs((r.start_km + r.end_km) / 2 - mileage_km))
    return nearest, True


WEATHER_COLUMN_ALIASES = {
    "task_id": ["task_id", "任務編號"],
    # Deliberately excludes the incident dispatch column ("發送").  Paper
    # artifacts must carry the timestamp of the weather observation itself.
    "observation_time": [
        "weather_observation_timestamp", "weather_timestamp",
        "observation_timestamp", "observation_time", "obs_timestamp", "obs_time",
        "station_timestamp", "timestamp", "datetime",
        "觀測時間", "觀測日期時間", "資料時間", "氣象時間",
    ],
    "temp_c": ["temp_c", "temperature_c", "temperature", "溫度", "氣溫", "氣溫(℃)", "氣溫(°C)"],
    "humidity_pct": ["humidity_pct", "humidity", "濕度", "溼度", "相對濕度", "相對溼度", "相對溼度(%)", "相對濕度(%)"],
    "wind_speed_mps": ["wind_speed_mps", "wind_speed", "風速", "平均風速", "風速(m/s)"],
    "rain_mm": ["rain_mm", "rainfall_mm", "rainfall", "降雨量", "降水量", "累積雨量", "降水量(mm)", "降雨量(mm)"],
    "visibility_km": ["visibility_km", "visibility", "visible_km", "可見度", "能見度", "可見度(km)", "能見度(km)"],
}


def normalize_column_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return text.replace("溼", "濕")


def resolve_alias_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lowered = {normalize_column_key(c): c for c in df.columns}
    for alias in aliases:
        key = normalize_column_key(alias)
        if key in lowered:
            return lowered[key]
    return None


def normalize_task_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def humidity_to_pct(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if not valid.empty and valid.max() <= 1.5:
        return values * 100.0
    return values


def indicator_with_availability(series: pd.Series, mask: pd.Series) -> tuple[pd.Series, pd.Series]:
    available = series.notna().astype(int)
    values = mask.where(series.notna()).astype("float")
    return values, available


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_local_wall_time(series: pd.Series) -> pd.Series:
    """Parse timestamps as Taiwan local wall time without timezone metadata."""
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series, errors="coerce")

    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    if parsed.dtype == object:
        def _normalize_one(value: object) -> pd.Timestamp | pd.NaT:
            ts = pd.to_datetime(value, errors="coerce")
            if pd.isna(ts):
                return pd.NaT
            ts = pd.Timestamp(ts)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("Asia/Taipei").tz_localize(None)
            return ts

        parsed = parsed.map(_normalize_one)
        parsed = pd.to_datetime(parsed, errors="coerce")
    return parsed


def _weather_value_rows(weather_df: pd.DataFrame, col_task: str) -> pd.DataFrame:
    """Convert weather measurements to binary value/availability feature rows."""
    col_temp = resolve_alias_column(weather_df, WEATHER_COLUMN_ALIASES["temp_c"])
    col_humidity = resolve_alias_column(weather_df, WEATHER_COLUMN_ALIASES["humidity_pct"])
    col_wind = resolve_alias_column(weather_df, WEATHER_COLUMN_ALIASES["wind_speed_mps"])
    col_rain = resolve_alias_column(weather_df, WEATHER_COLUMN_ALIASES["rain_mm"])
    col_visibility = resolve_alias_column(weather_df, WEATHER_COLUMN_ALIASES["visibility_km"])

    missing = [
        name
        for name, col in [("task_id", col_task), ("temp_c", col_temp), ("humidity_pct", col_humidity),
                          ("wind_speed_mps", col_wind), ("rain_mm", col_rain)]
        if col is None
    ]
    if missing:
        raise ValueError("Weather CSV is missing required columns (or aliases): " + ", ".join(missing))

    temp_c = pd.to_numeric(weather_df[col_temp], errors="coerce")
    humidity_pct = humidity_to_pct(weather_df[col_humidity])
    wind_speed_mps = pd.to_numeric(weather_df[col_wind], errors="coerce")
    rain_mm = pd.to_numeric(weather_df[col_rain], errors="coerce")
    if col_visibility is None:
        visibility_km = pd.Series(np.nan, index=weather_df.index)
        LOGGER.warning("weather CSV has no visibility column; low_visibility will be unavailable")
    else:
        visibility_km = pd.to_numeric(weather_df[col_visibility], errors="coerce")

    out = pd.DataFrame()
    out["task_id"] = normalize_task_id(weather_df[col_task])
    rules = {
        "low_temp": (temp_c, temp_c < WEATHER_LOW_TEMP_C),
        "high_humidity": (humidity_pct, humidity_pct >= WEATHER_HIGH_HUMIDITY_PCT),
        "strong_wind": (wind_speed_mps, wind_speed_mps >= WEATHER_STRONG_WIND_MPS),
        "raining": (rain_mm, rain_mm > WEATHER_RAIN_MM_MIN),
        "low_visibility": (visibility_km, visibility_km <= WEATHER_LOW_VISIBILITY_KM),
    }
    for feature, (series, mask) in rules.items():
        out[feature], out[f"avail_{feature}"] = indicator_with_availability(series, mask.astype(int))
    return out


def _weather_availability_summary(table: pd.DataFrame, denominator: int) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for feature in WEATHER_FEATURE_COLUMNS:
        col = f"avail_{feature}"
        series = table[col] if col in table.columns else pd.Series(0, index=table.index)
        available = int(pd.to_numeric(series, errors="coerce").fillna(0).sum())
        summary[feature] = {
            "available_count": available,
            "available_ratio": available / max(denominator, 1),
        }
    return summary


def _staleness_summary(values: pd.Series) -> dict[str, float | None]:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(valid.min()),
        "median": float(valid.median()),
        "p95": float(valid.quantile(0.95)),
        "max": float(valid.max()),
    }


def build_weather_feature_table(
    weather_csv: Path,
    accident_context: pd.DataFrame,
    *,
    observation_time_column: str | None = None,
    match_direction: str = DEFAULT_WEATHER_MATCH_DIRECTION,
    max_staleness_minutes: float = DEFAULT_WEATHER_MAX_STALENESS_MINUTES,
    allow_unverified_weather: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Match per-task weather candidates to the latest observation before dispatch.

    Contract for a causal paper artifact: the weather CSV contains ``task_id`` (or
    ``任務編號``), an actual observation timestamp, and one spatially resolved
    weather row per task/timestamp.  The incident ``發送`` timestamp is supplied by
    ``accident_context`` and is never accepted as a weather-observation timestamp.
    """
    if match_direction != "backward":
        raise ValueError("Paper weather matching only supports direction='backward'.")
    if max_staleness_minutes <= 0:
        raise ValueError("weather max staleness must be greater than zero minutes")

    weather_df = pd.read_csv(weather_csv, dtype=str)
    col_task = resolve_alias_column(weather_df, WEATHER_COLUMN_ALIASES["task_id"])
    if col_task is None:
        raise ValueError("Weather CSV is missing required task_id/任務編號 column")

    value_rows = _weather_value_rows(weather_df, col_task)
    context = accident_context[["task_id", "dispatch_timestamp"]].copy()
    context["task_id"] = normalize_task_id(context["task_id"])
    context["dispatch_timestamp"] = parse_local_wall_time(context["dispatch_timestamp"])
    if context["task_id"].duplicated().any():
        duplicates = context.loc[context["task_id"].duplicated(keep=False), "task_id"].unique()[:5]
        raise ValueError(f"Accident context has duplicate task_id values: {duplicates.tolist()}")

    if observation_time_column is not None:
        if observation_time_column not in weather_df.columns:
            raise ValueError(
                f"--weather-time-column {observation_time_column!r} is not present in the weather CSV"
            )
        col_observation_time = observation_time_column
    else:
        col_observation_time = resolve_alias_column(
            weather_df, WEATHER_COLUMN_ALIASES["observation_time"]
        )

    source_hash = sha256_file(weather_csv)
    if col_observation_time is None:
        if not allow_unverified_weather:
            raise ValueError(
                "Weather CSV has no actual observation timestamp. A causal paper artifact "
                "requires one of: weather_observation_timestamp, weather_timestamp, "
                "observation_timestamp, observation_time, obs_timestamp, obs_time, "
                "station_timestamp, timestamp, datetime, 觀測時間, 觀測日期時間, "
                "資料時間, or an explicit --weather-time-column. The incident 發送 time "
                "is not evidence of when weather was observed. For legacy diagnostics only, "
                "pass --allow-unverified-weather."
            )

        LOGGER.warning(
            "LEGACY DIAGNOSTIC ONLY: weather rows have no observation timestamp; "
            "causal availability cannot be verified"
        )
        dedup = value_rows.drop_duplicates(subset=["task_id"], keep="last")
        selected = context[["task_id"]].merge(dedup, on="task_id", how="inner")
        matched_ids = set(selected["task_id"])
        matches = context.copy()
        matches["weather_observation_timestamp"] = pd.NaT
        matches["weather_staleness_minutes"] = np.nan
        matches["weather_match_status"] = matches["task_id"].map(
            lambda task_id: "matched_task_id_unverified" if task_id in matched_ids else "no_task_rows"
        )
        matches["has_future_candidate"] = pd.NA
        matches["weather_any_value_available"] = matches["task_id"].map(
            selected.set_index("task_id")[[f"avail_{c}" for c in WEATHER_FEATURE_COLUMNS]]
            .max(axis=1)
            .to_dict()
        ).fillna(0).astype(int)
        audit: dict[str, Any] = {
            "mode": "legacy_task_id_unverified",
            "paper_valid": False,
            "causal_validated": False,
            "source": str(weather_csv),
            "source_sha256": source_hash,
            "source_rows": int(len(weather_df)),
            "source_task_ids": int(value_rows["task_id"].nunique()),
            "observation_time_column": None,
            "dispatch_time_column": "發送",
            "match_direction": "task_id_only_unverified",
            "max_staleness_minutes": None,
            "accidents_requested": int(len(context)),
            "valid_dispatch_timestamps": int(context["dispatch_timestamp"].notna().sum()),
            "matched_timestamp_count": 0,
            "matched_task_id_unverified_count": int(len(selected)),
            "matched_ratio": len(selected) / max(len(context), 1),
            "future_timestamp_violations": None,
            "staleness_minutes": _staleness_summary(matches["weather_staleness_minutes"]),
            "status_counts": {
                str(k): int(v) for k, v in matches["weather_match_status"].value_counts().items()
            },
            "availability": _weather_availability_summary(selected, len(context)),
        }
        return selected, matches, audit

    observation_times = parse_local_wall_time(weather_df[col_observation_time])
    candidates = value_rows.copy()
    candidates["weather_observation_timestamp"] = observation_times
    invalid_observation_time_count = int(candidates["weather_observation_timestamp"].isna().sum())
    candidates = candidates.dropna(subset=["weather_observation_timestamp"]).copy()
    duplicate_pairs = candidates.duplicated(
        subset=["task_id", "weather_observation_timestamp"], keep=False
    )
    if duplicate_pairs.any():
        example = candidates.loc[
            duplicate_pairs, ["task_id", "weather_observation_timestamp"]
        ].head(5).to_dict("records")
        raise ValueError(
            "Weather CSV must contain one spatially resolved row per task_id/observation "
            f"timestamp; duplicate examples: {example}"
        )
    groups = {
        str(task_id): group.sort_values("weather_observation_timestamp").reset_index(drop=True)
        for task_id, group in candidates.groupby("task_id", sort=False)
    }

    selected_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    tolerance = pd.Timedelta(minutes=float(max_staleness_minutes))
    for row in context.itertuples(index=False):
        task_id = str(row.task_id)
        dispatch = row.dispatch_timestamp
        record: dict[str, Any] = {
            "task_id": task_id,
            "dispatch_timestamp": dispatch,
            "weather_observation_timestamp": pd.NaT,
            "weather_staleness_minutes": np.nan,
            "weather_match_status": "no_task_rows",
            "has_future_candidate": False,
            "weather_any_value_available": 0,
        }
        if pd.isna(dispatch):
            record["weather_match_status"] = "invalid_dispatch_timestamp"
            match_rows.append(record)
            continue
        group = groups.get(task_id)
        if group is None or group.empty:
            match_rows.append(record)
            continue

        times = group["weather_observation_timestamp"]
        record["has_future_candidate"] = bool((times > dispatch).any())
        latest_position = int(times.searchsorted(dispatch, side="right")) - 1
        if latest_position < 0:
            record["weather_match_status"] = "no_prior_observation"
            match_rows.append(record)
            continue
        latest = group.iloc[latest_position]
        observation_time = latest["weather_observation_timestamp"]
        staleness = dispatch - observation_time
        record["weather_observation_timestamp"] = observation_time
        record["weather_staleness_minutes"] = float(staleness.total_seconds() / 60.0)
        if staleness > tolerance:
            record["weather_match_status"] = "stale_prior_observation"
            match_rows.append(record)
            continue

        record["weather_match_status"] = "matched"
        feature_record = {"task_id": task_id}
        for col in [*WEATHER_FEATURE_COLUMNS, *[f"avail_{c}" for c in WEATHER_FEATURE_COLUMNS]]:
            feature_record[col] = latest[col]
        record["weather_any_value_available"] = int(
            max(int(feature_record[f"avail_{c}"]) for c in WEATHER_FEATURE_COLUMNS)
        )
        selected_rows.append(feature_record)
        match_rows.append(record)

    selected_columns = [
        "task_id", *WEATHER_FEATURE_COLUMNS,
        *[f"avail_{c}" for c in WEATHER_FEATURE_COLUMNS],
    ]
    selected = pd.DataFrame(selected_rows, columns=selected_columns)
    matches = pd.DataFrame(match_rows)
    is_matched = matches["weather_match_status"].eq("matched")
    future_violations = int((
        is_matched
        & (matches["weather_observation_timestamp"] > matches["dispatch_timestamp"])
    ).sum())
    stale_violations = int((
        is_matched
        & (matches["weather_staleness_minutes"] > float(max_staleness_minutes) + 1e-9)
    ).sum())
    if future_violations or stale_violations:
        raise AssertionError(
            "Causal weather validation failed: "
            f"future={future_violations}, over_staleness_limit={stale_violations}"
        )

    matched_staleness = matches.loc[is_matched, "weather_staleness_minutes"]
    audit = {
        "mode": "causal_backward",
        "paper_valid": True,
        "causal_validated": True,
        "source": str(weather_csv),
        "source_sha256": source_hash,
        "source_rows": int(len(weather_df)),
        "source_task_ids": int(value_rows["task_id"].nunique()),
        "invalid_observation_timestamp_rows": invalid_observation_time_count,
        "observation_time_column": col_observation_time,
        "dispatch_time_column": "發送",
        "match_direction": match_direction,
        "max_staleness_minutes": float(max_staleness_minutes),
        "accidents_requested": int(len(context)),
        "valid_dispatch_timestamps": int(context["dispatch_timestamp"].notna().sum()),
        "matched_timestamp_count": int(is_matched.sum()),
        "matched_ratio": float(is_matched.mean()) if len(matches) else 0.0,
        "future_timestamp_violations": future_violations,
        "over_staleness_limit_violations": stale_violations,
        "staleness_minutes": _staleness_summary(matched_staleness),
        "status_counts": {
            str(k): int(v) for k, v in matches["weather_match_status"].value_counts().items()
        },
        "availability": _weather_availability_summary(selected, len(context)),
    }
    return selected, matches, audit


# ---------------------------------------------------------------------------
# Main feature-building logic
# ---------------------------------------------------------------------------


def build_feature_table(
    accident_csv: Path,
    geometry_dir: Path,
    weather_csv: Path | None,
    *,
    weather_time_column: str | None = None,
    weather_match_direction: str = DEFAULT_WEATHER_MATCH_DIRECTION,
    weather_max_staleness_minutes: float = DEFAULT_WEATHER_MAX_STALENESS_MINUTES,
    allow_unverified_weather: bool = False,
) -> tuple[pd.DataFrame, pd.Series, str, int, pd.DataFrame, dict[str, Any]]:
    acc = pd.read_csv(accident_csv, dtype=str)

    out = pd.DataFrame()
    out["task_id"] = normalize_task_id(acc["任務編號"])

    mileage = pd.to_numeric(acc.get("里程數", np.nan), errors="coerce")
    injuries_raw = pd.to_numeric(acc.get("受傷人數", np.nan), errors="coerce")
    fatalities_raw = pd.to_numeric(acc.get("死亡人數", np.nan), errors="coerce")
    injuries = injuries_raw.fillna(0.0)
    fatalities = fatalities_raw.fillna(0.0)
    num_lanes = pd.to_numeric(acc.get("主線車道數", np.nan), errors="coerce")
    remain_lanes = pd.to_numeric(acc.get("剩餘車道數", np.nan), errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        lane_blockage_rate = 1.0 - (remain_lanes / num_lanes)
    lane_blockage_rate = lane_blockage_rate.clip(lower=LANE_BLOCKAGE_MIN, upper=LANE_BLOCKAGE_MAX).fillna(0.0)

    out["mileage"] = mileage
    out["injuries"] = injuries
    out["fatalities"] = fatalities
    out["num_lanes"] = num_lanes.fillna(0.0)
    out["lane_blockage_rate"] = lane_blockage_rate

    direction_code = acc.get("方向", "").map(parse_direction_code).fillna("")
    out["direction_south"] = (direction_code == "S").astype(int)
    out["direction_north"] = (direction_code == "N").astype(int)
    out["direction_east"] = (direction_code == "E").astype(int)
    out["direction_west"] = (direction_code == "W").astype(int)

    highway_num = acc.get("國道", "").map(parse_highway_number).fillna("")
    out["highway_1"] = (highway_num == "1").astype(int)
    out["highway_3"] = (highway_num == "3").astype(int)
    out["highway_4"] = (highway_num == "4").astype(int)
    out["highway_6"] = (highway_num == "6").astype(int)

    accident_class = acc.get("事故類別", "").astype(str).str.strip().fillna("")
    out["accident_A1"] = (accident_class == "A1").astype(int)
    out["accident_A2"] = (accident_class == "A2").astype(int)
    out["accident_A3"] = (accident_class == "A3").astype(int)

    # fillna('') before concat: a missing component must not collapse the whole
    # blob to NaN (string + NaN = NaN), which would silently disable
    # special-accident detection for any row with a blank text field.
    def _text_col(name: str) -> pd.Series:
        col = acc.get(name)
        if col is None:
            return pd.Series("", index=acc.index)
        return col.fillna("").astype(str)

    text_blob = (
        _text_col("性質") + " " + _text_col("事故類別") + " "
        + _text_col("車種") + " " + _text_col("位置") + " "
        + _text_col("路段")
    )
    special_mask = pd.Series(False, index=acc.index)
    for pattern in SPECIAL_INCIDENT_PATTERNS:
        special_mask = special_mask | text_blob.str.contains(pattern, na=False, regex=True)
    out["special_accident"] = special_mask.astype(int)

    report_time_raw = acc["發送"] if "發送" in acc.columns else pd.Series(pd.NaT, index=acc.index)
    report_time = parse_local_wall_time(report_time_raw)
    accident_context = pd.DataFrame({
        "task_id": out["task_id"],
        "dispatch_timestamp": report_time,
    })

    weather_source = "fallback_zero"
    if weather_csv is not None:
        if not weather_csv.exists():
            raise FileNotFoundError(f"Weather CSV does not exist: {weather_csv}")
        weather_table, weather_matches, weather_audit = build_weather_feature_table(
            weather_csv,
            accident_context,
            observation_time_column=weather_time_column,
            match_direction=weather_match_direction,
            max_staleness_minutes=weather_max_staleness_minutes,
            allow_unverified_weather=allow_unverified_weather,
        )
        out = out.merge(weather_table, on="task_id", how="left")
        weather_source = str(weather_csv)
    else:
        for c in WEATHER_FEATURE_COLUMNS:
            out[c] = 0
            out[f"avail_{c}"] = 0
        weather_matches = accident_context.copy()
        weather_matches["weather_observation_timestamp"] = pd.NaT
        weather_matches["weather_staleness_minutes"] = np.nan
        weather_matches["weather_match_status"] = "weather_omitted"
        weather_matches["has_future_candidate"] = pd.NA
        weather_matches["weather_any_value_available"] = 0
        weather_audit = {
            "mode": "omitted",
            "paper_valid": True,
            "causal_validated": True,
            "source": None,
            "source_sha256": None,
            "source_rows": 0,
            "source_task_ids": 0,
            "observation_time_column": None,
            "dispatch_time_column": "發送",
            "match_direction": weather_match_direction,
            "max_staleness_minutes": float(weather_max_staleness_minutes),
            "accidents_requested": int(len(accident_context)),
            "valid_dispatch_timestamps": int(report_time.notna().sum()),
            "matched_timestamp_count": 0,
            "matched_ratio": 0.0,
            "future_timestamp_violations": 0,
            "staleness_minutes": _staleness_summary(
                weather_matches["weather_staleness_minutes"]
            ),
            "status_counts": {"weather_omitted": int(len(accident_context))},
            "availability": _weather_availability_summary(
                pd.DataFrame(index=accident_context.index), len(accident_context)
            ),
        }

    for c in WEATHER_FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = 0
        if f"avail_{c}" not in out.columns:
            out[f"avail_{c}"] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
        out[f"avail_{c}"] = pd.to_numeric(out[f"avail_{c}"], errors="coerce").fillna(0).astype(int)

    geom_lookup = load_geometry_records(geometry_dir) if geometry_dir.exists() else {}
    geom_cols = [
        "tunnel", "bridge", "lit", "shoulder",
        "curve_left", "curve_right", "curve_straight",
        "pavement_flexible", "pavement_rigid",
        "sharp_curve", "steep_slope",
    ]
    for c in geom_cols:
        out[c] = 0

    geometry_fallback_count = 0
    for i in range(len(out)):
        hwy = highway_num.iloc[i]
        dcode = direction_code.iloc[i]
        km = mileage.iloc[i]
        records = geom_lookup.get(hwy, {}).get(dcode, [])
        rec, used_fallback = select_geometry_record(records, km)
        if rec is None:
            continue
        if used_fallback:
            geometry_fallback_count += 1
        out.at[i, "tunnel"] = rec.tunnel
        out.at[i, "bridge"] = rec.bridge
        out.at[i, "lit"] = rec.lit
        out.at[i, "shoulder"] = rec.shoulder
        out.at[i, "curve_left"] = rec.curve_left
        out.at[i, "curve_right"] = rec.curve_right
        out.at[i, "curve_straight"] = rec.curve_straight
        out.at[i, "pavement_flexible"] = rec.pavement_flexible
        out.at[i, "pavement_rigid"] = rec.pavement_rigid
        out.at[i, "sharp_curve"] = rec.sharp_curve
        out.at[i, "steep_slope"] = rec.steep_slope

    completed_time_raw = (
        acc["處理完成"] if "處理完成" in acc.columns
        else pd.Series(pd.NaT, index=acc.index)
    )
    completed_time = parse_local_wall_time(completed_time_raw)
    impact_minutes = (completed_time - report_time).dt.total_seconds() / 60.0
    impact_minutes = impact_minutes.where(impact_minutes >= 0)

    # Source availability indicators (1 = dispatch reported the field).
    # Dispatch is the authoritative operational source, so incident-detail
    # fields are considered available whenever the source cell is non-blank.
    accident_type_avail = (~accident_class.isin(["", "nan", "None", "NaN"])).astype(int)
    avail_values = {
        "avail_injuries": injuries_raw.notna().astype(int),
        "avail_fatalities": fatalities_raw.notna().astype(int),
        "avail_lane_blockage_rate": (num_lanes.notna() & remain_lanes.notna()).astype(int),
        # special_accident is derived from the incident text; absence of a keyword
        # means "no special accident", so it is treated as always determinable.
        "avail_special_accident": pd.Series(1, index=out.index, dtype=int),
        "avail_accident_A1": accident_type_avail,
        "avail_accident_A2": accident_type_avail,
        "avail_accident_A3": accident_type_avail,
        "avail_low_temp": out["avail_low_temp"],
        "avail_high_humidity": out["avail_high_humidity"],
        "avail_strong_wind": out["avail_strong_wind"],
        "avail_raining": out["avail_raining"],
        "avail_low_visibility": out["avail_low_visibility"],
        # Genuine mainline (主線車道數 >= 2). NaN/1-lane -> 0 (ramp/shoulder/special),
        # where lane_blockage_rate is unreliable. See upstream_extent §4 / planning §8.
        "avail_mainline": (num_lanes >= 2).fillna(False).astype(int),
    }
    for col, series in avail_values.items():
        out[col] = series.to_numpy()

    out = out[OUTPUT_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in OUTPUT_COLUMNS:
        if col not in {"task_id", *NUMERIC_COLUMNS}:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    return (
        out,
        impact_minutes,
        weather_source,
        geometry_fallback_count,
        weather_matches,
        weather_audit,
    )


def build_statistics_table(features: pd.DataFrame, impact_minutes: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_total = len(features)
    for col in OUTPUT_COLUMNS:
        if col == "task_id":
            rows.append({
                "feature": col, "dtype": "identifier",
                "missing": 0, "missing_ratio": 0.0,
                "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
                "ones": np.nan, "zeros": np.nan, "corr_with_impact_min": np.nan,
            })
            continue

        series = features[col]
        missing = int(series.isna().sum())
        valid = pd.to_numeric(series, errors="coerce")

        pair = pd.DataFrame({"x": valid, "y": impact_minutes}).dropna()
        corr = np.nan
        if len(pair) >= 2 and pair["x"].nunique() > 1 and pair["y"].nunique() > 1:
            corr = float(pair["x"].corr(pair["y"]))

        is_binary = set(valid.dropna().unique().tolist()).issubset({0, 1})
        rows.append({
            "feature": col,
            "dtype": "binary" if is_binary else "numeric",
            "missing": missing,
            "missing_ratio": missing / max(n_total, 1),
            "mean": float(valid.mean()) if len(valid) else np.nan,
            "std": float(valid.std()) if len(valid) else np.nan,
            "min": float(valid.min()) if len(valid) else np.nan,
            "max": float(valid.max()) if len(valid) else np.nan,
            "ones": int((valid == 1).sum()) if is_binary else np.nan,
            "zeros": int((valid == 0).sum()) if is_binary else np.nan,
            "corr_with_impact_min": corr,
        })
    return pd.DataFrame(rows)


def write_markdown_summary(
    path: Path,
    features: pd.DataFrame,
    stats: pd.DataFrame,
    weather_source: str,
    impact_minutes: pd.Series,
    weather_audit: dict[str, Any],
    feature_sha256: str,
) -> None:
    corr_df = stats.dropna(subset=["corr_with_impact_min"]).copy()
    corr_df = corr_df[corr_df["feature"] != "task_id"]
    corr_sorted = corr_df.sort_values("corr_with_impact_min")
    top_negative = corr_sorted.head(5)
    top_positive = corr_sorted.tail(5).sort_values("corr_with_impact_min", ascending=False)

    lines: list[str] = []
    lines.append("# Incident Feature Engineering Summary")
    lines.append("")
    lines.append(f"- samples: {len(features)}")
    model_feature_dim = len([c for c in OUTPUT_COLUMNS if c not in {"task_id", "avail_mainline"}])
    lines.append(
        f"- columns: {len(features.columns)} (expected {len(OUTPUT_COLUMNS)}: "
        f"{len(FEATURE_COLUMNS)} value incl. task_id + {len(AVAIL_COLUMNS)} avail + "
        f"{len(QC_AVAIL_COLUMNS)} QC)"
    )
    lines.append(f"- model_feature_dim: {model_feature_dim} (excludes task_id and QC-only columns)")
    lines.append(f"- feature_csv_sha256: `{feature_sha256}`")
    lines.append(f"- weather_source: {weather_source}")
    lines.append(f"- weather_mode: {weather_audit['mode']}")
    lines.append(f"- weather_paper_valid: {str(bool(weather_audit['paper_valid'])).lower()}")
    lines.append(f"- valid_impact_minutes: {int(impact_minutes.notna().sum())}")
    lines.append("")
    lines.append("## Weather Matching Audit")
    lines.append("")
    lines.append(f"- match_direction: {weather_audit['match_direction']}")
    lines.append(f"- max_staleness_minutes: {weather_audit['max_staleness_minutes']}")
    lines.append(f"- observation_time_column: {weather_audit['observation_time_column']}")
    lines.append(f"- matched_timestamp_count: {weather_audit['matched_timestamp_count']}")
    lines.append(f"- matched_ratio: {weather_audit['matched_ratio']:.4f}")
    lines.append(f"- future_timestamp_violations: {weather_audit['future_timestamp_violations']}")
    staleness = weather_audit["staleness_minutes"]
    lines.append(
        "- matched_staleness_minutes: "
        f"min={staleness['min']}, median={staleness['median']}, "
        f"p95={staleness['p95']}, max={staleness['max']}"
    )
    lines.append("")
    lines.append("| match status | count |")
    lines.append("| --- | ---: |")
    for status, count in weather_audit["status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.append("")
    lines.append("| weather feature | available | ratio of all incidents |")
    lines.append("| --- | ---: | ---: |")
    for feature, availability in weather_audit["availability"].items():
        lines.append(
            f"| {feature} | {availability['available_count']} | "
            f"{availability['available_ratio']:.4f} |"
        )
    lines.append("")
    lines.append("## Top Positive Correlations With Impact Time")
    lines.append("")
    lines.append("| feature | corr_with_impact_min |")
    lines.append("| --- | ---: |")
    for _, row in top_positive.iterrows():
        lines.append(f"| {row['feature']} | {row['corr_with_impact_min']:.4f} |")
    lines.append("")
    lines.append("## Top Negative Correlations With Impact Time")
    lines.append("")
    lines.append("| feature | corr_with_impact_min |")
    lines.append("| --- | ---: |")
    for _, row in top_negative.iterrows():
        lines.append(f"| {row['feature']} | {row['corr_with_impact_min']:.4f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    if weather_source == "fallback_zero":
        lines.append("- Weather was omitted; weather values and availability indicators are zero.")
        lines.append(
            "- A paper-valid weather CSV must include an actual observation timestamp so the "
            "builder can select the latest observation at or before dispatch."
        )
    elif not weather_audit["paper_valid"]:
        lines.append(
            "- **Not paper-valid:** weather was joined by task ID without an auditable "
            "observation timestamp. Use this artifact only for legacy diagnostics."
        )
    else:
        lines.append(
            "- Weather features use the latest observation at or before dispatch, subject "
            "to the configured maximum staleness."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build incident features and descriptive statistics")
    parser.add_argument("--accident-csv", default=str(DEFAULT_ACCIDENT_CSV))
    parser.add_argument("--geometry-dir", default=str(DEFAULT_GEOMETRY_DIR))
    parser.add_argument(
        "--weather-csv",
        default=None,
        help=(
            "Per-task candidate weather observations. Paper mode requires an actual "
            "observation timestamp; omit this option to emit availability=0."
        ),
    )
    parser.add_argument(
        "--weather-time-column",
        default=None,
        help="Exact weather observation-time column name (otherwise recognized aliases are used)",
    )
    parser.add_argument(
        "--weather-match-direction",
        default=DEFAULT_WEATHER_MATCH_DIRECTION,
        choices=["backward"],
        help="Causal as-of direction; paper protocol is backward-only",
    )
    parser.add_argument(
        "--weather-max-staleness-minutes",
        type=float,
        default=DEFAULT_WEATHER_MAX_STALENESS_MINUTES,
        help="Maximum age of the latest pre-dispatch weather observation (default: 90)",
    )
    parser.add_argument(
        "--allow-unverified-weather",
        action="store_true",
        help=(
            "LEGACY DIAGNOSTIC ONLY: allow task-ID weather with no observation timestamp; "
            "the manifest marks the artifact paper_valid=false"
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME,
                        help="Feature CSV filename (availability-mask schema; filename kept for backward CLI compatibility)")
    parser.add_argument(
        "--immutable",
        action="store_true",
        help=(
            "Refuse to overwrite any artifact and require a versioned output name; use for "
            "paper builds (for example incident_features_paper_trb_2026.csv)"
        ),
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="[%(levelname)s] %(message)s")

    accident_csv = Path(args.accident_csv)
    geometry_dir = Path(args.geometry_dir)
    weather_csv = Path(args.weather_csv) if args.weather_csv else None
    output_dir = Path(args.output_dir)

    if not accident_csv.exists():
        raise FileNotFoundError(f"Accident CSV does not exist: {accident_csv}")
    if not geometry_dir.exists():
        LOGGER.warning("geometry directory does not exist: %s (geometry features will be zeros)", geometry_dir)

    output_name = Path(args.output_name)
    if output_name.name != args.output_name or output_name.suffix.lower() != ".csv":
        raise ValueError("--output-name must be a plain .csv filename; use --output-dir for its directory")
    if args.immutable and args.output_name == DEFAULT_OUTPUT_NAME:
        raise ValueError(
            "--immutable requires an explicit versioned --output-name; "
            "recommended: incident_features_paper_trb_2026.csv"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    feature_csv = output_dir / args.output_name
    stem = output_name.stem
    stats_csv = output_dir / f"{stem}_statistics.csv"
    summary_md = output_dir / f"{stem}_summary.md"
    weather_matches_csv = output_dir / f"{stem}_weather_matches.csv"
    manifest_json = output_dir / f"{stem}_manifest.json"
    artifact_paths = [
        feature_csv, stats_csv, summary_md, weather_matches_csv, manifest_json,
    ]
    if args.immutable:
        existing = [str(path) for path in artifact_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Immutable paper artifact already exists; choose a new versioned output name: "
                + ", ".join(existing)
            )

    (
        features,
        impact_minutes,
        weather_source,
        geometry_fallback_count,
        weather_matches,
        weather_audit,
    ) = build_feature_table(
        accident_csv=accident_csv,
        geometry_dir=geometry_dir,
        weather_csv=weather_csv,
        weather_time_column=args.weather_time_column,
        weather_match_direction=args.weather_match_direction,
        weather_max_staleness_minutes=args.weather_max_staleness_minutes,
        allow_unverified_weather=args.allow_unverified_weather,
    )
    stats = build_statistics_table(features, impact_minutes)

    features.to_csv(feature_csv, index=False, encoding="utf-8-sig")
    stats.to_csv(stats_csv, index=False, encoding="utf-8-sig")
    weather_matches.to_csv(weather_matches_csv, index=False, encoding="utf-8-sig")
    feature_hash = sha256_file(feature_csv)
    write_markdown_summary(
        summary_md,
        features,
        stats,
        weather_source,
        impact_minutes,
        weather_audit,
        feature_hash,
    )

    geometry_files = sorted(geometry_dir.glob("國*_道路*變數.csv")) if geometry_dir.exists() else []
    manifest = {
        "artifact_type": "incident_feature_table",
        "artifact_id": stem,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_valid": bool(weather_audit["paper_valid"]),
        "immutable_write": bool(args.immutable),
        "configuration": {
            "weather_mode": weather_audit["mode"],
            "weather_match_direction": args.weather_match_direction,
            "weather_max_staleness_minutes": float(args.weather_max_staleness_minutes),
            "weather_time_column_override": args.weather_time_column,
            "allow_unverified_weather": bool(args.allow_unverified_weather),
        },
        "sources": {
            "accident_csv": {
                "path": _portable_path(accident_csv),
                "sha256": sha256_file(accident_csv),
            },
            "geometry_files": [
                {"path": _portable_path(path), "sha256": sha256_file(path)}
                for path in geometry_files
            ],
            "weather": weather_audit,
        },
        "feature_schema": {
            "rows": int(len(features)),
            "unique_task_ids": int(features["task_id"].nunique()),
            "columns": [
                {"name": str(column), "dtype": str(features[column].dtype)}
                for column in features.columns
            ],
            "model_feature_columns": [
                column for column in features.columns
                if column not in {"task_id", "avail_mainline"}
            ],
        },
        "geometry_nearest_segment_fallback_count": int(geometry_fallback_count),
        "artifacts": {
            "features": {"path": _portable_path(feature_csv), "sha256": feature_hash},
            "statistics": {"path": _portable_path(stats_csv), "sha256": sha256_file(stats_csv)},
            "summary": {"path": _portable_path(summary_md), "sha256": sha256_file(summary_md)},
            "weather_matches": {
                "path": _portable_path(weather_matches_csv),
                "sha256": sha256_file(weather_matches_csv),
            },
        },
    }
    manifest_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    LOGGER.info("wrote %s", feature_csv)
    LOGGER.info("wrote %s", stats_csv)
    LOGGER.info("wrote %s", summary_md)
    LOGGER.info("wrote %s", weather_matches_csv)
    LOGGER.info("wrote %s", manifest_json)
    LOGGER.info("rows=%s | columns=%s", len(features), len(features.columns))
    LOGGER.info(
        "weather mode=%s | direction=%s | max staleness=%s min | matched=%s/%s | future violations=%s",
        weather_audit["mode"],
        weather_audit["match_direction"],
        weather_audit["max_staleness_minutes"],
        weather_audit["matched_timestamp_count"],
        weather_audit["accidents_requested"],
        weather_audit["future_timestamp_violations"],
    )
    LOGGER.info("feature sha256=%s", feature_hash)
    if weather_source == "fallback_zero":
        LOGGER.warning("weather omitted: features and availability indicators are zero")
    elif not weather_audit["paper_valid"]:
        LOGGER.warning("weather artifact is NOT paper-valid (unverified observation timestamps)")
    if geometry_fallback_count > 0:
        LOGGER.warning("geometry nearest-segment fallback used %s times", geometry_fallback_count)


if __name__ == "__main__":
    main()
