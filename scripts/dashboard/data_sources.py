"""Cached data accessors for the dashboard.

All cross-script imports go through this module so caching boundaries are
explicit. Filters slice the cached frames in-memory; we never re-read parquet
when the user toggles a checkbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pyarrow.lib import ArrowInvalid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_VDXML = DEFAULT_DATA_ROOT / "metadata" / "vd_static.xml"
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "output" / "models"
PROFILE_REQUIRED_COLUMNS = {
    "vdid",
    "direction",
    "mileage_m",
    "day_class",
    "peak_class",
    "minute_bucket",
    "mean_speed_x10",
}


def _import_repo_modules() -> None:
    """Make `scripts.*` importable when streamlit launches us from anywhere."""
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


_import_repo_modules()


# ---------------------------------------------------------------------------
# VD static (lat/lon)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=24 * 3600, show_spinner="載入 VD 靜態資料 (vd_static.xml)…")
def load_vd_records(xml_path: str = str(DEFAULT_VDXML)) -> list[dict[str, Any]]:
    """Parse vd_static.xml via the existing build_vd_station_map.parse_vd_xml."""
    from scripts.visualization.build_vd_station_map import parse_vd_xml

    records = parse_vd_xml(Path(xml_path))
    out = [r for r in records if not (r["lat"] == 0 and r["lon"] == 0)]
    return out


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_vd_records_df(xml_path: str = str(DEFAULT_VDXML)) -> pd.DataFrame:
    """Same as load_vd_records but as a DataFrame for vectorized filtering."""
    return pd.DataFrame(load_vd_records(xml_path))


# ---------------------------------------------------------------------------
# Curated metadata
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="載入 VD 維度表 (vd_dim.parquet)…")
def load_vd_dim(data_root: str = str(DEFAULT_DATA_ROOT)) -> pd.DataFrame:
    from scripts.lib.data_loaders import read_vd_dim

    return read_vd_dim(Path(data_root))


@st.cache_data(show_spinner="載入事故資料 (accident_data.csv)…")
def load_accidents(data_root: str = str(DEFAULT_DATA_ROOT)) -> pd.DataFrame:
    from scripts.lib.data_loaders import read_accidents

    df = read_accidents(Path(data_root))
    df = df[df["report_time"].notna() & df["mileage_km"].notna()].copy()
    df = df[df["direction"].isin(["N", "S", "E", "W"])]
    df = df[df["freeway"].isin(["N1", "N3", "N4", "N6"])]
    df["report_date"] = df["report_time"].dt.date
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Station / profile data
# ---------------------------------------------------------------------------


@st.cache_data(max_entries=16, show_spinner="讀取站台速度資料…")
def load_station_day(date_str: str, resolution_min: int = 5,
                     data_root: str = str(DEFAULT_DATA_ROOT)) -> pd.DataFrame:
    from scripts.lib.data_loaders import read_station_day

    try:
        df = read_station_day(date_str, resolution_min=resolution_min,
                              data_root=Path(data_root))
    except FileNotFoundError:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["speed_kmh"] = df["speed_x10"].astype(float) / 10.0
    df["mileage_km"] = df["mileage_m"].astype(float) / 1000.0
    return df


@st.cache_data(max_entries=8, show_spinner="讀取歷史 baseline…")
def load_historical_profile(freeway: str, resolution_min: int = 5,
                            data_root: str = str(DEFAULT_DATA_ROOT)) -> pd.DataFrame:
    from scripts.lib.data_loaders import read_historical_profile

    try:
        profile = read_historical_profile(resolution_min=resolution_min,
                                          data_root=Path(data_root),
                                          freeway=freeway)
    except (FileNotFoundError, ArrowInvalid, ValueError, OSError):
        return pd.DataFrame()
    missing = PROFILE_REQUIRED_COLUMNS.difference(profile.columns)
    if missing:
        return pd.DataFrame()
    return profile


# ---------------------------------------------------------------------------
# Coverage helpers
# ---------------------------------------------------------------------------


def latest_station_date(vd_dim: pd.DataFrame) -> str | None:
    """Most recent date with curated station data, derived from vd_dim."""
    if vd_dim.empty or "last_source_date" not in vd_dim.columns:
        return None
    last = pd.to_datetime(vd_dim["last_source_date"]).max()
    if pd.isna(last):
        return None
    return last.strftime("%Y-%m-%d")
