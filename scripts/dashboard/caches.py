"""LRU-cached accessors for data loaded on first use.

All arguments are serializable primitives (strings, ints) so functools.lru_cache works.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from pyarrow.lib import ArrowInvalid

PROFILE_REQUIRED_COLUMNS = {
    "vdid", "direction", "mileage_m", "day_class",
    "peak_class", "minute_bucket", "mean_speed_x10",
}


@lru_cache(maxsize=16)
def load_station_day(
    date_str: str,
    resolution_min: int = 5,
    data_root_str: str | None = None,
) -> pd.DataFrame:
    from scripts.lib.data_loaders import read_station_day

    data_root = Path(data_root_str) if data_root_str else None
    try:
        df = read_station_day(date_str, resolution_min=resolution_min, data_root=data_root)
    except FileNotFoundError:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["speed_kmh"] = df["speed_x10"].astype(float) / 10.0
    df["mileage_km"] = df["mileage_m"].astype(float) / 1000.0
    return df


@lru_cache(maxsize=8)
def load_historical_profile(
    freeway: str,
    resolution_min: int = 5,
    data_root_str: str | None = None,
) -> pd.DataFrame:
    from scripts.lib.data_loaders import read_historical_profile

    data_root = Path(data_root_str) if data_root_str else None
    try:
        profile = read_historical_profile(
            resolution_min=resolution_min,
            data_root=data_root,
            freeway=freeway,
        )
    except (FileNotFoundError, ArrowInvalid, ValueError, OSError):
        return pd.DataFrame()
    missing = PROFILE_REQUIRED_COLUMNS.difference(profile.columns)
    if missing:
        return pd.DataFrame()
    return profile
