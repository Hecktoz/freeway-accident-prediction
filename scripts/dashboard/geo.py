"""Derive accident lat/lon by interpolating between adjacent VDs along a route.

Accident CSV has only (freeway, direction, mileage_km). VD static XML carries
(lat, lon, mileage_km) per VD. We bracket each accident's mileage between its
two nearest VDs on the same (freeway, direction) and linearly interpolate.

Pre-jitter VD lat/lon by a deterministic per-vdid offset so VD markers never
collide with accident-derived markers — needed because streamlit-folium only
returns lat/lng on click, and we look up marker identity in a dict keyed by
rounded lat/lng.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

JITTER_SCALE = 1e-7  # ≈1 cm — visually invisible, lookup-safe.


def _jitter_for_vdid(vdid: str) -> tuple[float, float]:
    h = hashlib.sha1(vdid.encode("utf-8")).digest()
    lat_off = (int.from_bytes(h[0:2], "big") % 200 - 100) * JITTER_SCALE
    lon_off = (int.from_bytes(h[2:4], "big") % 200 - 100) * JITTER_SCALE
    return lat_off, lon_off


def jitter_vd_records(records: pd.DataFrame) -> pd.DataFrame:
    out = records.copy()
    if out.empty:
        return out
    offsets = out["vdid"].map(_jitter_for_vdid)
    out["lat"] = out["lat"] + offsets.map(lambda x: x[0])
    out["lon"] = out["lon"] + offsets.map(lambda x: x[1])
    return out


def build_route_index(vd_records: pd.DataFrame,
                      vd_dim_vdids: tuple[str, ...]) -> dict[tuple[str, str], pd.DataFrame]:
    """Per-route lookup of (mileage_km -> lat/lon).

    Restrict to VDs that exist in the curated dataset (vd_dim) so accident
    interpolation lands on routes we actually have speed data for.
    """
    if vd_records.empty:
        return {}
    in_dim = set(vd_dim_vdids)
    df = vd_records[vd_records["vdid"].isin(in_dim)].copy()
    if df.empty:
        df = vd_records.copy()
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for (fw, direction), group in df.groupby(["freeway", "direction"]):
        sub = group[["mileage_km", "lat", "lon", "vdid"]].sort_values("mileage_km").reset_index(drop=True)
        out[(str(fw), str(direction))] = sub
    return out


def accident_latlon(freeway: str, direction: str, mileage_km: float,
                    route_index: dict) -> tuple[float, float] | None:
    df = route_index.get((freeway, direction))
    if df is None or df.empty or pd.isna(mileage_km):
        return None
    lo = df[df["mileage_km"] <= mileage_km].tail(1)
    hi = df[df["mileage_km"] >= mileage_km].head(1)
    if lo.empty and hi.empty:
        return None
    if lo.empty:
        return float(hi["lat"].iat[0]), float(hi["lon"].iat[0])
    if hi.empty:
        return float(lo["lat"].iat[0]), float(lo["lon"].iat[0])
    if lo.index[0] == hi.index[0]:
        return float(lo["lat"].iat[0]), float(lo["lon"].iat[0])
    m0, m1 = float(lo["mileage_km"].iat[0]), float(hi["mileage_km"].iat[0])
    t = 0.0 if m1 == m0 else (mileage_km - m0) / (m1 - m0)
    lat = float(lo["lat"].iat[0] + t * (hi["lat"].iat[0] - lo["lat"].iat[0]))
    lon = float(lo["lon"].iat[0] + t * (hi["lon"].iat[0] - lo["lon"].iat[0]))
    return lat, lon


def attach_accident_latlon(accidents: pd.DataFrame,
                           route_index: dict) -> pd.DataFrame:
    """Vectorized: add lat/lon columns to filtered accident dataframe.

    Rows without a resolvable position are dropped; callers see a count diff.
    """
    if accidents.empty:
        return accidents.assign(lat=pd.Series(dtype=float), lon=pd.Series(dtype=float))
    lats: list[float] = []
    lons: list[float] = []
    keep: list[int] = []
    for idx, row in accidents.iterrows():
        ll = accident_latlon(row["freeway"], row["direction"], float(row["mileage_km"]), route_index)
        if ll is None:
            continue
        lats.append(ll[0])
        lons.append(ll[1])
        keep.append(idx)
    if not keep:
        return accidents.iloc[0:0].assign(lat=pd.Series(dtype=float), lon=pd.Series(dtype=float))
    out = accidents.loc[keep].copy()
    out["lat"] = lats
    out["lon"] = lons
    return out.reset_index(drop=True)


def round_key(lat: float, lon: float, ndigits: int = 6) -> tuple[float, float]:
    return round(float(lat), ndigits), round(float(lon), ndigits)
