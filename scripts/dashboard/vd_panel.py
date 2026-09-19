"""Render the VD detail side panel.

Shown when the user clicks a VD marker on the map. Displays:
  - VD metadata (route, mileage, lane info, last-source-date)
  - Recent-day speed time series for this VD with historical baseline overlay
"""

from __future__ import annotations

from datetime import time as dtime

import numpy as np
import pandas as pd
import streamlit as st

from . import data_sources as ds
from .map_layer import DIRECTION_LABEL
from .plots import line_chart


def _baseline_for_vd(vdid: str, freeway: str, target_date: pd.Timestamp,
                     resolution_min: int = 5) -> pd.DataFrame | None:
    profile = ds.load_historical_profile(freeway, resolution_min=resolution_min)
    if profile is None or profile.empty:
        return None
    sub = profile[profile["vdid"] == vdid].copy()
    if sub.empty or "minute_bucket" not in sub.columns:
        return None
    # If profile carries day_class / peak_class, prefer the matching slice for
    # the chosen date; otherwise fall back to mean per minute_bucket.
    if "day_class" in sub.columns:
        from scripts.lib.calendar import day_class
        cls = day_class(target_date.date())
        day_slice = sub[sub["day_class"] == cls]
        if not day_slice.empty:
            sub = day_slice
    grouped = sub.groupby("minute_bucket", as_index=False)["mean_speed_x10"].mean()
    grouped["speed_kmh"] = grouped["mean_speed_x10"] / 10.0
    grouped = grouped.sort_values("minute_bucket").reset_index(drop=True)
    return grouped[["minute_bucket", "speed_kmh"]]


def render(vdid: str, vd_records_df: pd.DataFrame, vd_dim: pd.DataFrame) -> None:
    st.subheader(f"VD 詳細：{vdid}")

    rec_match = vd_records_df[vd_records_df["vdid"] == vdid]
    dim_match = vd_dim[vd_dim["vdid"] == vdid] if not vd_dim.empty else pd.DataFrame()

    if rec_match.empty:
        st.warning("找不到該 VD 的靜態資料。")
        return
    rec = rec_match.iloc[0]

    cols = st.columns(3)
    cols[0].metric("路線", f"{rec['freeway']} {DIRECTION_LABEL.get(rec['direction'], rec['direction'])}")
    cols[1].metric("里程", f"{rec['mileage_km']:.3f} K")
    cols[2].metric("車道數", f"{rec.get('lane_num', 0)}（實際 {rec.get('actual_lane_num', 0)}）")

    if not dim_match.empty:
        d = dim_match.iloc[0]
        st.caption(
            f"資料期間：{d.get('first_source_date')} ~ {d.get('last_source_date')} ｜ "
            f"涵蓋天數：{int(d.get('n_source_days', 0))} ｜ "
            f"列數：{int(d.get('n_rows_station_1m', 0)):,}"
        )
    else:
        st.caption("此 VD 不在 vd_station 訓練資料集中（僅出現在靜態 XML）。")

    # ---- Recent-day speed series ----
    chart_key = f"vd-series:{vdid}"
    btn_cols = st.columns([1, 1])
    if btn_cols[0].button("載入速度時序", key=f"load-{chart_key}", type="primary",
                          use_container_width=True):
        st.session_state[chart_key] = True
    if btn_cols[1].button("收起速度時序", key=f"hide-{chart_key}",
                          use_container_width=True):
        st.session_state[chart_key] = False

    if not st.session_state.get(chart_key, False):
        st.info("VD 基本資料已載入。按「載入速度時序」後才會讀取最近一日 station Parquet。")
        return

    last_date = ds.latest_station_date(vd_dim)
    if last_date is None:
        st.info("尚未有可用的 station 資料日期。")
        return

    target_date = pd.Timestamp(last_date)
    df = ds.load_station_day(last_date, resolution_min=5)
    sub = df[df["vdid"] == vdid] if not df.empty else df
    if sub.empty:
        st.info(f"{last_date} 沒有此 VD 的速度資料。")
        return

    sub = sub.sort_values("timestamp")
    series = {"觀測 (該日)": sub["speed_kmh"].to_numpy()}

    baseline_df = _baseline_for_vd(vdid, str(rec["freeway"]), target_date)
    if baseline_df is not None and not baseline_df.empty:
        # Align baseline (per minute_bucket) onto the day's timestamps.
        minute_bucket = (sub["timestamp"].dt.hour * 60 + sub["timestamp"].dt.minute).astype(int)
        bucket_step = 5
        minute_bucket = (minute_bucket // bucket_step) * bucket_step
        baseline_lookup = dict(zip(baseline_df["minute_bucket"], baseline_df["speed_kmh"]))
        baseline_arr = minute_bucket.map(baseline_lookup).to_numpy(dtype=float)
        # Replace missing with NaN for plotly gap rendering.
        if np.any(~np.isnan(baseline_arr)):
            series["歷史 baseline"] = baseline_arr

    fig = line_chart(
        x=sub["timestamp"].dt.strftime("%H:%M").tolist(),
        series=series,
        title=f"{last_date}　{vdid} 全日速度",
    )
    st.plotly_chart(fig, use_container_width=True)
