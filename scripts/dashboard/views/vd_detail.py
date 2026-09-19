"""VD Browser detail sub-tab.

Shows: static metadata header, speed time-series with historical baseline,
and a list of related accidents within ±2 km.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.dashboard import caches
from scripts.dashboard.plotting import empty_figure, line_chart
from scripts.dashboard.state import AppState

DIRECTION_LABEL = {"N": "北向", "S": "南向", "E": "東向", "W": "西向"}


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _header_html(vdid: str, app_state: AppState) -> str:
    rec_match = app_state.vd_records_df[app_state.vd_records_df["vdid"] == vdid]
    dim_match = (
        app_state.vd_dim_df[app_state.vd_dim_df["vdid"] == vdid]
        if not app_state.vd_dim_df.empty else pd.DataFrame()
    )
    if rec_match.empty:
        return f"<p>找不到 VD <b>{vdid}</b> 的靜態資料。</p>"

    rec = rec_match.iloc[0]
    route = f"{rec['freeway']} {DIRECTION_LABEL.get(rec['direction'], rec['direction'])}"
    mileage = f"{rec['mileage_km']:.3f} K"
    lanes = f"{int(rec.get('lane_num', 0))} 道（實際 {int(rec.get('actual_lane_num', 0))} 道）"
    road_sec = rec.get("road_section", "")

    dim_info = ""
    if not dim_match.empty:
        d = dim_match.iloc[0]
        dim_info = (
            f"資料期間 {d.get('first_source_date', '—')} ～ {d.get('last_source_date', '—')} ｜ "
            f"涵蓋 {int(d.get('n_source_days', 0))} 天"
        )
    else:
        dim_info = "此 VD 不在 vd_station 訓練資料集中"

    cells = [("路線", route), ("里程", mileage), ("車道", lanes), ("路段", road_sec)]
    tds = "".join(
        f'<td style="padding:4px 14px;text-align:center">'
        f'<div style="font-size:11px;color:#64748b">{k}</div>'
        f'<div style="font-size:15px;font-weight:600">{v}</div>'
        f'</td>'
        for k, v in cells
    )
    return (
        f'<h3 style="margin:0 0 6px 0">VD：{vdid}</h3>'
        f'<table style="border-collapse:collapse"><tr>{tds}</tr></table>'
        f'<p style="margin:4px 0;font-size:12px;color:#475569">{dim_info}</p>'
    )


def _related_accidents_html(vdid: str, app_state: AppState) -> str:
    rec_match = app_state.vd_records_df[app_state.vd_records_df["vdid"] == vdid]
    if rec_match.empty or app_state.accidents_df.empty:
        return ""
    rec = rec_match.iloc[0]
    fw, direction, mileage_km = str(rec["freeway"]), str(rec["direction"]), float(rec["mileage_km"])
    df = app_state.accidents_df
    nearby = df[
        (df["freeway"] == fw)
        & (df["direction"] == direction)
        & (df["mileage_km"].between(mileage_km - 2.0, mileage_km + 2.0))
    ].sort_values("report_time", ascending=False).head(20)

    if nearby.empty:
        return "<p style='font-size:13px;color:#64748b'>此 VD ±2 km 範圍內無事故紀錄。</p>"

    rows_html = "".join(
        f'<tr data-accident-id="{row["accident_id"]}">'
        f'<td style="padding:3px 8px">{row["accident_id"]}</td>'
        f'<td style="padding:3px 8px">{pd.Timestamp(row["report_time"]).strftime("%Y-%m-%d %H:%M")}</td>'
        f'<td style="padding:3px 8px">{row["mileage_km"]:.3f} K</td>'
        f'</tr>'
        for _, row in nearby.iterrows()
    )
    return (
        f'<p style="margin:8px 0 4px;font-size:13px;font-weight:600">附近事故（±2 km，最多 20 筆）</p>'
        f'<table style="border-collapse:collapse;font-size:12px;width:100%">'
        f'<tr style="background:#f1f5f9"><th style="padding:3px 8px">事故 ID</th>'
        f'<th style="padding:3px 8px">發送時間</th><th style="padding:3px 8px">里程</th></tr>'
        f'{rows_html}</table>'
    )


# ---------------------------------------------------------------------------
# Speed series computation
# ---------------------------------------------------------------------------

def _baseline_for_vd(
    vdid: str, freeway: str, target_date: pd.Timestamp,
    data_root_str: str, resolution_min: int = 5,
) -> pd.DataFrame | None:
    profile = caches.load_historical_profile(freeway, resolution_min, data_root_str)
    if profile is None or profile.empty:
        return None
    sub = profile[profile["vdid"] == vdid].copy()
    if sub.empty or "minute_bucket" not in sub.columns:
        return None
    if "day_class" in sub.columns:
        from scripts.lib.calendar import day_class
        cls = day_class(target_date.date())
        day_slice = sub[sub["day_class"] == cls]
        if not day_slice.empty:
            sub = day_slice
    grouped = sub.groupby("minute_bucket", as_index=False)["mean_speed_x10"].mean()
    grouped["speed_kmh"] = grouped["mean_speed_x10"] / 10.0
    return grouped.sort_values("minute_bucket").reset_index(drop=True)[["minute_bucket", "speed_kmh"]]


def render_vd_detail(vdid: str, app_state: AppState) -> tuple:
    """Return (header_html, speed_fig, related_html) for the given VD."""
    from scripts.dashboard.loaders import latest_station_date

    header = _header_html(vdid, app_state)
    related = _related_accidents_html(vdid, app_state)

    last_date = latest_station_date(app_state.vd_dim_df)
    if last_date is None:
        speed_fig = empty_figure("速度時序", "尚未有可用的 station 資料")
        return header, speed_fig, related

    target_date = pd.Timestamp(last_date)
    data_root_str = str(app_state.data_root)
    df = caches.load_station_day(last_date, resolution_min=5, data_root_str=data_root_str)
    sub = df[df["vdid"] == vdid] if not df.empty else df

    if sub.empty:
        speed_fig = empty_figure("速度時序", f"{last_date} 沒有此 VD 的速度資料")
        return header, speed_fig, related

    sub = sub.sort_values("timestamp")
    series: dict[str, np.ndarray | list] = {"觀測 (該日)": sub["speed_kmh"].to_numpy()}

    rec_match = app_state.vd_records_df[app_state.vd_records_df["vdid"] == vdid]
    if not rec_match.empty:
        freeway = str(rec_match.iloc[0]["freeway"])
        baseline_df = _baseline_for_vd(vdid, freeway, target_date, data_root_str)
        if baseline_df is not None and not baseline_df.empty:
            minute_bucket = (sub["timestamp"].dt.hour * 60 + sub["timestamp"].dt.minute).astype(int)
            bucket_step = 5
            minute_bucket = (minute_bucket // bucket_step) * bucket_step
            lookup = dict(zip(baseline_df["minute_bucket"], baseline_df["speed_kmh"]))
            baseline_arr = minute_bucket.map(lookup).to_numpy(dtype=float)
            if np.any(~np.isnan(baseline_arr)):
                series["歷史 baseline"] = baseline_arr

    speed_fig = line_chart(
        x=sub["timestamp"].dt.strftime("%H:%M").tolist(),
        series=series,
        title=f"{last_date}　{vdid} 全日速度",
    )
    return header, speed_fig, related
