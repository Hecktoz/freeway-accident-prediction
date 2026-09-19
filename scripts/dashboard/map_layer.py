"""Folium map construction for the dashboard.

Builds a Leaflet map with two FeatureGroups (VD / accident) and a marker_index
dict so click events (lat/lng only, per streamlit-folium) can be reverse-mapped
to a (kind, id) pair.
"""

from __future__ import annotations

from typing import Any

import folium
from branca.element import Element
import pandas as pd

from .geo import round_key

FREEWAY_COLORS = {"N1": "#e74c3c", "N3": "#3498db", "N4": "#27ae60", "N6": "#f39c12"}
FREEWAY_LABELS = {"N1": "國道1號", "N3": "國道3號", "N4": "國道4號", "N6": "國道6號"}
DIRECTION_LABEL = {"N": "北向", "S": "南向", "E": "東向", "W": "西向"}


def _vd_popup(rec: dict) -> str:
    return (
        f'<div style="font-size:12px;line-height:1.5;min-width:180px">'
        f'<b style="font-size:13px">{rec["vdid"]}</b><br>'
        f'路線：{rec["freeway"]} {DIRECTION_LABEL.get(rec["direction"], rec["direction"])}<br>'
        f'里程：{rec["mileage_km"]:.3f} K<br>'
        f'路段：{rec.get("road_section", "")}<br>'
        f'</div>'
    )


def _accident_popup(row: pd.Series) -> str:
    rt = pd.Timestamp(row["report_time"])
    return (
        f'<div style="font-size:12px;line-height:1.5;min-width:200px">'
        f'<b style="font-size:13px">事故 {row["accident_id"]}</b><br>'
        f'發送時間：{rt.strftime("%Y-%m-%d %H:%M")}<br>'
        f'路線：{row["freeway"]} {DIRECTION_LABEL.get(row["direction"], row["direction"])}<br>'
        f'里程：{row["mileage_km"]:.3f} K<br>'
        f'</div>'
    )


def build_map(
    vd_df: pd.DataFrame,
    accident_df: pd.DataFrame,
    show_vd: bool,
    show_accident: bool,
    center: tuple[float, float] = (23.8, 121.0),
    zoom: int = 8,
) -> tuple[folium.Map, dict[tuple[float, float], tuple[str, str]]]:
    m = folium.Map(
        location=list(center),
        zoom_start=zoom,
        tiles=None,
        prefer_canvas=True,
    )
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
        name="底圖",
        subdomains="abcd",
        max_zoom=20,
    ).add_to(m)

    marker_index: dict[tuple[float, float], tuple[str, str]] = {}

    vd_group = folium.FeatureGroup(name="VD 站點", show=show_vd)
    if show_vd and not vd_df.empty:
        for rec in vd_df.to_dict(orient="records"):
            color = FREEWAY_COLORS.get(rec["freeway"], "#888")
            folium.CircleMarker(
                location=[rec["lat"], rec["lon"]],
                radius=4,
                color="#fff",
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=folium.Popup(_vd_popup(rec), max_width=260),
                tooltip=rec["vdid"],
            ).add_to(vd_group)
            marker_index[round_key(rec["lat"], rec["lon"])] = ("vd", rec["vdid"])
    vd_group.add_to(m)

    acc_group = folium.FeatureGroup(name="事故", show=show_accident)
    if show_accident and not accident_df.empty:
        for _, row in accident_df.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color="#7f0000",
                weight=1,
                fill=True,
                fill_color="#e53935",
                fill_opacity=0.95,
                popup=folium.Popup(_accident_popup(row), max_width=280),
                tooltip=f"事故 {row['accident_id']}",
            ).add_to(acc_group)
            # Accident wins over VD on lat/lng collision.
            marker_index[round_key(row["lat"], row["lon"])] = ("accident", str(row["accident_id"]))
    acc_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # --- Map legend ---
    _add_legend(m, show_vd=show_vd, show_accident=show_accident)

    return m, marker_index


def _add_legend(m: folium.Map, *, show_vd: bool, show_accident: bool) -> None:
    """Inject a fixed-position HTML legend into the folium map."""
    items: list[str] = []
    if show_vd:
        for fw, color in FREEWAY_COLORS.items():
            label = FREEWAY_LABELS.get(fw, fw)
            items.append(
                f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'border-radius:50%;background:{color};border:1px solid #fff;'
                f'flex-shrink:0"></span>'
                f'<span>VD — {label}</span></div>'
            )
    if show_accident:
        items.append(
            '<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
            '<span style="display:inline-block;width:14px;height:14px;'
            'border-radius:50%;background:#e53935;border:2px solid #7f0000;'
            'flex-shrink:0"></span>'
            '<span>事故位置</span></div>'
        )
    if not items:
        return
    html = (
        '<div style="position:fixed;bottom:30px;right:12px;z-index:1000;'
        'background:rgba(255,255,255,0.92);border:1px solid #aaa;'
        'border-radius:6px;padding:10px 14px;font-size:13px;'
        'font-family:sans-serif;line-height:1.6;box-shadow:0 2px 6px rgba(0,0,0,.15);">'
        '<div style="font-weight:700;margin-bottom:4px;font-size:14px">圖例</div>'
        + "\n".join(items)
        + "</div>"
    )
    m.get_root().html.add_child(Element(html))


def lookup_marker(marker_index: dict, lat: float, lon: float) -> tuple[str, str] | None:
    return marker_index.get(round_key(lat, lon))
