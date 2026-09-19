"""VD-only Folium map sub-tab.

Renders the 1 334 VDs that appear in vd_dim as coloured circle markers.
No accident layer, no marker capping, no cross-filter.
Clicking a marker calls `on_vd_select_fn(vdid)` if provided.
"""

from __future__ import annotations

from typing import Callable

import gradio as gr
import pandas as pd


FREEWAY_COLORS = {"N1": "#e74c3c", "N3": "#3498db", "N4": "#27ae60", "N6": "#f39c12"}
FREEWAY_LABELS = {"N1": "國道1號", "N3": "國道3號", "N4": "國道4號", "N6": "國道6號"}
DIRECTION_LABEL = {"N": "北向", "S": "南向", "E": "東向", "W": "西向"}


def _build_folium_map(vd_records_df: pd.DataFrame, vd_dim_vdids: set[str]) -> str:
    """Return an HTML string of the Folium map (embedded in an iframe later)."""
    try:
        import folium
        from branca.element import Element
    except ImportError:
        return "<p>Folium is not installed. Run <code>pip install folium branca</code>.</p>"

    in_dim = vd_dim_vdids or set(vd_records_df["vdid"].tolist())
    df = vd_records_df[vd_records_df["vdid"].isin(in_dim)].copy()
    if df.empty:
        df = vd_records_df.copy()

    center_lat = df["lat"].mean() if not df.empty else 23.8
    center_lon = df["lon"].mean() if not df.empty else 121.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles=None,
        prefer_canvas=True,
    )
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr='© OpenStreetMap contributors © CARTO',
        subdomains="abcd",
        max_zoom=20,
    ).add_to(m)

    for _, rec in df.iterrows():
        color = FREEWAY_COLORS.get(str(rec["freeway"]), "#888")
        popup_html = (
            f'<b>{rec["vdid"]}</b><br>'
            f'{rec["freeway"]} {DIRECTION_LABEL.get(rec["direction"], rec["direction"])}<br>'
            f'{rec["mileage_km"]:.3f} K'
        )
        folium.CircleMarker(
            location=[rec["lat"], rec["lon"]],
            radius=4,
            color="#fff",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=rec["vdid"],
        ).add_to(m)

    return m._repr_html_()


def build(vd_records_df: pd.DataFrame, vd_dim_df: pd.DataFrame) -> dict:
    """Build the VD map sub-tab. Map HTML is deferred to first tab activation."""
    in_dim = set(vd_dim_df["vdid"].astype(str).tolist()) if not vd_dim_df.empty else set()

    map_container = gr.HTML(
        "<p style='color:#64748b;padding:20px'>地圖載入中…請稍候（首次開啟需幾秒鐘）</p>"
    )
    _loaded = {"done": False}

    def load_map():
        if _loaded["done"]:
            return gr.update()
        html = _build_folium_map(vd_records_df, in_dim)
        _loaded["done"] = True
        return gr.update(value=html)

    return {"map_container": map_container, "load_map_fn": load_map}
