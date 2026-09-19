"""VD Browser list sub-tab: filters + DataFrame.

All filtering is in-memory. Returns component handles for cross-wiring.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from scripts.dashboard.state import AppState

_FREEWAYS = ["N1", "N3", "N4", "N6"]
_DIRECTIONS = ["N", "S", "E", "W"]


def _make_vd_display_df(vd_records_df: pd.DataFrame, vd_dim_df: pd.DataFrame) -> pd.DataFrame:
    """Merge vd_records + vd_dim into a single display DataFrame."""
    base = vd_records_df[["vdid", "freeway", "direction", "mileage_km",
                           "road_section", "lane_num", "actual_lane_num"]].copy()
    if not vd_dim_df.empty:
        dim_cols = ["vdid"]
        for col in ("n_source_days", "first_source_date", "last_source_date"):
            if col in vd_dim_df.columns:
                dim_cols.append(col)
        merged = base.merge(vd_dim_df[dim_cols], on="vdid", how="left")
    else:
        merged = base.copy()
        for col in ("n_source_days", "first_source_date", "last_source_date"):
            merged[col] = pd.NA
    merged["mileage_km"] = merged["mileage_km"].round(3)
    return merged.reset_index(drop=True)


def apply_vd_filters(
    full_df: pd.DataFrame,
    freeways: list[str],
    directions: list[str],
    mileage_lo: float,
    mileage_hi: float,
    vdid_search: str,
) -> tuple[pd.DataFrame, list[str]]:
    df = full_df
    if freeways:
        df = df[df["freeway"].isin(freeways)]
    if directions:
        df = df[df["direction"].isin(directions)]
    df = df[df["mileage_km"].between(mileage_lo, mileage_hi)]
    if vdid_search.strip():
        df = df[df["vdid"].str.contains(vdid_search.strip(), case=False, na=False)]
    df = df.sort_values(["freeway", "direction", "mileage_km"]).reset_index(drop=True)
    return df, df["vdid"].tolist()


def build(app_state: AppState) -> dict:
    """Build the VD list section. Returns component handles."""
    full_df = _make_vd_display_df(app_state.vd_records_df, app_state.vd_dim_df)

    m_min = float(full_df["mileage_km"].min()) if not full_df.empty else 0.0
    m_max = float(full_df["mileage_km"].max()) if not full_df.empty else 200.0

    with gr.Row():
        fw_filter = gr.CheckboxGroup(choices=_FREEWAYS, value=_FREEWAYS, label="國道", scale=1)
        dir_filter = gr.CheckboxGroup(choices=_DIRECTIONS, value=_DIRECTIONS, label="方向", scale=1)
        mileage_lo_filter = gr.Slider(
            minimum=round(m_min, 1), maximum=round(m_max, 1),
            value=round(m_min, 1),
            label="里程下限 (km)", step=1.0, scale=1,
        )
        mileage_hi_filter = gr.Slider(
            minimum=round(m_min, 1), maximum=round(m_max, 1),
            value=round(m_max, 1),
            label="里程上限 (km)", step=1.0, scale=1,
        )
        vdid_search = gr.Textbox(label="VDID 搜尋", placeholder="輸入 VDID 關鍵字…", scale=2)

    with gr.Row():
        row_count = gr.Textbox(value=f"{len(full_df):,} 筆", label="篩選結果",
                               interactive=False, scale=1)

    init_df, init_ids = apply_vd_filters(
        full_df, _FREEWAYS, _DIRECTIONS,
        round(m_min, 1), round(m_max, 1), "",
    )

    table = gr.DataFrame(value=init_df, label="VD 清單", interactive=False, wrap=False)
    filtered_vd_ids_state = gr.State(init_ids)

    def on_filter_change(fws, dirs, m_lo, m_hi, search):
        disp, ids = apply_vd_filters(
            full_df, list(fws or _FREEWAYS), list(dirs or _DIRECTIONS),
            float(m_lo), float(m_hi), search or "",
        )
        return disp, ids, f"{len(ids):,} 筆"

    filter_inputs = [fw_filter, dir_filter, mileage_lo_filter, mileage_hi_filter, vdid_search]
    for widget in filter_inputs:
        widget.change(
            fn=on_filter_change,
            inputs=filter_inputs,
            outputs=[table, filtered_vd_ids_state, row_count],
        )

    return {
        "table": table,
        "filtered_vd_ids_state": filtered_vd_ids_state,
        "full_df": full_df,
    }
