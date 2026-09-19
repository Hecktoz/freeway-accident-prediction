"""Accident table sub-tab: filters + DataFrame.

All filtering is in-memory pandas. Returns component handles for cross-wiring
in app.py.
"""

from __future__ import annotations

from datetime import date

import gradio as gr
import pandas as pd

from scripts.dashboard.state import AppState

_FREEWAYS = ["N1", "N3", "N4", "N6"]
_DIRECTIONS = ["N", "S", "E", "W"]
_DIR_LABELS = {"N": "北 (N)", "S": "南 (S)", "E": "東 (E)", "W": "西 (W)"}

DISPLAY_COLS = [
    "accident_id", "report_time", "freeway", "direction", "mileage_km",
    "性質", "事故類別", "車種", "死亡人數", "受傷人數", "processing_min",
    "has_heatmap", "has_npz",
]


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series("", index=df.index)


def _make_display_df(df: pd.DataFrame, heatmap_ids: set[str], sample_ids: set[str]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["accident_id"] = df["accident_id"].astype(str)
    out["report_time"] = df["report_time"].dt.strftime("%Y-%m-%d %H:%M")
    out["freeway"] = df["freeway"].astype(str)
    out["direction"] = df["direction"].astype(str)
    out["mileage_km"] = df["mileage_km"].round(3)
    for col in ("性質", "事故類別", "車種"):
        out[col] = _safe_col(df, col).fillna("").astype(str)
    for col in ("死亡人數", "受傷人數"):
        out[col] = _safe_col(df, col).fillna(0).astype(int)
    out["processing_min"] = _safe_col(df, "processing_min").astype("Int64")
    out["has_heatmap"] = out["accident_id"].isin(heatmap_ids)
    out["has_npz"] = out["accident_id"].isin(sample_ids)
    return out


def apply_filters(
    app_state: AppState,
    freeways: list[str],
    directions: list[str],
    date_lo: date,
    date_hi: date,
    mileage_lo: float,
    mileage_hi: float,
    has_casualties: bool,
    has_heatmap: bool,
    has_npz: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (display_df, filtered_accident_ids) for the current filter set."""
    df = app_state.accidents_df
    if not freeways:
        freeways = _FREEWAYS
    if not directions:
        directions = _DIRECTIONS

    mask = (
        df["freeway"].isin(freeways)
        & df["direction"].isin(directions)
        & (df["mileage_km"] >= mileage_lo)
        & (df["mileage_km"] <= mileage_hi)
        & (df["report_date"] >= date_lo)
        & (df["report_date"] <= date_hi)
    )
    if has_casualties:
        death_col = _safe_col(df, "死亡人數").fillna(0).astype(float)
        inj_col = _safe_col(df, "受傷人數").fillna(0).astype(float)
        mask = mask & ((death_col > 0) | (inj_col > 0))

    filtered = df[mask].copy()
    heatmap_ids = set(app_state.heatmap_png_index.keys())
    sample_ids = set(app_state.sample_id_index.keys())

    if has_heatmap:
        filtered = filtered[filtered["accident_id"].astype(str).isin(heatmap_ids)]
    if has_npz:
        filtered = filtered[filtered["accident_id"].astype(str).isin(sample_ids)]

    filtered = filtered.sort_values("report_time").reset_index(drop=True)
    display_df = _make_display_df(filtered, heatmap_ids, sample_ids)
    ids = display_df["accident_id"].tolist()
    return display_df, ids


def build(app_state: AppState) -> dict:
    """Build accident table section. Returns component handles."""
    df = app_state.accidents_df

    min_d = df["report_date"].min() if not df.empty else date.today()
    max_d = df["report_date"].max() if not df.empty else date.today()
    m_min = float(df["mileage_km"].min()) if not df.empty else 0.0
    m_max = float(df["mileage_km"].max()) if not df.empty else 200.0

    heatmap_ids = set(app_state.heatmap_png_index.keys())
    sample_ids = set(app_state.sample_id_index.keys())

    with gr.Row():
        fw_filter = gr.CheckboxGroup(
            choices=_FREEWAYS, value=_FREEWAYS, label="國道", scale=1,
        )
        dir_filter = gr.CheckboxGroup(
            choices=list(_DIR_LABELS.keys()),
            value=list(_DIR_LABELS.keys()),
            label="方向", scale=1,
        )
        date_lo_filter = gr.DateTime(
            value=str(min_d),
            label="開始日期",
            include_time=False,
            type="string",
            scale=1,
        )
        date_hi_filter = gr.DateTime(
            value=str(max_d),
            label="結束日期",
            include_time=False,
            type="string",
            scale=1,
        )
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

    with gr.Row():
        casualties_filter = gr.Checkbox(value=False, label="僅傷亡事故")
        heatmap_filter = gr.Checkbox(value=False, label="僅有熱圖 PNG")
        npz_filter = gr.Checkbox(value=False, label="僅有 NPZ 樣本")
        row_count = gr.Textbox(value="", label="篩選結果", interactive=False, scale=2)

    init_display, init_ids = apply_filters(
        app_state, _FREEWAYS, _DIRECTIONS,
        min_d, max_d, round(m_min, 1), round(m_max, 1),
        False, False, False,
    )

    table = gr.DataFrame(
        value=init_display,
        label="事故清單",
        interactive=False,
        wrap=False,
    )

    filtered_ids_state = gr.State(init_ids)

    def on_filter_change(fws, dirs, d_lo_str, d_hi_str, m_lo, m_hi,
                         casualties, has_heat, has_npz_chk):
        def _parse_date(s, fallback):
            try:
                return date(*[int(x) for x in str(s).split("-")[:3]])
            except Exception:
                return fallback
        d_lo = _parse_date(d_lo_str, min_d)
        d_hi = _parse_date(d_hi_str, max_d)
        disp, ids = apply_filters(
            app_state, list(fws or _FREEWAYS), list(dirs or _DIRECTIONS),
            d_lo, d_hi, float(m_lo), float(m_hi),
            bool(casualties), bool(has_heat), bool(has_npz_chk),
        )
        return disp, ids, f"{len(ids):,} 筆"

    filter_inputs = [fw_filter, dir_filter, date_lo_filter, date_hi_filter,
                     mileage_lo_filter, mileage_hi_filter,
                     casualties_filter, heatmap_filter, npz_filter]
    for widget in filter_inputs:
        widget.change(
            fn=on_filter_change,
            inputs=filter_inputs,
            outputs=[table, filtered_ids_state, row_count],
        )

    row_count.value = f"{len(init_ids):,} 筆"

    return {
        "table": table,
        "filtered_ids_state": filtered_ids_state,
        "filter_inputs": filter_inputs,
        "on_filter_change": on_filter_change,
    }
