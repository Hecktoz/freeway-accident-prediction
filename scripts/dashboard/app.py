"""Gradio Blocks entry: top-level tabs + all state wiring.

Launch via `uv run scripts/dashboard/run.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr

from scripts.dashboard import loaders, model_runtime
from scripts.dashboard.state import AppState
from scripts.dashboard.views import (
    accident_detail as ad_view,
    accident_gallery,
    accident_table,
    vd_detail as vd_view,
    vd_list,
    vd_map,
)
from scripts.dashboard.plotting import empty_figure

_EMPTY_FIG = empty_figure("等待", "請選擇一筆事故", height=340)
_EMPTY_VD_FIG = empty_figure("等待", "請選擇一個 VD", height=320)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_app(app_state: AppState) -> gr.Blocks:
    with gr.Blocks(title="高速公路事故速度預測分析儀表板") as demo:

        # ── Global gr.State ──────────────────────────────────────────────
        selected_accident_id = gr.State(None)
        accident_list_ids = gr.State([])
        selected_vd_id = gr.State(None)
        vd_list_ids = gr.State([])

        gr.Markdown("# 高速公路事故速度預測分析儀表板")

        with gr.Tabs() as main_tabs:

            # ============================================================
            # TAB 1: Accidents
            # ============================================================
            with gr.Tab("事故"):
                with gr.Tabs() as accident_subtabs:

                    # ── Table ────────────────────────────────────────────
                    with gr.Tab("清單") as acc_table_tab:
                        tbl_handles = accident_table.build(app_state)
                        acc_table = tbl_handles["table"]
                        filtered_ids_state = tbl_handles["filtered_ids_state"]

                    # ── Gallery ──────────────────────────────────────────
                    with gr.Tab("圖庫") as acc_gallery_tab:
                        gal_handles = accident_gallery.build(app_state, filtered_ids_state)
                        acc_gallery = gal_handles.get("gallery")
                        gal_page_info = gal_handles.get("page_info")

                    # ── Detail ───────────────────────────────────────────
                    with gr.Tab("詳情") as acc_detail_tab:
                        with gr.Row():
                            prev_accident_btn = gr.Button("← 上一筆", size="sm",
                                                          interactive=False, scale=1)
                            with gr.Column(scale=6):
                                detail_title = gr.Markdown("### 請從清單或圖庫選擇一筆事故")
                            next_accident_btn = gr.Button("下一筆 →", size="sm",
                                                          interactive=False, scale=1)

                        detail_header = gr.HTML("")

                        obs_plot = gr.Plot(value=_EMPTY_FIG, label="觀測速度（NPZ）")
                        baseline_plot = gr.Plot(value=_EMPTY_FIG, label="歷史 baseline")

                        # Re-slice button (heavy parquet path)
                        reslice_btn = gr.Button("重新切片（Parquet 路徑）", size="sm")
                        reslice_obs_plot = gr.Plot(value=None, label="觀測速度（重切）", visible=False)
                        reslice_status = gr.HTML("")

                        # Model prediction tabs
                        with gr.Tabs():
                            model_components: dict[str, dict] = {}
                            for mname, mdisplay in model_runtime.MODEL_DISPLAY.items():
                                with gr.Tab(mdisplay):
                                    run_model_btn = gr.Button(f"執行 {mdisplay}", size="sm")
                                    with gr.Row():
                                        m_obs_plot = gr.Plot(label="觀測速度")
                                        m_pred_plot = gr.Plot(label="預測速度")
                                        m_diff_plot = gr.Plot(label="差值")
                                    m_metrics_html = gr.HTML("")
                                    m_recovery_plot = gr.Plot(label="恢復曲線")
                                    m_status_html = gr.HTML("")
                                    model_components[mname] = {
                                        "btn": run_model_btn,
                                        "obs": m_obs_plot,
                                        "pred": m_pred_plot,
                                        "diff": m_diff_plot,
                                        "metrics": m_metrics_html,
                                        "recovery": m_recovery_plot,
                                        "status": m_status_html,
                                    }

                        related_vds_html = gr.HTML("")

            # ============================================================
            # TAB 2: VD Browser
            # ============================================================
            with gr.Tab("VD 瀏覽器"):
                with gr.Tabs() as vd_subtabs:

                    # ── List ─────────────────────────────────────────────
                    with gr.Tab("清單") as vd_list_tab:
                        vdl_handles = vd_list.build(app_state)
                        vd_table = vdl_handles["table"]
                        filtered_vd_ids_state = vdl_handles["filtered_vd_ids_state"]

                    # ── Map ──────────────────────────────────────────────
                    with gr.Tab("地圖（VD）") as vd_map_tab:
                        vdmap_handles = vd_map.build(app_state.vd_records_df, app_state.vd_dim_df)

                    # ── Detail ───────────────────────────────────────────
                    with gr.Tab("詳情") as vd_detail_tab:
                        with gr.Row():
                            prev_vd_btn = gr.Button("← 上一筆", size="sm",
                                                     interactive=False, scale=1)
                            with gr.Column(scale=6):
                                vd_detail_title = gr.Markdown("### 請從清單選擇一個 VD")
                            next_vd_btn = gr.Button("下一筆 →", size="sm",
                                                     interactive=False, scale=1)

                        vd_header_html = gr.HTML("")
                        vd_speed_plot = gr.Plot(value=_EMPTY_VD_FIG, label="速度時序")
                        vd_related_html = gr.HTML("")

            # ============================================================
            # TAB 3: About / Status
            # ============================================================
            with gr.Tab("關於 / 狀態"):
                _build_about_tab(app_state)

        # ================================================================
        # WIRING — Accident navigation
        # ================================================================

        def _render_accident(accident_id: str, ids_list: list[str]) -> tuple:
            """Render detail for accident_id. Return all accident detail outputs."""
            if not accident_id:
                return (
                    None, "### 請選擇一筆事故", "",
                    _EMPTY_FIG, _EMPTY_FIG, "",
                    gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(selected="詳情"),
                )
            header, obs_f, base_f, related = ad_view.render_accident_detail(
                accident_id, app_state
            )
            pos = ids_list.index(accident_id) if accident_id in ids_list else -1
            can_prev = pos > 0
            can_next = 0 <= pos < len(ids_list) - 1
            title_md = f"### 事故 {accident_id}　（{pos + 1} / {len(ids_list)}）"
            return (
                accident_id,
                title_md,
                header,
                obs_f,
                base_f,
                related,
                gr.update(interactive=can_prev),
                gr.update(interactive=can_next),
                gr.update(selected="詳情"),
            )

        _detail_outputs = [
            selected_accident_id,
            detail_title,
            detail_header,
            obs_plot,
            baseline_plot,
            related_vds_html,
            prev_accident_btn,
            next_accident_btn,
            accident_subtabs,
        ]

        # Table row click
        def on_table_select(ids: list[str], evt: gr.SelectData):
            if evt is None or evt.index is None:
                return (None,) + (gr.update(),) * (len(_detail_outputs) - 1)
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if row_idx >= len(ids):
                return (None,) + (gr.update(),) * (len(_detail_outputs) - 1)
            return _render_accident(ids[row_idx], ids)

        acc_table.select(
            fn=on_table_select,
            inputs=[filtered_ids_state],
            outputs=_detail_outputs,
        )

        # Gallery image click
        if acc_gallery is not None:
            def on_gallery_select(ids: list[str], page: int, evt: gr.SelectData):
                if evt is None:
                    return (None,) + (gr.update(),) * (len(_detail_outputs) - 1)
                idx_in_page = evt.index
                full_idx = page * accident_gallery._PAGE_SIZE + idx_in_page
                heatmap_ids = [
                    aid for aid in ids if aid in app_state.heatmap_png_index
                ]
                if full_idx >= len(heatmap_ids):
                    return (None,) + (gr.update(),) * (len(_detail_outputs) - 1)
                return _render_accident(heatmap_ids[full_idx], ids)

            page_state = gal_handles.get("page_state", gr.State(0))
            acc_gallery.select(
                fn=on_gallery_select,
                inputs=[filtered_ids_state, page_state],
                outputs=_detail_outputs,
            )

        # Prev / Next accident
        def on_prev_accident(accident_id: str, ids: list[str]):
            if not accident_id or accident_id not in ids:
                return (gr.update(),) * len(_detail_outputs)
            pos = ids.index(accident_id)
            new_pos = max(0, pos - 1)
            return _render_accident(ids[new_pos], ids)

        def on_next_accident(accident_id: str, ids: list[str]):
            if not accident_id or accident_id not in ids:
                return (gr.update(),) * len(_detail_outputs)
            pos = ids.index(accident_id)
            new_pos = min(len(ids) - 1, pos + 1)
            return _render_accident(ids[new_pos], ids)

        prev_accident_btn.click(
            fn=on_prev_accident,
            inputs=[selected_accident_id, filtered_ids_state],
            outputs=_detail_outputs,
        )
        next_accident_btn.click(
            fn=on_next_accident,
            inputs=[selected_accident_id, filtered_ids_state],
            outputs=_detail_outputs,
        )

        # Re-slice button
        def on_reslice(accident_id: str):
            if not accident_id:
                return gr.update(value=None, visible=False), ""
            df = app_state.accidents_df
            m = df[df["accident_id"].astype(str) == str(accident_id)]
            if m.empty:
                return gr.update(value=None, visible=False), "找不到事故"
            row = m.iloc[0]
            pack = ad_view._cached_observed_parquet(
                str(accident_id),
                str(row["report_time"].isoformat()),
                str(row["freeway"]),
                str(row["direction"]),
                str(row["mileage_km"]),
                str(app_state.data_root),
            )
            if pack is None:
                return gr.update(value=None, visible=False), "此事故時段缺少 station 資料"
            from scripts.dashboard.plotting import speed_heatmap
            fig = speed_heatmap(
                pack["matrix"], pack["space_km"], pack["time_min"],
                "觀測速度（Parquet 重切）", report_time=row["report_time"],
            )
            missing = pack.get("missing_dates", [])
            status = f"完成。缺少日期：{missing}" if missing else "完成。"
            return gr.update(value=fig, visible=True), status

        reslice_btn.click(
            fn=on_reslice,
            inputs=[selected_accident_id],
            outputs=[reslice_obs_plot, reslice_status],
        )

        # Model prediction buttons
        for mname, comps in model_components.items():
            _mname = mname  # capture

            def _make_model_fn(model_name):
                def on_run_model(accident_id: str):
                    if not accident_id:
                        ef = empty_figure(model_runtime.MODEL_DISPLAY[model_name], "請先選擇事故")
                        return ef, ef, ef, "", ef, ""
                    return ad_view.render_model_prediction(accident_id, model_name, app_state)
                return on_run_model

            comps["btn"].click(
                fn=_make_model_fn(_mname),
                inputs=[selected_accident_id],
                outputs=[
                    comps["obs"], comps["pred"], comps["diff"],
                    comps["metrics"], comps["recovery"], comps["status"],
                ],
            )

        # ================================================================
        # WIRING — VD navigation
        # ================================================================

        def _render_vd(vdid: str, ids_list: list[str]) -> tuple:
            if not vdid:
                return (
                    None, "### 請選擇一個 VD", "", _EMPTY_VD_FIG, "",
                    gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(selected="詳情"),
                )
            header, speed_fig, related = vd_view.render_vd_detail(vdid, app_state)
            pos = ids_list.index(vdid) if vdid in ids_list else -1
            can_prev = pos > 0
            can_next = 0 <= pos < len(ids_list) - 1
            title_md = f"### VD {vdid}　（{pos + 1} / {len(ids_list)}）"
            return (
                vdid, title_md, header, speed_fig, related,
                gr.update(interactive=can_prev),
                gr.update(interactive=can_next),
                gr.update(selected="詳情"),
            )

        _vd_detail_outputs = [
            selected_vd_id,
            vd_detail_title,
            vd_header_html,
            vd_speed_plot,
            vd_related_html,
            prev_vd_btn,
            next_vd_btn,
            vd_subtabs,
        ]

        def on_vd_table_select(ids: list[str], evt: gr.SelectData):
            if evt is None or evt.index is None:
                return (None,) + (gr.update(),) * (len(_vd_detail_outputs) - 1)
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if row_idx >= len(ids):
                return (None,) + (gr.update(),) * (len(_vd_detail_outputs) - 1)
            return _render_vd(ids[row_idx], ids)

        vd_table.select(
            fn=on_vd_table_select,
            inputs=[filtered_vd_ids_state],
            outputs=_vd_detail_outputs,
        )

        def on_prev_vd(vdid: str, ids: list[str]):
            if not vdid or vdid not in ids:
                return (gr.update(),) * len(_vd_detail_outputs)
            pos = ids.index(vdid)
            return _render_vd(ids[max(0, pos - 1)], ids)

        def on_next_vd(vdid: str, ids: list[str]):
            if not vdid or vdid not in ids:
                return (gr.update(),) * len(_vd_detail_outputs)
            pos = ids.index(vdid)
            return _render_vd(ids[min(len(ids) - 1, pos + 1)], ids)

        prev_vd_btn.click(
            fn=on_prev_vd,
            inputs=[selected_vd_id, filtered_vd_ids_state],
            outputs=_vd_detail_outputs,
        )
        next_vd_btn.click(
            fn=on_next_vd,
            inputs=[selected_vd_id, filtered_vd_ids_state],
            outputs=_vd_detail_outputs,
        )

        # ================================================================
        # LAZY LOADING — deferred heavy renders on first tab activate
        # ================================================================

        # VD Folium map — 2 710 markers, ~5s; render on first tab click
        vd_map_tab.select(
            fn=vdmap_handles["load_map_fn"],
            inputs=[],
            outputs=[vdmap_handles["map_container"]],
        )

        # Accident gallery — 100 PNG paths; populate on first tab click
        gal_page_state = gal_handles.get("page_state", gr.State(0))
        load_gal_fn = gal_handles.get("load_gallery_fn")
        if acc_gallery is not None and load_gal_fn is not None:
            acc_gallery_tab.select(
                fn=load_gal_fn,
                inputs=[],
                outputs=[acc_gallery, gal_page_info, gal_page_state],
            )

    return demo


# ---------------------------------------------------------------------------
# About / Status tab
# ---------------------------------------------------------------------------

def _build_about_tab(app_state: AppState) -> None:
    ckpts = app_state.checkpoints
    ckpt_lines = "\n".join(
        f"- **{model_runtime.MODEL_DISPLAY.get(k, k)}**: "
        + (f"`{v.parent.parent.name}`" if v else "尚未可用")
        for k, v in ckpts.items()
    )
    gr.Markdown(f"""
## 資料狀態

| 項目 | 數量 |
|---|---|
| 事故 | {len(app_state.accidents_df):,} 筆 |
| VD 記錄 (XML) | {len(app_state.vd_records_df):,} 筆 |
| VD 維度表 | {len(app_state.vd_dim_df):,} 筆 |
| NPZ 樣本 | {len(app_state.samples) if app_state.samples else "未載入"} 筆 |
| 熱圖 PNG | {len(app_state.heatmap_png_index):,} 筆 |

## 模型 Checkpoint 狀態

{ckpt_lines}

## 使用說明

- **事故 → 清單**：套用篩選條件，點擊列即進入詳情頁。
- **事故 → 圖庫**：瀏覽熱圖縮圖，點擊圖片即進入詳情頁（與清單共用篩選狀態）。
- **事故 → 詳情**：觀測熱圖來自 NPZ（即時）；可按「重新切片」從 Parquet 重新讀取；模型預測需按各模型的執行按鈕。
- **VD 瀏覽器 → 清單**：篩選 1 334 個 VD，點擊列進入詳情頁。
- **VD 瀏覽器 → 詳情**：顯示最新一日的速度時序與歷史 baseline。
""")
