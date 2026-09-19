"""Accident gallery sub-tab: thumbnails from output/accident_heatmaps_5m/.

Shares filter state with the accident table. Returns component handles.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from scripts.dashboard.state import AppState

_PAGE_SIZE = 100


def _gallery_items(
    app_state: AppState,
    filtered_ids: list[str],
    page: int = 0,
) -> list[tuple[str, str]]:
    """Return (path_str, label) tuples for the current page."""
    start = page * _PAGE_SIZE
    end = start + _PAGE_SIZE
    page_ids = filtered_ids[start:end]
    items: list[tuple[str, str]] = []
    df = app_state.accidents_df
    id_map: dict[str, pd.Series] = {}
    if not df.empty:
        for _, row in df[df["accident_id"].astype(str).isin(set(page_ids))].iterrows():
            id_map[str(row["accident_id"])] = row

    for aid in page_ids:
        png_path = app_state.heatmap_png_index.get(aid)
        if png_path is None:
            continue
        row = id_map.get(aid)
        if row is not None:
            rt = pd.Timestamp(row["report_time"])
            label = (
                f"{aid} | {row['freeway']}{row['direction']} "
                f"{row['mileage_km']:.1f}k | {rt.strftime('%m-%d %H:%M')}"
            )
        else:
            label = aid
        items.append((str(png_path), label))
    return items


def build(app_state: AppState, filtered_ids_state: gr.State) -> dict:
    """Build the gallery section. Returns component handles."""
    heatmap_dir = next(iter(app_state.heatmap_png_index.values()), None)
    if not app_state.heatmap_png_index:
        gr.Markdown(
            "**熱圖 PNG 未找到。**\n\n"
            "請先執行：\n```\nuv run scripts/visualization/accident_heatmap.py\n```"
        )
        page_ids_state = gr.State([])
        return {"page_ids_state": page_ids_state}

    page_state = gr.State(0)
    page_ids_state = gr.State([])

    init_ids = app_state.accidents_df["accident_id"].astype(str).tolist()
    total_pages = max(1, (len(init_ids) + _PAGE_SIZE - 1) // _PAGE_SIZE)
    _init_page_info = f"第 1 / {total_pages} 頁（共 {len(init_ids):,} 筆）"
    _loaded_gal = {"done": False}

    with gr.Row():
        prev_page_btn = gr.Button("← 上一頁", size="sm", scale=1)
        page_info = gr.Textbox(
            value=_init_page_info,
            interactive=False, show_label=False, scale=4,
        )
        next_page_btn = gr.Button("下一頁 →", size="sm", scale=1)

    gallery = gr.Gallery(
        value=[],
        label="事故熱圖瀏覽（切換至此頁籤後載入）",
        columns=4,
        height=560,
        object_fit="contain",
        show_label=True,
    )

    def load_gallery_fn():
        if _loaded_gal["done"]:
            return gr.update(), gr.update(), gr.update()
        items = _gallery_items(app_state, init_ids, 0)
        _loaded_gal["done"] = True
        return items, _init_page_info, 0

    def update_gallery(filtered_ids: list[str], page: int) -> tuple:
        items = _gallery_items(app_state, filtered_ids, page)
        n = len(filtered_ids)
        total = max(1, (n + _PAGE_SIZE - 1) // _PAGE_SIZE)
        info = f"第 {page + 1} / {total} 頁（共 {n:,} 筆）"
        return items, info, filtered_ids

    def on_prev_page(filtered_ids: list[str], page: int) -> tuple:
        new_page = max(0, page - 1)
        items, info, _ = update_gallery(filtered_ids, new_page)
        return items, info, new_page

    def on_next_page(filtered_ids: list[str], page: int) -> tuple:
        total = max(1, (len(filtered_ids) + _PAGE_SIZE - 1) // _PAGE_SIZE)
        new_page = min(total - 1, page + 1)
        items, info, _ = update_gallery(filtered_ids, new_page)
        return items, info, new_page

    prev_page_btn.click(
        fn=on_prev_page,
        inputs=[filtered_ids_state, page_state],
        outputs=[gallery, page_info, page_state],
    )
    next_page_btn.click(
        fn=on_next_page,
        inputs=[filtered_ids_state, page_state],
        outputs=[gallery, page_info, page_state],
    )

    filtered_ids_state.change(
        fn=lambda ids: (_gallery_items(app_state, ids, 0),
                        f"第 1 / {max(1,(len(ids)+_PAGE_SIZE-1)//_PAGE_SIZE)} 頁（共 {len(ids):,} 筆）",
                        0),
        inputs=[filtered_ids_state],
        outputs=[gallery, page_info, page_state],
    )

    return {
        "gallery": gallery,
        "page_info": page_info,
        "page_ids_state": page_ids_state,
        "page_state": page_state,
        "load_gallery_fn": load_gallery_fn,
    }
