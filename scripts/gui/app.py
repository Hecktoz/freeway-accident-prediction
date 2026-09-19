# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gradio>=4.44",
#     "httpx[http2]",
#     "lxml",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "torch",
#     "tqdm",
# ]
# ///
"""Phase F GUI — Highway Bureau accident-impact desktop app.

Single-operator, single-machine. Inputs an accident's metadata; outputs the
predicted speed spacetime heatmap, impact duration, and per-mileage recovery
curves. Wraps `scripts.lib.inference.predict_accident_impact`.

Supports three time-progressive stages that map to the trained checkpoints:
    t+0  — initial dispatch info (no incident details yet)
    t+5  — patrol arrives: accident type + special-accident flag available
    t+15 — full assessment: injuries, fatalities, lane blockage confirmed

Run locally:
    uv run scripts/gui/app.py

See docs/DEPLOYMENT.md for the full delivery contract.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

import os as _os
if _os.environ.get("MPLBACKEND", "").startswith("module://"):
    _os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.lib.inference import (
    AccidentInput,
    AccidentPrediction,
    ModelNotAvailableError,
    predict_accident_impact,
)
from scripts.lib.windows import DEFAULT_WINDOW, absolute_mileage_axis_m

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
FREEWAYS = ["N1", "N3", "N4", "N6"]
DIRECTIONS = ["N", "S", "E", "W"]

# Checkpoint identifiers for each time-progressive stage.
# t15 (champion) is loaded via reports/v1_champion.json (name=None, run_id=None).
_STAGE_MODEL: dict[str, tuple[str | None, str | None, str | None]] = {
    "t+0 初始通報": ("unet_feature_fusion", "20260521T150042Z", "direct_concat"),
    "t+5 巡邏抵達": ("unet_feature_fusion", "20260521T150048Z", "direct_concat"),
    "t+15 完整資訊": (None, None, None),
}
STAGE_LABELS = list(_STAGE_MODEL)


def _build_incident_features(
    stage: str,
    accident_type: str | None,
    special_accident: bool,
    injuries: float | None,
    fatalities: float | None,
    lane_blockage_rate: float,
    mileage_km: float,
) -> dict[str, Any]:
    """Assemble the incident_features dict from operator-entered form values.

    Zeroes out fields that are unavailable at each stage, matching the
    training-time masking used for t+0 and t+5 models.
    """
    feats: dict[str, Any] = {"mileage": float(mileage_km)}

    # Always include direction/highway stubs (mileage already covers location).
    # Geometry and weather (static) are not yet fetched from vd_dim/API here;
    # they default to 0 and the model handles the absence gracefully.

    if stage in ("t+5 巡邏抵達", "t+15 完整資訊"):
        if accident_type == "A1（死亡）":
            feats.update({"accident_A1": 1.0, "accident_A2": 0.0, "accident_A3": 0.0})
        elif accident_type == "A2（受傷）":
            feats.update({"accident_A1": 0.0, "accident_A2": 1.0, "accident_A3": 0.0})
        elif accident_type == "A3（財損）":
            feats.update({"accident_A1": 0.0, "accident_A2": 0.0, "accident_A3": 1.0})
        feats["special_accident"] = 1.0 if special_accident else 0.0

    if stage == "t+15 完整資訊":
        feats["injuries"] = float(injuries) if injuries is not None else 0.0
        feats["fatalities"] = float(fatalities) if fatalities is not None else 0.0
        feats["lane_blockage_rate"] = float(lane_blockage_rate)

    return feats


def _render_heatmap(prediction: AccidentPrediction, accident: AccidentInput, stage: str) -> Image.Image:
    window = prediction.window
    abs_mileage_km = absolute_mileage_axis_m(accident.mileage_m, accident.direction, window) / 1000.0
    time_offsets = window.time_offsets_min

    fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
    im = ax.imshow(
        prediction.heatmap,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0,
        vmax=140,
        extent=[time_offsets[0], time_offsets[-1], abs_mileage_km[0], abs_mileage_km[-1]],
    )
    if accident.direction in ("S", "E"):
        ax.invert_yaxis()
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="通報時刻")
    ax.scatter([0], [accident.mileage_km], color="red", s=50, zorder=5)
    ax.set_xlabel("距通報時刻（分鐘）")
    ax.set_ylabel("里程（km）")
    ax.set_title(
        f"{accident.report_time:%Y-%m-%d %H:%M} | {accident.freeway}-{accident.direction} | "
        f"{accident.mileage_km:.1f}K | 影響 {prediction.duration_min:.0f} 分鐘 [{stage}]"
    )
    fig.colorbar(im, ax=ax, label="預測速度（km/h）")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _format_coverage(coverage: dict, stage: str) -> str:
    if not coverage:
        return ""
    lines: list[str] = [f"模型階段：{stage}"]
    pre = coverage.get("filled_ratio_pre")
    if pre is not None:
        lines.append(f"事故前 VD 覆蓋率：{pre:.1%}")
        if pre < 0.8:
            lines.append("⚠ 事故前資料覆蓋率過低，模型結果可能不可靠。")
    base = coverage.get("filled_ratio_baseline")
    if base is not None:
        lines.append(f"歷史 baseline 覆蓋率：{base:.1%}")
    if "vd_count" in coverage:
        lines.append(f"使用的 VD 數：{coverage['vd_count']}")
    status = coverage.get("live_fetch_status")
    if isinstance(status, dict):
        downloaded = status.get("minutes_downloaded")
        requested = status.get("minutes_requested")
        if downloaded is not None and requested is not None:
            lines.append(f"即時資料：{downloaded}/{requested} 分鐘下載成功")
        if status.get("minutes_error", 0):
            lines.append(f"⚠ 下載錯誤：{status['minutes_error']} 分鐘")
    return "\n".join(lines)


def _format_model_input(model_input, stage: str) -> str:
    if model_input is None:
        return "資料管線未完成。"
    lines = [
        f"資料管線執行成功，但 [{stage}] 的 checkpoint 尚未找到。",
        f"事故前 VD 覆蓋率：{model_input.coverage_pre.get('filled_ratio', 0):.1%}",
        f"歷史 baseline 覆蓋率：{model_input.coverage_baseline.get('filled_ratio', 0):.1%}",
    ]
    if model_input.fetch_status:
        downloaded = model_input.fetch_status.get("minutes_downloaded")
        requested = model_input.fetch_status.get("minutes_requested")
        if downloaded is not None and requested is not None:
            lines.append(f"即時資料：{downloaded}/{requested} 分鐘下載成功")
    return "\n".join(lines)


def run_prediction(
    report_date: str,
    report_time_str: str,
    freeway: str,
    direction: str,
    mileage_km: float,
    stage: str,
    accident_type: str | None,
    special_accident: bool,
    injuries: float | None,
    fatalities: float | None,
    lane_blockage_rate: float,
    data_root: str,
):
    """Gradio handler: validate inputs, call inference, render outputs."""
    try:
        ts = datetime.strptime(f"{report_date} {report_time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None, "", "日期或時間格式錯誤，請使用 YYYY-MM-DD 與 HH:MM。"

    if ts > datetime.now():
        return None, "", "事故發送時間不能晚於現在。"

    if mileage_km <= 0:
        return None, "", "里程必須為正數。"

    feats = _build_incident_features(
        stage=stage,
        accident_type=accident_type,
        special_accident=special_accident,
        injuries=injuries,
        fatalities=fatalities,
        lane_blockage_rate=lane_blockage_rate,
        mileage_km=mileage_km,
    )

    accident = AccidentInput(
        report_time=ts,
        freeway=freeway,
        direction=direction,
        mileage_km=float(mileage_km),
        features=feats,
    )

    model_name, run_id, experiment = _STAGE_MODEL.get(stage, (None, None, None))

    try:
        prediction = predict_accident_impact(
            accident=accident,
            data_root=Path(data_root),
            model_name=model_name,
            run_id=run_id,
            experiment=experiment,
        )
    except ModelNotAvailableError as e:
        return None, "", _format_model_input(e.model_input, stage)
    except Exception as e:
        return None, "", f"預測失敗：{e}"

    image = _render_heatmap(prediction, accident, stage)
    duration = f"事故影響時間估計：{prediction.duration_min:.0f} 分鐘"
    coverage = _format_coverage(prediction.coverage, stage)
    return image, duration, coverage


def _stage_visibility(stage: str):
    """Return gr.update pairs for t+5 and t+15 field groups."""
    show_t5 = stage in ("t+5 巡邏抵達", "t+15 完整資訊")
    show_t15 = stage == "t+15 完整資訊"
    return gr.update(visible=show_t5), gr.update(visible=show_t15)


def build_app(data_root: Path) -> gr.Blocks:
    with gr.Blocks(title="高速公路事故影響預測") as demo:
        gr.Markdown(
            "# 高速公路事故影響預測\n"
            "輸入事故通報資訊，系統預測事故後的速度時空 heatmap、影響時間，以及各里程的恢復曲線。\n\n"
            "依資訊到位情況選擇資訊階段：**t+0** 在通報後即可預測，**t+5**/**t+15** 隨派遣資訊更新。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 事故基本資料")
                report_date = gr.Textbox(
                    label="事故發送日期 (YYYY-MM-DD)",
                    value=datetime.now().strftime("%Y-%m-%d"),
                )
                report_time_in = gr.Textbox(
                    label="事故發送時間 (HH:MM，台灣本地時間)",
                    value=datetime.now().strftime("%H:%M"),
                )
                freeway = gr.Dropdown(label="國道", choices=FREEWAYS, value="N1")
                direction = gr.Dropdown(label="方向", choices=DIRECTIONS, value="S")
                mileage_km = gr.Number(label="里程 (km)", value=147.0, precision=1)

                gr.Markdown("### 資訊階段")
                stage = gr.Radio(
                    choices=STAGE_LABELS,
                    value=STAGE_LABELS[0],
                    label="選擇目前可確認的資訊階段",
                )

                with gr.Group(visible=False) as t5_group:
                    gr.Markdown("**t+5 資訊（巡邏抵達後）**")
                    accident_type = gr.Radio(
                        choices=["A1（死亡）", "A2（受傷）", "A3（財損）"],
                        label="事故類別",
                        value=None,
                    )
                    special_accident = gr.Checkbox(label="特殊事故（翻覆 / 火燒 / 危品）", value=False)

                with gr.Group(visible=False) as t15_group:
                    gr.Markdown("**t+15 資訊（現場評估後）**")
                    injuries = gr.Number(label="傷者人數", value=None, minimum=0, precision=0)
                    fatalities = gr.Number(label="死亡人數", value=None, minimum=0, precision=0)
                    lane_blockage = gr.Slider(
                        label="車道阻斷比例（0 = 無阻斷，1 = 全阻斷）",
                        minimum=0.0, maximum=1.0, step=0.05, value=0.0,
                    )

                data_root_box = gr.Textbox(
                    label="資料根目錄",
                    value=str(data_root),
                    visible=False,
                )
                submit = gr.Button("預測", variant="primary")

            with gr.Column(scale=2):
                heatmap_out = gr.Image(label="預測速度時空 heatmap", type="pil")
                duration_out = gr.Textbox(label="影響時間", interactive=False)
                coverage_out = gr.Textbox(label="覆蓋率 / 狀態", interactive=False, lines=5)

        stage.change(
            fn=_stage_visibility,
            inputs=[stage],
            outputs=[t5_group, t15_group],
        )

        submit.click(
            fn=run_prediction,
            inputs=[
                report_date, report_time_in, freeway, direction, mileage_km,
                stage, accident_type, special_accident, injuries, fatalities, lane_blockage,
                data_root_box,
            ],
            outputs=[heatmap_out, duration_out, coverage_out],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Highway Bureau accident-impact GUI")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Open a public Gradio share link")
    args = parser.parse_args()

    demo = build_app(args.data_root)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
