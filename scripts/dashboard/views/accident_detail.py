"""Accident detail sub-tab.

Three sections:
  1. Observed heatmap — from NPZ (instant); optional re-slice button
  2. Historical baseline heatmap — first-use load of historical profile parquet
  3. Model predictions — one inner tab per checkpoint (button-triggered)

Returns component handles and render functions for wiring in app.py.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.dashboard import caches, metrics, model_runtime
from scripts.dashboard.plotting import (
    diff_heatmap,
    empty_figure,
    recovery_curve_plot,
    speed_heatmap,
)
from scripts.dashboard.state import AppState

DIRECTION_LABEL = {"N": "北向", "S": "南向", "E": "東向", "W": "西向"}

_DEFAULT_EMPTY_FIG = empty_figure("等待", "請選擇一筆事故", height=340)


# ---------------------------------------------------------------------------
# Header HTML
# ---------------------------------------------------------------------------

def _header_html(row: pd.Series) -> str:
    rt = pd.Timestamp(row["report_time"])
    inj = row.get("受傷人數", 0)
    death = row.get("死亡人數", 0)
    try:
        inj_n = int(inj) if pd.notna(inj) else 0
    except (ValueError, TypeError):
        inj_n = 0
    try:
        death_n = int(death) if pd.notna(death) else 0
    except (ValueError, TypeError):
        death_n = 0

    proc_min = row.get("processing_min", None)
    proc_str = f"{proc_min} 分" if pd.notna(proc_min) and proc_min else "—"

    cells = [
        ("事故 ID", str(row["accident_id"])),
        ("發送時間", rt.strftime("%Y-%m-%d %H:%M")),
        ("路線", f"{row['freeway']} {DIRECTION_LABEL.get(row['direction'], row['direction'])}"),
        ("里程", f"{row['mileage_km']:.3f} K"),
        ("傷亡", f"{inj_n} 傷 / {death_n} 亡"),
        ("處理時長", proc_str),
    ]
    tds = "".join(
        f'<td style="padding:4px 14px;text-align:center">'
        f'<div style="font-size:11px;color:#64748b">{k}</div>'
        f'<div style="font-size:15px;font-weight:600">{v}</div>'
        f'</td>'
        for k, v in cells
    )
    extras = []
    for k in ("性質", "事故類別", "車種", "位置", "路段"):
        v = row.get(k)
        if pd.notna(v) and str(v).strip():
            extras.append(f"{k}：{v}")
    extra_html = (
        f'<p style="margin:4px 0;font-size:12px;color:#475569">{"　｜　".join(extras)}</p>'
        if extras else ""
    )
    return (
        f'<table style="border-collapse:collapse"><tr>{tds}</tr></table>'
        f'{extra_html}'
    )


# ---------------------------------------------------------------------------
# NPZ observed heatmap
# ---------------------------------------------------------------------------

def _npz_observed(accident_id: str, app_state: AppState) -> dict[str, Any] | None:
    """Return dict with matrix, space_axis_km, time_offsets_min; or None."""
    idx = app_state.sample_id_index.get(accident_id)
    if idx is None or app_state.samples is None:
        return None
    matrix = app_state.samples.matrix[idx].copy()
    space_km = (app_state.samples.space_offsets_m.astype(float) / 1000.0).tolist()
    time_min = app_state.samples.time_offsets_min.astype(int).tolist()
    return {"matrix": matrix, "space_km": space_km, "time_min": time_min}


# ---------------------------------------------------------------------------
# Parquet re-slice (button-triggered, heavy)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _cached_observed_parquet(
    accident_id: str,
    report_time_iso: str,
    freeway: str,
    direction: str,
    mileage_km_str: str,
    data_root_str: str,
) -> dict[str, Any] | None:
    from scripts.lib.windows import DEFAULT_WINDOW, coverage, rasterize_window, slice_station_df

    window = DEFAULT_WINDOW
    report_time = pd.Timestamp(report_time_iso)
    mileage_km = float(mileage_km_str)
    mileage_m = int(round(mileage_km * 1000))

    start = report_time - pd.Timedelta(minutes=window.time_before_min)
    end = report_time + pd.Timedelta(minutes=window.time_after_min)
    dates = [
        ts.strftime("%Y-%m-%d")
        for ts in pd.date_range(start.normalize(), end.normalize(), freq="D")
    ]

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for d in dates:
        df = caches.load_station_day(d, resolution_min=window.resolution_min,
                                     data_root_str=data_root_str)
        if df.empty:
            missing.append(d)
        else:
            frames.append(df)
    if not frames:
        return None

    station_df = pd.concat(frames, ignore_index=True)
    sliced = slice_station_df(
        station_df,
        accident_mileage_m=mileage_m,
        direction=direction,
        freeway=freeway,
        report_time=report_time,
        window=window,
    )
    matrix = rasterize_window(sliced, window, value_col="speed_x10")
    space_km = (window.space_offsets_m.astype(float) / 1000.0).tolist()
    time_min = window.time_offsets_min.astype(int).tolist()
    return {
        "matrix": matrix.astype(np.float32),
        "space_km": space_km,
        "time_min": time_min,
        "coverage": coverage(matrix),
        "missing_dates": missing,
    }


# ---------------------------------------------------------------------------
# Historical baseline
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _cached_baseline(
    accident_id: str,
    report_time_iso: str,
    freeway: str,
    direction: str,
    mileage_km_str: str,
    space_n: int,
    time_n: int,
    data_root_str: str,
) -> np.ndarray | None:
    from scripts.lib.inference import AccidentInput, build_historical_baseline
    from scripts.lib.windows import DEFAULT_WINDOW

    window = DEFAULT_WINDOW
    profile = caches.load_historical_profile(freeway, window.resolution_min, data_root_str)
    if profile.empty:
        return None
    accident = AccidentInput(
        report_time=pd.Timestamp(report_time_iso).to_pydatetime(),
        freeway=freeway,
        direction=direction,
        mileage_km=float(mileage_km_str),
    )
    baseline, _ = build_historical_baseline(accident, window, profile)
    if np.count_nonzero(~np.isnan(baseline)) == 0:
        return None
    return baseline.astype(np.float32)


# ---------------------------------------------------------------------------
# Model prediction (button-triggered, heaviest)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _cached_prediction(
    accident_id: str,
    report_time_iso: str,
    freeway: str,
    direction: str,
    mileage_km_str: str,
    model_name: str,
    ckpt_str: str,
    data_root_str: str,
) -> dict[str, Any] | str:
    """Returns result dict or error string."""
    from scripts.lib.inference import AccidentInput, predict_accident_impact
    from scripts.lib.windows import DEFAULT_WINDOW

    try:
        forward_fn = model_runtime.get_forward_fn(model_name, ckpt_str)
    except Exception as exc:  # noqa: BLE001
        return f"模型載入失敗：{exc}"

    window = DEFAULT_WINDOW
    try:
        window = model_runtime.training_window_for(model_name, Path(ckpt_str)) or window
    except Exception:  # noqa: BLE001
        pass

    def fetch_live(start, end):
        from scripts.lib.inference import FetchResult
        dates = pd.date_range(pd.Timestamp(start).normalize(),
                              pd.Timestamp(end).normalize(), freq="D")
        frames: list[pd.DataFrame] = []
        for ts in dates:
            df = caches.load_station_day(ts.strftime("%Y-%m-%d"), resolution_min=5,
                                         data_root_str=data_root_str)
            if not df.empty:
                frames.append(df)
        if frames:
            allf = pd.concat(frames, ignore_index=True)
            allf = allf[
                (allf["timestamp"] >= pd.Timestamp(start))
                & (allf["timestamp"] <= pd.Timestamp(end))
            ]
        else:
            allf = pd.DataFrame()
        return FetchResult(station_5m=allf, status={"source": "vd_station_5m", "rows": len(allf)})

    accident = AccidentInput(
        report_time=pd.Timestamp(report_time_iso).to_pydatetime(),
        freeway=freeway,
        direction=direction,
        mileage_km=float(mileage_km_str),
    )
    try:
        pred = predict_accident_impact(
            accident=accident,
            data_root=Path(data_root_str),
            fetch_live_vd=fetch_live,
            forward_fn=forward_fn,
            model_name=model_name,
            window=window,
        )
    except Exception as exc:  # noqa: BLE001
        return f"推論失敗：{exc}"

    return {
        "heatmap": pred.heatmap.astype(np.float32),
        "duration_min": float(pred.duration_min),
        "window": window,
    }


# ---------------------------------------------------------------------------
# Related VDs
# ---------------------------------------------------------------------------

def _related_vds_html(row: pd.Series, app_state: AppState) -> str:
    fw = str(row["freeway"])
    direction = str(row["direction"])
    mileage_km = float(row["mileage_km"])
    dim = app_state.vd_dim_df
    if dim.empty:
        return ""
    nearby = app_state.vd_records_df[
        (app_state.vd_records_df["freeway"] == fw)
        & (app_state.vd_records_df["direction"] == direction)
        & (app_state.vd_records_df["mileage_km"].between(mileage_km - 2.0, mileage_km + 2.0))
    ].merge(dim[["vdid"]].assign(_in_dim=True), on="vdid", how="left")

    nearby_dim = nearby[nearby["_in_dim"] == True].head(10)  # noqa: E712
    if nearby_dim.empty:
        return "<p style='font-size:13px;color:#64748b'>此事故 ±2 km 範圍內無 vd_dim VD。</p>"

    rows_html = "".join(
        f'<tr>'
        f'<td style="padding:3px 8px">{r["vdid"]}</td>'
        f'<td style="padding:3px 8px">{r["mileage_km"]:.3f} K</td>'
        f'</tr>'
        for _, r in nearby_dim.iterrows()
    )
    return (
        f'<p style="margin:8px 0 4px;font-size:13px;font-weight:600">相關 VD（±2 km）</p>'
        f'<table style="border-collapse:collapse;font-size:12px">'
        f'<tr style="background:#f1f5f9"><th style="padding:3px 8px">VDID</th>'
        f'<th style="padding:3px 8px">里程</th></tr>'
        f'{rows_html}</table>'
    )


# ---------------------------------------------------------------------------
# Resample helper (same as the old accident_panel)
# ---------------------------------------------------------------------------

def _resample(observed: np.ndarray, obs_space: list, obs_time: list,
              tgt_space: list, tgt_time: list) -> np.ndarray:
    if observed.size == 0:
        return np.full((len(tgt_space), len(tgt_time)), np.nan, dtype=np.float32)
    src_t = np.asarray(obs_time, dtype=float)
    tgt_t = np.asarray(tgt_time, dtype=float)
    out_t = np.full((observed.shape[0], len(tgt_t)), np.nan, dtype=np.float32)
    for i in range(observed.shape[0]):
        r = observed[i]
        mask = ~np.isnan(r)
        if mask.sum() >= 2:
            out_t[i] = np.interp(tgt_t, src_t[mask], r[mask], left=np.nan, right=np.nan)
    src_s = np.asarray(obs_space, dtype=float)
    tgt_s = np.asarray(tgt_space, dtype=float)
    out = np.full((len(tgt_s), out_t.shape[1]), np.nan, dtype=np.float32)
    for j in range(out_t.shape[1]):
        col = out_t[:, j]
        mask = ~np.isnan(col)
        if mask.sum() >= 2:
            out[:, j] = np.interp(tgt_s, src_s[mask], col[mask], left=np.nan, right=np.nan)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Main render entry point
# ---------------------------------------------------------------------------

def render_accident_detail(
    accident_id: str,
    app_state: AppState,
) -> tuple:
    """
    Return (header_html, observed_fig, baseline_fig, related_vds_html).

    All returned values are ready to be assigned to Gradio component values.
    """
    df = app_state.accidents_df
    matches = df[df["accident_id"].astype(str) == str(accident_id)]
    if matches.empty:
        empty = _DEFAULT_EMPTY_FIG
        return (
            f"<p>找不到事故 <b>{accident_id}</b>。</p>",
            empty, empty, "",
        )
    row = matches.iloc[0]

    header = _header_html(row)
    related = _related_vds_html(row, app_state)

    # Observed heatmap from NPZ
    npz = _npz_observed(str(accident_id), app_state)
    if npz is not None:
        obs_fig = speed_heatmap(
            npz["matrix"], npz["space_km"], npz["time_min"],
            "觀測速度（NPZ）", report_time=row["report_time"],
        )
    else:
        obs_fig = empty_figure("觀測速度", "NPZ 中無此事故資料（未執行 build_accident_samples.py）")

    # Baseline heatmap
    if npz is not None:
        baseline_matrix = _cached_baseline(
            str(accident_id),
            pd.Timestamp(row["report_time"]).isoformat(),
            str(row["freeway"]),
            str(row["direction"]),
            str(row["mileage_km"]),
            len(npz["space_km"]),
            len(npz["time_min"]),
            str(app_state.data_root),
        )
        if baseline_matrix is not None:
            baseline_fig = speed_heatmap(
                baseline_matrix, npz["space_km"], npz["time_min"],
                "歷史 baseline", report_time=row["report_time"],
            )
        else:
            baseline_fig = empty_figure("歷史 baseline", "無歷史 baseline 資料（profile 不存在）")
    else:
        baseline_fig = empty_figure("歷史 baseline", "需要 NPZ 才能對齊 baseline 軸")

    return header, obs_fig, baseline_fig, related


def render_model_prediction(
    accident_id: str,
    model_name: str,
    app_state: AppState,
) -> tuple:
    """
    Return (obs_fig, pred_fig, diff_fig, metrics_html, recovery_fig, status_html).
    Heavy — button-triggered.
    """
    ckpt = app_state.checkpoints.get(model_name)
    display = model_runtime.MODEL_DISPLAY.get(model_name, model_name)

    if ckpt is None:
        msg = f"{display} 尚未有 checkpoint，預測暫不可用。"
        ef = empty_figure(display, msg)
        return ef, ef, ef, f"<p>{msg}</p>", ef, msg

    df = app_state.accidents_df
    matches = df[df["accident_id"].astype(str) == str(accident_id)]
    if matches.empty:
        ef = empty_figure(display, "找不到事故")
        return ef, ef, ef, "", ef, "找不到事故"
    row = matches.iloc[0]

    result = _cached_prediction(
        str(accident_id),
        pd.Timestamp(row["report_time"]).isoformat(),
        str(row["freeway"]),
        str(row["direction"]),
        str(row["mileage_km"]),
        model_name,
        str(ckpt),
        str(app_state.data_root),
    )

    if isinstance(result, str):
        ef = empty_figure(display, result)
        return ef, ef, ef, f"<p style='color:#b91c1c'>{result}</p>", ef, result

    pred_matrix = result["heatmap"]
    window = result["window"]
    space_km = (window.space_offsets_m.astype(float) / 1000.0).tolist()
    time_min = window.time_offsets_min.astype(int).tolist()

    # Get observed for this window (re-slice from parquet)
    obs_pack = _cached_observed_parquet(
        str(accident_id),
        pd.Timestamp(row["report_time"]).isoformat(),
        str(row["freeway"]),
        str(row["direction"]),
        str(row["mileage_km"]),
        str(app_state.data_root),
    )
    if obs_pack is None:
        obs_resampled = np.full_like(pred_matrix, np.nan)
    else:
        obs_resampled = _resample(
            obs_pack["matrix"], obs_pack["space_km"], obs_pack["time_min"],
            space_km, time_min,
        )

    rt = row["report_time"]
    obs_fig = speed_heatmap(obs_resampled, space_km, time_min, "觀測速度", report_time=rt)
    pred_fig = speed_heatmap(pred_matrix, space_km, time_min, "預測速度", report_time=rt)
    diff_fig = diff_heatmap(pred_matrix - obs_resampled, space_km, time_min,
                            "差值（預測 - 觀測）", report_time=rt)

    mets = metrics.per_accident_metrics(obs_resampled, pred_matrix)
    space_arr = np.asarray(space_km, dtype=float)
    space_idx = int(np.argmin(np.abs(space_arr)))
    impact_obs = metrics.impact_time_offset(obs_resampled, space_idx, time_min)
    impact_pred = metrics.impact_time_offset(pred_matrix, space_idx, time_min)
    mets_html = metrics.metrics_to_html(mets, impact_observed_min=impact_obs,
                                        impact_predicted_min=impact_pred)

    series = {"觀測": obs_resampled[space_idx], "預測": pred_matrix[space_idx]}
    baseline_matrix = _cached_baseline(
        str(accident_id),
        pd.Timestamp(row["report_time"]).isoformat(),
        str(row["freeway"]),
        str(row["direction"]),
        str(row["mileage_km"]),
        len(space_km), len(time_min),
        str(app_state.data_root),
    )
    if baseline_matrix is not None:
        bl_resampled = _resample(
            baseline_matrix, space_km, time_min, space_km, time_min,
        )
        series["歷史 baseline"] = bl_resampled[space_idx]

    rec_fig = recovery_curve_plot(
        time_min, series,
        "事故位置速度恢復曲線",
        impact_obs_min=impact_obs,
        impact_pred_min=impact_pred,
    )

    dur = result["duration_min"]
    status = f"預測恢復時間：{dur:.0f} 分　checkpoint: {ckpt.parent.parent.name}"
    return obs_fig, pred_fig, diff_fig, mets_html, rec_fig, status
