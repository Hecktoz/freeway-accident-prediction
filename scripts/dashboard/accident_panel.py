"""Render the accident detail side panel.

The panel is intentionally progressive: metadata renders immediately, while
observed matrices and model predictions load only when the user asks for them.
This keeps map clicks and filter changes responsive even when Drive-backed
Parquet reads or PyTorch checkpoint loads are slow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from . import data_sources as ds
from . import metrics_panel
from . import model_loaders
from .map_layer import DIRECTION_LABEL
from .plots import PLOTLY_CONFIG, diff_heatmap, recovery_curve_plot, speed_heatmap


WindowKey = tuple[int, int, int, int, int, int]
AccidentKey = tuple[str, str, str, str, float]


def _window_key(window) -> WindowKey:
    return (
        int(window.upstream_km),
        int(window.downstream_km),
        int(window.time_before_min),
        int(window.time_after_min),
        int(window.spatial_resolution_km),
        int(window.resolution_min),
    )


def _window_from_key(key: WindowKey):
    from scripts.lib.windows import SampleWindow

    return SampleWindow(
        upstream_km=key[0],
        downstream_km=key[1],
        time_before_min=key[2],
        time_after_min=key[3],
        spatial_resolution_km=key[4],
        resolution_min=key[5],
    )


def _accident_key(row: pd.Series) -> AccidentKey:
    return (
        str(row["accident_id"]),
        pd.Timestamp(row["report_time"]).isoformat(),
        str(row["freeway"]),
        str(row["direction"]),
        float(row["mileage_km"]),
    )


def _accident_series(key: AccidentKey) -> pd.Series:
    accident_id, report_time, freeway, direction, mileage_km = key
    return pd.Series(
        {
            "accident_id": accident_id,
            "report_time": pd.Timestamp(report_time),
            "freeway": freeway,
            "direction": direction,
            "mileage_km": float(mileage_km),
            "mileage_m": int(round(float(mileage_km) * 1000)),
        }
    )


def _relative_space_axis(window) -> list[float]:
    return (window.space_offsets_m.astype(float) / 1000.0).tolist()


def _absolute_space_axis(row: pd.Series, window) -> list[float]:
    from scripts.lib.windows import absolute_mileage_axis_m

    abs_m = absolute_mileage_axis_m(
        int(round(float(row["mileage_km"]) * 1000)),
        str(row["direction"]),
        window,
    )
    return [round(v / 1000.0, 2) for v in abs_m.tolist()]


def _required_dates(report_time: pd.Timestamp, window) -> tuple[str, ...]:
    start = report_time - pd.Timedelta(minutes=window.time_before_min)
    end = report_time + pd.Timedelta(minutes=window.time_after_min)
    return tuple(
        ts.strftime("%Y-%m-%d")
        for ts in pd.date_range(start.normalize(), end.normalize(), freq="D")
    )


@st.cache_data(max_entries=64, show_spinner="載入事故視窗資料…")
def _cached_observed_heatmap(
    accident_key: AccidentKey,
    window_key: WindowKey,
    data_root: str,
) -> dict[str, Any] | None:
    """Build and cache the observed `(n_space, n_time)` matrix."""
    from scripts.lib.windows import coverage, rasterize_window, slice_station_df

    row = _accident_series(accident_key)
    window = _window_from_key(window_key)
    report_time = pd.Timestamp(row["report_time"])

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for date_str in _required_dates(report_time, window):
        df = ds.load_station_day(date_str, resolution_min=window.resolution_min,
                                 data_root=data_root)
        if df.empty:
            missing.append(date_str)
        else:
            frames.append(df)

    if not frames:
        return None

    station_df = pd.concat(frames, ignore_index=True)
    sliced = slice_station_df(
        station_df,
        accident_mileage_m=int(round(float(row["mileage_km"]) * 1000)),
        direction=str(row["direction"]),
        freeway=str(row["freeway"]),
        report_time=report_time,
        window=window,
    )
    matrix = rasterize_window(sliced, window, value_col="speed_x10")
    cov = coverage(matrix)
    cov["vd_count"] = int(sliced["vdid"].nunique()) if not sliced.empty else 0
    cov["rows"] = int(len(sliced))
    cov["missing_station_dates"] = missing
    return {
        "matrix": matrix.astype(np.float32),
        "space_axis_km": _relative_space_axis(window),
        "absolute_space_axis_km": _absolute_space_axis(row, window),
        "time_offsets_min": window.time_offsets_min.astype(int).tolist(),
        "coverage": cov,
        "window": window.to_dict(),
    }


def _historical_fetcher_for_accident(accident_row: pd.Series):
    """Build a `fetch_live_vd` callback that reads from local vd_station_5m."""
    from scripts.lib.inference import FetchResult

    def fetch(start, end):
        dates = pd.date_range(pd.Timestamp(start).normalize(),
                              pd.Timestamp(end).normalize(), freq="D")
        frames: list[pd.DataFrame] = []
        for ts in dates:
            date_str = ts.strftime("%Y-%m-%d")
            df = ds.load_station_day(date_str, resolution_min=5)
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
        status = {
            "source": "vd_station_5m (local)",
            "dates": [ts.strftime("%Y-%m-%d") for ts in dates],
            "rows": int(len(allf)),
        }
        return FetchResult(station_5m=allf, status=status)

    return fetch


def _predict(accident_row: pd.Series, model_name: str, ckpt: Path,
             window) -> Any:
    """Run inference and return AccidentPrediction."""
    from scripts.lib.inference import AccidentInput, predict_accident_impact

    forward_fn = model_loaders.get_forward_fn(model_name, str(ckpt))
    accident = AccidentInput(
        report_time=pd.Timestamp(accident_row["report_time"]).to_pydatetime(),
        freeway=str(accident_row["freeway"]),
        direction=str(accident_row["direction"]),
        mileage_km=float(accident_row["mileage_km"]),
        features=None,
    )
    fetch_live_vd = _historical_fetcher_for_accident(accident_row)
    return predict_accident_impact(
        accident=accident,
        data_root=Path(ds.DEFAULT_DATA_ROOT),
        fetch_live_vd=fetch_live_vd,
        forward_fn=forward_fn,
        model_name=model_name,
        window=window,
    )


@st.cache_data(max_entries=32, show_spinner="執行模型推論…")
def _cached_prediction(
    accident_key: AccidentKey,
    model_name: str,
    ckpt_str: str,
    window_key: WindowKey,
    data_root: str,
) -> dict[str, Any]:
    row = _accident_series(accident_key)
    window = _window_from_key(window_key)
    pred = _predict(row, model_name, Path(ckpt_str), window)
    return {
        "heatmap": pred.heatmap.astype(np.float32),
        "duration_min": float(pred.duration_min),
        "coverage": pred.coverage,
        "input_echo": pred.input_echo,
        "window": pred.window.to_dict(),
        "data_root": data_root,
    }


@st.cache_data(max_entries=32, show_spinner="建立歷史 baseline…")
def _cached_baseline_matrix(
    accident_key: AccidentKey,
    window_key: WindowKey,
    data_root: str,
) -> dict[str, Any] | None:
    """Build the historical baseline matrix, treating missing profile as optional."""
    from scripts.lib.inference import AccidentInput, build_historical_baseline

    row = _accident_series(accident_key)
    window = _window_from_key(window_key)
    profile_df = ds.load_historical_profile(
        str(row["freeway"]),
        resolution_min=window.resolution_min,
        data_root=data_root,
    )
    if profile_df is None or profile_df.empty:
        return None

    accident = AccidentInput(
        report_time=pd.Timestamp(row["report_time"]).to_pydatetime(),
        freeway=str(row["freeway"]),
        direction=str(row["direction"]),
        mileage_km=float(row["mileage_km"]),
    )
    baseline, cov = build_historical_baseline(accident, window, profile_df)
    if np.count_nonzero(~np.isnan(baseline)) == 0:
        return None
    return {"matrix": baseline.astype(np.float32), "coverage": cov}


def _resample_to_inference_grid(
    observed: np.ndarray,
    obs_space_axis_km: list[float],
    obs_time_offsets: list[int],
    target_space_axis_km: list[float],
    target_time_offsets: list[int],
) -> np.ndarray:
    """Resample observed matrix onto the model window grid."""
    if observed.size == 0:
        return np.full(
            (len(target_space_axis_km), len(target_time_offsets)),
            np.nan,
            dtype=np.float32,
        )

    src_time = np.asarray(obs_time_offsets, dtype=float)
    tgt_time = np.asarray(target_time_offsets, dtype=float)
    out_time = np.full((observed.shape[0], len(tgt_time)), np.nan, dtype=np.float32)
    for i in range(observed.shape[0]):
        row = observed[i]
        mask = ~np.isnan(row)
        if mask.sum() < 2:
            continue
        out_time[i] = np.interp(tgt_time, src_time[mask], row[mask],
                                left=np.nan, right=np.nan)

    src_space = np.asarray(obs_space_axis_km, dtype=float)
    tgt_space = np.asarray(target_space_axis_km, dtype=float)
    out = np.full((len(tgt_space), out_time.shape[1]), np.nan, dtype=np.float32)
    for j in range(out_time.shape[1]):
        col = out_time[:, j]
        mask = ~np.isnan(col)
        if mask.sum() < 2:
            continue
        out[:, j] = np.interp(tgt_space, src_space[mask], col[mask],
                              left=np.nan, right=np.nan)
    return out.astype(np.float32)


def _render_metadata_header(row: pd.Series) -> None:
    rt = pd.Timestamp(row["report_time"])
    cols = st.columns(4)
    cols[0].metric("發送時間", rt.strftime("%Y-%m-%d %H:%M"))
    cols[1].metric("路線", f"{row['freeway']} {DIRECTION_LABEL.get(row['direction'], row['direction'])}")
    cols[2].metric("里程", f"{row['mileage_km']:.3f} K")
    inj = row.get("受傷人數", None)
    death = row.get("死亡人數", None)
    cols[3].metric(
        "傷亡",
        f"{int(inj) if pd.notna(inj) else 0} 傷 / {int(death) if pd.notna(death) else 0} 亡",
    )
    extras = []
    for k in ("性質", "事故類別", "車種", "位置", "路段"):
        v = row.get(k)
        if pd.notna(v) and str(v).strip():
            extras.append(f"{k}：{v}")
    if extras:
        st.caption("　｜　".join(extras))


def _render_observed_preview(accident_row: pd.Series, accident_key: AccidentKey) -> None:
    from scripts.lib.windows import DEFAULT_WINDOW

    observed_key = f"observed:{accident_key[0]}"
    cols = st.columns([1, 1])
    if cols[0].button("載入觀測熱圖", key=f"load-{observed_key}", use_container_width=True):
        st.session_state[observed_key] = True
    if cols[1].button("清除觀測熱圖", key=f"clear-{observed_key}", use_container_width=True):
        st.session_state[observed_key] = False

    if not st.session_state.get(observed_key, False):
        st.info("事故基本資料已載入。按「載入觀測熱圖」後才會讀取 Drive 上的 station Parquet。")
        return

    observed_pack = _cached_observed_heatmap(
        accident_key,
        _window_key(DEFAULT_WINDOW),
        str(ds.DEFAULT_DATA_ROOT),
    )
    if observed_pack is None:
        st.warning("此事故時段缺少 station 資料（檔案不存在或時段空），無法顯示熱圖。")
        return
    fig = speed_heatmap(
        observed_pack["matrix"],
        observed_pack["space_axis_km"],
        observed_pack["time_offsets_min"],
        "觀測速度",
        report_time=accident_key[1],
        absolute_space_axis_km=observed_pack["absolute_space_axis_km"],
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    with st.expander("觀測資料覆蓋率"):
        st.json(observed_pack["coverage"])


def _render_model_tab(
    accident_row: pd.Series,
    accident_key: AccidentKey,
    model_name: str,
    ckpt: Path | None,
) -> None:
    label = model_loaders.MODEL_DISPLAY.get(model_name, model_name)
    if ckpt is None:
        st.info(f"{label} 尚未有 checkpoint，預測暫不可用。")
        return

    try:
        window = model_loaders.training_window_for(model_name, ckpt)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"無法讀取此 checkpoint 的訓練視窗：{exc}")
        window = None
    if window is None:
        from scripts.lib.windows import DEFAULT_WINDOW

        window = DEFAULT_WINDOW

    run_key = f"prediction:{accident_key[0]}:{model_name}:{ckpt}"
    st.caption(
        f"checkpoint: `{ckpt.parent.parent.name}` ｜ "
        f"window: 上游 {window.upstream_km} km / 下游 {window.downstream_km} km, "
        f"{window.time_before_min} 分前到 {window.time_after_min} 分後"
    )
    cols = st.columns([1, 1])
    if cols[0].button("執行 / 載入此模型", key=f"run-{run_key}", type="primary",
                      use_container_width=True):
        st.session_state[run_key] = True
    if cols[1].button("收起結果", key=f"hide-{run_key}", use_container_width=True):
        st.session_state[run_key] = False

    if not st.session_state.get(run_key, False):
        st.info("此模型結果尚未載入。按按鈕後才會讀取事故視窗並執行推論。")
        return

    window_key = _window_key(window)
    observed_pack = _cached_observed_heatmap(
        accident_key,
        window_key,
        str(ds.DEFAULT_DATA_ROOT),
    )
    if observed_pack is None:
        st.warning("此事故時段缺少 station 資料，無法計算觀測 / 預測比較。")
        return

    try:
        prediction = _cached_prediction(
            accident_key,
            model_name,
            str(ckpt),
            window_key,
            str(ds.DEFAULT_DATA_ROOT),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"模型推論失敗：{exc}")
        return

    rel_space_axis = _relative_space_axis(window)
    abs_space_axis = _absolute_space_axis(accident_row, window)
    time_offsets = window.time_offsets_min.astype(int).tolist()
    pred_matrix = prediction["heatmap"]

    observed_resampled = _resample_to_inference_grid(
        observed_pack["matrix"],
        observed_pack["space_axis_km"],
        observed_pack["time_offsets_min"],
        rel_space_axis,
        time_offsets,
    )

    chart_cols = st.columns(3)
    with chart_cols[0]:
        st.plotly_chart(
            speed_heatmap(
                observed_resampled,
                rel_space_axis,
                time_offsets,
                "觀測速度",
                report_time=accident_key[1],
                absolute_space_axis_km=abs_space_axis,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with chart_cols[1]:
        st.plotly_chart(
            speed_heatmap(
                pred_matrix,
                rel_space_axis,
                time_offsets,
                "預測速度",
                report_time=accident_key[1],
                absolute_space_axis_km=abs_space_axis,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with chart_cols[2]:
        st.plotly_chart(
            diff_heatmap(
                pred_matrix - observed_resampled,
                rel_space_axis,
                time_offsets,
                "差值（預測 - 觀測）",
                report_time=accident_key[1],
                absolute_space_axis_km=abs_space_axis,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    metrics = metrics_panel.per_accident_metrics(observed_resampled, pred_matrix)
    space_idx = int(np.argmin(np.abs(np.asarray(rel_space_axis, dtype=float))))
    impact_obs = metrics_panel.impact_time_offset(observed_resampled, space_idx, time_offsets)
    impact_pred = metrics_panel.impact_time_offset(pred_matrix, space_idx, time_offsets)

    st.markdown("**模型表現（本事故）**")
    metrics_panel.render(metrics, impact_observed_min=impact_obs,
                         impact_predicted_min=impact_pred)
    st.caption(f"預測恢復時間（模型 free-flow 閾值）：{prediction['duration_min']:.0f} 分")

    baseline_pack = _cached_baseline_matrix(accident_key, window_key, str(ds.DEFAULT_DATA_ROOT))
    series = {
        "觀測": observed_resampled[space_idx],
        "預測": pred_matrix[space_idx],
    }
    if baseline_pack is not None:
        series["歷史 baseline"] = baseline_pack["matrix"][space_idx]

    st.plotly_chart(
        recovery_curve_plot(
            time_offsets,
            series,
            "事故位置速度恢復曲線",
            impact_obs_min=impact_obs,
            impact_pred_min=impact_pred,
        ),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    with st.expander("覆蓋率 / fetch 狀態"):
        st.json(
            {
                "observed": observed_pack.get("coverage"),
                "prediction.coverage": prediction.get("coverage"),
                "historical_baseline": None if baseline_pack is None else baseline_pack.get("coverage"),
                "training_window": window.to_dict(),
            }
        )


def render(accident_row: pd.Series, checkpoints: dict[str, Path | None]) -> None:
    accident_key = _accident_key(accident_row)
    st.subheader(f"事故 {accident_key[0]}")
    _render_metadata_header(accident_row)

    observed_tab, *model_tabs = st.tabs(
        ["觀測熱圖", *[model_loaders.MODEL_DISPLAY[m] for m in model_loaders.ADAPTERS]]
    )
    with observed_tab:
        _render_observed_preview(accident_row, accident_key)

    for tab, model_name in zip(model_tabs, model_loaders.ADAPTERS):
        with tab:
            _render_model_tab(
                accident_row,
                accident_key,
                model_name,
                checkpoints.get(model_name),
            )
