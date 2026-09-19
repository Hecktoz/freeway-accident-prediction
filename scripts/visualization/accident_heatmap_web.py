# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "dash",
#     "numpy",
#     "pandas",
#     "plotly",
#     "pyarrow",
# ]
# ///
# ruff: noqa: E402
"""
Interactive accident spacetime heatmap browser.

Usage:
  uv run scripts/visualization/accident_heatmap_web.py
  uv run scripts/visualization/accident_heatmap_web.py --host 0.0.0.0 --port 8050

In Colab, expose the server with the notebook proxy or a tunnel after running:
  uv run scripts/visualization/accident_heatmap_web.py --host 0.0.0.0 --port 8050
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from dash import Dash, Input, Output, State, dcc, html
from pyarrow.lib import ArrowInvalid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.calendar import day_class as taiwan_day_class
from scripts.lib.data_loaders import read_accidents, read_station_dates
from scripts.dashboard.plots import (
    PLOTLY_CONFIG,
    add_event_lines,
    empty_figure as shared_empty_figure,
    speed_heatmap,
)
from scripts.lib.windows import (
    SampleWindow,
    absolute_mileage_axis_m,
    coverage,
    rasterize_window,
    slice_station_df,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
STATION_COLUMNS = ["timestamp", "vdid", "freeway", "direction", "mileage_m", "speed_x10", "n_lanes_valid"]
PROFILE_COLUMNS = [
    "vdid",
    "direction",
    "mileage_m",
    "day_class",
    "peak_class",
    "minute_bucket",
    "mean_speed_x10",
    "n_obs",
    "coverage_ratio",
]
@dataclass(frozen=True)
class AppConfig:
    data_root: Path
    resolution_min: int
    max_cache_days: int


class StationCache:
    def __init__(self, config: AppConfig):
        self.config = config
        self._days: dict[str, pd.DataFrame] = {}
        self._order: list[str] = []

    def get(self, dates: tuple[str, ...]) -> pd.DataFrame:
        for date_str in dates:
            if date_str in self._days:
                continue
            try:
                frame = read_station_dates(
                    [date_str],
                    resolution_min=self.config.resolution_min,
                    data_root=self.config.data_root,
                    columns=STATION_COLUMNS,
                )
                if not frame.empty:
                    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
                    frame = frame[frame["timestamp"].notna()].copy()
            except FileNotFoundError:
                frame = pd.DataFrame(columns=STATION_COLUMNS)
            self._days[date_str] = frame
            self._order.append(date_str)
            while len(self._order) > self.config.max_cache_days:
                old = self._order.pop(0)
                self._days.pop(old, None)

        parts = [self._days[d] for d in dates if d in self._days and not self._days[d].empty]
        if not parts:
            return pd.DataFrame(columns=STATION_COLUMNS)
        return pd.concat(parts, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch an interactive accident heatmap browser.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Data root containing accident and VD parquet data")
    parser.add_argument("--resolution", type=int, default=5, choices=[1, 5], help="Station data resolution in minutes")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-cache-days", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true", help="Build one accident view and exit without starting Dash")
    parser.add_argument("--accident-id", default=None, help="Accident id for --smoke-test")
    return parser.parse_args()


def required_dates(report_time: pd.Timestamp, window: SampleWindow) -> tuple[str, ...]:
    start = report_time - pd.Timedelta(minutes=window.time_before_min)
    end = report_time + pd.Timedelta(minutes=window.time_after_min)
    return tuple(ts.strftime("%Y-%m-%d") for ts in pd.date_range(start.normalize(), end.normalize(), freq="D"))


def day_class(ts: pd.Timestamp) -> str:
    return taiwan_day_class(ts)


def peak_class(ts: pd.Timestamp) -> str:
    minute = ts.hour * 60 + ts.minute
    if (7 * 60 <= minute < 9 * 60) or (17 * 60 <= minute < 19 * 60):
        return "peak"
    return "nonpeak"


def minute_bucket(ts: pd.Timestamp, resolution_min: int) -> int:
    minute = ts.hour * 60 + ts.minute
    return int(minute // resolution_min * resolution_min)


def mileage_bounds(accident: pd.Series, window: SampleWindow) -> tuple[int, int]:
    mileage_m = int(accident["mileage_m"])
    if accident["direction"] in ("S", "E"):
        return mileage_m - window.upstream_km * 1000, mileage_m + window.downstream_km * 1000
    return mileage_m - window.downstream_km * 1000, mileage_m + window.upstream_km * 1000


def load_profile_freeway(data_root: Path, resolution_min: int, freeway: str) -> tuple[pd.DataFrame | None, str]:
    base = data_root / f"historical_speed_profile_{resolution_min}m"
    if not base.exists():
        return None, f"找不到 {base.relative_to(PROJECT_ROOT) if base.is_relative_to(PROJECT_ROOT) else base}"
    if not any(base.rglob("*.parquet")):
        return None, f"{base.relative_to(PROJECT_ROOT) if base.is_relative_to(PROJECT_ROOT) else base} 目前沒有 parquet 檔，歷史平均速度暫不可用"

    freeway_dir = base / f"freeway={freeway}"
    try:
        if freeway_dir.exists():
            files = sorted(freeway_dir.glob("*.parquet"))
            if not files:
                return None, f"{freeway_dir} 沒有 parquet 檔"
            frame = pd.concat([pd.read_parquet(path, columns=PROFILE_COLUMNS) for path in files], ignore_index=True)
            return frame, f"已載入 {freeway} historical profile ({len(frame):,} rows)"

        dataset = ds.dataset(str(base), format="parquet", partitioning="hive")
        table = dataset.to_table(columns=PROFILE_COLUMNS + ["freeway"])
        frame = table.to_pandas()
        frame = frame[frame["freeway"] == freeway].drop(columns=["freeway"], errors="ignore").copy()
        if frame.empty:
            return None, f"historical profile 中沒有 {freeway}"
        return frame, f"已載入 {freeway} historical profile ({len(frame):,} rows)"
    except (ArrowInvalid, FileNotFoundError, ValueError, OSError) as exc:
        return None, f"historical profile 欄位或檔案格式不符合目前 dashboard 需求，暫不顯示歷史平均速度：{exc}"


def profile_cache_key(data_root: Path, resolution_min: int, freeway: str) -> tuple[str, int, str]:
    return str(data_root.resolve()), resolution_min, freeway


@lru_cache(maxsize=16)
def cached_profile(data_root_str: str, resolution_min: int, freeway: str) -> tuple[pd.DataFrame | None, str]:
    return load_profile_freeway(Path(data_root_str), resolution_min, freeway)


def build_historical_matrix(
    accident: pd.Series,
    window: SampleWindow,
    data_root: Path,
    resolution_min: int,
) -> tuple[np.ndarray | None, str]:
    profile, status = cached_profile(*profile_cache_key(data_root, resolution_min, str(accident["freeway"])))
    if profile is None or profile.empty:
        return None, status

    m_low, m_high = mileage_bounds(accident, window)
    route = profile[
        (profile["direction"] == accident["direction"])
        & (profile["mileage_m"] >= m_low)
        & (profile["mileage_m"] <= m_high)
        & (profile["day_class"] == day_class(pd.Timestamp(accident["report_time"])))
    ].copy()
    if route.empty:
        return None, "historical profile 找不到同方向、同日型與同空間範圍的資料"

    accident_m = int(accident["mileage_m"])
    if accident["direction"] in ("S", "E"):
        route["space_offset_m"] = accident_m - route["mileage_m"].astype(int)
    else:
        route["space_offset_m"] = route["mileage_m"].astype(int) - accident_m

    space_axis = window.space_offsets_m.astype(float)
    matrix = np.full((window.n_space, window.n_time), np.nan, dtype=np.float32)
    report_time = pd.Timestamp(accident["report_time"])
    for t_idx, offset in enumerate(window.time_offsets_min):
        ts = report_time + pd.Timedelta(minutes=int(offset))
        rows = route[
            (route["minute_bucket"] == minute_bucket(ts, resolution_min))
            & (route["peak_class"] == peak_class(ts))
        ]
        if rows.empty:
            continue
        speeds = rows.groupby("space_offset_m", sort=True)["mean_speed_x10"].mean() / 10.0
        x = speeds.index.to_numpy(dtype=float)
        y = speeds.to_numpy(dtype=float)
        if len(x) == 1:
            nearest = int(np.argmin(np.abs(space_axis - x[0])))
            if abs(space_axis[nearest] - x[0]) <= window.spatial_resolution_km * 500:
                matrix[nearest, t_idx] = y[0]
            continue
        in_range = (space_axis >= x.min()) & (space_axis <= x.max())
        if in_range.any():
            matrix[in_range, t_idx] = np.interp(space_axis[in_range], x, y)

    filled = int(np.count_nonzero(~np.isnan(matrix)))
    if filled == 0:
        return None, "historical profile 已載入，但此事故視窗沒有可填入的歷史速度"
    return matrix, f"{status}；視窗填補 {filled:,}/{matrix.size:,} cells"


def build_observed_matrix(
    accident: pd.Series,
    window: SampleWindow,
    station_cache: StationCache,
) -> tuple[np.ndarray, dict, str]:
    dates = required_dates(pd.Timestamp(accident["report_time"]), window)
    station_df = station_cache.get(dates)
    sliced = slice_station_df(
        station_df,
        accident_mileage_m=int(accident["mileage_m"]),
        direction=str(accident["direction"]),
        freeway=str(accident["freeway"]),
        report_time=pd.Timestamp(accident["report_time"]),
        window=window,
    )
    matrix = rasterize_window(sliced, window, value_col="speed_x10")
    cov = coverage(matrix)
    status = f"station dates: {', '.join(dates)}；VD rows: {len(sliced):,}"
    return matrix, cov, status


def format_time(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def accident_label(row: pd.Series) -> str:
    return (
        f"{row['accident_id']} | {format_time(row['report_time'])} | "
        f"{row['freeway']}-{row['direction']} {float(row['mileage_km']):.1f}K"
    )


def empty_figure(title: str, message: str):
    return shared_empty_figure(title, message, height=420)


def heatmap_figure(
    matrix: np.ndarray | None,
    accident: pd.Series,
    window: SampleWindow,
    title: str,
    zmin: float = 20.0,
    zmax: float = 140.0,
):
    if matrix is None:
        return empty_figure(title, "沒有可顯示的資料")

    x = window.time_offsets_min.astype(int)
    y = window.space_offsets_m.astype(float) / 1000.0
    report_time = pd.Timestamp(accident["report_time"])
    abs_axis = absolute_mileage_axis_m(
        int(accident["mileage_m"]),
        str(accident["direction"]),
        window,
    )
    fig = speed_heatmap(
        matrix,
        y.tolist(),
        x.tolist(),
        title=title,
        speed_min=zmin,
        speed_max=zmax,
        report_time=report_time,
        absolute_space_axis_km=(abs_axis.astype(float) / 1000.0).tolist(),
    )
    add_event_lines(
        fig,
        report_time=report_time,
        time_offsets_min=x.tolist(),
        events={
            "arrived_time": (accident.get("arrived_time"), "到達", "#2563eb"),
            "cleared_time": (accident.get("cleared_time"), "開始清理", "#7c3aed"),
            "completed_time": (accident.get("completed_time"), "處理完成", "#059669"),
        },
    )
    return fig


def detail_item(label: str, value: object) -> html.Div:
    return html.Div([html.Div(label, className="detail-label"), html.Div(str(value), className="detail-value")], className="detail-item")


def details_panel(accident: pd.Series, observed_cov: dict, observed_status: str, profile_status: str, obs_mean: float | None, hist_mean: float | None) -> html.Div:
    original_fields = [
        ("路段", "路段"),
        ("位置", "位置"),
        ("事故類別", "事故類別"),
        ("車種", "車種"),
        ("主線車道數", "主線車道數"),
        ("輔助車道數", "輔助車道數"),
        ("剩餘車道數", "剩餘車道數"),
        ("受傷人數", "受傷人數"),
        ("死亡人數", "死亡人數"),
    ]
    items = [
        detail_item("任務編號", accident["accident_id"]),
        detail_item("路線/方向/里程", f"{accident['freeway']}-{accident['direction']} {float(accident['mileage_km']):.1f}K"),
        detail_item("發送", format_time(accident["report_time"])),
        detail_item("到達", format_time(accident.get("arrived_time"))),
        detail_item("開始清理", format_time(accident.get("cleared_time"))),
        detail_item("處理完成", format_time(accident.get("completed_time"))),
    ]
    for source_col, label in original_fields:
        if source_col in accident.index and pd.notna(accident[source_col]):
            value = accident[source_col]
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            items.append(detail_item(label, value))

    filled = observed_cov.get("filled_cells", 0)
    total = observed_cov.get("total_cells", 0)
    ratio = observed_cov.get("filled_ratio")
    ratio_text = "-" if ratio is None else f"{ratio:.1%}"
    items.extend(
        [
            detail_item("事故窗平均速度", "-" if obs_mean is None else f"{obs_mean:.1f} km/h"),
            detail_item("歷史平均速度", "-" if hist_mean is None else f"{hist_mean:.1f} km/h"),
            detail_item("事故窗覆蓋率", f"{filled:,}/{total:,} ({ratio_text})"),
            detail_item("事故窗資料", observed_status),
            detail_item("歷史速度資料", profile_status),
        ]
    )
    return html.Div(items, className="details-grid")


def safe_nanmean(matrix: np.ndarray | None) -> float | None:
    if matrix is None or not np.isfinite(matrix).any():
        return None
    return float(np.nanmean(matrix))


def make_app(config: AppConfig) -> Dash:
    accidents = read_accidents(data_root=config.data_root)
    accidents = accidents.dropna(subset=["accident_id", "freeway", "direction", "mileage_m", "report_time"]).copy()
    accidents = accidents[accidents["direction"].isin(["N", "S", "E", "W"])].sort_values(["report_time", "accident_id"]).reset_index(drop=True)
    if accidents.empty:
        raise SystemExit("No valid accidents found in data/accident_data.csv")

    accident_options = [{"label": accident_label(row), "value": str(row["accident_id"])} for _, row in accidents.iterrows()]
    first_accident = accident_options[0]["value"]
    station_cache = StationCache(config)

    app = Dash(__name__, title="事故熱力圖瀏覽器")
    app.layout = html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.H1("事故熱力圖瀏覽器"),
                            html.P("調整上下游與時間窗，檢視事故速度演化、事故資訊與歷史平均速度。"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div("資料根目錄", className="meta-label"),
                            html.Div(str(config.data_root), className="meta-value"),
                        ],
                        className="meta-box",
                    ),
                ],
                className="app-header",
            ),
            html.Main(
                [
                    html.Aside(
                        [
                            html.Label("事故", className="control-label"),
                            dcc.Dropdown(
                                id="accident-id",
                                options=accident_options,
                                value=first_accident,
                                clearable=False,
                                searchable=True,
                                maxHeight=420,
                            ),
                            html.Div(className="control-spacer"),
                            html.Label("上游範圍 (km)", className="control-label"),
                            dcc.Slider(id="upstream-km", min=0, max=80, step=1, value=50, marks={0: "0", 25: "25", 50: "50", 80: "80"}, tooltip={"placement": "bottom"}),
                            html.Label("下游範圍 (km)", className="control-label"),
                            dcc.Slider(id="downstream-km", min=0, max=30, step=1, value=10, marks={0: "0", 10: "10", 20: "20", 30: "30"}, tooltip={"placement": "bottom"}),
                            html.Label("事故前時間 (min)", className="control-label"),
                            dcc.Slider(id="time-before-min", min=0, max=180, step=5, value=60, marks={0: "0", 60: "60", 120: "120", 180: "180"}, tooltip={"placement": "bottom"}),
                            html.Label("事故後時間 (min)", className="control-label"),
                            dcc.Slider(id="time-after-min", min=0, max=360, step=5, value=240, marks={0: "0", 120: "120", 240: "240", 360: "360"}, tooltip={"placement": "bottom"}),
                            html.Div(
                                [
                                    html.Button("套用", id="apply-button", n_clicks=0, className="primary-button"),
                                    html.Div(f"{config.resolution_min} 分鐘資料", className="resolution-pill"),
                                ],
                                className="button-row",
                            ),
                        ],
                        className="control-panel",
                    ),
                    html.Section(
                        [
                            dcc.Loading(
                                [
                                    html.Div(id="details"),
                                    dcc.Graph(id="observed-heatmap", config=PLOTLY_CONFIG),
                                    dcc.Graph(id="historical-heatmap", config=PLOTLY_CONFIG),
                                ],
                                type="circle",
                            )
                        ],
                        className="content-panel",
                    ),
                ],
                className="app-main",
            ),
        ],
        className="app-shell",
    )

    app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                color-scheme: light;
                --text: #0f172a;
                --muted: #64748b;
                --line: #dbe3ef;
                --panel: #ffffff;
                --bg: #eef3f8;
                --accent: #0f766e;
                --accent-dark: #115e59;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: "Noto Sans TC", "Microsoft JhengHei", "Segoe UI", Arial, sans-serif;
                background: var(--bg);
                color: var(--text);
            }
            .app-header {
                display: grid;
                grid-template-columns: minmax(0, 1fr) minmax(280px, 420px);
                gap: 24px;
                align-items: end;
                padding: 24px 28px 18px;
                border-bottom: 1px solid var(--line);
                background: #ffffff;
            }
            h1 {
                margin: 0 0 6px;
                font-size: 28px;
                line-height: 1.18;
                letter-spacing: 0;
            }
            p { margin: 0; color: var(--muted); line-height: 1.5; }
            .meta-box {
                border: 1px solid var(--line);
                border-radius: 6px;
                padding: 10px 12px;
                background: #f8fafc;
                overflow-wrap: anywhere;
            }
            .meta-label, .detail-label { color: var(--muted); font-size: 12px; line-height: 1.3; }
            .meta-value { font-size: 13px; line-height: 1.35; margin-top: 3px; }
            .app-main {
                display: grid;
                grid-template-columns: 360px minmax(0, 1fr);
                gap: 18px;
                padding: 18px;
            }
            .control-panel, .content-panel {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
            }
            .control-panel {
                position: sticky;
                top: 14px;
                height: calc(100vh - 36px);
                overflow-y: auto;
                padding: 18px 16px;
            }
            .content-panel { min-width: 0; padding: 14px; }
            .control-label {
                display: block;
                margin: 14px 0 8px;
                font-size: 13px;
                font-weight: 700;
            }
            .control-spacer { height: 4px; }
            .button-row {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-top: 20px;
            }
            .primary-button {
                border: 0;
                border-radius: 6px;
                background: var(--accent);
                color: white;
                font-size: 14px;
                font-weight: 700;
                padding: 10px 18px;
                cursor: pointer;
            }
            .primary-button:hover { background: var(--accent-dark); }
            .resolution-pill {
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: 6px 10px;
                color: var(--muted);
                font-size: 12px;
                white-space: nowrap;
            }
            .details-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(160px, 1fr));
                gap: 10px;
                margin-bottom: 14px;
            }
            .detail-item {
                border: 1px solid var(--line);
                border-radius: 6px;
                padding: 9px 10px;
                min-height: 58px;
                background: #fbfdff;
            }
            .detail-value {
                margin-top: 4px;
                font-size: 14px;
                line-height: 1.35;
                overflow-wrap: anywhere;
            }
            @media (max-width: 980px) {
                .app-header, .app-main { grid-template-columns: 1fr; }
                .control-panel { position: static; height: auto; }
                .details-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }
            @media (max-width: 620px) {
                .app-header, .app-main { padding-left: 12px; padding-right: 12px; }
                .details-grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

    @app.callback(
        Output("observed-heatmap", "figure"),
        Output("historical-heatmap", "figure"),
        Output("details", "children"),
        Input("apply-button", "n_clicks"),
        State("accident-id", "value"),
        State("upstream-km", "value"),
        State("downstream-km", "value"),
        State("time-before-min", "value"),
        State("time-after-min", "value"),
    )
    def update_view(_n_clicks, accident_id, upstream_km, downstream_km, time_before_min, time_after_min):
        row = accidents[accidents["accident_id"].astype(str) == str(accident_id)]
        if row.empty:
            message = f"找不到事故 {accident_id}"
            return empty_figure("事故速度熱力圖", message), empty_figure("歷史平均速度", message), html.Div(message)

        accident = row.iloc[0]
        window = SampleWindow(
            upstream_km=int(upstream_km),
            downstream_km=int(downstream_km),
            time_before_min=int(time_before_min),
            time_after_min=int(time_after_min),
            spatial_resolution_km=1,
            resolution_min=config.resolution_min,
        )
        observed, cov, observed_status = build_observed_matrix(accident, window, station_cache)
        historical, profile_status = build_historical_matrix(accident, window, config.data_root, config.resolution_min)

        observed_title = f"事故速度熱力圖 | {accident['accident_id']} | {accident['freeway']}-{accident['direction']} {float(accident['mileage_km']):.1f}K"
        historical_title = "歷史平均速度 | 同路線方向、同日型、同時段 minute bucket"
        obs_fig = heatmap_figure(observed, accident, window, observed_title)
        hist_fig = heatmap_figure(historical, accident, window, historical_title) if historical is not None else empty_figure(historical_title, profile_status)
        details = details_panel(
            accident=accident,
            observed_cov=cov,
            observed_status=observed_status,
            profile_status=profile_status,
            obs_mean=safe_nanmean(observed),
            hist_mean=safe_nanmean(historical),
        )
        return obs_fig, hist_fig, details

    return app


def run_smoke_test(config: AppConfig, accident_id: str | None = None) -> None:
    accidents = read_accidents(data_root=config.data_root)
    accidents = accidents.dropna(subset=["accident_id", "freeway", "direction", "mileage_m", "report_time"]).copy()
    accidents = accidents[accidents["direction"].isin(["N", "S", "E", "W"])].sort_values(["report_time", "accident_id"]).reset_index(drop=True)
    if accident_id:
        accidents = accidents[accidents["accident_id"].astype(str) == str(accident_id)].copy()
    if accidents.empty:
        raise SystemExit("No valid accident found for smoke test.")

    accident = accidents.iloc[0]
    window = SampleWindow(upstream_km=50, downstream_km=10, time_before_min=60, time_after_min=240, spatial_resolution_km=1, resolution_min=config.resolution_min)
    station_cache = StationCache(config)
    observed, cov, observed_status = build_observed_matrix(accident, window, station_cache)
    historical, profile_status = build_historical_matrix(accident, window, config.data_root, config.resolution_min)
    obs_fig = heatmap_figure(observed, accident, window, "smoke observed")
    hist_fig = heatmap_figure(historical, accident, window, "smoke historical") if historical is not None else empty_figure("smoke historical", profile_status)

    print(f"accident_id={accident['accident_id']}")
    print(f"observed_shape={observed.shape} filled={cov['filled_cells']}/{cov['total_cells']} ratio={cov['filled_ratio']:.3f}")
    print(f"observed_mean={safe_nanmean(observed)}")
    print(f"observed_status={observed_status}")
    print(f"historical_shape={None if historical is None else historical.shape}")
    print(f"historical_mean={safe_nanmean(historical)}")
    print(f"profile_status={profile_status}")
    print(f"plotly_traces={len(obs_fig.data) + len(hist_fig.data)}")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    config = AppConfig(data_root=data_root, resolution_min=args.resolution, max_cache_days=args.max_cache_days)
    if args.smoke_test:
        run_smoke_test(config, accident_id=args.accident_id)
        return
    app = make_app(config)
    print(f"Starting accident heatmap browser: http://{args.host}:{args.port}")
    print(f"Data root: {data_root}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
