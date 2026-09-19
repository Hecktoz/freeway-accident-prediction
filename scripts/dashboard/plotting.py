"""Shared Plotly helpers for dashboard charts.

Matrix shape convention: (n_space, n_time), x = minutes from report, y = km offset.

Speed / diff constants are imported from ``scripts.lib.visualization`` so that
the dashboard always uses the same values as the static export scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.lib.visualization import (
    DIFF_ABS_MAX as DIFF_RANGE,
    SPEED_MAX,
    SPEED_MIN,
)

PLOT_BGCOLOR = "#f8fafc"
PAPER_BGCOLOR = "white"
GRID_COLOR = "#e2e8f0"
AXIS_COLOR = "#111827"

PLOTLY_CONFIG = {"displayModeBar": True, "responsive": True}
DEFAULT_HEATMAP_MARGIN = dict(l=72, r=24, t=58, b=58)
DEFAULT_LINE_MARGIN = dict(l=60, r=20, t=50, b=50)


def responsive_heatmap_height(
    matrix: np.ndarray | tuple[int, int] | None,
    *,
    min_height: int = 340,
    max_height: int = 620,
    cell_px: int = 8,
    chrome_px: int = 116,
) -> int:
    if matrix is None:
        return min_height
    if isinstance(matrix, tuple):
        n_space = int(matrix[0]) if matrix else 0
    else:
        n_space = int(matrix.shape[0]) if matrix.ndim >= 1 else 0
    if n_space <= 0:
        return min_height
    return int(max(min_height, min(max_height, chrome_px + n_space * cell_px)))


def empty_figure(title: str, message: str, *, height: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font={"size": 15, "color": "#475569"},
    )
    fig.update_layout(
        title=title, height=height, margin=DEFAULT_HEATMAP_MARGIN,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
    )
    return fig


def _timeline_customdata(
    n_space: int,
    time_offsets_min: Sequence[int],
    report_time: object | None,
    absolute_space_axis_km: Sequence[float] | None,
) -> np.ndarray | None:
    if report_time is None and absolute_space_axis_km is None:
        return None
    n_time = len(time_offsets_min)
    actual_times = [""] * n_time
    if report_time is not None:
        rt = pd.Timestamp(report_time)
        actual_times = [
            (rt + pd.Timedelta(minutes=int(offset))).strftime("%Y-%m-%d %H:%M")
            for offset in time_offsets_min
        ]
    abs_axis = list(absolute_space_axis_km) if absolute_space_axis_km is not None else [np.nan] * n_space
    custom = np.empty((n_space, n_time, 2), dtype=object)
    custom[:, :, 0] = np.tile(np.asarray(actual_times, dtype=object), (n_space, 1))
    custom[:, :, 1] = np.tile(np.asarray(abs_axis, dtype=object).reshape(-1, 1), (1, n_time))
    return custom


def _heatmap_hovertemplate(value_label: str, value_suffix: str, has_customdata: bool) -> str:
    time_line = "時間 %{x} 分<br>"
    abs_line = ""
    if has_customdata:
        time_line += "實際時間 %{customdata[0]}<br>"
        abs_line = "絕對里程 %{customdata[1]:.1f} K<br>"
    return (
        time_line + "相對位置 %{y:.1f} km<br>" + abs_line
        + f"{value_label} %{{z:.1f}} {value_suffix}<extra></extra>"
    )


def _apply_heatmap_layout(fig: go.Figure, title: str, *, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title="距事故時間 (分鐘)",
        yaxis_title="相對事故位置 (km；上游為正、下游為負)",
        height=height, margin=DEFAULT_HEATMAP_MARGIN,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        legend=dict(orientation="h", yanchor="bottom", y=-0.24, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


def add_accident_reference(
    fig: go.Figure,
    *,
    time_offsets_min: Sequence[int],
    space_axis_km: Sequence[float],
    show_marker: bool = True,
    label: str = "發送",
) -> go.Figure:
    if len(time_offsets_min) == 0 or len(space_axis_km) == 0:
        return fig
    x_min, x_max = float(min(time_offsets_min)), float(max(time_offsets_min))
    y_min, y_max = float(min(space_axis_km)), float(max(space_axis_km))
    shapes = list(fig.layout.shapes or [])
    annotations = list(fig.layout.annotations or [])
    shapes.extend([
        dict(type="line", x0=0, x1=0, y0=y_min, y1=y_max,
             line=dict(color=AXIS_COLOR, width=2)),
        dict(type="line", x0=x_min, x1=x_max, y0=0, y1=0,
             line=dict(color=AXIS_COLOR, width=1, dash="dot")),
    ])
    annotations.append(dict(x=0, y=1.04, xref="x", yref="paper",
                            text=label, showarrow=False))
    fig.update_layout(shapes=shapes, annotations=annotations)
    if show_marker:
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers",
            marker=dict(size=10, color=AXIS_COLOR, line=dict(width=2, color="white")),
            name="事故點",
            hovertemplate="事故發送時間<br>相對位置 0 km<extra></extra>",
        ))
    return fig


def speed_heatmap(
    matrix: np.ndarray | None,
    space_axis_km: Sequence[float],
    time_offsets_min: Sequence[int],
    title: str,
    *,
    speed_min: float = SPEED_MIN,
    speed_max: float = SPEED_MAX,
    report_time: object | None = None,
    absolute_space_axis_km: Sequence[float] | None = None,
    show_accident_reference: bool = True,
    height: int | None = None,
) -> go.Figure:
    if matrix is None:
        return empty_figure(title, "沒有可顯示的資料", height=height or 380)
    matrix = np.asarray(matrix, dtype=float)
    height = height or responsive_heatmap_height(matrix)
    customdata = _timeline_customdata(matrix.shape[0], time_offsets_min, report_time, absolute_space_axis_km)
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=list(time_offsets_min), y=list(space_axis_km),
        colorscale="RdYlGn", zmin=speed_min, zmax=speed_max,
        colorbar=dict(title="速度 (km/h)"),
        customdata=customdata,
        hovertemplate=_heatmap_hovertemplate("速度", "km/h", customdata is not None),
        hoverongaps=False,
    ))
    if show_accident_reference:
        add_accident_reference(fig, time_offsets_min=time_offsets_min, space_axis_km=space_axis_km)
    return _apply_heatmap_layout(fig, title, height=height)


def diff_heatmap(
    diff: np.ndarray | None,
    space_axis_km: Sequence[float],
    time_offsets_min: Sequence[int],
    title: str,
    *,
    span: float = DIFF_RANGE,
    report_time: object | None = None,
    absolute_space_axis_km: Sequence[float] | None = None,
    show_accident_reference: bool = True,
    height: int | None = None,
) -> go.Figure:
    if diff is None:
        return empty_figure(title, "沒有可顯示的資料", height=height or 380)
    diff = np.asarray(diff, dtype=float)
    height = height or responsive_heatmap_height(diff)
    customdata = _timeline_customdata(diff.shape[0], time_offsets_min, report_time, absolute_space_axis_km)
    fig = go.Figure(data=go.Heatmap(
        z=diff, x=list(time_offsets_min), y=list(space_axis_km),
        colorscale="RdBu", zmin=-span, zmax=span, zmid=0,
        colorbar=dict(title="差值 (km/h)"),
        customdata=customdata,
        hovertemplate=_heatmap_hovertemplate("差值", "km/h", customdata is not None),
        hoverongaps=False,
    ))
    if show_accident_reference:
        add_accident_reference(fig, time_offsets_min=time_offsets_min, space_axis_km=space_axis_km)
    return _apply_heatmap_layout(fig, title, height=height)


def recovery_curve_plot(
    time_offsets_min: Sequence[int],
    series: dict[str, np.ndarray],
    title: str,
    *,
    impact_obs_min: float | None = None,
    impact_pred_min: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    color_map = {"觀測": "#1f77b4", "預測": "#d62728", "歷史 baseline": "#2ca02c"}
    for label, arr in series.items():
        if arr is None:
            continue
        fig.add_trace(go.Scatter(
            x=list(time_offsets_min), y=list(arr), mode="lines", name=label,
            line=dict(color=color_map.get(label), width=2), connectgaps=False,
        ))
    fig.add_vline(x=0, line_dash="dot", line_color="#888",
                  annotation_text="事故發生", annotation_position="top")
    if impact_obs_min is not None and not np.isnan(impact_obs_min):
        fig.add_vline(x=float(impact_obs_min), line_dash="dash", line_color="#1f77b4",
                      annotation_text=f"觀測恢復 {impact_obs_min:.0f} 分",
                      annotation_position="bottom")
    if impact_pred_min is not None and not np.isnan(impact_pred_min):
        fig.add_vline(x=float(impact_pred_min), line_dash="dash", line_color="#d62728",
                      annotation_text=f"預測恢復 {impact_pred_min:.0f} 分",
                      annotation_position="bottom")
    fig.update_layout(
        title=title, xaxis_title="距事故時間 (分鐘)", yaxis_title="速度 (km/h)",
        height=320, margin=DEFAULT_LINE_MARGIN,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


def line_chart(
    x: Sequence,
    series: dict[str, np.ndarray | list],
    title: str,
    *,
    x_label: str = "時間",
    y_label: str = "速度 (km/h)",
) -> go.Figure:
    fig = go.Figure()
    for label, arr in series.items():
        fig.add_trace(go.Scatter(x=list(x), y=list(arr), mode="lines", name=label, connectgaps=False))
    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title=y_label,
        height=320, margin=DEFAULT_LINE_MARGIN,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig
