"""Heatmap renderers for accident spacetime matrices.

Canonical matrix layout: ``(n_space, n_time)``.
Spatial axis: upstream-positive km (positive = upstream of accident, negative = downstream).
Time axis: offset-from-report minutes (negative = before accident).

Speed / diff constants (``SPEED_MIN``, ``SPEED_MAX``, ``DIFF_ABS_MAX``, ``SPEED_CMAP``,
``DIFF_CMAP``) are exported here and imported by ``dashboard/plotting.py`` to keep
a single source of truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import os as _os
if _os.environ.get("MPLBACKEND", "").startswith("module://"):
    _os.environ["MPLBACKEND"] = "Agg"

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Shared constants (re-exported; imported by dashboard/plotting.py) ────────
SPEED_MIN: float = 0.0
SPEED_MAX: float = 140.0
DIFF_ABS_MAX: float = 40.0
SPEED_CMAP: str = "RdYlGn"
DIFF_CMAP: str = "RdBu"


# ── Style presets ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HeatmapStyle:
    """Visual parameters for a heatmap figure.

    Use the named presets ``STYLE_NOTEBOOK``, ``STYLE_REPORT``, ``STYLE_SLIDE``
    rather than constructing this directly.  Override individual fields via
    ``dataclasses.replace(STYLE_REPORT, dpi=200)`` when needed.
    """

    figsize: tuple[float, float] = (12.0, 7.0)
    dpi: int = 100
    font_size: int = 10
    title_size: int = 11
    speed_min: float = SPEED_MIN
    speed_max: float = SPEED_MAX
    cmap: str = SPEED_CMAP
    diff_cmap: str = DIFF_CMAP
    diff_abs_max: float = DIFF_ABS_MAX


# Colab inline / quick exploration
STYLE_NOTEBOOK = HeatmapStyle(figsize=(12.0, 7.0), dpi=100, font_size=10, title_size=11)
# PDF / LaTeX reports, journal figures
STYLE_REPORT   = HeatmapStyle(figsize=(14.0, 8.0), dpi=150, font_size=11, title_size=12)
# Slide embeds — larger text, same aspect ratio (paste at natural size)
STYLE_SLIDE    = HeatmapStyle(figsize=(14.0, 8.0), dpi=150, font_size=13, title_size=15)

STYLE_MAP: dict[str, HeatmapStyle] = {
    "notebook": STYLE_NOTEBOOK,
    "report":   STYLE_REPORT,
    "slide":    STYLE_SLIDE,
}

DEFAULT_STYLE = STYLE_REPORT


# ── CJK font setup (single canonical copy) ───────────────────────────────────
def configure_cjk_font(font_path: str | Path | None = None) -> str | None:
    """Configure matplotlib for CJK rendering. Idempotent; safe to call multiple times."""

    def _apply(name: str) -> None:
        existing = plt.rcParams.get("font.sans-serif", [])
        if isinstance(existing, str):
            existing = [existing]
        merged = [name, *list(existing), "DejaVu Sans"]
        deduped: list[str] = []
        for item in merged:
            if item and item not in deduped:
                deduped.append(item)
        plt.rcParams["font.family"] = ["sans-serif"]
        plt.rcParams["font.sans-serif"] = deduped
        plt.rcParams["axes.unicode_minus"] = False

    if font_path:
        resolved = Path(font_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Font path not found: {resolved}")
        fm.fontManager.addfont(str(resolved))
        name = fm.FontProperties(fname=str(resolved)).get_name()
        _apply(name)
        return name

    preferred = [
        "Noto Sans CJK TC", "Noto Sans CJK SC", 
        "Microsoft JhengHei", "Microsoft YaHei", "MingLiU", "SimHei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            _apply(name)
            return name

    keyword_preferred = [
        "Noto Sans CJK", "Source Han Sans", "Microsoft JhengHei",
        "Microsoft YaHei", "WenQuanYi", "PingFang", "Heiti",
        "MingLiU", "PMingLiU", "SimHei",
    ]
    for name in sorted(available):
        if any(k in name for k in keyword_preferred):
            _apply(name)
            return name

    for fp in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ):
        if Path(fp).exists():
            fm.fontManager.addfont(fp)
            name = fm.FontProperties(fname=fp).get_name()
            _apply(name)
            return name

    plt.rcParams["axes.unicode_minus"] = False
    return None


configure_cjk_font()


# ── Internal helpers ──────────────────────────────────────────────────────────
def _imshow(
    ax: plt.Axes,
    matrix: np.ndarray,
    space_axis_km: np.ndarray,
    time_offsets_min: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
):
    """Render ``(n_space, n_time)`` matrix on *ax*."""
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")
    return ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=[
            float(time_offsets_min[0]),
            float(time_offsets_min[-1]),
            float(space_axis_km[0]),
            float(space_axis_km[-1]),
        ],
    )


def _decorate(
    ax: plt.Axes,
    fig: plt.Figure,
    im,
    space_axis_km: np.ndarray,
    time_offsets_min: np.ndarray,
    accident_meta: dict | None,
    *,
    title: str | None,
    cbar_label: str,
    style: HeatmapStyle,
    resolution_min: int,
) -> None:
    fs = style.font_size

    # Reference lines
    ax.axvline(0, color="red", linestyle="--", linewidth=2.0, alpha=0.9)
    ax.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)

    # X-axis: absolute time ticks when report_time is known, else relative offsets
    report_time = None
    if accident_meta:
        report_time = accident_meta.get("report_time")

    if report_time is not None:
        rt = pd.Timestamp(report_time)
        tick_step = 10 if resolution_min == 1 else 15
        first_tick = int(math.ceil(int(time_offsets_min[0]) / tick_step) * tick_step)
        x_ticks = list(range(first_tick, int(time_offsets_min[-1]) + 1, tick_step))
        x_labels = [(rt + pd.Timedelta(minutes=t)).strftime("%H:%M") for t in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=fs)
        ax.set_xlabel(f"時間（{resolution_min} 分鐘）", fontsize=fs + 1, fontweight="bold")
    else:
        ax.set_xlabel(
            f"事故時間偏移（分鐘，{resolution_min} 分鐘解析度）",
            fontsize=fs + 1,
            fontweight="bold",
        )

    ax.set_ylabel("相對事故位置（km，上游為正）", fontsize=fs + 1, fontweight="bold")
    ax.tick_params(labelsize=fs)

    # Event lines and accident marker
    if accident_meta:
        if report_time is not None:
            rt = pd.Timestamp(report_time)
            t_min = float(time_offsets_min[0])
            t_max = float(time_offsets_min[-1])
            for key, color, label in (
                ("arrived_time",   "#2563eb", "到達"),
                ("completed_time", "#16a34a", "處理完成"),
            ):
                val = accident_meta.get(key)
                if val is not None and pd.notna(val):
                    offset = (pd.Timestamp(val) - rt).total_seconds() / 60
                    if t_min <= offset <= t_max:
                        ax.axvline(
                            offset, color=color, linewidth=1.5,
                            linestyle="-.", alpha=0.85, label=label,
                        )
        ax.plot(0, 0, "o", color="red", markersize=8, zorder=5, label="事故發送")
        ax.legend(loc="upper left", fontsize=max(fs - 1, 8), framealpha=0.85)

    # Title
    if title:
        ax.set_title(title, fontsize=style.title_size, fontweight="bold", pad=8)
    elif accident_meta:
        parts = []
        if report_time is not None:
            parts.append(f"{pd.Timestamp(report_time):%Y-%m-%d %H:%M}")
        fw = accident_meta.get("freeway", "?")
        dr = accident_meta.get("direction", "?")
        mk = accident_meta.get("mileage_km")
        mk_str = f"{mk:.1f}K" if mk is not None else "?"
        aid = accident_meta.get("accident_id", "?")
        parts.append(f"{fw}-{dr} | {mk_str} | {aid}")
        ax.set_title(" | ".join(parts), fontsize=style.title_size, fontweight="bold", pad=8)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=fs)
    cbar.ax.tick_params(labelsize=max(fs - 1, 8))


# ── Public rendering API ──────────────────────────────────────────────────────
def heatmap(
    matrix: np.ndarray,
    space_axis_km: np.ndarray,
    time_offsets_min: np.ndarray,
    accident_meta: dict | None = None,
    *,
    title: str | None = None,
    style: HeatmapStyle | None = None,
    resolution_min: int = 5,
    output_path: str | Path | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Render a speed heatmap.

    Parameters
    ----------
    matrix:
        Shape ``(n_space, n_time)``, dtype float, NaN = missing cell.
    space_axis_km:
        Shape ``(n_space,)``. Upstream-positive km from accident point.
        Typically ``[-downstream_km, …, 0, …, +upstream_km]``.
    time_offsets_min:
        Shape ``(n_time,)``. Minutes from report time.  Negative = pre-accident.
    accident_meta:
        Optional dict with any subset of: ``accident_id``, ``freeway``,
        ``direction``, ``mileage_km``, ``report_time``, ``arrived_time``,
        ``completed_time``.
    style:
        One of ``STYLE_NOTEBOOK``, ``STYLE_REPORT``, ``STYLE_SLIDE``.
        Defaults to ``STYLE_REPORT``.
    """
    style = style or DEFAULT_STYLE
    own_fig = fig is None
    if own_fig:
        fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)
    im = _imshow(ax, matrix, space_axis_km, time_offsets_min,
                 cmap=style.cmap, vmin=style.speed_min, vmax=style.speed_max)
    _decorate(ax, fig, im, space_axis_km, time_offsets_min, accident_meta,
              title=title, cbar_label="速度 (km/h)", style=style,
              resolution_min=resolution_min)
    if own_fig:
        fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), bbox_inches="tight")
    return fig


def diff_heatmap(
    matrix: np.ndarray,
    space_axis_km: np.ndarray,
    time_offsets_min: np.ndarray,
    accident_meta: dict | None = None,
    *,
    title: str | None = None,
    style: HeatmapStyle | None = None,
    resolution_min: int = 5,
    output_path: str | Path | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Render a signed-difference heatmap (e.g. observed − historical).

    Same signature as :func:`heatmap` but uses the diverging colormap and
    ``±diff_abs_max`` range.
    """
    style = style or DEFAULT_STYLE
    own_fig = fig is None
    if own_fig:
        fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)
    im = _imshow(ax, matrix, space_axis_km, time_offsets_min,
                 cmap=style.diff_cmap, vmin=-style.diff_abs_max, vmax=style.diff_abs_max)
    _decorate(ax, fig, im, space_axis_km, time_offsets_min, accident_meta,
              title=title, cbar_label="速度差值 (km/h)", style=style,
              resolution_min=resolution_min)
    if own_fig:
        fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), bbox_inches="tight")
    return fig


def compare_heatmaps(
    matrices: Sequence[np.ndarray],
    titles: Sequence[str],
    space_axis_km: np.ndarray,
    time_offsets_min: np.ndarray,
    accident_meta: dict | None = None,
    *,
    style: HeatmapStyle | None = None,
    resolution_min: int = 5,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Render multiple heatmaps side by side in one figure.

    Matrices with any negative finite values automatically use the diverging
    colormap; others use the speed colormap.
    """
    style = style or DEFAULT_STYLE
    if len(matrices) != len(titles):
        raise ValueError("matrices and titles must have the same length")
    n = len(matrices)
    fig, axes = plt.subplots(
        1, n,
        figsize=(style.figsize[0] * n / 1.6, style.figsize[1]),
        dpi=style.dpi,
        squeeze=False,
    )
    for i, (m, t) in enumerate(zip(matrices, titles)):
        ax = axes[0, i]
        finite_min = float(np.nanmin(m)) if np.isfinite(m).any() else 0.0
        if finite_min < 0:
            diff_heatmap(m, space_axis_km, time_offsets_min, accident_meta,
                         title=t, style=style, resolution_min=resolution_min,
                         fig=fig, ax=ax)
        else:
            heatmap(m, space_axis_km, time_offsets_min, accident_meta,
                    title=t, style=style, resolution_min=resolution_min,
                    fig=fig, ax=ax)
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), bbox_inches="tight")
    return fig
