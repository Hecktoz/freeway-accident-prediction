# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "scipy",
# ]
# ///
# ruff: noqa: E402
"""
Generate slide materials for F1–F6 accident filtering logic.

Outputs (all under output/slide_materials/):
  filter_summary_bar.png   — per-filter count bar chart
  filter_funnel.png        — waterfall: total → each active filter → kept
  overlap_matrix.png       — pairwise co-occurrence heatmap
  distributions.png        — speed_drop / post_min / hist_deficit KDE: kept vs removed
  f1_examples.png          — 3 representative heatmaps for F1 group
  f2_examples.png          — F2
  f3_examples.png          — F3
  f4_examples.png          — F4
  f5_examples.png          — F5
  f6_examples.png          — F6
  kept_examples.png        — 3 clear-congestion cases that pass all filters

Usage:
  uv run scripts/visualization/filter_slide_materials.py
  uv run scripts/visualization/filter_slide_materials.py --output-dir output/my_slides
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import os as _os
if _os.environ.get("MPLBACKEND", "").startswith("module://"):
    _os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.visualization import (
    STYLE_SLIDE,
    configure_cjk_font,
    heatmap as lib_heatmap,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_NPZ      = PROJECT_ROOT / "data" / "accident_samples_5m.npz"
DEFAULT_QUALITY  = PROJECT_ROOT / "output" / "quality" / "accident_quality.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "accident_samples_5m_metadata.parquet"
DEFAULT_OUT_DIR  = PROJECT_ROOT / "output" / "slide_materials"

FLAG_COLS = [
    "f1_no_low_speed",
    "f2_small_drop",
    "f3_pre_congested",
    "f4_overnight",
    "f5_low_cov",
    "f6_like_historical",
]
FLAG_LABELS = ["F1\n無低速", "F2\n降速小", "F3\n事前壅塞", "F4\n深夜", "F5\n涵蓋率低", "F6\n類歷史"]
FLAG_SHORT  = ["F1", "F2", "F3", "F4", "F5", "F6"]

FILTER_DESCS = {
    "f1_no_low_speed":    "F1：事後上游無低於 60 km/h 讀值（無明顯壅塞）",
    "f2_small_drop":      "F2：速降 < 15 km/h 且 < 15%（矩陣內差異太小）",
    "f3_pre_congested":   "F3：事前探針中位速 < 60 km/h（事前已壅塞）",
    "f4_overnight":       "F4：報案時間 23:00–05:00（深夜，流量模式不同）",
    "f5_low_cov":         "F5：事後上游 VD 涵蓋率 < 30%（資料不足）",
    "f6_like_historical": "F6：速度赤字 vs 歷史 p50 < 8 km/h（與平常日無異）",
}

ACTIVE_FLAGS = {"f1_no_low_speed", "f2_small_drop", "f6_like_historical"}
FLAG_COLORS  = ["#e74c3c", "#e67e22", "#9b59b6", "#3498db", "#1abc9c", "#f39c12"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(str(path), bbox_inches="tight", dpi=fig.get_dpi())
    plt.close(fig)
    print(f"  saved → {path.relative_to(PROJECT_ROOT)}")


def _pick_examples(
    quality_df: pd.DataFrame,
    flag_col: str,
    n: int = 3,
    exclude_flags: list[str] | None = None,
    sort_col: str | None = None,
    ascending: bool = True,
) -> list[str]:
    mask = quality_df[flag_col]
    if exclude_flags:
        for ef in exclude_flags:
            mask = mask & ~quality_df[ef]
    subset = quality_df[mask]
    if sort_col and sort_col in subset.columns:
        subset = subset.dropna(subset=[sort_col]).sort_values(sort_col, ascending=ascending)
    return subset["accident_id"].astype(str).tolist()[:n]


def _render_one_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    space_km: np.ndarray,
    time_min: np.ndarray,
    meta: dict | None,
    title: str,
) -> None:
    from scripts.lib.visualization import _imshow, _decorate, STYLE_SLIDE
    style = STYLE_SLIDE
    im = _imshow(ax, matrix, space_km, time_min,
                 cmap=style.cmap, vmin=style.speed_min, vmax=style.speed_max)
    _decorate(ax, ax.get_figure(), im, space_km, time_min, meta,
              title=title, cbar_label="速度 (km/h)", style=style, resolution_min=5)


# ── 1. Filter summary bar chart ───────────────────────────────────────────────

def plot_filter_summary(quality_df: pd.DataFrame, out_dir: Path) -> None:
    n_total = len(quality_df)
    counts = [quality_df[f].sum() for f in FLAG_COLS]
    n_kept = (~quality_df["filtered_out"]).sum()
    n_removed = quality_df["filtered_out"].sum()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # Left: individual filter counts
    ax = axes[0]
    bars = ax.barh(FLAG_SHORT[::-1], [counts[i] for i in range(5, -1, -1)],
                   color=[FLAG_COLORS[i] for i in range(5, -1, -1)], alpha=0.85, height=0.6)
    ax.set_xlabel("事故筆數", fontsize=12)
    ax.set_title(f"各篩選條件命中數（共 {n_total} 筆）", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(counts) * 1.2)
    for bar, cnt in zip(bars, [counts[i] for i in range(5, -1, -1)]):
        pct = cnt / n_total * 100
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{cnt}  ({pct:.1f}%)", va="center", fontsize=10)
    # Mark active filters
    yticklabels = [f"[*] {l}" if FLAG_COLS[5 - i] in ACTIVE_FLAGS else l
                   for i, l in enumerate(FLAG_SHORT[::-1])]
    ax.set_yticks(range(len(FLAG_SHORT)))
    ax.set_yticklabels(yticklabels, fontsize=11)
    ax.text(0.98, 0.02, "[*] = 預設啟用", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="gray")
    ax.grid(axis="x", alpha=0.3)

    # Right: pie — kept vs removed by active filters
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        [n_kept, n_removed],
        labels=[f"保留\n{n_kept} 筆", f"過濾\n{n_removed} 筆"],
        colors=["#2ecc71", "#e74c3c"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
        wedgeprops={"alpha": 0.85},
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
    ax2.set_title(f"主動篩選（F1+F2+F6）後留存率\n（共 {n_total} 筆）", fontsize=13, fontweight="bold")

    _save(fig, out_dir / "filter_summary_bar.png")


# ── 2. Funnel / waterfall chart ───────────────────────────────────────────────

def plot_filter_funnel(quality_df: pd.DataFrame, out_dir: Path) -> None:
    n_total = len(quality_df)

    # Compute cumulative removal for active flags (sequential union)
    active_ordered = ["f1_no_low_speed", "f2_small_drop", "f6_like_historical"]
    # Also show optional flags as additional info
    all_ordered = active_ordered + ["f3_pre_congested", "f4_overnight", "f5_low_cov"]

    steps = [("原始資料", n_total)]
    removed_so_far = pd.Series(False, index=quality_df.index)

    for flag in active_ordered:
        removed_so_far |= quality_df[flag]
        n_kept = (~removed_so_far).sum()
        label = flag.replace("f1_no_low_speed", "F1 移除後").replace(
            "f2_small_drop", "F2 移除後").replace("f6_like_historical", "F6 移除後")
        steps.append((label, n_kept))

    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    x = np.arange(len(steps))
    bar_colors = ["#3498db"] + ["#e74c3c"] * (len(steps) - 2) + ["#2ecc71"]
    bars = ax.bar(x, values, color=bar_colors, alpha=0.85, width=0.55)

    # Annotate each bar
    for i, (bar, val) in enumerate(zip(bars, values)):
        pct = val / n_total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    # Add drop annotations between bars
    for i in range(1, len(steps)):
        drop = values[i - 1] - values[i]
        if drop > 0:
            mid_x = (x[i - 1] + x[i]) / 2
            mid_y = (values[i - 1] + values[i]) / 2
            ax.annotate(f"-{drop}", xy=(mid_x, mid_y), ha="center", fontsize=10,
                        color="#c0392b", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("事故筆數", fontsize=12)
    ax.set_ylim(0, n_total * 1.15)
    ax.set_title("主動篩選條件逐步縮減漏斗圖（F1 → F2 → F6）", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Side annotation: optional filters
    opt_text = "選用篩選條件（預設關閉）：\n"
    for flag in ["f3_pre_congested", "f4_overnight", "f5_low_cov"]:
        cnt = quality_df[flag].sum()
        name = {"f3_pre_congested": "F3 事前壅塞",
                 "f4_overnight": "F4 深夜",
                 "f5_low_cov": "F5 低涵蓋率"7��flag]
        opt_text += f"  {name}: {cnt} 筆\n"
    ax.text(0.98, 0.95, opt_text.strip(), transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="gray",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    _save(fig, out_dir / "filter_funnel.png")


# ── 3. Overlap matrix ───────────────────────────────────────────────────────────────────

def plot_overlap_matrix(quality_df: pd.DataFrame, out_dir: Path) -> None:
    n = len(FLAG_COLS)
    mat = np.zeros((n, n), dtype=int)
    for i, ci in enumerate(FLAG_COLS):
        for j, cj in enumerate(FLAG_COLS):
            mat[i, j] = (quality_df[ci] & quality_df[cj]).sum()

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax, label="共同命中數")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(FLAG_SHORT, fontsize=12)
    ax.set_yticklabels(FLAG_SHORT, fontsize=12)
    for i in range(n):
        for j in range(n):
            color = "white" if mat[i, j] > mat.max() * 0.6 else "black"
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)
    ax.set_title("篩選條件兩兩共同命中數（對角線 = 各條件總數）", fontsize=12, fontweight="bold")
    _save(fig, out_dir / "overlap_matrix.png")


# ── 4. Distribution comparison plots ─────────────────────────────────────────────────────────

def plot_distributions(quality_df: pd.DataFrame, out_dir: Path) -> None:
    kept    = quality_df[~quality_df["filtered_out"]]
    removed = quality_df[quality_df["filtered_out"]]

    metrics = [
        ("speed_drop_kmh",   "速降幅度 (km/h)\n(事前均速 − 事後最低速)",   0,   120, False),
        ("post_min_kmh",     "事後上游最低速度 (km/h)",                      0,   120, False),
        ("hist_deficit_kmh", "速度赃字 vs 歷史 p50 (km/h)\n（正值 = 比平常慢）", -30, 80, False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    for ax, (col, xlabel, xmin, xmax, log_y) in zip(axes, metrics):
        kept_vals    = kept[col].dropna()
        removed_vals = removed[col].dropna()

        bins = np.linspace(xmin, xmax, 35)
        ax.hist(kept_vals,    bins=bins, density=True, alpha=0.65,
                color="#2ecc71", label=f"保留 ({len(kept_vals)} 筆)", edgecolor="white")
        ax.hist(removed_vals, bins=bins, density=True, alpha=0.65,
                color="#e74c3c", label=f"過忾 ({len(removed_vals)} 筆)", edgecolor="white")

        # Median lines
        for vals, color in [(kept_vals, "#27ae60"), (removed_vals, "#c0392b")]:
            if len(vals):
                ax.axvline(vals.median(), color=color, linestyle="--", linewidth=2)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("密度", fontsize=11)
        ax.legend(fontsize=10, framealpha=0.85)
        ax.set_xlim(xmin, xmax)
        ax.grid(alpha=0.3)

    axes[0].set_title("速降幅度分布", fontsize=12, fontweight="bold")
    axes[1].set_title("事後最低速度分布", fontsize=12, fontweight="bold")
    axes[2].set_title("F6 速度赃字分布（藊線 = 中位數）", fontsize=12, fontweight="bold")

    fig.suptitle("保留 vs 過濾事故的速度指標分布比較（主勝篩選：F1+F2+F6）",
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, out_dir / "distributions.png")


# ── 5. Example heatmaps per filter ───────────────────────────────────────────────

def _load_npz(npz_path: Path) -> dict:
    raw = np.load(str(npz_path), allow_pickle=False)
    return {
        "matrices":    raw["matrix"],           # (N, n_space, n_time)
        "ids":         raw["accident_ids"].astype(str),
        "space_km":    raw["space_offsets_m"].astype(float) / 1000.0,
        "time_min":    raw["time_offsets_min"].astype(float),
    }


def _get_matrix(npz: dict, accident_id: str) -> np.ndarray | None:
    idx_arr = np.where(npz["ids"] == accident_id)[0]
    if len(idx_arr) == 0:
        return None
    return npz["matrices"][idx_arr[0]]  # (n_space, n_time)


def _meta_for(quality_df: pd.DataFrame, aid: str) -> dict:
    row = quality_df[quality_df["accident_id"].astype(str) == aid]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "accident_id": str(r["accident_id"]),
        "freeway":    str(r.get("freeway", "")),
        "direction":  str(r.get("direction", "")),
        "mileage_km": float(r.get("mileage_km") or 0),
        "report_time": r.get("report_time"),
    }


_FILTER_ANNOTATION_TMPL: dict[str, str] = {
    "f1_no_low_speed":    "事後上游最低速度：{post_min_kmh:.0f} km/h\n（閾值：< 60 km/h 才算壅塞）",
    "f2_small_drop":      "速降幅度：{speed_drop_kmh:.0f} km/h（{speed_drop_ratio:.0%}）\n（閾值：< 15 km/h 且 < 15%）",
    "f3_pre_congested":   "事故前 15 分鐘探韝中位速：{probe_median_kmh:.0f} km/h\n（閾值：< 60 km/h 即認定事前已壅塞）",
    "f4_overnight":       "倱案時間：{report_hour}（深夜 23:00–05:00）\n（夊間浅量態與日間不同）",
    "f5_low_cov":         "事後上游 VD 涵蓋率：{post_zone_coverage:.0%}\n（閾值：< 30% 資料不足）",
    "f6_like_historical": "速度赤字平均'\n羉綪中數’",
    "_kept":              "速降幅度：{speed_drop_kmh:.0f} km/h　事後最低速：{post_min_kmh:.0f} km/h",
}


def _filter_annotation(row: pd.Series, flag_col: str) -> str:
    tmpl = _FILTER_ANNOTATION_TMPL.get(flag_col, "")
    if not tmpl:
        return ""
    try:
        report_time = row.get("report_time")
        report_hour = pd.Timestamp(report_time).strftime("%H:%M") if pd.notna(report_time) else "?"
        return tmpl.format(
            post_min_kmh    = float(row.get("post_min_kmh")     or 0),
            speed_drop_kmh  = float(row.get("speed_drop_kmh")   or 0),
            speed_drop_ratio= float(row.get("speed_drop_ratio") or 0),
            probe_median_kmh= float(row.get("probe_median_kmh") or 0),
            post_zone_coverage= float(row.get("post_zone_coverage") or 0),
            hist_deficit_kmh= float(row.get("hist_deficit_kmh") or 0),
            report_hour     = report_hour,
        )
    except Exception:
        return ""


def _add_filter_annotation(ax: plt.Axes, text: str, color: str = "#fff3cd") -> None:
    if not text:
        return
    ax.text(
        0.5, -0.18, text,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.9, edgecolor="#aaa"),
        linespacing=1.5,
    )


def render_filter_examples(
    flag_col: str,
    quality_df: pd.DataFrame,
    npz: dict,
    out_dir: Path,
    n: int = 3,
) -> None:
    name = flag_col  # e.g. f1_no_low_speed
    short = name.split("_")[0].upper()  # F1

    # Pick accident ids
    exclude = [f for f in FLAG_COLS if f != flag_col]

    # Sort strategy per filter
    sort_strategies = {
        "f1_no_low_speed":    ("post_min_kmh",     False, None),   # highest post_min (most normal)
        "f2_small_drop":      ("speed_drop_kmh",   True,  None),   # smallest drop
        "f3_pre_congested":   ("probe_median_kmh", True,  None),   # lowest probe (most congested)
        "f4_overnight":       ("report_time",      True,  None),   # any order
        "f5_low_cov":         ("post_zone_coverage", True, None),  # lowest coverage
        "f6_like_historical": ("hist_deficit_kmh", True,  None),   # smallest deficit
    }
    sort_col, asc, _ = sort_strategies.get(flag_col, (None, True, None))

    # First try: exclusive examples (only this filter fires)
    ids = _pick_examples(quality_df, flag_col, n=n, exclude_flags=exclude,
                         sort_col=sort_col, ascending=asc)
    # Fall back: any accident with this flag
    if len(ids) < n:
        ids_all = _pick_examples(quality_df, flag_col, n=n,
                                 sort_col=sort_col, ascending=asc)
        combined = ids + [x for x in ids_all if x not in ids]
        ids = combined[:n]

    if not ids:
        print(f"  [SKIP] No examples found for {flag_col}")
        return

    ncols = len(ids)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 8.5), dpi=130)
    if ncols == 1:
        axes = [axes]

    for ax, aid in zip(axes, ids):
        mat = _get_matrix(npz, aid)
        if mat is None:
            ax.text(0.5, 0.5, f"{short}\n(not in NPZ)", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        row = quality_df[quality_df["accident_id"].astype(str) == aid]
        meta = _meta_for(quality_df, aid)
        # title=None → _decorate generates: "日期 | 路線-方向 | 里程 | 任務編號"
        _render_one_heatmap(ax, mat, npz["space_km"], npz["time_min"], meta, title=None)
        ann = _filter_annotation(row.iloc[0], flag_col) if not row.empty else ""
        _add_filter_annotation(ax, ann, color="#fff3cd")

    desc = FILTER_DESCS.get(flag_col, flag_col)
    fig.suptitle(f"{short} 過忾篩例\n{desc}", fontsize=13, fontweight="bold", y=1.02)
    fname = f"{short.lower()}_examples.png"
    fig.subplots_adjust(bottom=0.22)
    _save(fig, out_dir / fname, tight=False)


def render_kept_examples(quality_df: pd.DataFrame, npz: dict, out_dir: Path, n: int = 3) -> None:
    ids = _pick_examples(quality_df, "filtered_out",
                         n=n, sort_col="speed_drop_kmh", ascending=False)
    # filtered_out = False means kept → invert
    kept_ids = (
        quality_df[~quality_df["filtered_out"]]
        .dropna(subset=["speed_drop_kmh"])
        .nlargest(n, "speed_drop_kmh")["accident_id"].astype(str).tolist()
    )

    ncols = len(kept_ids)
    if ncols == 0:
        return
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 8.5), dpi=130)
    if ncols == 1:
        axes = [axes]

    for ax, aid in zip(axes, kept_ids):
        mat = _get_matrix(npz, aid)
        if mat is None:
            continue
        row = quality_df[quality_df["accident_id"].astype(str) == aid]
        meta = _meta_for(quality_df, aid)
        _render_one_heatmap(ax, mat, npz["space_km"], npz["time_min"], meta, title=None)
        ann = _filter_annotation(row.iloc[0], "_kept") if not row.empty else ""
        _add_filter_annotation(ax, ann, color="#d4edda")

    fig.suptitle("遚過篩選之代表事故（速降幅度最大前三筆）", fontsize=13, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.22)
    _save(fig, out_dir / "kept_examples.png", tight=False)


# ── 6. Per-filter stat annotation figure ─────────────────────────────────────

def plot_filter_stats_table(quality_df: pd.DataFrame, out_dir: Path) -> None:
    """Text table: for each filter — count, %, key metric median."""
    n_total = len(quality_df)
    kept    = quality_df[~quality_df["filtered_out"]]
    removed = quality_df[quality_df["filtered_out"]]

    rows = []
    for flag, label in zip(FLAG_COLS, FLAG_LABELS):
        cnt = quality_df[flag].sum()
        pct = cnt / n_total * 100
        active_str = "✓" if flag in ACTIVE_FLAGS else "—"
        rows.append([FLAG_SHORT[FLAG_COLS.index(flag)], active_str,
                     f"{cnt}", f"{pct:.1f}%",
                     FILTER_DESCS[flag]])

    # Add summary row
    n_removed = removed.shape[0]
    n_kept    = kept.shape[0]
    rows.append(["聯集", "✓",
                 f"{n_removed}", f"{n_removed/n_total:.1%}",
                 f"主動過濾危移除（F1+F2+F6 聯集）"])

    col_headers = ["篩選", "啟用", "命中數", "比例", "說明"]

    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="left",
        loc="center",
        colColours=["#2c3e50"] * 5,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_facecolor("#2c3e50")
        elif rows[r - 1][1] == "✓":
            cell.set_facecolor("#ffeaa7")
        elif r % 2 == 0:
            cell.set_facecolor("#f8f9fa")

    fig.suptitle(f"F1–F6 篩選條件摘要表（總計 {n_total} 筆，保留 {n_kept} 筆）",
                 fontsize=13, fontweight="bold", y=0.98)
    _save(fig, out_dir / "filter_stats_table.png")


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate slide materials for F1–F6 filtering.")
    p.add_argument("--npz",       default=str(DEFAULT_NPZ))
    p.add_argument("--quality",   default=str(DEFAULT_QUALITY))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--examples-per-filter", type=int, default=3)
    p.add_argument("--no-heatmaps", action="store_true", help="Skip heatmap rendering (faster)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_cjk_font()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading quality CSV: {args.quality}")
    quality_df = pd.read_csv(args.quality, encoding="utf-8-sig")
    print(f"  {len(quality_df)} accidents loaded")

    if not args.no_heatmaps:
        print(f"Loading NPZ: {args.npz}")
        npz = _load_npz(Path(args.npz))
        print(f"  {len(npz['ids'])} matrices, shape {npz['matrices'].shape[1:]}")
    else:
        npz = None

    print("\n[1/6] Filter summary bar chart …")
    plot_filter_summary(quality_df, out_dir)

    print("[2/6] Filter funnel chart …")
    plot_filter_funnel(quality_df, out_dir)

    print("[3/6] Overlap matrix …")
    plot_overlap_matrix(quality_df, out_dir)

    print("[4/6] Distribution comparison …")
    plot_distributions(quality_df, out_dir)

    print("[5/6] Filter stats table …")
    plot_filter_stats_table(quality_df, out_dir)

    if npz is not None:
        print(f"[6/6] Example heatmaps (n={args.examples_per_filter} per filter) …")
        for flag in FLAG_COLS:
            short = flag.split("_")[0].upper()
            print(f"  {short} …")
            render_filter_examples(flag, quality_df, npz, out_dir, n=args.examples_per_filter)
        print("  Kept examples …")
        render_kept_examples(quality_df, npz, out_dir, n=args.examples_per_filter)
    else:
        print("[6/6] Heatmaps skipped (--no-heatmaps)")

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
