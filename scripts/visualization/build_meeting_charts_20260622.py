# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.0",
#   "matplotlib>=3.8",
#   "scipy>=1.11",
# ]
# ///
"""Decoupled, single-purpose charts for the 2026-06 monthly meeting deck.

Every output is *one chart per file* with **no baked-in title / explanatory
prose** (axis labels, colorbars, and on-chart data guides only), so the slides
can be laid out and captioned by hand. Companion narrative lives in
``reports/slides_monthly_20260622_outline.md``.

Charts (-> output/slide_materials/monthly_20260622/):
  Topic A — origin_cc mask
    mask1_observed / mask2_baseline / mask3_deficit / mask4_candidates /
    mask5_components / mask6_origin_cc   (6-stage construction, one PNG each)
    loss_weight_schematic                (downstream 0 / upstream 1 / origin_cc w=4 / >20km 0)
    wsweep_tradeoff                       (w-sweep: congestion MAE vs impact/recovery AE)
  Topic B — upstream 20 km
    extent_abs_hist                       (naive abs extent — bimodal / contaminated)
    extent_origin_cc_hist                 (origin_cc extent — clean, 20 km cap line)
    window_neutrality                     (train x eval window MAE matrix, ~flat)

Reuses the locked mask logic in ``scripts/lib/congestion_mask.py`` and the speed
colormap / range constants in ``scripts/lib/visualization.py`` so figures match
training/eval exactly. Ablation numbers are transcribed from the locked report
``reports/loss_window_decision_20260603.md`` (val).

Usage
-----
  uv run scripts/visualization/build_meeting_charts_20260622.py
  uv run scripts/visualization/build_meeting_charts_20260622.py --accident-id 2024110787
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.lib.congestion_mask import (  # noqa: E402
    base_abs_mask,
    origin_cc_mask,
    origin_cc_congestion_mask,
    DELTA_MIN_DEFAULT,
    CONG_KMH_DEFAULT,
    PERSIST_N_DEFAULT,
    ANCHOR_T_MAX_DEFAULT,
    ANCHOR_S_KM_MAX_DEFAULT,
)
from scripts.lib import visualization as viz  # noqa: E402  (configures CJK font on import)

plt.rcParams["axes.unicode_minus"] = False  # JhengHei lacks U+2212

SPEED_CMAP = plt.get_cmap(viz.SPEED_CMAP).copy()
SPEED_CMAP.set_bad("white")
CONG_UPSTREAM_KM = 20  # locked loss/eval upstream cap
WEIGHT_W = 4           # locked origin_cc congested-cell weight

CSV_ORIGIN_CC = "output/analysis/upstream_extent_per_accident_origin_cc.csv"

# Locked w-sweep (val), reports/loss_window_decision_20260603.md + loss_window_ablation summary.csv.
WSWEEP = {
    "label":        ["cell_MAE", "w=4", "w=7", "w=10"],
    "origin_cc":    [30.66, 28.29, 24.79, 22.15],
    "recovery_ae":  [13.70, 14.60, 18.46, 25.11],
    "f1":           [0.465, 0.478, 0.476, 0.479],
}
# Train-window x eval-window origin_cc_MAE (val), same report §1 (accuracy-neutral).
WIN_TRAIN = [10, 15, 20]
WIN_EVAL = [10, 15, 20, 25]
WIN_MAE = np.array([
    [21.66, 21.86, 21.89, 21.89],
    [21.53, 21.85, 21.86, 21.86],
    [21.62, 22.08, 22.15, 22.15],
])


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ── shared spacetime helpers (mask stages + schematic) ───────────────────────
def _imshow(ax, M, extent, **kw):
    return ax.imshow(np.ma.masked_invalid(M), aspect="auto", interpolation="nearest",
                     origin="lower", extent=extent, **kw)


def _ref_lines(ax):
    ax.axvline(0, color="red", linestyle="--", linewidth=1.6, alpha=0.85, zorder=4)
    ax.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.5, zorder=4)


def _anchor_patch(t_max, s_max):
    from matplotlib.patches import Rectangle
    return Rectangle((0.0, 0.0), t_max, s_max, fill=False, edgecolor="#111111",
                     linewidth=1.8, linestyle=":", zorder=6)


def _axes(ax, extent, *, xlabel=True, ylabel=True):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if xlabel:
        ax.set_xlabel("事故時間偏移（分鐘）", fontsize=12)
    if ylabel:
        ax.set_ylabel("相對事故位置（km，上游為正）", fontsize=12)
    ax.tick_params(labelsize=10)


def _crop(s_km, t_min, final, *, s_pad=5.0, t_lo_pad=30.0, t_hi_pad=25.0):
    if final.any():
        ss = s_km[final.any(axis=1)]
        tt = t_min[final.any(axis=0)]
        s_lo, s_hi = ss.min() - s_pad, ss.max() + s_pad
        t_lo, t_hi = tt.min() - t_lo_pad, tt.max() + t_hi_pad
    else:
        s_lo, s_hi, t_lo, t_hi = -8, 22, -30, 150
    s_lo, s_hi = min(s_lo, -5.0), max(s_hi, 10.0)
    t_lo, t_hi = min(t_lo, -30.0), max(t_hi, 60.0)
    # cap display to the research window (upstream 25 km / post 145 min) for a tight,
    # consistent view that matches the prediction slides
    s_lo, s_hi = max(s_lo, -10.0), min(s_hi, 25.0)
    t_lo, t_hi = max(t_lo, -30.0), min(t_hi, 145.0)
    si = (s_km >= s_lo) & (s_km <= s_hi)
    ti = (t_min >= t_lo) & (t_min <= t_hi)
    return si, ti


def _load_accident(data: Path, aid: str):
    obs = np.load(data / "accident_samples_5m.npz", allow_pickle=False)
    base = np.load(data / "historical_matrices_5m.npz", allow_pickle=False)
    ids = obs["accident_ids"].astype(str)
    s_m = obs["space_offsets_m"].astype(np.float64)
    t_min = obs["time_offsets_min"].astype(np.float64)
    i = int(np.where(ids == aid)[0][0])
    O = obs["matrix"][i].astype(np.float32)
    b_ids = base["accident_ids"].astype(str)
    j = int(np.where(b_ids == aid)[0][0])
    B = base["matrix"][j].astype(np.float32)
    return O, B, s_m, t_min


# ── Topic A: mask construction (6 individual stages) ─────────────────────────
def mask_stages(data: Path, out: Path, aid: str) -> None:
    O, B, s_m, t_min = _load_accident(data, aid)
    s_km = s_m / 1000.0
    abs_m = base_abs_mask(O, B)
    labels, n_cc = ndi.label(abs_m, structure=np.ones((3, 3), bool))
    final = origin_cc_mask(abs_m, s_m, t_min)
    si, ti = _crop(s_km, t_min, final)
    sk, tk = s_km[si], t_min[ti]
    extent = [tk[0], tk[-1], sk[0], sk[-1]]
    ta, sa = float(ANCHOR_T_MAX_DEFAULT), float(ANCHOR_S_KM_MAX_DEFAULT)

    def cw(A):
        return A[np.ix_(si, ti)]

    Oc, Bc = cw(O), cw(B)
    deficit = np.where(np.isfinite(Oc) & np.isfinite(Bc), Bc - Oc, np.nan)
    absc, labc, finalc = cw(abs_m).astype(float), cw(labels), cw(final)

    def new_ax():
        fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=150)
        return fig, ax

    # 1 observed
    fig, ax = new_ax()
    im = _imshow(ax, Oc, extent, cmap=SPEED_CMAP, vmin=viz.SPEED_MIN, vmax=viz.SPEED_MAX)
    _ref_lines(ax); ax.add_patch(_anchor_patch(ta, sa)); ax.plot(0, 0, "o", color="red", ms=8, zorder=7)
    _axes(ax, extent); fig.colorbar(im, ax=ax, pad=0.02, label="km/h"); _save(fig, out / "mask1_observed.png")

    # 2 baseline
    fig, ax = new_ax()
    im = _imshow(ax, Bc, extent, cmap=SPEED_CMAP, vmin=viz.SPEED_MIN, vmax=viz.SPEED_MAX)
    _ref_lines(ax); _axes(ax, extent); fig.colorbar(im, ax=ax, pad=0.02, label="km/h")
    _save(fig, out / "mask2_baseline.png")

    # 3 deficit
    fig, ax = new_ax()
    im = _imshow(ax, deficit, extent, cmap="RdBu_r", vmin=-viz.DIFF_ABS_MAX, vmax=viz.DIFF_ABS_MAX)
    if np.isfinite(deficit).any():
        ax.contour(tk, sk, np.nan_to_num(deficit, nan=-999.0), levels=[DELTA_MIN_DEFAULT],
                   colors="black", linewidths=1.6)
    _ref_lines(ax); _axes(ax, extent); fig.colorbar(im, ax=ax, pad=0.02, label="base - obs (km/h)")
    _save(fig, out / "mask3_deficit.png")

    # 4 candidate cells
    fig, ax = new_ax()
    ax.set_facecolor("#f4f4f4")
    _imshow(ax, np.where(absc > 0, 1.0, np.nan), extent, cmap=ListedColormap(["#c0392b"]), vmin=0, vmax=1)
    _ref_lines(ax); ax.add_patch(_anchor_patch(ta, sa)); _axes(ax, extent)
    _save(fig, out / "mask4_candidates.png")

    # 5 connected components: kept vs dropped
    fig, ax = new_ax()
    ax.set_facecolor("#f4f4f4")
    disp = np.full(labc.shape, np.nan)
    disp[(labc > 0) & (~finalc)] = 0.0
    disp[finalc] = 1.0
    _imshow(ax, disp, extent, cmap=ListedColormap(["#9aa0a6", "#c0392b"]), vmin=-0.5, vmax=1.5)
    if finalc.any():
        ax.contour(tk, sk, finalc.astype(float), levels=[0.5], colors="black", linewidths=2.2)
    _ref_lines(ax); ax.add_patch(_anchor_patch(ta, sa)); _axes(ax, extent)
    _save(fig, out / "mask5_components.png")

    # 6 origin_cc
    fig, ax = new_ax()
    im = _imshow(ax, Oc, extent, cmap=SPEED_CMAP, vmin=viz.SPEED_MIN, vmax=viz.SPEED_MAX, alpha=0.45)
    if finalc.any():
        ax.contourf(tk, sk, finalc.astype(float), levels=[0.5, 1.5], colors=["#1f4e79"], alpha=0.30)
        ax.contour(tk, sk, finalc.astype(float), levels=[0.5], colors="#1f4e79", linewidths=2.6)
    _ref_lines(ax); ax.add_patch(_anchor_patch(ta, sa)); ax.plot(0, 0, "o", color="red", ms=8, zorder=7)
    _axes(ax, extent); fig.colorbar(im, ax=ax, pad=0.02, label="km/h")
    _save(fig, out / "mask6_origin_cc.png")

    print(f"  mask stages: n_cc={n_cc}, origin_cc cells={int(final.sum())}")


# ── Topic A: loss-weight schematic ───────────────────────────────────────────
def loss_weight_schematic(data: Path, out: Path, aid: str) -> None:
    O, B, s_m, t_min = _load_accident(data, aid)
    s_km = s_m / 1000.0
    occ = origin_cc_congestion_mask(O, B, s_m, t_min)
    post = (t_min >= 0) & (t_min <= 145)  # research/eval post window
    sp, tp = s_km, t_min[post]
    occ_p = occ[:, post]

    weight = np.zeros_like(occ_p, dtype=float)          # downstream + beyond-cap = 0
    up_zone = np.broadcast_to((sp[:, None] >= 0) & (sp[:, None] <= CONG_UPSTREAM_KM), occ_p.shape)
    weight[up_zone] = 1.0
    weight[up_zone & (occ_p > 0)] = float(WEIGHT_W)

    # crop to a readable upstream span around the cap
    si = (sp >= -10) & (sp <= 25)
    sk = sp[si]
    W = weight[si]
    extent = [tp[0], tp[-1], sk[0], sk[-1]]

    cmap = ListedColormap(["#e9e9e9", "#9ecae1", "#08519c"])  # 0, 1, w
    norm = BoundaryNorm([-0.5, 0.5, 1.5, float(WEIGHT_W) + 0.5], cmap.N)

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=150)
    im = ax.imshow(W, aspect="auto", interpolation="nearest", origin="lower",
                   extent=extent, cmap=cmap, norm=norm)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.axhline(CONG_UPSTREAM_KM, color="black", linestyle="--", linewidth=1.6)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.4, alpha=0.8)
    _axes(ax, extent)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, WEIGHT_W], pad=0.02)
    cbar.set_label("loss 權重", fontsize=11)
    cbar.ax.set_yticklabels(["0", "1", f"{WEIGHT_W}"])
    _save(fig, out / "loss_weight_schematic.png")


# ── Topic A: w-sweep tradeoff chart ──────────────────────────────────────────
def wsweep_tradeoff(out: Path) -> None:
    x = np.arange(len(WSWEEP["label"]))
    c_occ, c_rec, c_f1 = "#08519c", "#31a354", "#e6550d"
    fig, ax1 = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    ax1.plot(x, WSWEEP["origin_cc"], "-o", color=c_occ, lw=2.4, ms=8, label="origin_cc 壅塞 MAE (km/h)")
    ax1.set_ylabel("origin_cc 壅塞 MAE (km/h)", fontsize=12, color=c_occ)
    ax1.set_ylim(20, 32)
    ax1.tick_params(axis="y", labelcolor=c_occ, labelsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels(WSWEEP["label"], fontsize=11)
    ax1.set_xlabel("上游壅塞權重 w", fontsize=12)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, WSWEEP["recovery_ae"], "-^", color=c_rec, lw=2.0, ms=7, label="recovery AE (min)")
    ax2.set_ylabel("recovery 時間誤差 (min)", fontsize=12, color=c_rec)
    ax2.tick_params(axis="y", labelcolor=c_rec, labelsize=10)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 58))
    ax3.plot(x, WSWEEP["f1"], "-D", color=c_f1, lw=2.0, ms=7, label="congestion F1@20")
    ax3.set_ylabel("congestion F1@20", fontsize=12, color=c_f1)
    ax3.set_ylim(0.45, 0.49)
    ax3.tick_params(axis="y", labelcolor=c_f1, labelsize=10)

    ax1.axvspan(0.7, 1.3, color="#fff2cc", alpha=0.8, zorder=0)  # highlight w=4
    handles = [*ax1.get_lines(), *ax2.get_lines(), *ax3.get_lines()]
    ax1.legend(handles, [h.get_label() for h in handles], fontsize=9.5,
               loc="upper center", framealpha=0.9)
    _save(fig, out / "wsweep_tradeoff.png")


# ── Topic B: extent distributions ───────────────────────────────────────────
CDF_MARKS_KM = (10.0, 15.0, 20.0)  # candidate upstream caps to read off the ECDF


def _dist(values: np.ndarray, out: Path, color: str, cap_km: float | None,
          box_xy: tuple[float, float] = (0.60, 0.16)) -> None:
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=150)
    bins = np.arange(0, 52, 2)
    ax.hist(values, bins=bins, color=color, alpha=0.8, edgecolor="white")
    ax.set_xlabel("上游壅塞延伸距離 (km)", fontsize=12)
    ax.set_ylabel("事故數", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(-1, 51)

    ax2 = ax.twinx()
    xs = np.sort(values)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax2.plot(xs, ys, color="#444444", lw=2.0)
    ax2.set_ylabel("ECDF", fontsize=12)
    ax2.set_ylim(0, 1.02)
    ax2.tick_params(labelsize=10)

    # ECDF readouts at the candidate upstream caps (10 / 15 / 20 km): guide line +
    # marker on the curve, with the figures collected into one corner box so the
    # closely-spaced points stay legible.
    rows = []
    for km in CDF_MARKS_KM:
        frac = float((values <= km).mean())
        is_cap = cap_km is not None and abs(km - cap_km) < 1e-6
        lc = "#d62728" if is_cap else "#1f4e79"
        ax2.axvline(km, color=lc, linestyle="--",
                    lw=1.8 if is_cap else 1.1, alpha=0.9 if is_cap else 0.55, zorder=2)
        ax2.plot([km], [frac], "o", color=lc, ms=6.5, zorder=6)
        rows.append(f"<= {km:g} km: {frac:.0%}" + ("  (cap)" if is_cap else ""))

    x0, y0 = box_xy
    box = "ECDF\n" + "\n".join(rows)
    ax2.text(x0, y0, box, transform=ax2.transAxes, ha="left", va="bottom",
             fontsize=11, color="#222", linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#888", alpha=0.94),
             zorder=7)
    _save(fig, out)


def extent_distributions(out: Path) -> None:
    df = pd.read_csv(PROJECT_ROOT / CSV_ORIGIN_CC)
    df = df[df["filled_ratio"] >= 0.8]
    _dist(df["max_upstream_extent_km_abs"].to_numpy(float), out / "extent_abs_hist.png",
          "#bdbdbd", None)
    _dist(df["max_upstream_extent_km_origin_cc"].to_numpy(float), out / "extent_origin_cc_hist.png",
          "#3182bd", float(CONG_UPSTREAM_KM))
    occ = df["max_upstream_extent_km_origin_cc"].to_numpy(float)
    occ = occ[np.isfinite(occ)]
    print(f"  origin_cc extent: p50={np.percentile(occ,50):.0f} p90={np.percentile(occ,90):.0f} "
          f"p99={np.percentile(occ,99):.0f} max={occ.max():.0f} zero={np.mean(occ==0):.0%}")


# ── Topic B: window-neutrality matrix ────────────────────────────────────────
def window_neutrality(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=150)
    im = ax.imshow(WIN_MAE, cmap="YlGnBu", vmin=21.3, vmax=22.3, aspect="auto")
    ax.set_xticks(range(len(WIN_EVAL))); ax.set_xticklabels([f"{e}" for e in WIN_EVAL], fontsize=11)
    ax.set_yticks(range(len(WIN_TRAIN))); ax.set_yticklabels([f"{t}" for t in WIN_TRAIN], fontsize=11)
    ax.set_xlabel("評估窗 (km)", fontsize=12)
    ax.set_ylabel("訓練窗 (km)", fontsize=12)
    for i in range(WIN_MAE.shape[0]):
        for j in range(WIN_MAE.shape[1]):
            ax.text(j, i, f"{WIN_MAE[i, j]:.2f}", ha="center", va="center", fontsize=11, color="#222")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("origin_cc MAE (km/h)", fontsize=10)
    _save(fig, out / "window_neutrality.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--output-root", default=str(PROJECT_ROOT / "output" / "slide_materials" / "monthly_20260622"))
    ap.add_argument("--accident-id", default="2024110787", help="Accident id for mask construction + weight schematic.")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    data = Path(args.data_root)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    print("== Topic A: origin_cc mask ==")
    mask_stages(data, out, args.accident_id)
    loss_weight_schematic(data, out, args.accident_id)
    wsweep_tradeoff(out)
    print("== Topic B: upstream 20 km ==")
    extent_distributions(out)
    window_neutrality(out)
    print(f"\nDone. Charts in {out}")


if __name__ == "__main__":
    main()
