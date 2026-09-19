# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "numpy>=1.26",
#   "pandas>=2.0",
#   "pyarrow>=14",
#   "matplotlib>=3.8",
#   "scipy>=1.11",
#   "tqdm",
# ]
# ///
"""Case-study prediction heatmaps for the 2026-06 monthly meeting deck.

Offline inference with the **locked-spec** checkpoints from the loss/window
ablation (`reports/loss_window_decision_20260603.md`):
  - w=4 origin_cc-weighted : output/models/loss_window_ablation/.../w20_origin_cc_wt4
  - cell_MAE baseline      : output/models/loss_window_ablation/.../cellmae

Both are single-seed, 50-epoch matrix U-Nets (vd+hist, simple backbone, 25/10 km
window). They are **not** the final paper champion (the 3-seed/80-epoch re-run is
still pending) — label the slide as "locked-spec model, interim". They are the
exact models behind the w=4 decision numbers, so they are the honest visual
companion to the ablation table.

Reuses the training module's preprocessing (`scripts/models/unet_matrix.py`) so
the input tensor, padding, and target slicing match training/eval bit-for-bit.

Outputs one decoupled PNG per panel (no baked-in title) ->
output/slide_materials/monthly_20260622/:
  case_<id>_observed.png        (origin_cc contour overlay)
  case_<id>_historical.png
  case_<id>_predicted_w4.png    (origin_cc contour overlay)
  case_<id>_predicted_cellmae.png
  case_<id>_diff_w4.png         (predicted_w4 - observed)

Usage
-----
  uv run scripts/visualization/build_meeting_prediction_heatmaps_20260622.py
  uv run scripts/visualization/build_meeting_prediction_heatmaps_20260622.py --accident-ids 2025012209
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.data_loaders import load_samples  # noqa: E402
from scripts.lib.windows import SampleWindow  # noqa: E402
from scripts.lib.model_architectures import WindowConfig, build_matrix_unet  # noqa: E402
from scripts.lib.congestion_mask import origin_cc_congestion_mask  # noqa: E402
from scripts.lib import visualization as viz  # noqa: E402  (CJK font on import)
from scripts.models.unet_matrix import (  # noqa: E402
    TrainConfig,
    AccidentSampleDataset,
    filter_and_subset,
    load_hist_aligned,
    time_based_split,
)

plt.rcParams["axes.unicode_minus"] = False
SPEED_CMAP = plt.get_cmap(viz.SPEED_CMAP).copy(); SPEED_CMAP.set_bad("white")
DIFF_CMAP = plt.get_cmap(viz.DIFF_CMAP).copy(); DIFF_CMAP.set_bad("white")

CKPT_W4 = "output/models/loss_window_ablation/20260601T081743Z/cells/w20_origin_cc_wt4/best_model.pth"
CKPT_CELL = "output/models/loss_window_ablation/20260601T081743Z/cells/cellmae/best_model.pth"


def _render(M, space_km, time_min, out: Path, *, kind: str = "speed", occ=None) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=150)
    extent = [float(time_min[0]), float(time_min[-1]), float(space_km[0]), float(space_km[-1])]
    if kind == "diff":
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", interpolation="nearest",
                       origin="lower", extent=extent, cmap=DIFF_CMAP,
                       vmin=-viz.DIFF_ABS_MAX, vmax=viz.DIFF_ABS_MAX)
        clabel = "預測 - 觀測 (km/h)"
    elif kind == "loss":
        vmax = max(30.0, float(np.nanpercentile(M, 98))) if np.isfinite(M).any() else 80.0
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", interpolation="nearest",
                       origin="lower", extent=extent, cmap="YlOrRd", vmin=0.0, vmax=vmax)
        clabel = "加權誤差貢獻 w×|誤差| (km/h)"
    else:
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", interpolation="nearest",
                       origin="lower", extent=extent, cmap=SPEED_CMAP,
                       vmin=viz.SPEED_MIN, vmax=viz.SPEED_MAX)
        clabel = "速度 (km/h)"
    if occ is not None and occ.any():
        ax.contour(time_min, space_km, occ.astype(float), levels=[0.5],
                   colors="#1f4e79", linewidths=2.4)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("事故後時間（分鐘）", fontsize=12)
    ax.set_ylabel("相對事故位置（km，上游為正）", fontsize=12)
    ax.tick_params(labelsize=10)
    fig.colorbar(im, ax=ax, pad=0.02, label=clabel)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def _load_model(ckpt: Path, window_cfg: WindowConfig) -> torch.nn.Module:
    model = build_matrix_unet(window_cfg, input_type="vd+hist", backbone="simple")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--output-root", default=str(PROJECT_ROOT / "output" / "slide_materials" / "monthly_20260622"))
    ap.add_argument("--accident-ids", default="2025012209,2025020054,2025010429,2025051800",
                    help="Case studies: large / medium / typical + failure (2025051800, appendix).")
    ap.add_argument("--ckpt-w4", default=str(PROJECT_ROOT / CKPT_W4))
    ap.add_argument("--ckpt-cellmae", default=str(PROJECT_ROOT / CKPT_CELL))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    data = Path(args.data_root)
    out = Path(args.output_root)
    torch.set_grad_enabled(False)

    cfg = TrainConfig(input_type="vd+hist", backbone="simple", feature_set="t0",
                      congestion_upstream_km=20, congestion_weight=4.0, eval_protocol="v1")
    target_window = SampleWindow(upstream_km=cfg.upstream_km, downstream_km=cfg.downstream_km,
                                 time_before_min=cfg.time_before_min, time_after_min=cfg.time_after_min,
                                 spatial_resolution_km=cfg.spatial_resolution_km, resolution_min=cfg.resolution_min)
    window_cfg = WindowConfig(upstream_km=cfg.upstream_km, downstream_km=cfg.downstream_km,
                              time_before_min=cfg.time_before_min, time_after_min=cfg.time_after_min,
                              spatial_resolution_km=cfg.spatial_resolution_km, resolution_min=cfg.resolution_min)

    print("Loading samples + historical baseline ...")
    samples = load_samples(data_root=data, resolution_min=5)
    matrices, ids = filter_and_subset(samples, target_window, cfg.min_before_coverage, cfg.min_filled_ratio)
    hist = load_hist_aligned(ids, target_window, data, cfg.resolution_min, "")
    train_idx, val_idx, test_idx = time_based_split(ids, data, cfg.train_end_date, cfg.val_end_date)
    split_of = {}
    for i in train_idx: split_of[str(ids[i])] = "train"
    for i in val_idx: split_of[str(ids[i])] = "val"
    for i in test_idx: split_of[str(ids[i])] = "test"

    print(f"Loading checkpoints (CPU) ...")
    model_w4 = _load_model(Path(args.ckpt_w4), window_cfg)
    model_cell = _load_model(Path(args.ckpt_cellmae), window_cfg)

    n_before = window_cfg.n_before
    space_km = target_window.space_offsets_m.astype(float) / 1000.0
    time_post = target_window.time_offsets_min[n_before:].astype(float)

    for aid in [a for a in args.accident_ids.split(",") if a]:
        where = np.where(ids == aid)[0]
        if where.size == 0:
            print(f"[skip] {aid}: not in coverage-filtered set")
            continue
        idx = int(where[0])
        obs_full, hist_full = matrices[idx], hist[idx]
        ds = AccidentSampleDataset(matrices[idx:idx + 1], ids[idx:idx + 1], cfg, hist[idx:idx + 1])
        x, y, m, _ = ds[0]
        pred_w4 = model_w4(x.unsqueeze(0))[0].numpy() * viz.SPEED_MAX
        pred_cell = model_cell(x.unsqueeze(0))[0].numpy() * viz.SPEED_MAX

        valid = m.numpy() > 0
        obs_v = y.numpy() * viz.SPEED_MAX
        obs_post = np.where(valid, obs_v, np.nan)
        hist_post = hist_full[:, n_before:]
        occ_post = origin_cc_congestion_mask(
            obs_full, hist_full, target_window.space_offsets_m, target_window.time_offsets_min
        )[:, n_before:]
        diff_w4 = np.where(valid, pred_w4 - obs_v, np.nan)

        # loss-weight map (eval_protocol_v1): downstream/beyond-20km 0, upstream non-cc 1, origin_cc w
        up = (space_km >= 0) & (space_km <= cfg.congestion_upstream_km)
        w_map = np.zeros_like(obs_v, dtype=float)
        w_map[np.broadcast_to(up[:, None], w_map.shape)] = 1.0
        w_map[up[:, None] & occ_post.astype(bool)] = float(cfg.congestion_weight)
        lossmap_w4 = np.where(valid, w_map * np.abs(pred_w4 - obs_v), np.nan)

        # per-accident origin_cc MAE on the loss zone (0..+20 km), for the outline narrative
        cc = occ_post.astype(bool) & valid & up[:, None]
        tag = split_of.get(aid, "?")
        if cc.any():
            mae_w4 = float(np.abs(pred_w4[cc] - obs_v[cc]).mean())
            mae_cell = float(np.abs(pred_cell[cc] - obs_v[cc]).mean())
            upz = valid & up[:, None]
            cell_w4 = float(np.abs(pred_w4[upz] - obs_v[upz]).mean())
            print(f"{aid} [{tag}] occ_cells={int(cc.sum())} | origin_cc MAE w4={mae_w4:.1f} cell={mae_cell:.1f} | upstream cellMAE w4={cell_w4:.1f} km/h")
        else:
            print(f"{aid} [{tag}] origin_cc empty in loss zone")

        stem = f"case_{aid}"
        _render(obs_post, space_km, time_post, out / f"{stem}_observed.png", kind="speed", occ=occ_post)
        _render(hist_post, space_km, time_post, out / f"{stem}_historical.png", kind="speed")
        _render(pred_w4, space_km, time_post, out / f"{stem}_predicted_w4.png", kind="speed", occ=occ_post)
        _render(pred_cell, space_km, time_post, out / f"{stem}_predicted_cellmae.png", kind="speed", occ=occ_post)
        _render(diff_w4, space_km, time_post, out / f"{stem}_diff_w4.png", kind="diff")
        _render(lossmap_w4, space_km, time_post, out / f"{stem}_lossmap_w4.png", kind="loss", occ=occ_post)

    print(f"\nDone. Heatmaps in {out}")


if __name__ == "__main__":
    main()
