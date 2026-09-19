# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "torch-directml",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scikit-learn",
#     "tqdm",
#     "pyarrow",
#     "tabulate",
# ]
# ///
# ruff: noqa: E402
"""Loss / upstream-window ablation for the speed-matrix U-Net (eval_protocol_v1).

Goal
----
Empirically select the upstream loss-focus window (10 / 15 / 20 km) and the
congestion-weighted loss shape (mask = origin_cc vs speed<60, congested weight
`w`), confirming or revising the 2026-05-29 decision (planning_next_steps §0/§1/§5).

Design
------
- The matrix window is FIXED for every run (upstream 25 km / downstream 10 km), so
  every model predicts the same spatial extent and is directly comparable. Only the
  *loss-focus window* (`congestion_upstream_km`) and the *loss shape* vary.
- Data is loaded ONCE; the same train/val/test DataLoaders (time-based split) are
  reused across all grid cells. Per cell we only swap the loss + reinit the model,
  so the comparison isolates the loss/window effect.
- CRITICAL: the trainer couples the loss zone and its in-training metric zone to one
  `congestion_upstream_km`. That makes per-run upstream_MAE incomparable across
  windows. So selection does NOT use the in-training metric — instead every trained
  checkpoint is re-scored by a single `unified evaluator` on a FIXED metric panel
  (eval windows {10,15,20,25} km x masks {origin_cc, speed<60}, plus cell_MAE,
  recovery_MAE@10, impact_MAE@20, F1@20) on both val and test.

Grid (per --input-type, default vd+hist)
-----------------------------------------
- cell_mae baseline (window-independent loss)                               -> 1 run
- upstream_congestion_weighted x window{10,15,20} x mask{origin_cc,speed60}
  x w{2,4,7,10}                                                             -> 24 runs
Then the top-`--top-k` cells by val origin_cc_MAE@20 get `--extra-seeds` reruns.

Outputs
-------
output/models/loss_window_ablation/<run_ts>/
  cells/<cell_id>[_s<seed>]/best_model.pth + run_config.json
  per_accident_metrics.csv      one row per (cell, seed, split, accident)
  summary.csv                   one row per (cell, seed); aggregated val+test panel
  selection.csv                 seed-aggregated (mean+/-std) for ranking
  report.md                     human-readable selection writeup
reports/loss_window_ablation_<date>.md  (copy of report.md)

Usage
-----
  uv run scripts/experiments/run_loss_window_ablation.py --smoke      # 3 cells x 2 epochs, timing
  uv run scripts/experiments/run_loss_window_ablation.py              # full grid, 50 epochs
  uv run scripts/experiments/run_loss_window_ablation.py --epochs 40 --windows 10,15,20 --weights 2,4,7,10
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, date, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the canonical training building blocks so train/eval geometry is identical
# to scripts/models/unet_matrix.py (same SimpleUNet, losses, split, helpers).
from scripts.models.unet_matrix import ( # noqa: E402
    CONGESTION_KMH_DEFAULT,
    FREE_FLOW_KMH,
    SPEED_MAX,
    AccidentSampleDataset,
    SimpleUNet,
    TrainConfig,
    _impact_time,
    _recovery_time,
    _upstream_end_idx,
    _upstream_start_idx,
    build_loss,
    filter_and_subset,
    load_congestion_masks_aligned,
    load_hist_aligned,
    select_device,
    time_based_split,
)
from scripts.lib.data_loaders import load_samples  # noqa: E402
from scripts.lib.windows import SampleWindow  # noqa: E402

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "models" / "loss_window_ablation"
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_MASK_NPZ = DATA_ROOT / "congestion_masks_origin_cc_5m.npz"

# Fixed matrix window for ALL runs (research default; NPZ stores 50 km upstream).
FIXED_UPSTREAM_KM = 25
FIXED_DOWNSTREAM_KM = 10
FIXED_TIME_BEFORE = 30
FIXED_TIME_AFTER = 145
SPATIAL_RES_KM = 1
RES_MIN = 5

# Eval panel windows (km upstream) re-scored on every checkpoint.
EVAL_WINDOWS_KM = (10, 15, 20, 25)
# Headline operational selection metric: origin_cc upstream MAE on the locked 20 km zone.
PRIMARY_EVAL_WINDOW_KM = 20


# ────────────────────────────────────────────────────────────────────────────
# Grid definition
# ────────────────────────────────────────────────────────────────────────────


def build_grid(windows: list[int], weights: list[float], masks: list[str]) -> list[dict]:
    """Return the list of grid-cell specs (excluding seed reruns)."""
    cells: list[dict] = [{
        "cell_id": "cellmae",
        "loss_name": "cell_mae",
        "congestion_upstream_km": PRIMARY_EVAL_WINDOW_KM,  # only affects in-training metric, not loss
        "congestion_weight": 1.0,
        "mask": "none",
    }]
    for win in windows:
        for mask in masks:
            for w in weights:
                cells.append({
                    "cell_id": f"w{win}_{mask}_wt{int(w) if float(w).is_integer() else w}",
                    "loss_name": "upstream_congestion_weighted",
                    "congestion_upstream_km": win,
                    "congestion_weight": float(w),
                    "mask": mask,  # "origin_cc" | "speed60"
                })
    return cells


# ────────────────────────────────────────────────────────────────────────────
# Training (one grid cell)
# ───────────────────────────────────────────────────────────────────────────


def make_cell_cfg(base: TrainConfig, spec: dict, epochs: int, seed: int,
                  mask_npz: str) -> TrainConfig:
    use_mask = spec["mask"] == "origin_cc"
    return replace(
        base,
        epochs=epochs,
        random_state=seed,
        loss_name=spec["loss_name"],
        congestion_upstream_km=int(spec["congestion_upstream_km"]),
        congestion_weight=float(spec["congestion_weight"]),
        congestion_mask_npz=mask_npz if use_mask else "",
    )


def train_cell(
    cfg: TrainConfig,
    spec: dict,
    seed: int,
    device: torch.device,
    train_ds: AccidentSampleDataset,
    val_ds: AccidentSampleDataset,
    cong_by_id: dict[str, np.ndarray],
    ckpt_path: Path,
) -> dict:
    """Train one grid cell; checkpoint best-val-loss model. Returns small history dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = SimpleUNet(cfg.n_space, cfg.n_time, cfg.n_before, cfg.pad_h, cfg.pad_w,
                       in_channels=cfg.in_channels).to(device)
    criterion = build_loss(cfg.loss_name, cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    use_mask = spec["mask"] == "origin_cc"

    def _cmask(ids_batch):
        if not use_mask:
            return None
        stacked = np.stack([cong_by_id[str(a)] for a in ids_batch], axis=0)
        return torch.from_numpy(stacked).to(device)

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tl = 0.0
        for x, y, m, ids_b in train_loader:
            x = x.to(device); y = y.to(device); m = m.to(device)
            cmask = _cmask(ids_b)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y, m, cmask)
            loss.backward()
            optimizer.step()
            tl += loss.item()
        tl /= max(len(train_loader), 1)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y, m, ids_b in val_loader:
                x = x.to(device); y = y.to(device); m = m.to(device)
                cmask = _cmask(ids_b)
                vl += criterion(model(x), y, m, cmask).item()
        vl /= max(len(val_loader), 1)

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        if vl < best_val:
            best_val = vl
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt_path)
        scheduler.step()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"best_val_loss": best_val, **history}


# ────────────────────────────────────────────────────────────────────────────
# Unified evaluator (fixed metric panel, re-scored on every checkpoint)
# ────────────────────────────────────────────────────────────────────────────


def _masked_mae(pred: np.ndarray, target: np.ndarray, sel: np.ndarray) -> float:
    if sel.sum() == 0:
        return float("nan")
    return float(np.abs(pred[sel] - target[sel]).mean())


def _f1(pred_kmh: np.ndarray, target_kmh: np.ndarray, valid: np.ndarray,
        thresh: float = CONGESTION_KMH_DEFAULT) -> float:
    v = valid > 0
    if v.sum() == 0:
        return float("nan")
    pc = (pred_kmh < thresh) & v
    tc = (target_kmh < thresh) & v
    if not tc.any():
        return float("nan")
    tp = float((pc & tc).sum()); fp = float((pc & ~tc & v).sum()); fn = float((~pc & tc).sum())
    prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
    return float(2 * prec * rec / (prec + rec + 1e-8))


def panel_for_accident(
    pred_kmh: np.ndarray,      # (n_space, n_after)
    target_kmh: np.ndarray,
    valid: np.ndarray,         # 0/1
    cc_mask: np.ndarray,       # 0/1 origin_cc, post cols, same shape
    up_start: int,
) -> dict:
    """Fixed metric panel for one accident; NaN where a selection set is empty."""
    out: dict[str, float] = {}
    v = valid > 0
    cc = (cc_mask > 0) & v
    s60 = (target_kmh < CONGESTION_KMH_DEFAULT) & v

    for W in EVAL_WINDOWS_KM:
        e = up_start + W // SPATIAL_RES_KM + 1
        zsl = slice(up_start, e)
        p = pred_kmh[zsl]; t = target_kmh[zsl]
        out[f"origin_cc_MAE@{W}"] = _masked_mae(p, t, cc[zsl])
        out[f"speed60_MAE@{W}"] = _masked_mae(p, t, s60[zsl])
        out[f"upstream_MAE@{W}"] = _masked_mae(p, t, v[zsl])

    # window-independent diagnostics
    out["cell_MAE"] = _masked_mae(pred_kmh, target_kmh, v)
    out["congestion_F1@20"] = _f1(pred_kmh[slice(up_start, up_start + 20 + 1)],
                                  target_kmh[slice(up_start, up_start + 20 + 1)],
                                  valid[slice(up_start, up_start + 20 + 1)])

    # operational time metrics
    up_end_20 = up_start + 20 // SPATIAL_RES_KM + 1
    it_obs = _impact_time(target_kmh, up_start, RES_MIN, valid, FREE_FLOW_KMH, up_end_20)
    it_pred = _impact_time(pred_kmh, up_start, RES_MIN, valid, FREE_FLOW_KMH, up_end_20)
    out["impact_time_AE@20"] = abs(it_pred - it_obs)
    rt_obs = _recovery_time(target_kmh, up_start, RES_MIN, valid, FREE_FLOW_KMH)
    rt_pred = _recovery_time(pred_kmh, up_start, RES_MIN, valid, FREE_FLOW_KMH)
    out["recovery_AE@10"] = abs(rt_pred - rt_obs)
    out["has_origin_cc"] = bool(cc.any())
    out["has_speed60"] = bool(s60.any())
    return out


def evaluate_checkpoint(
    ckpt_path: Path,
    cfg: TrainConfig,
    device: torch.device,
    loaders: dict[str, DataLoader | None],
    cong_by_id: dict[str, np.ndarray],
    up_start: int,
) -> tuple[list[dict], dict]:
    """Load best model and re-score the fixed panel on every split. Returns
    (per_accident_rows, aggregated_dict)."""
    model = SimpleUNet(cfg.n_space, cfg.n_time, cfg.n_before, cfg.pad_h, cfg.pad_w,
                       in_channels=cfg.in_channels).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()

    rows: list[dict] = []
    agg: dict[str, float] = {}
    with torch.no_grad():
        for split, loader in loaders.items():
            if loader is None:
                continue
            split_rows: list[dict] = []
            for x, y, m, ids_b in loader:
                x = x.to(device)
                pred = model(x).cpu().numpy() * SPEED_MAX
                tgt = y.numpy() * SPEED_MAX
                val = m.numpy()
                for i, aid in enumerate(ids_b):
                    cc = cong_by_id.get(str(aid))
                    if cc is None:
                        cc = np.zeros_like(val[i], dtype=np.uint8)
                    pm = panel_for_accident(pred[i], tgt[i], val[i], cc, up_start)
                    pm["acc_id"] = str(aid); pm["split"] = split
                    split_rows.append(pm)
            rows.extend(split_rows)
            # macro aggregate (nanmean across accidents) for this split
            df = pd.DataFrame(split_rows)
            for col in df.columns:
                if col in ("acc_id", "split", "has_origin_cc", "has_speed60"):
                    continue
                agg[f"{split}/{col}"] = float(np.nanmean(df[col].to_numpy(dtype=float)))
            agg[f"{split}/n"] = int(len(split_rows))
            agg[f"{split}/n_origin_cc"] = int(df["has_origin_cc"].sum())
    return rows, agg


# ────────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--run-ts", default=None, help="Override run timestamp dir name.")
    p.add_argument("--input-type", default="vd+hist", choices=["vd", "hist", "vd+hist"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--windows", default="10,15,20", help="loss-focus upstream windows (km)")
    p.add_argument("--weights", default="2,4,7,10", help="upstream congested weights w")
    p.add_argument("--masks", default="origin_cc,speed60", help="congested-cell mask variants")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--extra-seeds", default="43,44", help="extra seeds for top-k cells")
    p.add_argument("--top-k", type=int, default=3, help="# top cells (by val origin_cc_MAE@20) to reseed")
    p.add_argument("--min-filled-ratio", type=float, default=0.8)
    p.add_argument("--mask-npz", default=str(DEFAULT_MASK_NPZ))
    p.add_argument("--hist-tag", default="")
    p.add_argument("--smoke", action="store_true",
                   help="Quick end-to-end: windows=20, masks=origin_cc, weights=4, +cellmae, 2 epochs, no reseed.")
    p.add_argument("--skip-reseed", action="store_true", help="Run the base grid only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_ts = args.run_ts or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_root) / run_ts
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        windows = [20]; weights = [4.0]; masks = ["origin_cc"]; epochs = 2
        extra_seeds: list[int] = []
    else:
        windows = [int(x) for x in args.windows.split(",") if x.strip()]
        weights = [float(x) for x in args.weights.split(",") if x.strip()]
        masks = [x.strip() for x in args.masks.split(",") if x.strip()]
        epochs = args.epochs
        extra_seeds = [] if args.skip_reseed else [int(x) for x in args.extra_seeds.split(",") if x.strip()]

    device, device_msg = select_device()
    print(f"Device: {device_msg}")
    print(f"Run dir: {out_dir}")

    # ── Load data ONCE ──────────────────────────────────────────────────────
    base_cfg = TrainConfig(
        upstream_km=FIXED_UPSTREAM_KM, downstream_km=FIXED_DOWNSTREAM_KM,
        time_before_min=FIXED_TIME_BEFORE, time_after_min=FIXED_TIME_AFTER,
        spatial_resolution_km=SPATIAL_RES_KM, resolution_min=RES_MIN,
        epochs=epochs, batch_size=args.batch_size, lr=args.lr,
        min_filled_ratio=args.min_filled_ratio,
        eval_protocol="v1", data_scope="dispatch_only", feature_set="t0",
        input_type=args.input_type, loss_name="upstream_congestion_weighted",
    )
    target_window = SampleWindow(
        upstream_km=base_cfg.upstream_km, downstream_km=base_cfg.downstream_km,
        time_before_min=base_cfg.time_before_min, time_after_min=base_cfg.time_after_min,
        spatial_resolution_km=base_cfg.spatial_resolution_km, resolution_min=base_cfg.resolution_min,
    )
    samples = load_samples(data_root=DATA_ROOT, resolution_min=RES_MIN)
    matrices, ids = filter_and_subset(samples, target_window, 0.0, base_cfg.min_filled_ratio)
    print(f"Samples after filled_ratio>={base_cfg.min_filled_ratio}: {len(matrices)}/{len(samples)}")

    hist = None
    if "hist" in base_cfg.input_type:
        hist = load_hist_aligned(ids, target_window, DATA_ROOT, RES_MIN, args.hist_tag)
    cong_aligned = load_congestion_masks_aligned(ids, target_window, Path(args.mask_npz))
    cong_by_id = {str(a): cong_aligned[i] for i, a in enumerate(ids)}
    print(f"origin_cc congested-cell share (all): {cong_aligned.mean():.4%} | "
          f"input_type={base_cfg.input_type} in_ch={base_cfg.in_channels}")

    train_idx, val_idx, test_idx = time_based_split(ids, DATA_ROOT, base_cfg.train_end_date, base_cfg.val_end_date)
    print(f"Time-based split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    def _h(idx):
        return hist[idx] if hist is not None else None

    train_ds = AccidentSampleDataset(matrices[train_idx], ids[train_idx], base_cfg, _h(train_idx))
    val_ds = AccidentSampleDataset(matrices[val_idx], ids[val_idx], base_cfg, _h(val_idx))
    test_ds = (AccidentSampleDataset(matrices[test_idx], ids[test_idx], base_cfg, _h(test_idx))
               if len(test_idx) else None)
    val_loader = DataLoader(val_ds, batch_size=base_cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = (DataLoader(test_ds, batch_size=base_cfg.batch_size, shuffle=False, num_workers=0)
                   if test_ds else None)
    eval_loaders = {"val": val_loader, "test": test_loader}
    up_start = _upstream_start_idx(base_cfg)

    grid = build_grid(windows, weights, masks)
    print(f"\nGrid cells: {len(grid)} | epochs={epochs} | base_seed={args.base_seed} | "
          f"extra_seeds(top-{args.top_k})={extra_seeds}\n")

    # ── Phase 1: base grid (one seed each) ───────────────────────────────────
    all_per_acc: list[dict] = []
    summary_rows: list[dict] = []

    def run_one(spec: dict, seed: int) -> dict:
        cell_id = spec["cell_id"] + ("" if seed == args.base_seed else f"_s{seed}")
        cell_dir = cells_dir / cell_id
        ckpt = cell_dir / "best_model.pth"
        cfg = make_cell_cfg(base_cfg, spec, epochs, seed, args.mask_npz)
        t0 = time.time()
        hist_dict = train_cell(cfg, spec, seed, device, train_ds, val_ds, cong_by_id, ckpt)
        per_acc, agg = evaluate_checkpoint(ckpt, cfg, device, eval_loaders, cong_by_id, up_start)
        secs = round(time.time() - t0, 1)
        for r in per_acc:
            r["cell_id"] = spec["cell_id"]; r["seed"] = seed
        all_per_acc.extend(per_acc)
        row = {
            "cell_id": spec["cell_id"], "seed": seed,
            "loss_name": spec["loss_name"], "mask": spec["mask"],
            "loss_window_km": spec["congestion_upstream_km"] if spec["loss_name"] != "cell_mae" else np.nan,
            "weight": spec["congestion_weight"] if spec["loss_name"] != "cell_mae" else np.nan,
            "best_val_loss": hist_dict["best_val_loss"], "train_sec": secs, **agg,
        }
        summary_rows.append(row)
        cell_dir.mkdir(parents=True, exist_ok=True)
        cfg_dump = asdict(cfg); cfg_dump["cell_id"] = cell_id; cfg_dump["train_sec"] = secs
        (cell_dir / "run_config.json").write_text(json.dumps(cfg_dump, indent=2), encoding="utf-8")
        primary = agg.get(f"val/origin_cc_MAE@{PRIMARY_EVAL_WINDOW_KM}", float("nan"))
        print(f"  [{cell_id}] {secs}s | best_val_loss={hist_dict['best_val_loss']:.4f} | "
              f"val origin_cc_MAE@{PRIMARY_EVAL_WINDOW_KM}={primary:.3f} | "
              f"val cell_MAE={agg.get('val/cell_MAE', float('nan')):.3f}")
        # incremental persistence so a crash mid-grid keeps partial results
        pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
        return row

    print("Phase 1 — base grid")
    for spec in tqdm(grid, desc="grid"):
        run_one(spec, args.base_seed)

    # ── Phase 2: reseed top-k by val origin_cc_MAE@20 ────────────────────────
    primary_key = f"val/origin_cc_MAE@{PRIMARY_EVAL_WINDOW_KM}"
    base_df = pd.DataFrame([r for r in summary_rows if r["seed"] == args.base_seed])
    if extra_seeds and not base_df.empty:
        ranked = base_df.dropna(subset=[primary_key]).sort_values(primary_key)
        top_ids = ranked.head(args.top_k)["cell_id"].tolist()
        print(f"\nPhase 2 — reseed top-{args.top_k}: {top_ids}")
        spec_by_id = {s["cell_id"]: s for s in grid}
        for cid in top_ids:
            for seed in extra_seeds:
                run_one(spec_by_id[cid], seed)

    # ── Persist + report ─────────────────────────────────────────────────────
    pd.DataFrame(all_per_acc).to_csv(out_dir / "per_accident_metrics.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    selection_df = build_selection_table(summary_df, primary_key)
    selection_df.to_csv(out_dir / "selection.csv", index=False)

    report = build_report(summary_df, selection_df, primary_key, base_cfg, windows, weights, masks,
                          epochs, device_msg, run_ts)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    report_copy = PROJECT_ROOT / "reports" / f"loss_window_ablation_{date.today():%Y%m%d}.md"
    shutil.copyfile(out_dir / "report.md", report_copy)
    print(f"\nDone. Summary: {out_dir/'summary.csv'}\nReport: {report_copy}")


def build_selection_table(summary_df: pd.DataFrame, primary_key: str) -> pd.DataFrame:
    """Seed-aggregate (mean+/-std) the key panel columns per cell."""
    keep = [primary_key, f"test/origin_cc_MAE@{PRIMARY_EVAL_WINDOW_KM}",
            "val/cell_MAE", "test/cell_MAE",
            f"val/upstream_MAE@{PRIMARY_EVAL_WINDOW_KM}",
            f"val/recovery_AE@10", f"val/impact_time_AE@20", f"val/congestion_F1@20"]
    keep = [k for k in keep if k in summary_df.columns]
    g = summary_df.groupby(["cell_id", "loss_name", "mask", "loss_window_km", "weight"], dropna=False)
    rows = []
    for key, sub in g:
        row = dict(zip(["cell_id", "loss_name", "mask", "loss_window_km", "weight"], key))
        row["n_seeds"] = len(sub)
        for k in keep:
            row[f"{k}_mean"] = float(np.nanmean(sub[k]))
            row[f"{k}_std"] = float(np.nanstd(sub[k]))
        rows.append(row)
    out = pd.DataFrame(rows)
    if f"{primary_key}_mean" in out.columns:
        out = out.sort_values(f"{primary_key}_mean")
    return out


def build_report(summary_df, selection_df, primary_key, base_cfg, windows, weights, masks,
                 epochs, device_msg, run_ts) -> str:
    L: list[str] = []
    L.append(f"# Loss / Upstream-Window Ablation — {date.today().isoformat()}\n")
    L.append(f"Run: `{run_ts}` | device: {device_msg} | epochs: {epochs} | "
             f"input_type: `{base_cfg.input_type}` | eval_protocol: v1 | data_scope: dispatch_only | feature_set: t0\n")
    L.append("## Design\n")
    L.append("- Matrix window FIXED for all runs: upstream 25 km / downstream 10 km "
             "(only the loss-focus window varies). NPZ lock (50 km) untouched.")
    L.append(f"- Loss-focus windows: {windows} km | masks: {masks} | weights w: {weights} | + `cell_mae` baseline.")
    L.append(f"- Headline selection metric: **origin_cc upstream MAE @ {PRIMARY_EVAL_WINDOW_KM} km** "
             "(km/h, lower=better), re-scored on a FIXED panel decoupled from each run's loss zone.")
    L.append("- Guards: `cell_MAE` (free-flow / full-matrix fidelity), `congestion_F1@20`, "
             "`recovery_AE@10` (min), `impact_time_AE@20` (min).")
    L.append("")

    # Selection ranking (seed-aggregated)
    L.append("## Selection ranking (seed-aggregated, by val origin_cc_MAE@20)\n")
    pk = f"{primary_key}_mean"
    cols = ["cell_id", "loss_window_km", "mask", "weight", "n_seeds", pk,
            f"test/origin_cc_MAE@{PRIMARY_EVAL_WINDOW_KM}_mean", "val/cell_MAE_mean",
            "val/recovery_AE@10_mean", "val/congestion_F1@20_mean"]
    cols = [c for c in cols if c in selection_df.columns]
    L.append(selection_df[cols].round(3).to_markdown(index=False))
    L.append("")

    # train-window x eval-window matrix for origin_cc mask (base seed only)
    L.append("## Train-window × eval-window matrix — origin_cc_MAE (val, base seed)\n")
    L.append("Rows = loss-focus window the model was trained with; columns = eval window the metric is scored on. "
             "Reveals whether a narrow loss window helps near-field at the cost of the far-field (or vice versa). "
             "Weight w is held at the per-window best.\n")
    base = summary_df[summary_df["seed"] == base_cfg.random_state]
    occ = base[(base["mask"] == "origin_cc")]
    if not occ.empty:
        # for each loss window, pick the weight with the best val origin_cc_MAE@20
        best_rows = []
        for win in sorted(occ["loss_window_km"].dropna().unique()):
            sub = occ[occ["loss_window_km"] == win]
            best = sub.loc[sub[primary_key].idxmin()]
            rec = {"train_window_km": int(win), "best_w": best["weight"]}
            for W in EVAL_WINDOWS_KM:
                rec[f"eval@{W}"] = round(float(best.get(f"val/origin_cc_MAE@{W}", float("nan"))), 3)
            best_rows.append(rec)
        L.append(pd.DataFrame(best_rows).to_markdown(index=False))
        L.append("")

    # weight sweep at the primary window
    L.append(f"## Weight sweep @ loss-window {PRIMARY_EVAL_WINDOW_KM} km (val, base seed)\n")
    wsub = base[(base["loss_window_km"] == PRIMARY_EVAL_WINDOW_KM)]
    if not wsub.empty:
        wt = wsub.pivot_table(index="weight", columns="mask",
                              values=f"val/origin_cc_MAE@{PRIMARY_EVAL_WINDOW_KM}")
        L.append(wt.round(3).to_markdown())
        L.append("")

    L.append("## Notes\n")
    L.append("- `cell_mae` baseline weights the full matrix (downstream included), loss window N/A; "
             "it anchors how much the congestion weighting trades cell-wide fidelity for upstream focus.")
    L.append("- Selection should pick the (window, mask, w) with the best val origin_cc_MAE@20 whose "
             "advantage holds on test and does not blow up cell_MAE or recovery_AE@10.")
    L.append("- Full per-cell panel: `summary.csv`; per-accident detail: `per_accident_metrics.csv`.")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
