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
"""Speed + true-pace hybrid loss ablation for the matrix U-Net.

This experiment isolates the proposed loss:

    L = lambda_speed * L_speed + lambda_pace * L_pace

where L_pace is true pace MAE, p(v)=60/max(v, v_min), over the locked upstream
0~+20 km zone. The evaluator is fixed across all cells and includes the existing
eval_protocol_v1 panel plus pace diagnostics.

Usage:
  uv run scripts/experiments/run_speed_pace_ablation.py --smoke
  uv run scripts/experiments/run_speed_pace_ablation.py --epochs 30
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
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

from scripts.lib.data_loaders import load_samples  # noqa: E402
from scripts.lib.windows import SampleWindow  # noqa: E402
from scripts.models.unet_matrix import (  # noqa: E402
    CONGESTION_KMH_DEFAULT,
    FREE_FLOW_KMH,
    SPEED_MAX,
    AccidentSampleDataset,
    SimpleUNet,
    TrainConfig,
    _impact_time,
    _recovery_time,
    _upstream_start_idx,
    build_loss,
    filter_and_subset,
    load_congestion_masks_aligned,
    load_hist_aligned,
    select_device,
    time_based_split,
)


DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "models" / "speed_pace_ablation"
DEFAULT_MASK_NPZ = DATA_ROOT / "congestion_masks_origin_cc_5m.npz"

FIXED_UPSTREAM_KM = 25
FIXED_DOWNSTREAM_KM = 10
FIXED_TIME_BEFORE = 30
FIXED_TIME_AFTER = 145
SPATIAL_RES_KM = 1
RES_MIN = 5
PRIMARY_WINDOW_KM = 20
EVAL_WINDOWS_KM = (10, 15, 20, 25)
DEFAULT_EVAL_PACE_V_MIN = 5.0


def _cell_float(x: float) -> str:
    return str(x).replace(".", "p").replace("-", "m")


def build_grid(pace_lambdas: list[float], vmins: list[float], include_locked: bool = True) -> list[dict]:
    cells: list[dict] = [{
        "cell_id": "cellmae",
        "loss_name": "cell_mae",
        "mask": "none",
        "congestion_weight": 1.0,
        "speed_lambda": 1.0,
        "pace_lambda": 0.0,
        "pace_v_min_kmh": DEFAULT_EVAL_PACE_V_MIN,
        "family": "baseline",
    }]
    if include_locked:
        cells.append({
            "cell_id": "locked_origin_cc_w4",
            "loss_name": "upstream_congestion_weighted",
            "mask": "origin_cc",
            "congestion_weight": 4.0,
            "speed_lambda": 1.0,
            "pace_lambda": 0.0,
            "pace_v_min_kmh": DEFAULT_EVAL_PACE_V_MIN,
            "family": "locked",
        })
    for lam in pace_lambdas:
        cells.append({
            "cell_id": f"pace_v5_lam{_cell_float(lam)}",
            "loss_name": "speed_pace_hybrid",
            "mask": "none",
            "congestion_weight": 1.0,
            "speed_lambda": 1.0,
            "pace_lambda": float(lam),
            "pace_v_min_kmh": 5.0,
            "family": "lambda_sweep",
        })
    for vmin in vmins:
        if abs(vmin - 5.0) < 1e-9:
            continue
        cells.append({
            "cell_id": f"pace_v{_cell_float(vmin)}_lam1",
            "loss_name": "speed_pace_hybrid",
            "mask": "none",
            "congestion_weight": 1.0,
            "speed_lambda": 1.0,
            "pace_lambda": 1.0,
            "pace_v_min_kmh": float(vmin),
            "family": "vmin_sweep",
        })
    return cells


def make_cfg(base: TrainConfig, spec: dict, epochs: int, seed: int, mask_npz: str) -> TrainConfig:
    return replace(
        base,
        epochs=epochs,
        random_state=seed,
        loss_name=spec["loss_name"],
        congestion_upstream_km=PRIMARY_WINDOW_KM,
        congestion_weight=float(spec["congestion_weight"]),
        congestion_mask_npz=mask_npz if spec["mask"] == "origin_cc" else "",
        speed_loss_lambda=float(spec["speed_lambda"]),
        pace_loss_lambda=float(spec["pace_lambda"]),
        pace_v_min_kmh=float(spec["pace_v_min_kmh"]),
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = SimpleUNet(cfg.n_space, cfg.n_time, cfg.n_before, cfg.pad_h, cfg.pad_w, in_channels=cfg.in_channels).to(device)
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
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            cmask = _cmask(ids_b)
            optimizer.zero_grad()
            loss = criterion(model(x), y, m, cmask)
            loss.backward()
            optimizer.step()
            tl += loss.item()
        tl /= max(len(train_loader), 1)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y, m, ids_b in val_loader:
                x = x.to(device)
                y = y.to(device)
                m = m.to(device)
                vl += criterion(model(x), y, m, _cmask(ids_b)).item()
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


def _masked_mae(pred: np.ndarray, target: np.ndarray, sel: np.ndarray) -> float:
    if sel.sum() == 0:
        return float("nan")
    return float(np.abs(pred[sel] - target[sel]).mean())


def _masked_pace_mae(pred: np.ndarray, target: np.ndarray, sel: np.ndarray, v_min: float) -> float:
    if sel.sum() == 0:
        return float("nan")
    pred_pace = 60.0 / np.maximum(pred, v_min)
    target_pace = 60.0 / np.maximum(target, v_min)
    return float(np.abs(pred_pace[sel] - target_pace[sel]).mean())


def _f1(pred_kmh: np.ndarray, target_kmh: np.ndarray, valid: np.ndarray, thresh: float = CONGESTION_KMH_DEFAULT) -> float:
    v = valid > 0
    if v.sum() == 0:
        return float("nan")
    pc = (pred_kmh < thresh) & v
    tc = (target_kmh < thresh) & v
    if not tc.any():
        return float("nan")
    tp = float((pc & tc).sum())
    fp = float((pc & ~tc & v).sum())
    fn = float((~pc & tc).sum())
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return float(2 * prec * rec / (prec + rec + 1e-8))


def panel_for_accident(
    pred_kmh: np.ndarray,
    target_kmh: np.ndarray,
    valid: np.ndarray,
    cc_mask: np.ndarray,
    up_start: int,
    eval_pace_v_min: float,
) -> dict:
    out: dict[str, float | bool] = {}
    v = valid > 0
    cc = (cc_mask > 0) & v
    s60 = (target_kmh < CONGESTION_KMH_DEFAULT) & v
    for win in EVAL_WINDOWS_KM:
        e = up_start + win // SPATIAL_RES_KM + 1
        z = slice(up_start, e)
        out[f"origin_cc_MAE@{win}"] = _masked_mae(pred_kmh[z], target_kmh[z], cc[z])
        out[f"speed60_MAE@{win}"] = _masked_mae(pred_kmh[z], target_kmh[z], s60[z])
        out[f"upstream_MAE@{win}"] = _masked_mae(pred_kmh[z], target_kmh[z], v[z])
        out[f"pace_MAE@{win}"] = _masked_pace_mae(pred_kmh[z], target_kmh[z], v[z], eval_pace_v_min)
        out[f"origin_cc_pace_MAE@{win}"] = _masked_pace_mae(pred_kmh[z], target_kmh[z], cc[z], eval_pace_v_min)
    out["cell_MAE"] = _masked_mae(pred_kmh, target_kmh, v)
    out["congestion_F1@20"] = _f1(
        pred_kmh[slice(up_start, up_start + PRIMARY_WINDOW_KM + 1)],
        target_kmh[slice(up_start, up_start + PRIMARY_WINDOW_KM + 1)],
        valid[slice(up_start, up_start + PRIMARY_WINDOW_KM + 1)],
    )
    up_end_20 = up_start + PRIMARY_WINDOW_KM // SPATIAL_RES_KM + 1
    it_obs = _impact_time(target_kmh, up_start, RES_MIN, valid, FREE_FLOW_KMH, up_end_20)
    it_pred = _impact_time(pred_kmh, up_start, RES_MIN, valid, FREE_FLOW_KMH, up_end_20)
    rt_obs = _recovery_time(target_kmh, up_start, RES_MIN, valid, FREE_FLOW_KMH)
    rt_pred = _recovery_time(pred_kmh, up_start, RES_MIN, valid, FREE_FLOW_KMH)
    out["impact_time_AE@20"] = abs(it_pred - it_obs)
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
    eval_pace_v_min: float,
) -> tuple[list[dict], dict]:
    model = SimpleUNet(cfg.n_space, cfg.n_time, cfg.n_before, cfg.pad_h, cfg.pad_w, in_channels=cfg.in_channels).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    up_start = _upstream_start_idx(cfg)
    rows: list[dict] = []
    agg: dict[str, float | int] = {}
    with torch.no_grad():
        for split, loader in loaders.items():
            if loader is None:
                continue
            split_rows: list[dict] = []
            for x, y, m, ids_b in loader:
                pred = model(x.to(device)).cpu().numpy() * SPEED_MAX
                tgt = y.numpy() * SPEED_MAX
                val = m.numpy()
                for i, aid in enumerate(ids_b):
                    cc = cong_by_id.get(str(aid), np.zeros_like(val[i], dtype=np.uint8))
                    rec = panel_for_accident(pred[i], tgt[i], val[i], cc, up_start, eval_pace_v_min)
                    rec["acc_id"] = str(aid)
                    rec["split"] = split
                    split_rows.append(rec)
            rows.extend(split_rows)
            df = pd.DataFrame(split_rows)
            for col in df.columns:
                if col in ("acc_id", "split", "has_origin_cc", "has_speed60"):
                    continue
                agg[f"{split}/{col}"] = float(np.nanmean(df[col].to_numpy(dtype=float)))
            agg[f"{split}/n"] = int(len(split_rows))
            agg[f"{split}/n_origin_cc"] = int(df["has_origin_cc"].sum())
    return rows, agg


def _composite(selection: pd.DataFrame, baseline_id: str) -> pd.DataFrame:
    metrics = {
        "origin": "val/origin_cc_MAE@20_mean",
        "cell": "val/cell_MAE_mean",
        "impact": "val/impact_time_AE@20_mean",
        "recovery": "val/recovery_AE@10_mean",
    }
    weights = {"origin": 0.50, "cell": 0.15, "impact": 0.20, "recovery": 0.15}
    out = selection.copy()
    base = out[out["cell_id"] == baseline_id]
    if base.empty:
        return out
    base = base.iloc[0]
    out["composite_vs_locked"] = 0.0
    for key, col in metrics.items():
        denom = float(base[col])
        out[f"ratio_{key}"] = out[col] / denom if denom and np.isfinite(denom) else np.nan
        out["composite_vs_locked"] += weights[key] * out[f"ratio_{key}"]
    return out


def seed_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in summary.columns if c.startswith(("val/", "test/")) or c == "best_val_loss"]
    keys = [
        "cell_id",
        "family",
        "loss_name",
        "mask",
        "speed_lambda",
        "pace_lambda",
        "pace_v_min_kmh",
        "congestion_weight",
    ]
    agg = {c: ["mean", "std"] for c in metric_cols}
    agg["seed"] = "count"
    out = summary.groupby(keys, dropna=False).agg(agg)
    out.columns = [f"{a}_{b}" if b else a for a, b in out.columns]
    out = out.rename(columns={"seed_count": "n_seeds"}).reset_index()
    return _composite(out, "locked_origin_cc_w4")


def _fmt(value: object, ndigits: int = 3) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "nan"
        return f"{float(value):.{ndigits}f}"
    return str(value)


def build_report(
    selection: pd.DataFrame,
    summary: pd.DataFrame,
    run_ts: str,
    epochs: int,
    device_msg: str,
    pace_lambdas: list[float],
    vmins: list[float],
    eval_pace_v_min: float,
) -> str:
    lines: list[str] = []
    lines.append(f"# Speed + True-Pace Loss Ablation — {date.today():%Y-%m-%d}\n")
    lines.append(f"Run: `{run_ts}` | device: {device_msg} | epochs: {epochs} | input=`vd+hist` | backbone=`simple`\n")
    lines.append("## Design\n")
    lines.append("- Matrix window fixed: upstream 25 km / downstream 10 km, `-30..+145 min` @ 5 min.")
    lines.append("- Evaluation fixed: eval_protocol v1, origin_cc mask, primary window 0~+20 km.")
    lines.append(f"- True pace definition: `pace(v)=60/max(v,{eval_pace_v_min:g})` min/km for reporting.")
    lines.append(f"- Lambda sweep: `lambda_pace={pace_lambdas}` with `v_min=5`; v_min sensitivity: `{vmins}` with `lambda_pace=1`.")
    lines.append("- Existing locked baseline is `upstream_congestion_weighted`, origin_cc mask, w=4.\n")

    rank = selection.sort_values("val/origin_cc_MAE@20_mean")
    cols = [
        "cell_id",
        "family",
        "n_seeds",
        "val/origin_cc_MAE@20_mean",
        "test/origin_cc_MAE@20_mean",
        "val/cell_MAE_mean",
        "val/impact_time_AE@20_mean",
        "val/recovery_AE@10_mean",
        "val/origin_cc_pace_MAE@20_mean",
        "val/congestion_F1@20_mean",
        "composite_vs_locked",
    ]
    lines.append("## Ranking by val origin_cc_MAE@20\n")
    lines.append(rank[cols].round(3).to_markdown(index=False))
    lines.append("")

    comp = selection.sort_values("composite_vs_locked")
    lines.append("## Ranking by locked-baseline composite\n")
    lines.append("Composite uses origin_cc_MAE@20 / cell_MAE / impact_AE@20 / recovery_AE@10 with weights 0.50 / 0.15 / 0.20 / 0.15, relative to `locked_origin_cc_w4` (=1.000).\n")
    lines.append(comp[cols].round(3).to_markdown(index=False))
    lines.append("")

    lambda_rows = selection[selection["family"] == "lambda_sweep"].sort_values("pace_lambda")
    if not lambda_rows.empty:
        lines.append("## Lambda sweep (`v_min=5`)\n")
        lcols = [
            "cell_id",
            "pace_lambda",
            "val/origin_cc_MAE@20_mean",
            "test/origin_cc_MAE@20_mean",
            "val/cell_MAE_mean",
            "val/origin_cc_pace_MAE@20_mean",
            "val/impact_time_AE@20_mean",
            "val/recovery_AE@10_mean",
            "val/congestion_F1@20_mean",
        ]
        lines.append(lambda_rows[lcols].round(3).to_markdown(index=False))
        lines.append("")

    vmin_rows = selection[selection["family"] == "vmin_sweep"].sort_values("pace_v_min_kmh")
    if not vmin_rows.empty:
        lines.append("## v_min sensitivity (`lambda_pace=1`)\n")
        vcols = [
            "cell_id",
            "pace_v_min_kmh",
            "val/origin_cc_MAE@20_mean",
            "test/origin_cc_MAE@20_mean",
            "val/cell_MAE_mean",
            "val/origin_cc_pace_MAE@20_mean",
            "val/congestion_F1@20_mean",
        ]
        lines.append(vmin_rows[vcols].round(3).to_markdown(index=False))
        lines.append("")

    best_origin = rank.iloc[0]
    best_comp = comp.iloc[0]
    lines.append("## Interpretation\n")
    lines.append(f"- Best val origin_cc_MAE@20: `{best_origin['cell_id']}` = {_fmt(best_origin['val/origin_cc_MAE@20_mean'])} km/h.")
    lines.append(f"- Best composite vs locked: `{best_comp['cell_id']}` = {_fmt(best_comp['composite_vs_locked'])}.")
    locked = selection[selection["cell_id"] == "locked_origin_cc_w4"]
    if not locked.empty:
        locked = locked.iloc[0]
        lines.append(
            f"- Locked baseline reference: val origin_cc={_fmt(locked['val/origin_cc_MAE@20_mean'])}, "
            f"cell={_fmt(locked['val/cell_MAE_mean'])}, impact={_fmt(locked['val/impact_time_AE@20_mean'])}, "
            f"recovery={_fmt(locked['val/recovery_AE@10_mean'])}."
        )
    lines.append("- Treat this as a loss-shape ablation, not a final champion update, until the top candidates are rerun with extra seeds.")
    lines.append("")

    lines.append("## Artifacts\n")
    lines.append(f"- `output/models/speed_pace_ablation/{run_ts}/summary.csv`")
    lines.append(f"- `output/models/speed_pace_ablation/{run_ts}/selection.csv`")
    lines.append(f"- `output/models/speed_pace_ablation/{run_ts}/per_accident_metrics.csv`")
    lines.append(f"- `output/models/speed_pace_ablation/{run_ts}/report.md`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--run-ts", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--pace-lambdas", default="0.25,0.5,1,2,4")
    p.add_argument("--vmins", default="3,5,10")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--extra-seeds", default="", help="Optional comma list rerun seeds for top-k cells.")
    p.add_argument("--top-k", type=int, default=0, help="Rerun top-k cells by val origin_cc_MAE@20 with extra seeds.")
    p.add_argument("--input-type", default="vd+hist", choices=["vd", "hist", "vd+hist"])
    p.add_argument("--hist-tag", default="")
    p.add_argument("--mask-npz", default=str(DEFAULT_MASK_NPZ))
    p.add_argument("--eval-pace-v-min-kmh", type=float, default=DEFAULT_EVAL_PACE_V_MIN)
    p.add_argument("--smoke", action="store_true", help="2 epochs, small grid.")
    p.add_argument("--resume", action="store_true",
                   help="Load existing summary/per-accident CSVs and skip completed (cell_id, seed) pairs.")
    return p.parse_args()


def _csv_float(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _csv_int(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    run_ts = args.run_ts or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.output_root) / run_ts
    cells_dir = out_root / "cells"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.csv"
    panel_path = out_root / "per_accident_metrics.csv"

    pace_lambdas = _csv_float(args.pace_lambdas)
    vmins = _csv_float(args.vmins)
    epochs = args.epochs
    extra_seeds = _csv_int(args.extra_seeds)
    top_k = args.top_k
    if args.smoke:
        pace_lambdas = [0.5, 1.0]
        vmins = [5.0]
        epochs = 2
        extra_seeds = []
        top_k = 0

    device, device_msg = select_device()
    print(f"Device: {device_msg}")
    print(f"Run: {run_ts} | epochs={epochs}")

    target_window = SampleWindow(
        upstream_km=FIXED_UPSTREAM_KM,
        downstream_km=FIXED_DOWNSTREAM_KM,
        time_before_min=FIXED_TIME_BEFORE,
        time_after_min=FIXED_TIME_AFTER,
        spatial_resolution_km=SPATIAL_RES_KM,
        resolution_min=RES_MIN,
    )
    base_cfg = TrainConfig(
        upstream_km=FIXED_UPSTREAM_KM,
        downstream_km=FIXED_DOWNSTREAM_KM,
        time_before_min=FIXED_TIME_BEFORE,
        time_after_min=FIXED_TIME_AFTER,
        spatial_resolution_km=SPATIAL_RES_KM,
        resolution_min=RES_MIN,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        random_state=args.base_seed,
        eval_protocol="v1",
        data_scope="dispatch_only",
        feature_set="t0",
        input_type=args.input_type,
        backbone="simple",
        loss_name="speed_pace_hybrid",
        congestion_upstream_km=PRIMARY_WINDOW_KM,
        min_filled_ratio=0.8,
    )

    samples = load_samples(data_root=DATA_ROOT, resolution_min=RES_MIN)
    matrices, ids = filter_and_subset(samples, target_window, base_cfg.min_before_coverage, base_cfg.min_filled_ratio)
    print(f"Samples after filter: {len(ids)}")
    hist = None
    if "hist" in args.input_type:
        hist = load_hist_aligned(ids, target_window, DATA_ROOT, RES_MIN, args.hist_tag)
        print(f"Hist matrices: {hist.shape}")
    cong_aligned = load_congestion_masks_aligned(ids, target_window, Path(args.mask_npz))
    cong_by_id = {str(a): cong_aligned[i] for i, a in enumerate(ids)}
    print(f"Origin_cc cell share={float(cong_aligned.mean()):.4%}")

    train_idx, val_idx, test_idx = time_based_split(ids, DATA_ROOT, base_cfg.train_end_date, base_cfg.val_end_date)
    _h = lambda idx: hist[idx] if hist is not None else None
    train_ds = AccidentSampleDataset(matrices[train_idx], ids[train_idx], base_cfg, _h(train_idx))
    val_ds = AccidentSampleDataset(matrices[val_idx], ids[val_idx], base_cfg, _h(val_idx))
    test_ds = AccidentSampleDataset(matrices[test_idx], ids[test_idx], base_cfg, _h(test_idx)) if len(test_idx) else None
    loaders = {
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if test_ds else None,
    }
    print(f"Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds) if test_ds else 0}")

    grid = build_grid(pace_lambdas, vmins, include_locked=True)
    all_rows: list[dict] = []
    all_panel: list[dict] = []
    completed: set[tuple[str, int]] = set()
    if args.resume and summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        all_rows = existing_summary.to_dict("records")
        completed = {(str(r["cell_id"]), int(r["seed"])) for r in all_rows}
        print(f"Resume: loaded {len(all_rows)} completed rows from {summary_path}")
    if args.resume and panel_path.exists():
        all_panel = pd.read_csv(panel_path).to_dict("records")
        print(f"Resume: loaded {len(all_panel)} per-accident rows from {panel_path}")

    def run_spec(spec: dict, seed: int) -> dict:
        cfg = make_cfg(base_cfg, spec, epochs, seed, args.mask_npz)
        cell_dir = cells_dir / f"{spec['cell_id']}_s{seed}"
        ckpt = cell_dir / "best_model.pth"
        print(f"\n=== {spec['cell_id']} | seed={seed} | loss={spec['loss_name']} ===")
        t0 = time.time()
        hist_dict = train_cell(cfg, spec, seed, device, train_ds, val_ds, cong_by_id, ckpt)
        panel_rows, agg = evaluate_checkpoint(ckpt, cfg, device, loaders, cong_by_id, args.eval_pace_v_min_kmh)
        secs = round(time.time() - t0, 1)
        for row in panel_rows:
            row.update({"cell_id": spec["cell_id"], "seed": seed})
        all_panel.extend(panel_rows)
        record = {
            **spec,
            "seed": seed,
            "epochs": epochs,
            "train_sec": secs,
            "best_val_loss": hist_dict["best_val_loss"],
            **agg,
        }
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "run_config.json").write_text(json.dumps({"config": asdict(cfg), "spec": spec}, indent=2), encoding="utf-8")
        print(
            f"  done {secs}s | val origin_cc={record.get('val/origin_cc_MAE@20', float('nan')):.3f} | "
            f"cell={record.get('val/cell_MAE', float('nan')):.3f} | "
            f"impact={record.get('val/impact_time_AE@20', float('nan')):.3f}"
        )
        return record

    for spec in grid:
        key = (spec["cell_id"], int(args.base_seed))
        if key in completed:
            print(f"Skip completed: {key[0]} seed={key[1]}")
            continue
        all_rows.append(run_spec(spec, args.base_seed))
        completed.add(key)
        pd.DataFrame(all_rows).to_csv(summary_path, index=False)
        pd.DataFrame(all_panel).to_csv(panel_path, index=False)

    if extra_seeds and top_k > 0:
        interim = seed_aggregate(pd.DataFrame(all_rows))
        top_ids = interim.sort_values("val/origin_cc_MAE@20_mean").head(top_k)["cell_id"].tolist()
        print(f"\nExtra-seed reruns for: {top_ids} | seeds={extra_seeds}")
        for spec in grid:
            if spec["cell_id"] not in top_ids:
                continue
            for seed in extra_seeds:
                key = (spec["cell_id"], int(seed))
                if key in completed:
                    print(f"Skip completed: {key[0]} seed={key[1]}")
                    continue
                all_rows.append(run_spec(spec, seed))
                completed.add(key)
                pd.DataFrame(all_rows).to_csv(summary_path, index=False)
                pd.DataFrame(all_panel).to_csv(panel_path, index=False)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(summary_path, index=False)
    panel = pd.DataFrame(all_panel)
    panel.to_csv(panel_path, index=False)
    selection = seed_aggregate(summary)
    selection.to_csv(out_root / "selection.csv", index=False)
    report = build_report(selection, summary, run_ts, epochs, device_msg, pace_lambdas, vmins, args.eval_pace_v_min_kmh)
    (out_root / "report.md").write_text(report, encoding="utf-8")
    report_copy = PROJECT_ROOT / "reports" / f"speed_pace_ablation_{date.today():%Y%m%d}.md"
    report_copy.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report] {out_root / 'report.md'}")
    print(f"[copy] {report_copy}")


if __name__ == "__main__":
    main()
