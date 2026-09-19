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
"""Ablation for origin_cc-weighted loss plus true-pace auxiliary.

Grid:
- cell_mae baseline
- origin_cc weighted baselines for w sweep
- origin_cc weighted + alpha * upstream pace MAE for each (w, alpha)
- one v_min sensitivity cell

The evaluator is fixed to the existing eval_protocol_v1 panel plus pace metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_speed_pace_ablation import (  # noqa: E402
    DATA_ROOT,
    DEFAULT_MASK_NPZ,
    DEFAULT_OUTPUT_ROOT,
    EVAL_WINDOWS_KM,
    FIXED_DOWNSTREAM_KM,
    FIXED_TIME_AFTER,
    FIXED_TIME_BEFORE,
    FIXED_UPSTREAM_KM,
    PRIMARY_WINDOW_KM,
    RES_MIN,
    SPATIAL_RES_KM,
    _cell_float,
    evaluate_checkpoint,
    seed_aggregate,
    train_cell,
)
from scripts.lib.data_loaders import load_samples  # noqa: E402
from scripts.lib.windows import SampleWindow  # noqa: E402
from scripts.models.unet_matrix import (  # noqa: E402
    AccidentSampleDataset,
    TrainConfig,
    filter_and_subset,
    load_congestion_masks_aligned,
    load_hist_aligned,
    select_device,
    time_based_split,
)


OUT_ROOT = PROJECT_ROOT / "output" / "models" / "origin_cc_pace_aux_ablation"


def _csv_float(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _csv_int(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _weight_id(w: float) -> str:
    value = float(w)
    if value.is_integer():
        return str(int(value))
    return _cell_float(value)


def build_grid(weights: list[float], alphas: list[float], vmin: float, vmin_sensitivity: list[float]) -> list[dict]:
    cells: list[dict] = [{
        "cell_id": "cellmae",
        "family": "baseline",
        "loss_name": "cell_mae",
        "mask": "none",
        "congestion_weight": 1.0,
        "speed_lambda": 1.0,
        "pace_lambda": 0.0,
        "pace_v_min_kmh": vmin,
    }]
    for w in weights:
        cells.append({
            "cell_id": f"origin_w{_weight_id(w)}",
            "family": "origin_weight",
            "loss_name": "upstream_congestion_weighted",
            "mask": "origin_cc",
            "congestion_weight": float(w),
            "speed_lambda": 1.0,
            "pace_lambda": 0.0,
            "pace_v_min_kmh": vmin,
        })
    for w in weights:
        for alpha in alphas:
            cells.append({
                "cell_id": f"origin_w{_weight_id(w)}_pace_a{_cell_float(alpha)}_v{_cell_float(vmin)}",
                "family": "origin_pace_aux",
                "loss_name": "origin_cc_pace_aux",
                "mask": "origin_cc",
                "congestion_weight": float(w),
                "speed_lambda": 1.0,
                "pace_lambda": float(alpha),
                "pace_v_min_kmh": vmin,
            })
    for v in vmin_sensitivity:
        if abs(float(v) - float(vmin)) < 1e-9:
            continue
        cells.append({
            "cell_id": f"origin_w{_weight_id(4.0)}_pace_a0p25_v{_cell_float(v)}",
            "family": "vmin_sensitivity",
            "loss_name": "origin_cc_pace_aux",
            "mask": "origin_cc",
            "congestion_weight": 4.0,
            "speed_lambda": 1.0,
            "pace_lambda": 0.25,
            "pace_v_min_kmh": float(v),
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


def _composite_against(selection: pd.DataFrame, baseline_id: str) -> pd.DataFrame:
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
    col_name = f"composite_vs_{baseline_id}"
    out[col_name] = 0.0
    for key, col in metrics.items():
        denom = float(base[col])
        out[f"ratio_{key}_vs_{baseline_id}"] = out[col] / denom if denom and np.isfinite(denom) else np.nan
        out[col_name] += weights[key] * out[f"ratio_{key}_vs_{baseline_id}"]
    return out


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    selection = seed_aggregate(summary)
    if "locked_origin_cc_w4" in set(selection["cell_id"]):
        # Older helper uses locked_origin_cc_w4; this ablation names it origin_w4.
        return selection
    return _composite_against(
        selection.drop(columns=[c for c in selection.columns if c.startswith("ratio_") or c.startswith("composite_vs_")], errors="ignore"),
        "origin_w4",
    )


def _fmt(value: object, ndigits: int = 3) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "nan"
        return f"{float(value):.{ndigits}f}"
    return str(value)


def build_report(
    selection: pd.DataFrame,
    run_ts: str,
    epochs: int,
    device_msg: str,
    weights: list[float],
    alphas: list[float],
    train_vmin: float,
    vmin_sensitivity: list[float],
    eval_pace_v_min: float,
) -> str:
    lines: list[str] = []
    lines.append(f"# Origin_cc + True-Pace Auxiliary Ablation - {date.today():%Y-%m-%d}\n")
    lines.append(f"Run: `{run_ts}` | device: {device_msg} | epochs: {epochs} | input=`vd+hist` | model=`unet_matrix/simple`\n")
    lines.append("## Design\n")
    lines.append("- Base loss: `upstream_congestion_weighted`, origin_cc mask, upstream 0..20 km.")
    lines.append("- Auxiliary term: normalized true-pace MAE over valid upstream 0..20 km cells.")
    lines.append(f"- Weight sweep: `w={weights}`.")
    lines.append(f"- Pace alpha sweep: `alpha={alphas}` with training `v_min={train_vmin:g}`.")
    lines.append(f"- V_min sensitivity cells: `{vmin_sensitivity}` at `w=4, alpha=0.25`.")
    lines.append(f"- Fixed evaluator: origin_cc_MAE@20, speed60/upstream/cell metrics, impact/recovery, F1, and pace metrics with `v_min={eval_pace_v_min:g}` on val/test.\n")

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
        "composite_vs_origin_w4",
    ]
    show_cols = [c for c in cols if c in selection.columns]
    lines.append("## Ranking by val origin_cc_MAE@20\n")
    lines.append(selection.sort_values("val/origin_cc_MAE@20_mean")[show_cols].round(3).to_markdown(index=False))
    lines.append("")

    if "composite_vs_origin_w4" in selection.columns:
        lines.append("## Ranking by composite vs `origin_w4`\n")
        lines.append("Composite weights: origin_cc_MAE@20 0.50, cell_MAE 0.15, impact_AE@20 0.20, recovery_AE@10 0.15. `origin_w4` = 1.000.\n")
        lines.append(selection.sort_values("composite_vs_origin_w4")[show_cols].round(3).to_markdown(index=False))
        lines.append("")

    origin_rows = selection[selection["family"] == "origin_weight"].sort_values("congestion_weight")
    if not origin_rows.empty:
        lines.append("## Origin_cc weight-only sweep\n")
        wcols = [
            "cell_id",
            "congestion_weight",
            "val/origin_cc_MAE@20_mean",
            "test/origin_cc_MAE@20_mean",
            "val/cell_MAE_mean",
            "val/impact_time_AE@20_mean",
            "val/recovery_AE@10_mean",
            "val/congestion_F1@20_mean",
        ]
        lines.append(origin_rows[wcols].round(3).to_markdown(index=False))
        lines.append("")

    aux_rows = selection[selection["family"] == "origin_pace_aux"].copy()
    if not aux_rows.empty:
        lines.append("## Pace auxiliary grid\n")
        acols = [
            "cell_id",
            "congestion_weight",
            "pace_lambda",
            "val/origin_cc_MAE@20_mean",
            "test/origin_cc_MAE@20_mean",
            "val/cell_MAE_mean",
            "val/impact_time_AE@20_mean",
            "val/recovery_AE@10_mean",
            "val/origin_cc_pace_MAE@20_mean",
            "val/congestion_F1@20_mean",
        ]
        lines.append(aux_rows.sort_values(["congestion_weight", "pace_lambda"])[acols].round(3).to_markdown(index=False))
        lines.append("")

    best_origin = selection.sort_values("val/origin_cc_MAE@20_mean").iloc[0]
    lines.append("## Early interpretation\n")
    lines.append(f"- Best val origin_cc_MAE@20 so far: `{best_origin['cell_id']}` = {_fmt(best_origin['val/origin_cc_MAE@20_mean'])} km/h.")
    if "composite_vs_origin_w4" in selection.columns:
        best_comp = selection.sort_values("composite_vs_origin_w4").iloc[0]
        lines.append(f"- Best composite vs origin_w4 so far: `{best_comp['cell_id']}` = {_fmt(best_comp['composite_vs_origin_w4'])}.")
    lines.append("- Treat this as a first-pass grid; rerun the top 2-3 cells with extra seeds before changing the locked training spec.")
    lines.append("")

    lines.append("## Artifacts\n")
    lines.append(f"- `output/models/origin_cc_pace_aux_ablation/{run_ts}/summary.csv`")
    lines.append(f"- `output/models/origin_cc_pace_aux_ablation/{run_ts}/selection.csv`")
    lines.append(f"- `output/models/origin_cc_pace_aux_ablation/{run_ts}/per_accident_metrics.csv`")
    lines.append(f"- `output/models/origin_cc_pace_aux_ablation/{run_ts}/report.md`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--run-ts", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weights", default="2,4,7,10")
    p.add_argument("--alphas", default="0.1,0.25,0.5,1")
    p.add_argument("--vmin", type=float, default=5.0)
    p.add_argument("--vmin-sensitivity", default="10")
    p.add_argument("--eval-pace-v-min-kmh", type=float, default=5.0)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--extra-seeds", default="")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--mask-npz", default=str(DEFAULT_MASK_NPZ))
    p.add_argument("--hist-tag", default="")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_ts = args.run_ts or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.output_root) / run_ts
    cells_dir = out_root / "cells"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.csv"
    panel_path = out_root / "per_accident_metrics.csv"

    weights = _csv_float(args.weights)
    alphas = _csv_float(args.alphas)
    vmin_sensitivity = _csv_float(args.vmin_sensitivity)
    epochs = args.epochs
    extra_seeds = _csv_int(args.extra_seeds)
    top_k = args.top_k
    if args.smoke:
        weights = [4.0]
        alphas = [0.25]
        vmin_sensitivity = []
        epochs = 2
        extra_seeds = []
        top_k = 0

    device, device_msg = select_device()
    print(f"Device: {device_msg}")
    print(f"Run: {run_ts} | epochs={epochs} | cells={1 + len(weights) + len(weights) * len(alphas) + len([v for v in vmin_sensitivity if abs(v - args.vmin) > 1e-9])}")

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
        input_type="vd+hist",
        backbone="simple",
        loss_name="origin_cc_pace_aux",
        congestion_upstream_km=PRIMARY_WINDOW_KM,
        min_filled_ratio=0.8,
    )

    samples = load_samples(data_root=DATA_ROOT, resolution_min=RES_MIN)
    matrices, ids = filter_and_subset(samples, target_window, base_cfg.min_before_coverage, base_cfg.min_filled_ratio)
    hist = load_hist_aligned(ids, target_window, DATA_ROOT, RES_MIN, args.hist_tag)
    cong_aligned = load_congestion_masks_aligned(ids, target_window, Path(args.mask_npz))
    cong_by_id = {str(a): cong_aligned[i] for i, a in enumerate(ids)}
    train_idx, val_idx, test_idx = time_based_split(ids, DATA_ROOT, base_cfg.train_end_date, base_cfg.val_end_date)
    _h = lambda idx: hist[idx]
    train_ds = AccidentSampleDataset(matrices[train_idx], ids[train_idx], base_cfg, _h(train_idx))
    val_ds = AccidentSampleDataset(matrices[val_idx], ids[val_idx], base_cfg, _h(val_idx))
    test_ds = AccidentSampleDataset(matrices[test_idx], ids[test_idx], base_cfg, _h(test_idx)) if len(test_idx) else None
    loaders = {
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if test_ds else None,
    }
    print(f"Samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds) if test_ds else 0} | origin_cc share={float(cong_aligned.mean()):.4%}")

    grid = build_grid(weights, alphas, args.vmin, vmin_sensitivity)
    all_rows: list[dict] = []
    all_panel: list[dict] = []
    completed: set[tuple[str, int]] = set()
    if args.resume and summary_path.exists():
        existing = pd.read_csv(summary_path)
        all_rows = existing.to_dict("records")
        completed = {(str(r["cell_id"]), int(r["seed"])) for r in all_rows}
        print(f"Resume: loaded {len(all_rows)} completed rows")
    if args.resume and panel_path.exists():
        all_panel = pd.read_csv(panel_path).to_dict("records")
        print(f"Resume: loaded {len(all_panel)} per-accident rows")

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
        rec = {**spec, "seed": seed, "epochs": epochs, "train_sec": secs, "best_val_loss": hist_dict["best_val_loss"], **agg}
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "run_config.json").write_text(json.dumps({"config": asdict(cfg), "spec": spec}, indent=2), encoding="utf-8")
        print(
            f"  done {secs}s | val origin_cc={rec.get('val/origin_cc_MAE@20', float('nan')):.3f} | "
            f"cell={rec.get('val/cell_MAE', float('nan')):.3f} | impact={rec.get('val/impact_time_AE@20', float('nan')):.3f}"
        )
        return rec

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
        interim = aggregate(pd.DataFrame(all_rows))
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
    pd.DataFrame(all_panel).to_csv(panel_path, index=False)
    selection = aggregate(summary)
    selection.to_csv(out_root / "selection.csv", index=False)
    report = build_report(
        selection,
        run_ts,
        epochs,
        device_msg,
        weights,
        alphas,
        args.vmin,
        vmin_sensitivity,
        args.eval_pace_v_min_kmh,
    )
    (out_root / "report.md").write_text(report, encoding="utf-8")
    report_copy = PROJECT_ROOT / "reports" / f"origin_cc_pace_aux_ablation_{date.today():%Y%m%d}.md"
    report_copy.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report] {out_root / 'report.md'}")
    print(f"[copy] {report_copy}")


if __name__ == "__main__":
    main()
