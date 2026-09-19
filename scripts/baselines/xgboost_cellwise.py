# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "xgboost",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "torch",
#     "matplotlib",
#     "scikit-learn",
#     "tqdm",
# ]
# ///
# ruff: noqa: E402
"""Baseline 1b — cell-wise gradient boosting (eval_protocol_v1).

Model class
-----------
A tabular regressor that predicts each cell/horizon speed independently from
flat features. Like the LSTM baseline it has NO spatial field view; unlike the
LSTM it is a tree ensemble over engineered features, which is the natural way to
fold in the structured t0 incident attributes. The gap to the U-Net candidates
isolates "tabular per-cell learner" vs "joint spatiotemporal field model".

One training row per (accident, mileage cell, post horizon):
- pre-accident VD sequence for the cell (+ last/mean/min/std/n_valid summaries),
- historical baseline at the target cell/horizon (`hist_now`) and its pre/post
  summaries — the level to recover toward,
- position (space_offset_km, |space|, is_upstream, horizon_min),
- the shared immutable t0 incident vector (geometry / direction / highway /
  mileage and causally available value/availability fields).

NaNs are passed through as XGBoost "missing".
The primary `full_field_congestion_weighted` objective assigns weight 1 to
every valid row and weight 4 to origin_cc rows in upstream 0--20 km. The
separate `upstream_congestion_weighted` sensitivity assigns upstream
non-congested rows weight 1, origin_cc rows weight 4, and rows outside the zone
weight 0, exactly matching the U-Net loss support.

Protocol: vd+hist + the shared immutable t0 vector, 3 seeds {42,7,123}, the same
explicit N1/N3 cohort, and either the chronological primary split or fixed
random robustness split. Scored on the shared metric panel.

Usage:
  uv run scripts/baselines/xgboost_cellwise.py --smoke
  uv run scripts/baselines/xgboost_cellwise.py                # 3 seeds
  uv run scripts/baselines/xgboost_cellwise.py --n-estimators 600 --max-depth 7
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines._harness import (  # noqa: E402
    CONGESTION_UPSTREAM_KM,
    CONGESTION_WEIGHT,
    DEFAULT_OUTPUT_ROOT,
    PAPER_SEEDS,
    SPEED_MAX,
    BaselineData,
    BaselineResult,
    add_protocol_arguments,
    artifact_manifest,
    congestion_weights,
    configure_reproducibility,
    load_baseline_data,
    make_run_id,
    objective_mae_kmh,
    print_seed_table,
    protocol_kwargs_from_args,
    score_predictions,
    write_result,
    write_seed_summary,
)


# ────────────────────────────────────────────────────────────────────────────
# Feature grid (one accident -> (n_space*n_after, F) tabular rows)
# ────────────────────────────────────────────────────────────────────────────


def _last_valid(pre: np.ndarray) -> np.ndarray:
    """Last non-NaN value along the time axis of a (n_space, n_before) block."""
    out = np.full(pre.shape[0], np.nan, dtype=np.float32)
    for col in range(pre.shape[1]):  # left→Rright, overwrite where observed
        v = pre[:, col]
        seen = ~np.isnan(v)
        out[seen] = v[seen]
    return out


def _accident_grid(data: BaselineData, idx: int, loss_name: str):
    """Return (X, target, valid, weight, feat_names) for one accident.

    Rows are row-major over (cell s, horizon j): row = s * n_after + j, so a
    split's predictions reshape to (n_space, n_after) per accident directly.
    """
    nb, na, ns = data.n_before, data.n_after, data.n_space
    vd = data.matrices[idx]            # (ns, n_time) km/h NaN
    hi = data.hist[idx]               # (ns, n_time) km/h NaN
    pre = vd[:, :nb]                  # (ns, nb)
    hist_post = hi[:, nb:]           # (ns, na)
    target = vd[:, nb:]              # (ns, na)
    valid = ~np.isnan(target)
    space_km = data.target_window.space_offsets_m.astype(np.float32) / 1000.0   # (ns,)
    horizon_min = (np.arange(na, dtype=np.float32) * data.target_window.resolution_min)  # (na,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows → nan
        pre_mean = np.nanmean(pre, axis=1)
        pre_min = np.nanmin(pre, axis=1)
        pre_std = np.nanstd(pre, axis=1)
        hist_post_mean = np.nanmean(hist_post, axis=1)
    pre_last = _last_valid(pre)
    pre_nvalid = (~np.isnan(pre)).sum(axis=1).astype(np.float32)
    hist_pre_last = hi[:, nb - 1] if nb > 0 else np.full(ns, np.nan, np.float32)

    def col(vec_s: np.ndarray) -> np.ndarray:                 # (ns,) -> (ns, na)
        return np.broadcast_to(vec_s.reshape(ns, 1), (ns, na))

    planes: dict[str, np.ndarray] = {}
    if data.base_cfg.input_type != "hist":
        for k in range(nb):
            planes[f"pre_vd_{k}"] = col(pre[:, k])
        planes["pre_vd_last"] = col(pre_last)
        planes["pre_vd_mean"] = col(pre_mean)
        planes["pre_vd_min"] = col(pre_min)
        planes["pre_vd_std"] = col(pre_std)
        planes["pre_vd_nvalid"] = col(pre_nvalid)
    planes["hist_pre_last"] = col(hist_pre_last)
    planes["hist_post_mean"] = col(hist_post_mean)
    planes["hist_now"] = hist_post                            # per (s, j)
    planes["space_km"] = col(space_km)
    planes["abs_space_km"] = col(np.abs(space_km))
    planes["is_upstream"] = col((space_km >= 0).astype(np.float32))
    planes["horizon_min"] = np.broadcast_to(horizon_min.reshape(1, na), (ns, na))
    for fi, name in enumerate(data.feat_names):
        planes[f"f_{name}"] = col(np.full(ns, data.feat[idx, fi], dtype=np.float32))

    names = list(planes.keys())
    X = np.stack([planes[n].reshape(-1) for n in names], axis=1).astype(np.float32)
    tgt = target.reshape(-1).astype(np.float32)
    val = valid.reshape(-1).astype(np.float32)
    if loss_name == "cell_mae":
        wt = val.copy()
    else:
        outside_weight = 1.0 if loss_name == "full_field_congestion_weighted" else 0.0
        wt = congestion_weights(data, idx, outside_weight=outside_weight).reshape(-1).astype(np.float32)
    return X, tgt, val, wt, names


def build_split(data: BaselineData, indices: np.ndarray, loss_name: str, train: bool):
    """Stack per-accident grids. For train, keep only weighted+valid rows."""
    Xs, ts, vs, ws = [], [], [], []
    names: list[str] | None = None
    for idx in indices:
        X, tgt, val, wt, nm = _accident_grid(data, int(idx), loss_name)
        if names is None:
            names = nm
        Xs.append(X)
        ts.append(tgt)
        vs.append(val)
        ws.append(wt)
    X = np.concatenate(Xs, axis=0)
    tgt = np.concatenate(ts, axis=0)
    val = np.concatenate(vs, axis=0)
    wt = np.concatenate(ws, axis=0)
    if train:
        keep = (wt > 0) & (val > 0)
        return X[keep], tgt[keep], wt[keep], names
    return X, tgt, val, wt, names


# ────────────────────────────────────────────────────────────────────────────
# Train + predict
# ───────────────────────────────────────────────────────────────────────────


def _make_model(seed: int, args: argparse.Namespace):
    import xgboost as xgb

    return xgb.XGBRegressor(
        objective="reg:absoluteerror",   # matches the MAE-style cell loss
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=args.n_jobs,
    )


def predict_field(model, data: BaselineData, indices: np.ndarray, loss_name: str) -> np.ndarray:
    """Predict a split's full field -> (n_acc, n_space, n_after) km/h."""
    X, _, _, _, _ = build_split(data, indices, loss_name, train=False)
    pred = model.predict(X)
    pred = np.clip(pred, 0.0, SPEED_MAX).astype(np.float32)
    return pred.reshape(len(indices), data.n_space, data.n_after)


# ────────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default=",".join(str(s) for s in PAPER_SEEDS))
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-weight", type=float, default=5.0)
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Paper default 1 avoids thread-scheduling variation; increase only for diagnostics.",
    )
    p.add_argument("--input-type", default="vd+hist", choices=["vd+hist", "hist"])
    p.add_argument(
        "--loss",
        default="full_field_congestion_weighted",
        choices=[
            "full_field_congestion_weighted",
            "upstream_congestion_weighted",
            "cell_mae",
        ],
        help=(
            "full_field_congestion_weighted is the primary per-cell objective; "
            "upstream_congestion_weighted is the exact U-Net-loss sensitivity."
        ),
    )
    p.add_argument("--no-features", action="store_true", help="Drop t0 incident features.")
    p.add_argument("--run-id", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Quick check: seed 42 only, 60 trees, subsample train accidents.")
    p.add_argument("--max-train-accidents", type=int, default=0)
    add_protocol_arguments(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        seeds = [42]
        args.n_estimators = min(args.n_estimators, 60)
        max_train = args.max_train_accidents or 150
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        max_train = args.max_train_accidents

    run_id = args.run_id or make_run_id("xgboost_cellwise" + ("_smoke" if args.smoke else ""))
    print(f"Run id: {run_id} | loss={args.loss} | seeds={seeds} | "
          f"n_estimators={args.n_estimators} max_depth={args.max_depth}")

    data = load_baseline_data(
        input_type=args.input_type,
        with_features=not args.no_features,
        **protocol_kwargs_from_args(args),
    )

    train_idx = data.train_idx
    if max_train and len(train_idx) > max_train:
        train_idx = train_idx[:max_train]
        print(f"[smoke] subsampled train accidents -> {len(train_idx)}")

    print("Building train rows ...")
    t0 = time.time()
    Xtr, ytr, wtr, names = build_split(data, train_idx, args.loss, train=True)
    print(f"  train rows={Xtr.shape[0]:,} | features={Xtr.shape[1]} | {time.time() - t0:.1f}s")

    summary_rows: list[dict] = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        t_seed = time.time()
        reproducibility = configure_reproducibility(seed, args.deterministic_mode)
        model = _make_model(seed, args)
        model.fit(Xtr, ytr, sample_weight=wtr)

        val_pred = predict_field(model, data, data.val_idx, args.loss)
        val_rows, val_agg = score_predictions(val_pred, data.val_idx, data, "val")
        val_objective_mae = objective_mae_kmh(val_pred, data.val_idx, data, args.loss)
        test_rows, test_agg = [], {}
        if len(data.test_idx):
            test_pred = predict_field(model, data, data.test_idx, args.loss)
            test_rows, test_agg = score_predictions(test_pred, data.test_idx, data, "test")

        seed_run_id = f"{run_id}_s{seed}"
        out_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
        checkpoint_dir = out_root / "xgboost_cellwise" / seed_run_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoint_dir / "model.json"
        model.save_model(str(model_path))
        model_artifact = artifact_manifest(model_path)

        result = BaselineResult(
            model="xgboost_cellwise", seed=seed, loss_name=args.loss,
            mask="origin_cc" if args.loss != "cell_mae" else "none",
            val_rows=val_rows, val_agg=val_agg, test_rows=test_rows, test_agg=test_agg,
            best_val_loss=val_objective_mae,
            extra_config={
                "n_estimators": args.n_estimators, "max_depth": args.max_depth,
                "learning_rate": args.learning_rate, "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree, "min_child_weight": args.min_child_weight,
                "input_type": args.input_type, "with_features": not args.no_features,
                "smoke": bool(args.smoke),
                "n_features": int(Xtr.shape[1]),
                "feature_names": list(names or []),
                "congestion_weight": CONGESTION_WEIGHT, "congestion_upstream_km": CONGESTION_UPSTREAM_KM,
                "n_train_accidents": int(len(train_idx)), "n_train_rows": int(Xtr.shape[0]),
                "best_val_loss_unit": "km/h",
                "fit_selection": {
                    "rule": "fixed_n_estimators_no_early_stopping",
                    "n_estimators": int(args.n_estimators),
                },
                "model_artifact": model_artifact,
                "xgboost_version": importlib.metadata.version("xgboost"),
                "n_jobs": args.n_jobs,
                "reproducibility": reproducibility,
                "paper_protocol": data.paper_protocol,
            },
        )
        out_dir = write_result(result, seed_run_id, args.output_root)
        summary_rows.append(result.summary_row())
        print(
            f"  seed {seed} done in {time.time() - t_seed:.1f}s | "
            f"val_objective_MAE={val_objective_mae:.3f} km/h -> {out_dir}"
        )

    print("\nPer-seed panel (val):")
    print_seed_table(summary_rows)
    summary_path = write_seed_summary(summary_rows, "xgboost_cellwise", run_id, args.output_root)
    print(f"\n[summary] {summary_path}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
