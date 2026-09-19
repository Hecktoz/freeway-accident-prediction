# /// script
# requires-python = ">=3.11"
# dependencies = [
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
"""Baselines 0a/0b - non-learning rows for the paper Table 2 panel.

0a `historical_mean` predicts each post-accident cell with the matched
historical baseline matrix.

0b `persistence` holds the last valid pre-accident speed at each mileage cell
constant over the post-accident horizon.

Both are scored by the shared eval_protocol_v1 harness, so their outputs match
`lstm_percell.py` and `xgboost_cellwise.py`:

  uv run scripts/baselines/nonlearning.py
  uv run scripts/baselines/nonlearning.py --models historical_mean,persistence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines._harness import (  # noqa: E402
    SPEED_MAX,
    BaselineData,
    BaselineResult,
    DEFAULT_OUTPUT_ROOT,
    add_protocol_arguments,
    load_baseline_data,
    make_run_id,
    print_seed_table,
    protocol_kwargs_from_args,
    score_predictions,
    write_result,
    write_seed_summary,
)


def _global_train_mean(data: BaselineData) -> float:
    vals = []
    for idx in data.train_idx:
        target, valid = data.post_observed(int(idx))
        if valid.any():
            vals.append(target[valid > 0])
    if not vals:
        return 90.0
    return float(np.nanmean(np.concatenate(vals)))


def _historical_predictions(data: BaselineData, indices: np.ndarray, fill_value: float) -> np.ndarray:
    if data.hist is None:
        raise SystemExit("historical_mean requires --input-type hist or vd+hist")
    pred = np.empty((len(indices), data.n_space, data.n_after), dtype=np.float32)
    for k, idx in enumerate(indices):
        h = data.hist[int(idx)][:, data.n_before:]
        pred[k] = np.nan_to_num(h, nan=fill_value).astype(np.float32)
    return np.clip(pred, 0.0, SPEED_MAX)


def _last_valid_per_cell(pre: np.ndarray) -> np.ndarray:
    out = np.full(pre.shape[0], np.nan, dtype=np.float32)
    for col in range(pre.shape[1]):
        v = pre[:, col]
        seen = np.isfinite(v)
        out[seen] = v[seen]
    return out


def _persistence_predictions(data: BaselineData, indices: np.ndarray, fill_value: float) -> np.ndarray:
    pred = np.empty((len(indices), data.n_space, data.n_after), dtype=np.float32)
    for k, idx in enumerate(indices):
        vd = data.matrices[int(idx)]
        last = _last_valid_per_cell(vd[:, :data.n_before])
        last = np.nan_to_num(last, nan=fill_value).astype(np.float32)
        pred[k] = np.broadcast_to(last.reshape(data.n_space, 1), (data.n_space, data.n_after))
    return np.clip(pred, 0.0, SPEED_MAX)


def _score_model(
    model: str,
    data: BaselineData,
    run_id: str,
    output_root: str | None,
    *,
    smoke: bool,
) -> dict:
    fill_value = _global_train_mean(data)
    if model == "historical_mean":
        pred_fn = _historical_predictions
    elif model == "persistence":
        pred_fn = _persistence_predictions
    else:
        raise ValueError(f"unknown non-learning baseline: {model}")

    val_pred = pred_fn(data, data.val_idx, fill_value)
    val_rows, val_agg = score_predictions(val_pred, data.val_idx, data, "val")
    test_rows, test_agg = [], {}
    if len(data.test_idx):
        test_pred = pred_fn(data, data.test_idx, fill_value)
        test_rows, test_agg = score_predictions(test_pred, data.test_idx, data, "test")

    result = BaselineResult(
        model=model,
        seed=0,
        loss_name="cell_mae",
        mask="none",
        val_rows=val_rows,
        val_agg=val_agg,
        test_rows=test_rows,
        test_agg=test_agg,
        best_val_loss=float(val_agg.get("val/cell_MAE", np.nan)),
        extra_config={
            "input_type": data.base_cfg.input_type,
            "fill_value_kmh": fill_value,
            "nonlearning_baseline": model,
            "smoke": bool(smoke),
            "paper_protocol": data.paper_protocol,
        },
    )
    out_dir = write_result(result, run_id, Path(output_root) if output_root else None)
    row = result.summary_row()
    write_seed_summary([row], model, run_id, Path(output_root) if output_root else None)
    print(f"[{model}] wrote {out_dir}")
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="historical_mean,persistence")
    p.add_argument("--input-type", default="vd+hist", choices=["vd+hist", "hist"])
    p.add_argument("--run-id", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--smoke", action="store_true", help="Quick check: score a small val/test subset.")
    p.add_argument("--max-eval-accidents", type=int, default=0)
    add_protocol_arguments(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    run_id = args.run_id or make_run_id("nonlearning")
    print(f"Run id: {run_id} | models={models} | input_type={args.input_type}")

    data = load_baseline_data(
        input_type=args.input_type,
        with_features=False,
        **protocol_kwargs_from_args(args),
    )
    max_eval = args.max_eval_accidents or (40 if args.smoke else 0)
    if max_eval:
        data.val_idx = data.val_idx[:max_eval]
        data.test_idx = data.test_idx[:max_eval]
        print(f"[smoke] scoring subset: val={len(data.val_idx)} test={len(data.test_idx)}")
    rows = [
        _score_model(model, data, run_id, args.output_root, smoke=args.smoke)
        for model in models
    ]

    print("\nNon-learning baseline panel (val):")
    print_seed_table(rows)

    out_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
    combined_dir = out_root / "nonlearning" / run_id
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_path = combined_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(combined_path, index=False)
    print(f"\n[combined summary] {combined_path}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
