# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "pyarrow",
# ]
# ///
# ruff: noqa: E402
"""
Baseline vs covariate-enhanced ridge benchmark on accident speed series.

Each accident is reduced to a spatial-mean speed time series (n_time floats).
Two ridge regressors compete:
- baseline_no_covariate : input = pre-accident context series only.
- xreg_covariate        : input = context + 33 incident features + dynamic blockage decay + cyclic time covariates.

Inputs:
- `data/accident_samples_5m.npz`
- `data/accident_samples_5m_metadata.parquet`
- `output/feature_engineering/incident_features_33.csv`

Outputs (under `output/models/timesfm_covariate/<run_id>/`):
- `comparison_summary.csv` and `comparison_summary.md`
- `per_sample_metrics.csv`
- `run_config.json`

If `timesfm` is installed it is reported, but the default benchmark stays a
deterministic ridge proxy so it runs without internet/GPU.

Usage:
  uv run scripts/models/timesfm_covariate.py
  uv run scripts/models/timesfm_covariate.py --max-samples 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.data_loaders import load_samples
from scripts.lib.windows import SampleWindow, subset


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "models" / "timesfm_covariate"
DEFAULT_FEATURE_CSV = PROJECT_ROOT / "output" / "feature_engineering" / "incident_features_33.csv"
CONGESTION_KMH = 60.0

LOGGER = logging.getLogger(__name__)


@dataclass
class RunConfig:
    upstream_km: int = 25
    downstream_km: int = 10
    time_before_min: int = 30
    time_after_min: int = 145
    spatial_resolution_km: int = 1
    resolution_min: int = 5
    val_split: float = 0.2
    random_state: int = 42
    ridge_alpha: float = 1.0
    max_samples: int | None = None
    min_before_coverage: float = 0.0
    min_filled_ratio: float = 0.8

    @property
    def n_before(self) -> int:
        return self.time_before_min // self.resolution_min

    @property
    def n_after(self) -> int:
        return self.time_after_min // self.resolution_min + 1


def fill_series_nan(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values.astype(float))
    s = s.interpolate(limit_direction="both")
    return s.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)


def compute_blockage_profile(
    lane_blockage_rate: float,
    report_time: pd.Timestamp,
    completed_time: pd.Timestamp,
    n_after: int,
    resolution_min: int,
) -> np.ndarray:
    if np.isnan(lane_blockage_rate):
        lane_blockage_rate = 0.0
    if pd.isna(report_time) or pd.isna(completed_time):
        return np.full(n_after, lane_blockage_rate, dtype=np.float32)
    total_min = max((completed_time - report_time).total_seconds() / 60.0, 1.0)
    elapsed = np.arange(n_after, dtype=float) * resolution_min
    decay = np.maximum(0.0, 1.0 - elapsed / total_min)
    return (lane_blockage_rate * decay).astype(np.float32)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    flat_t = y_true.reshape(-1)
    flat_p = y_pred.reshape(-1)
    mae = float(mean_absolute_error(flat_t, flat_p))
    rmse = float(np.sqrt(mean_squared_error(flat_t, flat_p)))
    f1 = float(f1_score((flat_t < CONGESTION_KMH).astype(int), (flat_p < CONGESTION_KMH).astype(int), zero_division=0))
    return {"mae_kmh": mae, "rmse_kmh": rmse, "congestion_f1": f1}


def build_samples(
    cfg: RunConfig,
    feature_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
):
    """Returns (task_ids, context_matrix, target_matrix, covariate_matrix, n_after)."""
    samples = load_samples(data_root=PROJECT_ROOT / "data", resolution_min=cfg.resolution_min)
    target_window = SampleWindow(
        upstream_km=cfg.upstream_km,
        downstream_km=cfg.downstream_km,
        time_before_min=cfg.time_before_min,
        time_after_min=cfg.time_after_min,
        spatial_resolution_km=cfg.spatial_resolution_km,
        resolution_min=cfg.resolution_min,
    )

    feat_index = feature_df.set_index("task_id")
    feat_cols = [c for c in feature_df.columns if c != "task_id"]
    meta_index = metadata_df.set_index("accident_id")

    task_ids: list[str] = []
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    covariates: list[np.ndarray] = []

    n_before = cfg.n_before
    n_after = cfg.n_after

    for i in range(len(samples)):
        accident_id = str(samples.accident_ids[i])
        if accident_id not in feat_index.index or accident_id not in meta_index.index:
            continue
        filled_ratio = float(meta_index.loc[accident_id].get("filled_ratio", 0.0) or 0.0)
        if filled_ratio < cfg.min_filled_ratio:
            continue
        m_full = samples.matrix[i]
        m, _ = subset(
            m_full,
            samples.window,
            upstream_km=cfg.upstream_km,
            downstream_km=cfg.downstream_km,
            time_before_min=cfg.time_before_min,
            time_after_min=cfg.time_after_min,
        )
        before_block = m[:, :n_before]
        if cfg.min_before_coverage > 0 and (
            before_block.size == 0 or float(np.isnan(before_block).mean()) > 1.0 - cfg.min_before_coverage
        ):
            continue

        # Spatial mean per time bin (NaN-safe).
        with np.errstate(invalid="ignore"):
            series = np.nanmean(m, axis=0)
        series = fill_series_nan(series)
        context = series[:n_before]
        target = series[n_before:]
        if context.size == 0 or target.size == 0:
            continue

        feat_row = feat_index.loc[accident_id]
        static_cov = pd.to_numeric(feat_row[feat_cols], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        meta_row = meta_index.loc[accident_id]
        report_time = pd.to_datetime(meta_row["report_time"], errors="coerce")
        completed_time = pd.to_datetime(meta_row.get("completed_time"), errors="coerce")
        blockage = compute_blockage_profile(
            lane_blockage_rate=float(feat_row.get("lane_blockage_rate", 0.0)),
            report_time=report_time,
            completed_time=completed_time,
            n_after=n_after,
            resolution_min=cfg.resolution_min,
        )

        hour = report_time.hour if pd.notna(report_time) else 0
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        weekend = int(pd.notna(report_time) and report_time.weekday() >= 5)

        cov = np.concatenate([
            context.astype(np.float32),
            static_cov,
            blockage,
            np.array([hour_sin, hour_cos, weekend], dtype=np.float32),
        ], axis=0)

        task_ids.append(accident_id)
        contexts.append(context.astype(np.float32))
        targets.append(target.astype(np.float32))
        covariates.append(cov)

    if cfg.max_samples is not None and len(task_ids) > cfg.max_samples:
        task_ids = task_ids[: cfg.max_samples]
        contexts = contexts[: cfg.max_samples]
        targets = targets[: cfg.max_samples]
        covariates = covariates[: cfg.max_samples]

    if not task_ids:
        raise SystemExit("No usable samples found after filtering and feature alignment.")

    return (
        task_ids,
        np.stack(contexts, axis=0),
        np.stack(targets, axis=0),
        np.stack(covariates, axis=0),
        n_after,
    )


def fit_predict(X: np.ndarray, y_train: np.ndarray, train_idx, val_idx, alpha: float, random_state: int) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X[train_idx])
    x_val = scaler.transform(X[val_idx])
    model = Ridge(alpha=alpha, random_state=random_state)
    model.fit(x_train, y_train[train_idx])
    return model.predict(x_val)


def write_summary_md(path: Path, summary_df: pd.DataFrame, native_timesfm: bool, n_samples: int, cfg: RunConfig) -> None:
    lines = [
        "# Covariate Benchmark (TimesFM stage)",
        "",
        f"- samples: {n_samples}",
        f"- native_timesfm_available: {native_timesfm}",
        f"- benchmark_backend: ridge_proxy (alpha={cfg.ridge_alpha})",
        f"- window: up={cfg.upstream_km} km / down={cfg.downstream_km} km / -{cfg.time_before_min}..+{cfg.time_after_min} min @ {cfg.resolution_min} min",
        "",
        "| model | MAE (km/h) | RMSE (km/h) | congestion F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['mae_kmh']:.3f} | {row['rmse_kmh']:.3f} | {row['congestion_f1']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `baseline_no_covariate`: ridge over pre-incident context only.",
        "- `xreg_covariate`: ridge over context + 33 incident features + blockage decay + cyclic time covariates.",
        "- This stage is intentionally lightweight and reproducible. Drop in native TimesFM by extending `fit_predict`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline vs covariate ridge benchmark on accident speed series.")
    p.add_argument("--feature-csv", default=str(DEFAULT_FEATURE_CSV))
    p.add_argument("--metadata-parquet", default=None, help="Default: data/accident_samples_<R>m_metadata.parquet")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--run-id", default=None)
    p.add_argument("--upstream-km", type=int, default=25)
    p.add_argument("--downstream-km", type=int, default=10)
    p.add_argument("--time-before-min", type=int, default=30)
    p.add_argument("--time-after-min", type=int, default=145)
    p.add_argument("--spatial-resolution-km", type=int, default=1)
    p.add_argument("--resolution-min", type=int, default=5)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--min-before-coverage",
        type=float,
        default=0.0,
        help="Optional pre-input coverage threshold; default 0 disables this extra filter.",
    )
    p.add_argument(
        "--min-filled-ratio",
        type=float,
        default=0.8,
        help="Minimum full sample filled_ratio from metadata (default: 0.8).",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    cfg = RunConfig(
        upstream_km=args.upstream_km,
        downstream_km=args.downstream_km,
        time_before_min=args.time_before_min,
        time_after_min=args.time_after_min,
        spatial_resolution_km=args.spatial_resolution_km,
        resolution_min=args.resolution_min,
        val_split=args.val_split,
        random_state=args.random_state,
        ridge_alpha=args.ridge_alpha,
        max_samples=args.max_samples,
        min_before_coverage=args.min_before_coverage,
        min_filled_ratio=args.min_filled_ratio,
    )
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_csv(args.feature_csv, dtype={"task_id": str})
    feature_df["task_id"] = feature_df["task_id"].astype(str).str.strip()

    metadata_path = Path(args.metadata_parquet) if args.metadata_parquet else (PROJECT_ROOT / "data" / f"accident_samples_{cfg.resolution_min}m_metadata.parquet")
    if not metadata_path.exists():
        raise SystemExit(f"metadata parquet not found: {metadata_path}")
    metadata_df = pd.read_parquet(metadata_path)
    metadata_df["accident_id"] = metadata_df["accident_id"].astype(str).str.strip()

    task_ids, context_matrix, target_matrix, covariate_matrix, n_after = build_samples(cfg, feature_df, metadata_df)
    LOGGER.info("samples=%d | n_before=%d | n_after=%d", len(task_ids), cfg.n_before, n_after)

    idx = np.arange(len(task_ids))
    train_idx, val_idx = train_test_split(idx, test_size=cfg.val_split, random_state=cfg.random_state)

    y_true = target_matrix[val_idx]
    y_pred_base = fit_predict(context_matrix, target_matrix, train_idx, val_idx, cfg.ridge_alpha, cfg.random_state)
    y_pred_cov = fit_predict(covariate_matrix, target_matrix, train_idx, val_idx, cfg.ridge_alpha, cfg.random_state)

    base_metrics = evaluate_predictions(y_true, y_pred_base)
    cov_metrics = evaluate_predictions(y_true, y_pred_cov)
    summary_df = pd.DataFrame([
        {"model": "baseline_no_covariate", **base_metrics},
        {"model": "xreg_covariate", **cov_metrics},
    ])

    val_task_ids = [task_ids[i] for i in val_idx]
    per_sample_rows = []
    for name, preds in (("baseline_no_covariate", y_pred_base), ("xreg_covariate", y_pred_cov)):
        for i, tid in enumerate(val_task_ids):
            m = evaluate_predictions(y_true[i : i + 1], preds[i : i + 1])
            per_sample_rows.append({"task_id": tid, "model": name, **m})
    per_sample = pd.DataFrame(per_sample_rows)

    summary_csv = output_dir / "comparison_summary.csv"
    per_sample_csv = output_dir / "per_sample_metrics.csv"
    report_md = output_dir / "comparison_summary.md"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    per_sample.to_csv(per_sample_csv, index=False, encoding="utf-8-sig")
    write_summary_md(
        path=report_md,
        summary_df=summary_df,
        native_timesfm=importlib.util.find_spec("timesfm") is not None,
        n_samples=len(task_ids),
        cfg=cfg,
    )
    (output_dir / "run_config.json").write_text(
        json.dumps({"run_id": run_id, "config": asdict(cfg)}, indent=2),
        encoding="utf-8",
    )

    print(f"\nSummary:\n{summary_df.to_string(index=False)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
