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
"""Shared harness for the paper baseline models (`scripts/baselines/`).

Why this module exists
----------------------
The whole point of the baselines is to be scored *identically* to the U-Net
candidates so their rows are directly comparable in Table 2 and feed
`scripts/experiments/select_composite.py`. Rather than re-implement the
protocol, this harness REUSES the canonical building blocks:

- data load + coverage filter + time-based split + hist/origin_cc alignment
  from `scripts/models/unet_matrix.py`,
- the fixed metric panel `panel_for_accident` from
  `scripts/experiments/run_loss_window_ablation.py`
  (origin_cc_MAE@{10,15,20,25}, cell_MAE, congestion_F1@20,
   impact_time_AE@20, recovery_AE@10).

So every baseline only has to produce predictions in km/h, shaped
`(N_split, n_space, n_after)` and aligned to a split's accident order; the
harness scores them on the shared panel and writes the standard outputs.

Shared evaluation protocol (matches the candidates / run_loss_window_ablation):
- matrix window upstream 25 km / downstream 10 km, t-30..+145 min @ 5 min,
- coverage filter `filled_ratio >= 0.8`,
- time split train <= 2024-11-30 / val <= 2025-02-28 / test after,
- origin_cc congestion mask, upstream loss/metric zone 0~+20 km,
- paper seeds {42, 7, 123}.

t0 feature set
--------------
Baselines that enable incident features use the same ordered immutable t0
vector as the feature-fusion U-Net. The canonical LSTM intentionally disables
this vector and all historical/static inputs; it receives only the six
pre-incident VD speed/validity steps.

Learned-baseline training objective
-----------------------------------
Independent per-cell models need an explicit choice of loss support. The
canonical past-30-minute LSTM holds full-field support fixed and compares
equal valid-cell weights, observed speed<60 weighting, and origin_cc AIC-mask
weighting. Historical full-field/upstream loss names remain implemented only
so earlier XGBoost/LSTM paper artifacts are reproducible; run manifests keep
all objective names distinct.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Canonical training/eval building blocks — identical geometry to the candidates.
from scripts.models.unet_matrix import (  # noqa: E402
    SPEED_MAX,
    TrainConfig,
    _upstream_end_idx,
    _upstream_start_idx,
    filter_and_subset,
    load_congestion_masks_aligned,
    load_hist_aligned,
)
from scripts.models.paper_protocol import (  # noqa: E402
    DEFAULT_ALLOWED_FREEWAYS,
    DEFAULT_SPLIT_SEED,
    artifact_manifest,
    cohort_manifest,
    configure_reproducibility,
    feature_schema_manifest,
    load_samples_artifact,
    parse_allowed_freeways,
    split_indices as build_split_indices,
)
# The fixed metric panel re-scored on every checkpoint in the ablation. Reusing
# it guarantees baseline metrics are computed exactly like the learned models'.
from scripts.experiments.run_loss_window_ablation import (  # noqa: E402
    EVAL_WINDOWS_KM,
    PRIMARY_EVAL_WINDOW_KM,
    panel_for_accident,
)
from scripts.lib.windows import SampleWindow  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_MASK_NPZ = DATA_ROOT / "congestion_masks_origin_cc_5m.npz"
FEATURES_CSV = PROJECT_ROOT / "output" / "feature_engineering" / "incident_features_33.csv"
DEFAULT_SAMPLES_NPZ = DATA_ROOT / "accident_samples_5m.npz"
DEFAULT_METADATA_PARQUET = DATA_ROOT / "accident_samples_5m_metadata.parquet"
DEFAULT_HISTORICAL_NPZ = DATA_ROOT / "historical_matrices_5m.npz"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "models" / "baselines"
PAPER_ARTIFACT_ROOT = DATA_ROOT / "paper_trb_2026"
PAPER_HISTORICAL_NPZ = PAPER_ARTIFACT_ROOT / "historical_matrices_5m_pre20240401_n1n3.npz"
PAPER_MASK_NPZ = PAPER_ARTIFACT_ROOT / "congestion_masks_origin_cc_5m_pre20240401_n1n3.npz"
PAPER_FEATURES_CSV = (
    PROJECT_ROOT / "output" / "feature_engineering" / "incident_features_paper_trb_2026.csv"
)

# Fixed matrix window for ALL baselines (identical to run_loss_window_ablation).
FIXED_UPSTREAM_KM = 25
FIXED_DOWNSTREAM_KM = 10
FIXED_TIME_BEFORE = 30
FIXED_TIME_AFTER = 145
SPATIAL_RES_KM_LOCAL = 1
RES_MIN = 5

# Locked loss/metric spec.
CONGESTION_WEIGHT = 4.0
SECONDARY_CONGESTION_WEIGHT = 2.0
CONGESTION_UPSTREAM_KM = 20

# Paper seeds (PAPER_ROADMAP §2 / Table 2 footnote).
PAPER_SEEDS = (42, 7, 123)


def add_protocol_arguments(parser) -> None:
    """Add the shared paper cohort/artifact/split CLI contract."""
    parser.add_argument("--samples-npz", default=str(DEFAULT_SAMPLES_NPZ))
    parser.add_argument("--samples-metadata-parquet", default=str(DEFAULT_METADATA_PARQUET))
    parser.add_argument(
        "--historical-matrices-npz",
        default=str(PAPER_HISTORICAL_NPZ),
        help="Causal pre-study historical matrices built by the paper Colab notebook.",
    )
    parser.add_argument(
        "--mask-npz",
        default=str(PAPER_MASK_NPZ),
        help="origin_cc masks regenerated from --historical-matrices-npz.",
    )
    parser.add_argument(
        "--feature-csv",
        default=str(PAPER_FEATURES_CSV),
        help="Immutable dispatch-time feature table shared with feature-fusion U-Net.",
    )
    parser.add_argument(
        "--allowed-freeways",
        default=",".join(DEFAULT_ALLOWED_FREEWAYS),
        help="Explicit paper cohort allow-list (locked default: N1,N3).",
    )
    parser.add_argument(
        "--split-mode",
        choices=["time", "random"],
        default="time",
        help="time is the primary analysis; random is a same-count robustness split.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Fixed random-split seed, separate from model training seeds.",
    )
    parser.add_argument("--train-end-date", default="2024-11-30")
    parser.add_argument("--val-end-date", default="2025-02-28")
    parser.add_argument(
        "--allow-missing-artifacts",
        action="store_true",
        help="Diagnostic only: do not fail when aligned historical/mask IDs are missing.",
    )
    parser.add_argument(
        "--deterministic-mode",
        choices=["warn", "error", "off"],
        default="warn",
        help="Deterministic-kernel policy for learned baselines.",
    )


def protocol_kwargs_from_args(args) -> dict:
    """Map shared CLI arguments to ``load_baseline_data`` keyword arguments."""
    return {
        "samples_npz": args.samples_npz,
        "samples_metadata_parquet": args.samples_metadata_parquet,
        "historical_matrices_npz": args.historical_matrices_npz,
        "mask_npz": args.mask_npz,
        "feature_csv": args.feature_csv,
        "allowed_freeways": args.allowed_freeways,
        "split_mode": args.split_mode,
        "split_seed": args.split_seed,
        "train_end_date": args.train_end_date,
        "val_end_date": args.val_end_date,
        "strict_artifacts": not args.allow_missing_artifacts,
    }

# ────────────────────────────────────────────────────────────────────────────
# Loaded data bundle
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class BaselineData:
    """Everything a baseline needs, loaded once and shared across seeds."""

    matrices: np.ndarray            # (N, n_space, n_time) observed VD km/h, NaN allowed
    ids: np.ndarray                 # (N,) accident id (str)
    hist: np.ndarray | None         # (N, n_space, n_time) historical baseline km/h, NaN allowed
    cong_by_id: dict[str, np.ndarray]  # acc_id -> (n_space, n_after) uint8 origin_cc mask (post cols)
    feat: np.ndarray                # (N, n_feat) float32 t0 features (raw, NaN-free)
    feat_names: list[str]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    base_cfg: TrainConfig
    target_window: SampleWindow
    up_start: int                   # accident-point row index (downstream_km cells)
    up_end: int                     # exclusive cap row for the 0~+20 km upstream zone
    paper_protocol: dict            # cohort/split/artifact/feature manifests

    @property
    def n_space(self) -> int:
        return self.target_window.n_space

    @property
    def n_time(self) -> int:
        return self.target_window.n_time

    @property
    def n_before(self) -> int:
        return self.target_window.time_before_min // self.target_window.resolution_min

    @property
    def n_after(self) -> int:
        return self.n_time - self.n_before

    def split_indices(self, split: str) -> np.ndarray:
        return {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}[split]

    def post_observed(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (target_kmh, valid) for the post-accident block of accident `idx`."""
        post = self.matrices[idx][:, self.n_before:]
        valid = (~np.isnan(post)).astype(np.float32)
        target = np.nan_to_num(post, nan=0.0).astype(np.float32)
        return target, valid

    def cong_mask(self, idx: int) -> np.ndarray:
        cc = self.cong_by_id.get(str(self.ids[idx]))
        if cc is None:
            return np.zeros((self.n_space, self.n_after), dtype=np.uint8)
        return cc


def load_t0_features(
    ids: np.ndarray,
    feature_csv: str | Path = FEATURES_CSV,
    *,
    return_manifest: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[np.ndarray, list[str], dict]:
    """Align the feature-fusion U-Net's exact ordered t0 vector to ``ids``.

    Columns are retained in artifact order, excluding only the ID and QC-only
    ``avail_mainline`` field. Values unavailable at dispatch and their paired
    availability indicators are zeroed exactly as in
    ``unet_feature_fusion._apply_feature_time_mask``. No baseline-only cyclical
    features or whole-dataset mean imputation are added.
    """
    feature_path = Path(feature_csv)
    if not feature_path.exists():
        raise SystemExit(
            f"t0 feature CSV not found: {feature_path}\n"
            f"Build it: uv run scripts/features/build_incident_features.py"
        )
    df = pd.read_csv(feature_path, encoding="utf-8-sig")
    if "task_id" not in df.columns:
        raise ValueError(f"feature table has no task_id column: {feature_path}")
    df["task_id"] = df["task_id"].astype(str).str.strip()
    duplicated = df["task_id"].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "task_id"].head(5).tolist()
        raise ValueError(f"duplicate task_id values in feature table, examples={examples}")

    feature_columns = [c for c in df.columns if c not in {"task_id", "avail_mainline"}]
    values = df.set_index("task_id")[feature_columns].apply(pd.to_numeric, errors="coerce")
    t0_unavailable = {
        "injuries", "fatalities", "lane_blockage_rate", "special_accident",
        "accident_A1", "accident_A2", "accident_A3",
    }
    for col in t0_unavailable | {f"avail_{c}" for c in t0_unavailable}:
        if col in values.columns:
            values[col] = 0.0

    aligned = values.reindex([str(acc_id).strip() for acc_id in ids])
    missing_rows = int(aligned.isna().all(axis=1).sum())
    if missing_rows:
        raise ValueError(
            f"{missing_rows}/{len(ids)} eligible accidents are absent from feature table {feature_path}"
        )
    feat = aligned.fillna(0.0).to_numpy(dtype=np.float32)
    manifest = feature_schema_manifest(df, feature_columns, feature_path)
    manifest["feature_stage"] = "t0"
    manifest["zeroed_at_t0"] = sorted(
        col
        for col in t0_unavailable | {f"avail_{c}" for c in t0_unavailable}
        if col in feature_columns
    )
    manifest["baseline_only_derived_features"] = []
    if return_manifest:
        return feat, feature_columns, manifest
    return feat, feature_columns


def load_baseline_data(
    input_type: str = "vd+hist",
    mask_npz: str | Path = DEFAULT_MASK_NPZ,
    min_filled_ratio: float = 0.8,
    hist_tag: str = "",
    with_features: bool = True,
    *,
    samples_npz: str | Path = DEFAULT_SAMPLES_NPZ,
    samples_metadata_parquet: str | Path | None = DEFAULT_METADATA_PARQUET,
    historical_matrices_npz: str | Path | None = None,
    feature_csv: str | Path = FEATURES_CSV,
    allowed_freeways: str | tuple[str, ...] = DEFAULT_ALLOWED_FREEWAYS,
    split_mode: str = "time",
    split_seed: int = DEFAULT_SPLIT_SEED,
    train_end_date: str = "2024-11-30",
    val_end_date: str = "2025-02-28",
    strict_artifacts: bool = True,
) -> BaselineData:
    """Load one fully specified paper baseline bundle.

    Chronological and random runs use the same eligible N1/N3 cohort. The
    random mode changes only split membership and keeps the chronological
    counts. Explicit artifact paths and their hashes are persisted so a paper
    run cannot silently consume the legacy historical profile or feature table.
    """
    allowed = parse_allowed_freeways(allowed_freeways)
    samples_path = Path(samples_npz)
    metadata_path = (
        Path(samples_metadata_parquet)
        if samples_metadata_parquet is not None and str(samples_metadata_parquet).strip()
        else None
    )
    mask_path = Path(mask_npz)
    historical_path = (
        Path(historical_matrices_npz)
        if historical_matrices_npz is not None and str(historical_matrices_npz).strip()
        else DATA_ROOT / f"historical_matrices_{RES_MIN}m{('_' + hist_tag) if hist_tag else ''}.npz"
    )
    feature_path = Path(feature_csv)

    base_cfg = TrainConfig(
        upstream_km=FIXED_UPSTREAM_KM, downstream_km=FIXED_DOWNSTREAM_KM,
        time_before_min=FIXED_TIME_BEFORE, time_after_min=FIXED_TIME_AFTER,
        spatial_resolution_km=SPATIAL_RES_KM_LOCAL, resolution_min=RES_MIN,
        min_filled_ratio=min_filled_ratio,
        eval_protocol="v1", data_scope="dispatch_only", feature_set="t0",
        input_type=input_type, loss_name="upstream_congestion_weighted",
        congestion_weight=CONGESTION_WEIGHT, congestion_upstream_km=CONGESTION_UPSTREAM_KM,
        congestion_mask_npz=str(mask_path),
        train_end_date=train_end_date, val_end_date=val_end_date,
    )
    target_window = SampleWindow(
        upstream_km=base_cfg.upstream_km, downstream_km=base_cfg.downstream_km,
        time_before_min=base_cfg.time_before_min, time_after_min=base_cfg.time_after_min,
        spatial_resolution_km=base_cfg.spatial_resolution_km, resolution_min=base_cfg.resolution_min,
    )
    samples = load_samples_artifact(samples_path, metadata_path)
    matrices, ids = filter_and_subset(
        samples, target_window, 0.0, min_filled_ratio, allowed_freeways=allowed
    )
    cohort = cohort_manifest(samples.accident_ids, ids, samples.metadata, allowed)
    print(
        f"Eligible cohort: {len(matrices)}/{len(samples)} | "
        f"freeways={list(allowed)} | filled_ratio>={min_filled_ratio} | "
        f"counts={cohort['retained_freeway_counts']}"
    )

    def _assert_npz_membership(path: Path, label: str) -> None:
        if not strict_artifacts:
            return
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(str(path), allow_pickle=False) as arr:
            if "accident_ids" not in arr.files:
                raise ValueError(f"{label} NPZ has no accident_ids key: {path}")
            available = {str(value).strip() for value in arr["accident_ids"]}
        missing = [str(value) for value in ids if str(value) not in available]
        if missing:
            raise ValueError(
                f"{label} NPZ is missing {len(missing)} eligible accidents; examples={missing[:5]}"
            )

    hist = None
    if "hist" in input_type:
        _assert_npz_membership(historical_path, "historical matrices")
        hist = load_hist_aligned(
            ids,
            target_window,
            DATA_ROOT,
            RES_MIN,
            hist_tag,
            historical_matrices_npz=historical_path,
        )
        if strict_artifacts and np.isnan(hist).all(axis=(1, 2)).any():
            bad = ids[np.isnan(hist).all(axis=(1, 2))].tolist()
            raise ValueError(f"historical matrices are entirely empty for eligible IDs: {bad[:5]}")

    _assert_npz_membership(mask_path, "origin_cc mask")
    cong_aligned = load_congestion_masks_aligned(ids, target_window, mask_path)
    cong_by_id = {str(a): cong_aligned[i] for i, a in enumerate(ids)}
    print(f"origin_cc congested-cell share (all): {cong_aligned.mean():.4%} | input_type={input_type}")

    train_idx, val_idx, test_idx, split_manifest = build_split_indices(
        ids,
        samples.metadata,
        split_mode=split_mode,
        split_seed=split_seed,
        train_end=base_cfg.train_end_date,
        val_end=base_cfg.val_end_date,
    )
    print(
        f"Split mode={split_mode}: train={len(train_idx)} val={len(val_idx)} "
        f"test={len(test_idx)} | split_seed={split_seed}"
    )

    if with_features:
        feat, feat_names, feature_manifest = load_t0_features(
            ids, feature_path, return_manifest=True
        )
        feature_source_manifest_path = feature_path.with_name(
            f"{feature_path.stem}_manifest.json"
        )
        feature_source_manifest = None
        if feature_source_manifest_path.exists():
            feature_source_manifest = json.loads(
                feature_source_manifest_path.read_text(encoding="utf-8")
            )
        if strict_artifacts:
            if feature_source_manifest is None:
                raise FileNotFoundError(
                    f"immutable feature manifest not found: {feature_source_manifest_path}"
                )
            if not bool(feature_source_manifest.get("paper_valid")):
                raise ValueError(
                    f"feature manifest is not paper-valid: {feature_source_manifest_path}"
                )
            if not bool(feature_source_manifest.get("immutable_write")):
                raise ValueError(
                    "formal feature artifact was not written immutably: "
                    f"{feature_source_manifest_path}"
                )
            declared_hash = (
                feature_source_manifest.get("artifacts", {})
                .get("features", {})
                .get("sha256")
            )
            actual_hash = feature_manifest["artifact"]["sha256"]
            if not declared_hash or declared_hash != actual_hash:
                raise ValueError(
                    "feature CSV hash does not match its immutable manifest: "
                    f"declared={declared_hash}, actual={actual_hash}"
                )
            declared_columns = (
                feature_source_manifest.get("feature_schema", {})
                .get("model_feature_columns")
            )
            if declared_columns != feat_names:
                raise ValueError(
                    "feature CSV model-column order does not match its immutable manifest"
                )
            declared_rows = feature_source_manifest.get("feature_schema", {}).get("rows")
            if declared_rows != feature_manifest["row_count"]:
                raise ValueError(
                    "feature CSV row count does not match its immutable manifest: "
                    f"declared={declared_rows}, actual={feature_manifest['row_count']}"
                )
            weather_cfg = feature_source_manifest.get("configuration", {})
            if weather_cfg.get("weather_match_direction") != "backward":
                raise ValueError(
                    "formal feature manifest must use backward weather matching "
                    "(latest observation at or before dispatch)"
                )
        feature_manifest["source_manifest"] = {
            "artifact": artifact_manifest(feature_source_manifest_path),
            "paper_valid": (
                feature_source_manifest.get("paper_valid")
                if feature_source_manifest is not None else None
            ),
            "weather_configuration": (
                feature_source_manifest.get("configuration")
                if feature_source_manifest is not None else None
            ),
        }
        print(f"t0 features: {feat.shape[1]} dims -> {feat_names}")
    else:
        feat, feat_names = np.zeros((len(ids), 0), dtype=np.float32), []
        feature_manifest = None
        feature_source_manifest_path = None

    actual_metadata_path = (
        metadata_path
        if metadata_path is not None
        else samples_path.with_name(f"{samples_path.stem}_metadata.parquet")
    )
    artifacts = {
        "samples_npz": artifact_manifest(samples_path),
        "samples_metadata_parquet": artifact_manifest(actual_metadata_path),
        "historical_matrices_npz": (
            artifact_manifest(historical_path) if "hist" in input_type else None
        ),
        "congestion_mask_npz": artifact_manifest(mask_path),
        "feature_csv": artifact_manifest(feature_path) if with_features else None,
        "feature_manifest_json": (
            artifact_manifest(feature_source_manifest_path) if with_features else None
        ),
    }
    paper_protocol = {
        "protocol_id": "paper_trb_2026_v1",
        "cohort": cohort,
        "split": split_manifest,
        "split_membership_ids": {
            "train": [str(ids[int(i)]) for i in train_idx],
            "val": [str(ids[int(i)]) for i in val_idx],
            "test": [str(ids[int(i)]) for i in test_idx],
        },
        "artifacts": artifacts,
        "feature_schema": feature_manifest,
        "strict_artifacts": bool(strict_artifacts),
        "min_filled_ratio": float(min_filled_ratio),
        "model_window": {
            "upstream_km": FIXED_UPSTREAM_KM,
            "downstream_km": FIXED_DOWNSTREAM_KM,
            "time_before_min": FIXED_TIME_BEFORE,
            "time_after_min": FIXED_TIME_AFTER,
            "spatial_resolution_km": SPATIAL_RES_KM_LOCAL,
            "resolution_min": RES_MIN,
        },
    }

    return BaselineData(
        matrices=matrices, ids=ids, hist=hist, cong_by_id=cong_by_id,
        feat=feat, feat_names=feat_names,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        base_cfg=base_cfg, target_window=target_window,
        up_start=_upstream_start_idx(base_cfg), up_end=_upstream_end_idx(base_cfg),
        paper_protocol=paper_protocol,
    )


# ────────────────────────────────────────────────────────────────────────────
# Congestion-weight scheme (shared by the learned baselines)
# ────────────────────────────────────────────────────────────────────────────


def congestion_weights(
    data: BaselineData,
    idx: int,
    outside_weight: float = 0.0,
    *,
    loss_name: str = "upstream_congestion_weighted",
) -> np.ndarray:
    """Per-cell loss weights `(n_space, n_after)` for accident `idx`.

    The canonical past-30-minute LSTM losses all supervise the full prediction
    field. ``full_field_speed60_weighted`` gives observed speed<60 cells weight
    4; ``full_field_aic_weighted`` gives origin_cc AIC-mask cells weight 4;
    every other valid cell has weight 1.

    Historical upstream-only names remain available for reproducibility of
    frozen paper artifacts. For those, cells outside the 0--20 km upstream
    focus receive ``outside_weight`` and every cell inside receives base 1.

    - ``upstream_congestion_weighted`` (and the full-field variant): origin_cc=4;
    - ``upstream_speed60_weighted``: observed speed<60 km/h=4;
    - ``upstream_aic_speed60_weighted``: origin_cc=4, other speed<60=2.

    For the hybrid, origin_cc takes precedence over the speed tier. Call with
    ``outside_weight=0`` for exact U-Net loss support or ``outside_weight=1``
    for full-field per-cell supervision.

    Target-validity masking is applied by the caller.
    """
    if outside_weight < 0:
        raise ValueError("outside_weight must be non-negative")
    if loss_name == "full_field_speed60_weighted":
        target, valid = data.post_observed(idx)
        speed60 = (target < 60.0) & (valid > 0)
        return np.where(speed60, CONGESTION_WEIGHT, 1.0).astype(np.float32)
    if loss_name == "full_field_aic_weighted":
        cc = data.cong_mask(idx) > 0
        return np.where(cc, CONGESTION_WEIGHT, 1.0).astype(np.float32)

    w = np.full((data.n_space, data.n_after), outside_weight, dtype=np.float32)
    s, e = data.up_start, data.up_end
    if loss_name in {"upstream_congestion_weighted", "full_field_congestion_weighted"}:
        cc = data.cong_mask(idx)[s:e, :] > 0
        focus_weight = np.where(cc, CONGESTION_WEIGHT, 1.0)
    elif loss_name == "upstream_speed60_weighted":
        target, valid = data.post_observed(idx)
        speed60 = (target[s:e, :] < 60.0) & (valid[s:e, :] > 0)
        focus_weight = np.where(speed60, CONGESTION_WEIGHT, 1.0)
    elif loss_name == "upstream_aic_speed60_weighted":
        target, valid = data.post_observed(idx)
        speed60 = (target[s:e, :] < 60.0) & (valid[s:e, :] > 0)
        cc = data.cong_mask(idx)[s:e, :] > 0
        focus_weight = np.where(
            cc,
            CONGESTION_WEIGHT,
            np.where(speed60, SECONDARY_CONGESTION_WEIGHT, 1.0),
        )
    else:
        raise ValueError(f"unsupported congestion-weighted loss: {loss_name}")
    w[s:e, :] = focus_weight.astype(np.float32)
    return w


def loss_design_manifest(loss_name: str) -> dict:
    """Machine-readable loss definition written into each baseline run."""
    designs = {
        "full_field_mae": {
            "support": "full_field",
            "base_weight": 1.0,
            "aic_weight": 1.0,
            "speed60_weight": 1.0,
            "overlap_rule": "not_applicable",
        },
        "cell_mae": {
            "support": "full_field",
            "base_weight": 1.0,
            "aic_weight": 1.0,
            "speed60_weight": 1.0,
            "overlap_rule": "not_applicable",
        },
        "full_field_congestion_weighted": {
            "support": "full_field",
            "base_weight": 1.0,
            "aic_weight": CONGESTION_WEIGHT,
            "speed60_weight": 1.0,
            "overlap_rule": "origin_cc",
        },
        "full_field_speed60_weighted": {
            "support": "full_field",
            "base_weight": 1.0,
            "aic_weight": None,
            "speed60_weight": CONGESTION_WEIGHT,
            "overlap_rule": "speed60_only",
        },
        "full_field_aic_weighted": {
            "support": "full_field",
            "base_weight": 1.0,
            "aic_weight": CONGESTION_WEIGHT,
            "speed60_weight": 1.0,
            "overlap_rule": "origin_cc_full_mask",
        },
        "upstream_congestion_weighted": {
            "support": "upstream_0_20_km",
            "base_weight": 1.0,
            "aic_weight": CONGESTION_WEIGHT,
            "speed60_weight": 1.0,
            "overlap_rule": "origin_cc",
        },
        "upstream_speed60_weighted": {
            "support": "upstream_0_20_km",
            "base_weight": 1.0,
            "aic_weight": None,
            "speed60_weight": CONGESTION_WEIGHT,
            "overlap_rule": "speed60_only",
        },
        "upstream_aic_speed60_weighted": {
            "support": "upstream_0_20_km",
            "base_weight": 1.0,
            "aic_weight": CONGESTION_WEIGHT,
            "speed60_weight": SECONDARY_CONGESTION_WEIGHT,
            "overlap_rule": "origin_cc_precedence",
        },
    }
    if loss_name not in designs:
        raise ValueError(f"unsupported loss design: {loss_name}")
    design = designs[loss_name]
    return {
        "loss_name": loss_name,
        "upstream_km": (
            CONGESTION_UPSTREAM_KM
            if design["support"] == "upstream_0_20_km"
            else None
        ),
        **design,
    }


# ────────────────────────────────────────────────────────────────────────────
# Panel scoring (reuses run_loss_window_ablation.panel_for_accident verbatim)
# ────────────────────────────────────────────────────────────────────────────

_NON_METRIC_COLS = ("acc_id", "split", "has_origin_cc", "has_speed60")


def score_predictions(
    pred_kmh: np.ndarray,
    split_indices: np.ndarray,
    data: BaselineData,
    split_name: str,
) -> tuple[list[dict], dict]:
    """Score a split's predictions on the shared fixed metric panel.

    `pred_kmh` is `(len(split_indices), n_space, n_after)` in km/h, aligned to
    `split_indices` order. Returns (per_accident_rows, aggregated_dict) with the
    SAME `{split}/<metric>` keys produced by run_loss_window_ablation.
    """
    if len(split_indices) == 0:
        return [], {f"{split_name}/n": 0, f"{split_name}/n_origin_cc": 0}
    if pred_kmh.shape[0] != len(split_indices):
        raise ValueError(f"pred rows {pred_kmh.shape[0]} != split size {len(split_indices)}")

    rows: list[dict] = []
    for k, idx in enumerate(split_indices):
        target, valid = data.post_observed(int(idx))
        cc = data.cong_mask(int(idx))
        pm = panel_for_accident(pred_kmh[k], target, valid, cc, data.up_start)
        pm["acc_id"] = str(data.ids[int(idx)])
        pm["split"] = split_name
        rows.append(pm)

    df = pd.DataFrame(rows)
    agg: dict[str, float] = {}
    for col in df.columns:
        if col in _NON_METRIC_COLS:
            continue
        agg[f"{split_name}/{col}"] = float(np.nanmean(df[col].to_numpy(dtype=float)))
    agg[f"{split_name}/n"] = int(len(rows))
    agg[f"{split_name}/n_origin_cc"] = int(df["has_origin_cc"].sum()) if "has_origin_cc" in df else 0
    return rows, agg


def objective_mae_kmh(
    pred_kmh: np.ndarray,
    split_indices: np.ndarray,
    data: BaselineData,
    loss_name: str,
) -> float:
    """Return the micro-averaged validation objective in km/h.

    This is the scalar used for learned-baseline checkpoint/reporting fields;
    unlike the old XGBoost implementation it is computed on validation targets,
    not the training rows.
    """
    if pred_kmh.shape[0] != len(split_indices):
        raise ValueError(f"pred rows {pred_kmh.shape[0]} != split size {len(split_indices)}")
    numerator = 0.0
    denominator = 0.0
    for k, idx_raw in enumerate(split_indices):
        idx = int(idx_raw)
        target, valid = data.post_observed(idx)
        if loss_name in {"cell_mae", "full_field_mae"}:
            weight = np.ones_like(valid, dtype=np.float32)
        else:
            outside_weight = 1.0 if loss_name.startswith("full_field_") else 0.0
            weight = congestion_weights(
                data,
                idx,
                outside_weight=outside_weight,
                loss_name=loss_name,
            )
        effective = valid * weight
        numerator += float((np.abs(pred_kmh[k] - target) * effective).sum())
        denominator += float(effective.sum())
    return numerator / denominator if denominator > 0 else float("nan")


# ────────────────────────────────────────────────────────────────────────────
# Output writing (standard layout + select_composite-compatible summary row)
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class BaselineResult:
    """One trained/scored baseline run (a single seed)."""

    model: str                      # e.g. "lstm_percell"
    seed: int
    loss_name: str                  # full-field/upstream congestion weighted | cell_mae
    mask: str                       # "origin_cc" | "none"
    val_rows: list[dict]
    val_agg: dict
    test_rows: list[dict] = field(default_factory=list)
    test_agg: dict = field(default_factory=dict)
    best_val_loss: float = float("nan")
    extra_config: dict = field(default_factory=dict)

    def summary_row(self) -> dict:
        """Flat row matching run_loss_window_ablation summary.csv / select_composite."""
        weighted = self.loss_name not in {"cell_mae", "full_field_mae"}
        design = loss_design_manifest(self.loss_name)
        protocol = self.extra_config.get("paper_protocol", {})
        split = protocol.get("split", {})
        row = {
            "cell_id": self.model,
            "seed": self.seed,
            "smoke": bool(self.extra_config.get("smoke", False)),
            "protocol_id": protocol.get("protocol_id"),
            "split_mode": split.get("mode"),
            "split_seed": split.get("split_seed"),
            "loss_name": self.loss_name,
            "mask": self.mask,
            "loss_window_km": (
                CONGESTION_UPSTREAM_KM
                if design.get("support") == "upstream_0_20_km"
                else np.nan
            ),
            "weight": max(
                float(design.get("aic_weight") or 0.0),
                float(design.get("speed60_weight") or 0.0),
            ) if weighted else np.nan,
            "aic_weight": design.get("aic_weight"),
            "speed60_weight": design.get("speed60_weight"),
            "best_val_loss": self.best_val_loss,
            **self.val_agg,
            **self.test_agg,
        }
        return row


def write_result(result: BaselineResult, run_id: str, output_root: Path | None = None) -> Path:
    """Write metrics_v1.json + per_accident.csv + run_config.json for one run.

    Layout: `output/models/baselines/<model>/<run_id>/`.
    """
    out_root = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
    out_dir = out_root / result.model / run_id
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    protocol = result.extra_config.get("paper_protocol", {})
    split = protocol.get("split", {})

    # per-accident metrics (val + test)
    per_acc = pd.DataFrame(result.val_rows + result.test_rows)
    per_acc.to_csv(eval_dir / "per_accident.csv", index=False)

    def _split_block(agg: dict, prefix: str) -> dict:
        out = {k[len(prefix) + 1:]: v for k, v in agg.items() if k.startswith(prefix + "/")}
        return out

    metrics_v1 = {
        "eval_protocol": "v1",
        "run_id": run_id,
        "model": result.model,
        "seed": result.seed,
        "protocol_id": protocol.get("protocol_id"),
        "split_mode": split.get("mode"),
        "split_seed": split.get("split_seed"),
        "loss_name": result.loss_name,
        "mask": result.mask,
        "congestion_weight": (
            CONGESTION_WEIGHT
            if result.loss_name not in {"cell_mae", "full_field_mae"}
            else 1.0
        ),
        "loss_design": loss_design_manifest(result.loss_name),
        "congestion_upstream_km": CONGESTION_UPSTREAM_KM,
        "best_val_loss": result.best_val_loss,
        "val": _split_block(result.val_agg, "val"),
        "test": _split_block(result.test_agg, "test"),
        "extra_config": result.extra_config,
    }
    (eval_dir / "metrics_v1.json").write_text(
        json.dumps(metrics_v1, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if protocol:
        (out_dir / "paper_protocol.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    split_summaries = split.get("splits", {})

    cfg = {
        "eval_protocol": "v1",
        "data_scope": "dispatch_only",
        "feature_set": "t0",
        "model": result.model,
        "seed": result.seed,
        "protocol_id": protocol.get("protocol_id"),
        "split_mode": split.get("mode"),
        "split_seed": split.get("split_seed"),
        "loss": result.loss_name,
        "mask": result.mask,
        "split_date_ranges": {
            name: [block.get("date_min"), block.get("date_max")]
            for name, block in split_summaries.items()
        },
        "matrix_window": {
            "upstream_km": FIXED_UPSTREAM_KM, "downstream_km": FIXED_DOWNSTREAM_KM,
            "time_before_min": FIXED_TIME_BEFORE, "time_after_min": FIXED_TIME_AFTER,
            "resolution_min": RES_MIN,
        },
        **result.extra_config,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir


def write_seed_summary(
    rows: list[dict], model: str, run_id: str, output_root: Path | None = None
) -> Path:
    """Write a per-seed summary.csv for one model (rows = summary_row() dicts)."""
    out_root = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
    out_dir = out_root / model / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def print_seed_table(rows: list[dict], primary: str | None = None) -> None:
    """Pretty per-seed metric print for the locked panel headline metrics."""
    primary = primary or f"val/origin_cc_MAE@{PRIMARY_EVAL_WINDOW_KM}"
    cols = [
        "cell_id", "seed", primary, "val/cell_MAE",
        "val/impact_time_AE@20", "val/recovery_AE@10", "val/congestion_F1@20",
    ]
    df = pd.DataFrame(rows)
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df[cols].round(3).to_string(index=False))


def make_run_id(prefix: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{prefix}" if prefix else ts


__all__ = [
    "BaselineData", "BaselineResult", "load_baseline_data", "load_t0_features",
    "add_protocol_arguments", "protocol_kwargs_from_args",
    "configure_reproducibility",
    "congestion_weights", "loss_design_manifest", "score_predictions", "objective_mae_kmh", "write_result", "write_seed_summary",
    "print_seed_table", "make_run_id",
    "PAPER_SEEDS", "CONGESTION_WEIGHT", "SECONDARY_CONGESTION_WEIGHT", "CONGESTION_UPSTREAM_KM", "SPEED_MAX",
    "EVAL_WINDOWS_KM", "PRIMARY_EVAL_WINDOW_KM", "DEFAULT_OUTPUT_ROOT", "DEFAULT_MASK_NPZ",
    "PAPER_HISTORICAL_NPZ", "PAPER_MASK_NPZ", "PAPER_FEATURES_CSV",
]


if __name__ == "__main__":
    # Self-test: load data, then score a trivial "predict the historical baseline"
    # forecast on val. Exercises the full load -> align -> panel-score path.
    import argparse

    p = argparse.ArgumentParser(description="Harness self-test (load + score hist baseline on val).")
    p.add_argument("--input-type", default="vd+hist")
    p.add_argument("--no-features", action="store_true")
    args = p.parse_args()

    data = load_baseline_data(input_type=args.input_type, with_features=not args.no_features)
    print(
        f"n_space={data.n_space} n_time={data.n_time} n_before={data.n_before} "
        f"n_after={data.n_after} up_start={data.up_start} up_end={data.up_end}"
    )
    if data.hist is None:
        raise SystemExit("self-test needs hist; run with --input-type vd+hist")

    val_idx = data.val_idx
    preds = np.empty((len(val_idx), data.n_space, data.n_after), dtype=np.float32)
    for k, idx in enumerate(val_idx):
        h = data.hist[idx][:, data.n_before:]
        fill = float(np.nanmean(h)) if np.isfinite(np.nanmean(h)) else 90.0
        preds[k] = np.nan_to_num(h, nan=fill)

    _, agg = score_predictions(preds, val_idx, data, "val")
    print("Historical-baseline val panel:")
    for key in (
        "val/origin_cc_MAE@20", "val/cell_MAE", "val/upstream_MAE@20",
        "val/impact_time_AE@20", "val/recovery_AE@10", "val/congestion_F1@20",
        "val/n", "val/n_origin_cc",
    ):
        print(f"  {key:28s} = {agg.get(key)}")
