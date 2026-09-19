# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.26",
#     "pandas>=2.2",
#     "pyarrow>=15",
# ]
# ///
"""Render the frozen-v7 aggregate and predeclared case-study paper figures.

Only formally persisted v7 artifacts, their hash-locked target/metadata inputs,
and the predeclared case-study rule are read. The script does not refit, tune,
rank, or select any model from temporal outcomes.

Example:
    uv run scripts/visualization/plot_cnn_refinement_v7.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V7_DIR = PROJECT_ROOT / "reports" / "paper_trb_2026" / "cnn_refinement_v7"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "reports" / "paper_trb_2026" / "figures" / "cnn_refinement_v7"
)
CASE_PROVENANCE_PATH = PROJECT_ROOT / "paper" / "TRB_MODEL_DEVELOPMENT_DRAFT.md"

# Exact canonical exports from scripts/lib/visualization.py. Keeping the values
# local avoids import-path differences between Windows and Colab script launch.
SPEED_MIN = 0.0
SPEED_MAX = 140.0
SPEED_CMAP = "RdYlGn"

CASE_ORDER = ("N1", "N3")
PREDECLARED_MEDIUM_CASES = {"N1": "2025051500", "N3": "2025030675"}
CASE_SELECTION_RULE = (
    "Model-independent temporal incidents with a nonempty origin contour were "
    "ranked within freeway by origin-contour cell count, divided into equal-count "
    "low/medium/high terciles, and the case nearest each tercile median was fixed; "
    "ties used proximity to median valid coverage and then accident ID. This compact "
    "panel uses the predeclared medium-tercile case for each freeway."
)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    unit: str
    higher_is_better: bool
    decimals: int


@dataclass(frozen=True)
class CaseBundle:
    accident_ids: np.ndarray
    target_kmh: np.ndarray
    valid_mask: np.ndarray
    origin_cc_mask: np.ndarray
    space_offsets_km: np.ndarray
    resolution_min: int
    xgboost_prediction_kmh: np.ndarray
    spatial_blend_prediction_kmh: np.ndarray
    matched_blend_prediction_kmh: np.ndarray
    metadata: pd.DataFrame
    blends_path: Path
    targets_path: Path
    metadata_path: Path


METRICS = (
    MetricSpec("origin_cc_MAE@20", "Origin-connected\nMAE@20", "km/h", False, 2),
    MetricSpec("cell_MAE", "Cell MAE", "km/h", False, 3),
    MetricSpec("impact_time_AE@20", "Impact-time\nAE@20", "min", False, 2),
    MetricSpec("recovery_AE@10", "Recovery\nAE@10", "min", False, 2),
    MetricSpec("congestion_F1@20", "Congestion\nF1@20", "score", True, 3),
)

MODEL_ORDER = (
    "xgboost",
    "spatial_cnn",
    "matched_pointwise_cnn",
    "spatial_blend",
    "matched_blend",
)

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "spatial_cnn": "Unblended XGBoost→spatial-CNN refiner",
    "matched_pointwise_cnn": "Unblended XGBoost→matched 1×1-CNN refiner",
    "spatial_blend": "Spatial blend",
    "matched_blend": "Matched 1×1 blend",
}

PLOT_LABELS = {
    "xgboost": "XGBoost",
    "spatial_cnn": "Unblended XGBoost→\nspatial-CNN refiner",
    "matched_pointwise_cnn": "Unblended XGBoost→\nmatched 1×1-CNN refiner",
    "spatial_blend": "Spatial blend",
    "matched_blend": "Matched 1×1 blend",
}

# Okabe-Ito-inspired colors, paired with distinct marker shapes.
MODEL_COLORS = {
    "xgboost": "#0072B2",
    "spatial_cnn": "#009E73",
    "matched_pointwise_cnn": "#E69F00",
    "spatial_blend": "#CC79A7",
    "matched_blend": "#7A7A7A",
}

MODEL_MARKERS = {
    "xgboost": "s",
    "spatial_cnn": "o",
    "matched_pointwise_cnn": "^",
    "spatial_blend": "D",
    "matched_blend": "P",
}

REFERENCE_ORDER = ("xgboost", "matched_blend")
REFERENCE_LABELS = {
    "xgboost": "vs XGBoost",
    "matched_blend": "vs matched 1×1 blend",
}
REFERENCE_COLORS = {
    "xgboost": MODEL_COLORS["xgboost"],
    "matched_blend": MODEL_COLORS["matched_blend"],
}
REFERENCE_MARKERS = {"xgboost": "o", "matched_blend": "s"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.0,
            "figure.titlesize": 11.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "paper-trb-2026-cnn-refinement-v7",
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require_columns(frame: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def _validate_manifest_output(
    manifest: dict[str, object], key: str, expected_path: Path
) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or key not in outputs:
        raise ValueError(f"Manifest is missing outputs.{key}")
    record = outputs[key]
    if not isinstance(record, dict):
        raise ValueError(f"Manifest outputs.{key} is not an object")
    expected_hash = str(record.get("sha256", ""))
    actual_hash = _sha256(expected_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Frozen hash mismatch for {expected_path}: {actual_hash} != {expected_hash}"
        )


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object at {label}")
    return value


def _resolve_locked_record(record: dict[str, object], label: str) -> Path:
    raw_value = str(record.get("path", ""))
    if not raw_value:
        raise ValueError(f"Missing path in frozen record {label}")

    candidates = [Path(raw_value)]
    normalized = raw_value.replace("\\", "/")
    for marker in ("/output/", "/data/", "/reports/", "/paper/"):
        index = normalized.rfind(marker)
        if index >= 0:
            candidates.append(PROJECT_ROOT / normalized[index + 1 :])

    path = next(
        (candidate.resolve() for candidate in candidates if candidate.exists()), None
    )
    if path is None:
        raise FileNotFoundError(f"Could not resolve frozen record {label}: {raw_value}")

    expected_hash = str(record.get("sha256", ""))
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Frozen hash mismatch for {label}: {actual_hash} != {expected_hash}"
        )
    expected_size = int(record.get("size_bytes", -1))
    if expected_size >= 0 and path.stat().st_size != expected_size:
        raise ValueError(f"Frozen byte-size mismatch for {label}: {path}")
    return path


def _validate_case_provenance(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    required_fragments = (
        "### 5.6 Case studies and failure modes",
        "origin-contour cell count",
        "low/medium/high terciles",
        "2025051500",
        "2025030675",
    )
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise ValueError(f"Case-selection provenance changed; missing text: {missing}")


def _load_case_bundle(
    temporal_manifest: dict[str, object],
    temporal_decision: dict[str, object],
    temporal_dir: Path,
) -> CaseBundle:
    blends_path = temporal_dir / "v7_temporal_fixed_blends.npz"
    _validate_manifest_output(temporal_manifest, "blends", blends_path)

    inputs = _mapping(temporal_manifest.get("inputs"), "temporal_manifest.inputs")
    xgboost = _mapping(inputs.get("xgboost"), "temporal_manifest.inputs.xgboost")
    spatial = _mapping(
        inputs.get("spatial_cnn"), "temporal_manifest.inputs.spatial_cnn"
    )
    matched = _mapping(
        inputs.get("matched_pointwise_cnn"),
        "temporal_manifest.inputs.matched_pointwise_cnn",
    )
    target_records = [
        _mapping(xgboost.get("target_npz"), "xgboost.target_npz"),
        _mapping(spatial.get("target_npz"), "spatial_cnn.target_npz"),
        _mapping(matched.get("target_npz"), "matched_pointwise_cnn.target_npz"),
    ]
    target_hashes = {str(record.get("sha256", "")) for record in target_records}
    if len(target_hashes) != 1:
        raise ValueError("Frozen v7 model inputs do not share one target artifact")
    targets_path = _resolve_locked_record(target_records[0], "temporal target NPZ")
    for index, record in enumerate(target_records[1:], start=1):
        if (
            _resolve_locked_record(record, f"temporal target NPZ copy {index}")
            != targets_path
        ):
            raise ValueError("Frozen target paths differ across model inputs")

    metadata_record = _mapping(
        inputs.get("strata_metadata"), "temporal_manifest.inputs.strata_metadata"
    )
    metadata_path = _resolve_locked_record(metadata_record, "temporal strata metadata")

    blend_keys = {
        "accident_ids",
        "xgboost_prediction_kmh",
        "spatial_blend_prediction_kmh",
        "matched_blend_prediction_kmh",
        "alpha",
        "clip_kmh",
        "membership_sha256",
        "ordered_sha256",
        "frozen_config_sha256",
        "evaluator_sha256",
    }
    with np.load(blends_path, allow_pickle=False) as archive:
        missing = sorted(blend_keys - set(archive.files))
        if missing:
            raise ValueError(f"Frozen blend NPZ lacks keys: {missing}")
        blend_ids = archive["accident_ids"].astype(str)
        xgboost_prediction = archive["xgboost_prediction_kmh"].astype(np.float32)
        spatial_blend_prediction = archive["spatial_blend_prediction_kmh"].astype(
            np.float32
        )
        matched_blend_prediction = archive["matched_blend_prediction_kmh"].astype(
            np.float32
        )
        alpha = float(archive["alpha"].item())
        clip_kmh = archive["clip_kmh"].astype(float)
        blend_membership_hash = str(archive["membership_sha256"].item())
        blend_ordered_hash = str(archive["ordered_sha256"].item())
        blend_config_hash = str(archive["frozen_config_sha256"].item())
        blend_evaluator_hash = str(archive["evaluator_sha256"].item())

    target_keys = {
        "accident_ids",
        "target_kmh",
        "valid_mask",
        "origin_cc_mask",
        "space_offsets_m",
        "space_offsets_km",
        "resolution_min",
    }
    with np.load(targets_path, allow_pickle=False) as archive:
        missing = sorted(target_keys - set(archive.files))
        if missing:
            raise ValueError(f"Frozen target NPZ lacks keys: {missing}")
        target_ids = archive["accident_ids"].astype(str)
        target_kmh = archive["target_kmh"].astype(np.float32)
        valid_mask = archive["valid_mask"].astype(bool)
        origin_cc_mask = archive["origin_cc_mask"].astype(bool)
        space_offsets_m = archive["space_offsets_m"].astype(np.int64)
        space_offsets_km = archive["space_offsets_km"].astype(np.float64)
        resolution_min = int(archive["resolution_min"].item())

    if not np.array_equal(blend_ids, target_ids):
        raise ValueError("Frozen blend and target incident order differs")
    if len(target_ids) != 473 or len(set(target_ids.tolist())) != 473:
        raise ValueError("Expected 473 unique temporal incident IDs")
    expected_shape = target_kmh.shape
    if expected_shape != (473, 36, 30):
        raise ValueError(f"Unexpected target grid shape: {expected_shape}")
    arrays = (
        valid_mask,
        origin_cc_mask,
        xgboost_prediction,
        spatial_blend_prediction,
        matched_blend_prediction,
    )
    if any(array.shape != expected_shape for array in arrays):
        raise ValueError(
            "A frozen mask/prediction grid does not match the target shape"
        )
    if np.any(origin_cc_mask & ~valid_mask):
        raise ValueError("Origin-connected mask contains invalid cells")
    if not np.isfinite(target_kmh[valid_mask]).all():
        raise ValueError("Observed target contains non-finite valid cells")
    for name, prediction in (
        ("xgboost", xgboost_prediction),
        ("spatial_blend", spatial_blend_prediction),
        ("matched_blend", matched_blend_prediction),
    ):
        if not np.isfinite(prediction[valid_mask]).all():
            raise ValueError(f"{name} contains non-finite valid predictions")
        if np.any(
            (prediction[valid_mask] < SPEED_MIN) | (prediction[valid_mask] > SPEED_MAX)
        ):
            raise ValueError(f"{name} predictions exceed the frozen clipping bounds")

    if resolution_min != 5:
        raise ValueError(f"Expected 5-minute resolution, found {resolution_min}")
    if space_offsets_km.shape != (36,) or not np.all(np.diff(space_offsets_km) > 0):
        raise ValueError("Invalid spatial grid")
    if not np.allclose(space_offsets_m / 1000.0, space_offsets_km, atol=1e-6):
        raise ValueError("Spatial grids disagree between metres and kilometres")
    if not math.isclose(alpha, 0.35, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Unexpected frozen blend alpha: {alpha}")
    if not np.array_equal(clip_kmh, np.array([SPEED_MIN, SPEED_MAX])):
        raise ValueError(f"Unexpected frozen clipping bounds: {clip_kmh}")

    membership = _mapping(temporal_decision.get("membership"), "temporal membership")
    frozen_contract = _mapping(
        temporal_decision.get("frozen_contract"), "temporal frozen contract"
    )
    expected_membership_hash = str(membership.get("membership_sha256", ""))
    expected_ordered_hash = str(membership.get("ordered_sha256", ""))
    if blend_membership_hash != expected_membership_hash:
        raise ValueError("Blend membership hash differs from the sealed report")
    if blend_ordered_hash != expected_ordered_hash:
        raise ValueError("Blend ordered-membership hash differs from the sealed report")
    if blend_config_hash != str(frozen_contract.get("config_sha256", "")):
        raise ValueError("Blend frozen-config hash differs from the sealed report")
    if blend_evaluator_hash != str(frozen_contract.get("evaluator_sha256", "")):
        raise ValueError("Blend evaluator hash differs from the sealed report")

    metadata = pd.read_parquet(metadata_path)
    _require_columns(
        metadata,
        {"accident_id", "freeway", "direction", "mileage_km", "report_time"},
        metadata_path,
    )
    metadata = metadata.copy()
    metadata["accident_id"] = metadata["accident_id"].astype(str).str.strip()
    if metadata["accident_id"].duplicated().any():
        raise ValueError("Duplicate accident IDs in frozen metadata")
    metadata = metadata.set_index("accident_id", drop=False)

    id_to_index = {accident_id: index for index, accident_id in enumerate(target_ids)}
    for freeway in CASE_ORDER:
        accident_id = PREDECLARED_MEDIUM_CASES[freeway]
        if accident_id not in id_to_index:
            raise ValueError(
                f"Predeclared case {accident_id} is absent from temporal arrays"
            )
        if accident_id not in metadata.index:
            raise ValueError(f"Predeclared case {accident_id} is absent from metadata")
        if str(metadata.loc[accident_id, "freeway"]) != freeway:
            raise ValueError(f"Predeclared case {accident_id} has the wrong freeway")
        if not origin_cc_mask[id_to_index[accident_id]].any():
            raise ValueError(
                f"Predeclared case {accident_id} has an empty origin contour"
            )

    return CaseBundle(
        accident_ids=target_ids,
        target_kmh=target_kmh,
        valid_mask=valid_mask,
        origin_cc_mask=origin_cc_mask,
        space_offsets_km=space_offsets_km,
        resolution_min=resolution_min,
        xgboost_prediction_kmh=xgboost_prediction,
        spatial_blend_prediction_kmh=spatial_blend_prediction,
        matched_blend_prediction_kmh=matched_blend_prediction,
        metadata=metadata,
        blends_path=blends_path,
        targets_path=targets_path,
        metadata_path=metadata_path,
    )


def _validate_protocol_chain(
    development_decision: dict[str, object],
    development_manifest: dict[str, object],
    temporal_decision: dict[str, object],
    temporal_manifest: dict[str, object],
    development_decision_path: Path,
    temporal_decision_path: Path,
) -> None:
    if development_decision.get("passed") is not True:
        raise ValueError("The frozen v7 development gate did not pass")
    if development_decision.get("temporal_scoring_authorized") is not True:
        raise ValueError("The frozen v7 decision did not authorize temporal reporting")
    if temporal_decision.get("stage") != "sealed_confirmatory_reporting":
        raise ValueError("Temporal artifact is not sealed confirmatory reporting")
    if temporal_decision.get("decision") != "report_only_no_selection":
        raise ValueError("Temporal artifact is not report-only")
    if temporal_decision.get("passed") is not None:
        raise ValueError(
            "Temporal report must not contain a post-outcome pass/fail decision"
        )

    dev_contract = development_decision.get("frozen_contract")
    temporal_contract = temporal_decision.get("frozen_contract")
    if not isinstance(dev_contract, dict) or not isinstance(temporal_contract, dict):
        raise ValueError("Missing frozen contract in a v7 decision")
    for key in ("config_sha256", "evaluator_sha256", "alpha", "formula"):
        if dev_contract.get(key) != temporal_contract.get(key):
            raise ValueError(f"Development/temporal frozen-contract mismatch: {key}")
    if not math.isclose(float(temporal_contract["alpha"]), 0.35, abs_tol=1e-12):
        raise ValueError("Unexpected v7 blend alpha")

    _validate_manifest_output(
        development_manifest, "decision", development_decision_path
    )
    _validate_manifest_output(temporal_manifest, "decision", temporal_decision_path)


def _ordered_metrics(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {"model_id", "n_incidents"}
    for metric in METRICS:
        required.update({metric.key, f"{metric.key}_n"})
    _require_columns(frame, required, path)
    if frame["model_id"].duplicated().any():
        raise ValueError(f"Duplicate model rows in {path}")
    by_model = frame.set_index("model_id", drop=False)
    missing = [model for model in MODEL_ORDER if model not in by_model.index]
    extra = sorted(set(by_model.index.astype(str)) - set(MODEL_ORDER))
    if missing or extra:
        raise ValueError(f"Unexpected model set: missing={missing}, extra={extra}")
    ordered = pd.DataFrame([by_model.loc[model] for model in MODEL_ORDER]).reset_index(
        drop=True
    )
    if not (ordered["n_incidents"].astype(int) == 473).all():
        raise ValueError("Temporal metric table is not the frozen 473-incident cohort")
    for metric in METRICS:
        values = ordered[metric.key].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite aggregate values for {metric.key}")
        counts = ordered[f"{metric.key}_n"].astype(int).unique()
        if len(counts) != 1:
            raise ValueError(f"Metric eligibility counts vary by model: {metric.key}")
    ordered["display_label"] = ordered["model_id"].map(MODEL_LABELS)
    return ordered


def _ordered_comparisons(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    _require_columns(
        frame,
        {
            "candidate_id",
            "reference_id",
            "metric",
            "improvement_definition",
            "paired_n",
            "candidate_mean",
            "reference_mean",
            "improvement",
            "ci_95_low",
            "ci_95_high",
            "bootstrap_repetitions",
            "bootstrap_seed",
            "bootstrap_stratum",
            "aggregation",
        },
        path,
    )
    selected = frame[frame["candidate_id"].astype(str) == "spatial_blend"].copy()
    expected = {
        (reference, metric.key) for reference in REFERENCE_ORDER for metric in METRICS
    }
    actual = set(
        zip(selected["reference_id"].astype(str), selected["metric"].astype(str))
    )
    if actual != expected or len(selected) != 10:
        raise ValueError(
            f"Unexpected paired-comparison set: missing={sorted(expected - actual)}"
        )
    rows: list[pd.Series] = []
    for metric in METRICS:
        for reference in REFERENCE_ORDER:
            match = selected[
                (selected["reference_id"].astype(str) == reference)
                & (selected["metric"].astype(str) == metric.key)
            ]
            if len(match) != 1:
                raise ValueError(f"Expected one row for {reference}/{metric.key}")
            row = match.iloc[0]
            expected_definition = (
                "candidate_minus_reference"
                if metric.higher_is_better
                else "reference_minus_candidate"
            )
            if str(row["improvement_definition"]) != expected_definition:
                raise ValueError(f"Direction mismatch for {reference}/{metric.key}")
            if int(row["bootstrap_repetitions"]) != 2000:
                raise ValueError("Expected 2,000 frozen bootstrap repetitions")
            if int(row["bootstrap_seed"]) != 20260715:
                raise ValueError("Unexpected frozen bootstrap seed")
            if str(row["bootstrap_stratum"]) != "freeway":
                raise ValueError("Expected freeway-stratified bootstrap")
            rows.append(row)
    result = pd.DataFrame(rows).reset_index(drop=True)
    result["reference_label"] = result["reference_id"].map(REFERENCE_LABELS)
    result["candidate_label"] = MODEL_LABELS["spatial_blend"]
    result["ci_excludes_zero"] = result["ci_95_low"].astype(float) > 0.0
    return result


def _shade_rows(ax: plt.Axes, count: int) -> None:
    for index in range(count):
        if index % 2:
            ax.axhspan(index - 0.5, index + 0.5, color="#F3F3F3", zorder=0)


def _format(value: float, metric: MetricSpec, *, signed: bool = False) -> str:
    rounded = round(float(value), metric.decimals)
    if rounded == 0:
        rounded = 0.0
    prefix = "+" if signed and rounded > 0 else ""
    return f"{prefix}{rounded:.{metric.decimals}f}"


def _save(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    paths: list[Path] = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        metadata: dict[str, object] = {
            "Creator": "scripts/visualization/plot_cnn_refinement_v7.py",
            "Title": stem.replace("_", " ").title(),
        }
        if extension == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        if extension == "svg":
            metadata["Date"] = None
        fig.savefig(
            path,
            dpi=dpi if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        paths.append(path)
    return paths


def plot_metric_comparison(
    metrics: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    y = np.arange(len(MODEL_ORDER), dtype=float)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15.7, 5.5), sharey=True)

    for metric, ax in zip(METRICS, axes):
        _shade_rows(ax, len(MODEL_ORDER))
        values = metrics[metric.key].to_numpy(dtype=float)
        best = int(np.argmax(values) if metric.higher_is_better else np.argmin(values))
        span = float(np.ptp(values))
        if span <= 0:
            span = max(abs(float(values[0])) * 0.05, 0.01)
        left = float(values.min() - 0.13 * span)
        right = float(values.max() + 0.42 * span)
        if left >= 0:
            left = max(0.0, left)
        if metric.higher_is_better:
            right = min(1.0, right)
        ax.set_xlim(left, right)
        label_offset = 0.025 * (right - left)

        for index, (model, value) in enumerate(zip(MODEL_ORDER, values)):
            ax.scatter(
                value,
                y[index],
                s=58,
                marker=MODEL_MARKERS[model],
                facecolor=MODEL_COLORS[model],
                edgecolor="#252525" if index == best else "white",
                linewidth=1.15 if index == best else 0.65,
                zorder=3,
            )
            ax.text(
                value + label_offset,
                y[index],
                _format(value, metric),
                va="center",
                ha="left",
                fontsize=7.4,
                fontweight="bold" if index == best else "normal",
                color="#252525",
            )

        count = int(metrics[f"{metric.key}_n"].iloc[0])
        direction = "↑" if metric.higher_is_better else "↓"
        ax.set_title(f"{metric.title}\n(n={count})", pad=8)
        ax.set_xlabel(f"{metric.unit} {direction}")
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(MODEL_ORDER) - 0.5, -0.5)

    axes[0].set_yticks(y, [PLOT_LABELS[model] for model in MODEL_ORDER])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    fig.suptitle(
        "Frozen-v7 performance on the temporal evaluation period",
        y=0.995,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Point estimates are incident-macro means from the sealed 473-incident report. "
        "The outlined mark is the best point estimate within each metric; no temporal selection was performed.",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#4A4A4A",
    )
    fig.subplots_adjust(left=0.238, right=0.995, top=0.80, bottom=0.15, wspace=0.34)
    paths = _save(fig, output_dir, "temporal_five_metric_comparison", formats, dpi)
    plt.close(fig)
    return paths


def plot_paired_forest(
    comparisons: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    y = np.arange(len(REFERENCE_ORDER), dtype=float)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15.7, 4.55), sharey=True)

    for metric, ax in zip(METRICS, axes):
        panel = comparisons[comparisons["metric"].astype(str) == metric.key].copy()
        panel = panel.set_index("reference_id").loc[list(REFERENCE_ORDER)].reset_index()
        lows = panel["ci_95_low"].to_numpy(dtype=float)
        highs = panel["ci_95_high"].to_numpy(dtype=float)
        radius = max(float(np.max(np.abs(np.concatenate([lows, highs])))), 1e-9) * 1.32
        ax.set_xlim(-radius, radius)
        _shade_rows(ax, len(REFERENCE_ORDER))
        ax.axvline(0.0, color="#252525", linewidth=0.9, zorder=1)

        for index, row in panel.iterrows():
            reference = str(row["reference_id"])
            mean = float(row["improvement"])
            low = float(row["ci_95_low"])
            high = float(row["ci_95_high"])
            color = REFERENCE_COLORS[reference]
            excludes = low > 0.0
            ax.errorbar(
                mean,
                y[index],
                xerr=np.array([[mean - low], [high - mean]]),
                fmt=REFERENCE_MARKERS[reference],
                color=color,
                ecolor=color,
                markerfacecolor=color if excludes else "white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                markersize=6.4,
                elinewidth=1.25,
                capsize=2.8,
                zorder=3,
            )
            x_text = high + 0.035 * (2 * radius)
            if x_text > radius * 0.92:
                x_text = mean - 0.035 * (2 * radius)
                horizontal = "right"
            else:
                horizontal = "left"
            ax.text(
                x_text,
                y[index],
                _format(mean, metric, signed=True),
                va="center",
                ha=horizontal,
                fontsize=7.3,
                color="#252525",
            )

        ax.set_title(metric.title, pad=8)
        ax.set_xlabel(f"Improvement ({metric.unit})")
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(REFERENCE_ORDER) - 0.5, -0.5)

    axes[0].set_yticks(y, [REFERENCE_LABELS[item] for item in REFERENCE_ORDER])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    legend = [
        Line2D(
            [0],
            [0],
            marker=REFERENCE_MARKERS[reference],
            color=REFERENCE_COLORS[reference],
            markerfacecolor=REFERENCE_COLORS[reference],
            linewidth=1.2,
            markersize=6,
            label=REFERENCE_LABELS[reference],
        )
        for reference in REFERENCE_ORDER
    ]
    legend.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="#252525",
                markerfacecolor="#252525",
                linewidth=0,
                markersize=6,
                label="95% CI excludes zero",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#252525",
                markerfacecolor="white",
                linewidth=0,
                markersize=6,
                label="95% CI crosses zero",
            ),
        ]
    )
    fig.suptitle(
        "Spatial-blend paired improvement on the temporal evaluation period",
        y=0.995,
        fontweight="bold",
    )
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.25,
    )
    fig.text(
        0.5,
        0.006,
        "Positive values favor the spatial blend. Whiskers are freeway-stratified paired incident-bootstrap "
        "95% intervals (2,000 resamples; seed 20260715).",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#4A4A4A",
    )
    fig.subplots_adjust(left=0.155, right=0.995, top=0.76, bottom=0.23, wspace=0.33)
    paths = _save(fig, output_dir, "temporal_paired_improvement_forest", formats, dpi)
    plt.close(fig)
    return paths


def _case_panel_data(
    bundle: CaseBundle,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    id_to_index = {
        accident_id: index for index, accident_id in enumerate(bundle.accident_ids)
    }
    time_offsets_min = (
        np.arange(bundle.target_kmh.shape[2], dtype=float) * bundle.resolution_min
    )
    panel_specs = (
        ("observed", "Observed target", bundle.target_kmh, "observed"),
        ("xgboost", "XGBoost", bundle.xgboost_prediction_kmh, "xgboost"),
        (
            "spatial_blend",
            "Spatial blend",
            bundle.spatial_blend_prediction_kmh,
            "0.65*xgboost + 0.35*spatial_refiner",
        ),
        (
            "matched_1x1_blend",
            "Matched 1×1 blend",
            bundle.matched_blend_prediction_kmh,
            "0.65*xgboost + 0.35*matched_1x1_refiner",
        ),
    )
    source_frames: list[pd.DataFrame] = []
    selected: dict[str, dict[str, np.ndarray]] = {}

    space_grid, time_grid = np.meshgrid(
        bundle.space_offsets_km, time_offsets_min, indexing="ij"
    )
    for freeway in CASE_ORDER:
        accident_id = PREDECLARED_MEDIUM_CASES[freeway]
        index = id_to_index[accident_id]
        valid = bundle.valid_mask[index]
        origin_mask = bundle.origin_cc_mask[index]
        metadata = bundle.metadata.loc[accident_id]
        selected[freeway] = {
            "index": np.array(index),
            "valid_mask": valid,
            "origin_cc_mask": origin_mask,
            "time_offsets_min": time_offsets_min,
        }
        for panel_id, panel_label, values, formula in panel_specs:
            matrix = values[index]
            selected[freeway][panel_id] = matrix
            source_frames.append(
                pd.DataFrame(
                    {
                        "accident_id": accident_id,
                        "freeway": freeway,
                        "direction": str(metadata["direction"]),
                        "mileage_km": float(metadata["mileage_km"]),
                        "report_time": pd.Timestamp(
                            metadata["report_time"]
                        ).isoformat(),
                        "selection_tercile": "medium",
                        "selection_model_independent": True,
                        "panel_id": panel_id,
                        "panel_label": panel_label,
                        "prediction_formula": formula,
                        "space_offset_km": space_grid.ravel(),
                        "time_offset_min": time_grid.ravel(),
                        "valid_cell": valid.ravel(),
                        "origin_cc_cell": origin_mask.ravel(),
                        "speed_kmh": np.where(valid, matrix, np.nan).ravel(),
                    }
                )
            )
    return pd.concat(source_frames, ignore_index=True), selected


def _axis_extent(values: np.ndarray) -> tuple[float, float]:
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Heatmap axes require at least two one-dimensional values")
    step = float(np.median(np.diff(values)))
    return float(values[0] - step / 2), float(values[-1] + step / 2)


def plot_predeclared_cases(
    bundle: CaseBundle,
    selected: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    figure = plt.figure(figsize=(15.7, 7.5))
    grid = figure.add_gridspec(
        2,
        5,
        width_ratios=(1.0, 1.0, 1.0, 1.0, 0.045),
        hspace=0.24,
        wspace=0.17,
    )
    axes: list[list[plt.Axes]] = []
    shared_axis: plt.Axes | None = None
    for row in range(2):
        row_axes: list[plt.Axes] = []
        for column in range(4):
            axis = figure.add_subplot(
                grid[row, column], sharex=shared_axis, sharey=shared_axis
            )
            if shared_axis is None:
                shared_axis = axis
            row_axes.append(axis)
        axes.append(row_axes)
    colorbar_axis = figure.add_subplot(grid[:, 4])

    column_specs = (
        ("observed", "Observed target"),
        ("xgboost", "XGBoost"),
        (
            "spatial_blend",
            "Spatial blend\n0.65 XGBoost + 0.35 spatial refiner",
        ),
        (
            "matched_1x1_blend",
            "Matched 1×1 blend\n0.65 XGBoost + 0.35 matched 1×1 refiner",
        ),
    )
    color_map = plt.get_cmap(SPEED_CMAP).copy()
    color_map.set_bad("#E6E6E6")
    last_image = None
    metadata_notes: list[str] = []

    for row, freeway in enumerate(CASE_ORDER):
        accident_id = PREDECLARED_MEDIUM_CASES[freeway]
        case = selected[freeway]
        valid = case["valid_mask"].astype(bool)
        origin_mask = case["origin_cc_mask"].astype(bool)
        time_offsets_min = case["time_offsets_min"].astype(float)
        x_extent = _axis_extent(time_offsets_min)
        y_extent = _axis_extent(bundle.space_offsets_km)
        metadata = bundle.metadata.loc[accident_id]
        metadata_notes.append(
            f"{freeway} {accident_id}: {freeway}-{metadata['direction']}, "
            f"{float(metadata['mileage_km']):.1f} km, "
            f"dispatch {pd.Timestamp(metadata['report_time']):%Y-%m-%d %H:%M}"
        )

        for column, (panel_id, title) in enumerate(column_specs):
            axis = axes[row][column]
            matrix = np.where(valid, case[panel_id].astype(float), np.nan)
            last_image = axis.imshow(
                np.ma.masked_invalid(matrix),
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                extent=(*x_extent, *y_extent),
                cmap=color_map,
                vmin=SPEED_MIN,
                vmax=SPEED_MAX,
            )
            axis.axvline(0.0, color="#D55E00", linestyle="--", linewidth=1.0, alpha=0.9)
            axis.axhline(0.0, color="#252525", linestyle=":", linewidth=0.8, alpha=0.8)
            if panel_id == "observed":
                axis.contour(
                    time_offsets_min,
                    bundle.space_offsets_km,
                    origin_mask.astype(float),
                    levels=[0.5],
                    colors=["#252525"],
                    linewidths=0.8,
                )
            if row == 0:
                axis.set_title(title, fontsize=8.8, fontweight="bold", pad=7)
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("Minutes after dispatch")
            if column == 0:
                axis.set_ylabel(
                    f"{freeway} · {accident_id}\nDistance from incident (km)\n(upstream positive)"
                )
            else:
                axis.tick_params(labelleft=False)
            axis.set_xticks([0, 30, 60, 90, 120, 150])
            axis.set_yticks([-10, 0, 10, 20])
            axis.set_xlim(x_extent)
            axis.set_ylim(y_extent)

    if last_image is None:
        raise RuntimeError("No case heatmap was rendered")
    colorbar = figure.colorbar(last_image, cax=colorbar_axis)
    colorbar.set_label("Speed (km/h)")
    colorbar.set_ticks(np.arange(0, 141, 20))

    figure.suptitle(
        "Predeclared medium-tercile temporal case studies",
        y=0.995,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.952,
        "Cases were fixed by the model-independent origin-contour-size rule before interpretation of v7 outcomes.",
        ha="center",
        va="top",
        fontsize=8.0,
        color="#4A4A4A",
    )
    figure.text(
        0.5,
        0.048,
        "  |  ".join(metadata_notes),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#4A4A4A",
    )
    figure.text(
        0.5,
        0.018,
        "Black outline: observed origin-connected congestion contour. Gray cells: unavailable target cells. "
        "All panels share the fixed 0–140 km/h scale.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#4A4A4A",
    )
    figure.subplots_adjust(left=0.105, right=0.94, top=0.86, bottom=0.15)
    paths = _save(
        figure, output_dir, "temporal_predeclared_case_heatmaps", formats, dpi
    )
    plt.close(figure)
    return paths


def _write_manifest(
    output_dir: Path,
    inputs: list[Path],
    outputs: list[Path],
    formats: tuple[str, ...],
    dpi: int,
    temporal_decision: dict[str, object],
    case_bundle: CaseBundle,
) -> Path:
    path = output_dir / "figure_manifest.json"
    payload = {
        "schema_version": "cnn_refinement_v7_static_figures_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "path": _relative(Path(__file__)),
            "sha256": _sha256(Path(__file__)),
        },
        "scope": (
            "frozen-v7 aggregate reporting plus predeclared model-independent medium-tercile "
            "case heatmaps; no refitting, tuning, or temporal-outcome-based selection"
        ),
        "temporal_reporting_policy": temporal_decision.get("decision"),
        "model_labels": MODEL_LABELS,
        "uncertainty": (
            "freeway-stratified paired incident bootstrap; two-sided percentile 95% CI; "
            "2,000 resamples; seed 20260715"
        ),
        "dpi": dpi,
        "formats": list(formats),
        "case_selection": {
            "provenance_path": _relative(CASE_PROVENANCE_PATH),
            "provenance_sha256": _sha256(CASE_PROVENANCE_PATH),
            "provenance_section": "5.6 Case studies and failure modes",
            "rule": CASE_SELECTION_RULE,
            "compact_panel_tercile": "medium",
            "case_ids": PREDECLARED_MEDIUM_CASES,
            "model_outcomes_used_for_selection": False,
        },
        "case_heatmap_contract": {
            "panels": [
                "observed target",
                "XGBoost",
                "spatial blend = 0.65 XGBoost + 0.35 spatial refiner",
                "matched 1×1 blend = 0.65 XGBoost + 0.35 matched 1×1 refiner",
            ],
            "speed_scale_kmh": [SPEED_MIN, SPEED_MAX],
            "speed_colormap": SPEED_CMAP,
            "space_axis": "distance from incident in km; upstream positive",
            "time_axis": "minutes after dispatch at 5-minute resolution",
            "origin_contour": "observed origin-connected congestion contour; shown on observed panel",
        },
        "case_sources": {
            "frozen_blends": {
                "path": _relative(case_bundle.blends_path),
                "sha256": _sha256(case_bundle.blends_path),
            },
            "frozen_targets": {
                "path": _relative(case_bundle.targets_path),
                "sha256": _sha256(case_bundle.targets_path),
            },
            "frozen_metadata": {
                "path": _relative(case_bundle.metadata_path),
                "sha256": _sha256(case_bundle.metadata_path),
            },
        },
        "inputs": [
            {
                "path": _relative(item),
                "size_bytes": item.stat().st_size,
                "sha256": _sha256(item),
            }
            for item in inputs
        ],
        "outputs": [
            {
                "path": _relative(item),
                "size_bytes": item.stat().st_size,
                "sha256": _sha256(item),
            }
            for item in outputs
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render frozen-v7 aggregate and predeclared case-study paper figures."
    )
    parser.add_argument(
        "--v7-dir",
        type=Path,
        default=DEFAULT_V7_DIR,
        help="Directory containing frozen v7 development_gate and temporal_reporting artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for paper figures, source CSVs, and manifest.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf,svg",
        help="Comma-separated subset of png,pdf,svg.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution (minimum 72)."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    v7_dir = args.v7_dir.resolve()
    development_dir = v7_dir / "development_gate"
    temporal_dir = v7_dir / "temporal_reporting"
    output_dir = args.output_dir.resolve()
    formats = tuple(
        item.strip().lower() for item in args.formats.split(",") if item.strip()
    )
    invalid = sorted(set(formats) - {"png", "pdf", "svg"})
    if not formats or invalid:
        raise ValueError(f"Invalid --formats entries: {invalid}")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")

    development_decision_path = development_dir / "v7_development_decision.json"
    development_manifest_path = development_dir / "v7_development_manifest.json"
    temporal_decision_path = temporal_dir / "v7_temporal_decision.json"
    temporal_manifest_path = temporal_dir / "v7_temporal_manifest.json"
    metrics_path = temporal_dir / "v7_temporal_metrics.csv"
    comparisons_path = temporal_dir / "v7_temporal_bootstrap_comparisons.csv"

    inputs = [
        development_decision_path,
        development_manifest_path,
        temporal_decision_path,
        temporal_manifest_path,
        metrics_path,
        comparisons_path,
        CASE_PROVENANCE_PATH,
    ]
    missing = [path for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen-v7 inputs: {missing}")

    development_decision = _load_json(development_decision_path)
    development_manifest = _load_json(development_manifest_path)
    temporal_decision = _load_json(temporal_decision_path)
    temporal_manifest = _load_json(temporal_manifest_path)
    _validate_protocol_chain(
        development_decision,
        development_manifest,
        temporal_decision,
        temporal_manifest,
        development_decision_path,
        temporal_decision_path,
    )
    _validate_manifest_output(temporal_manifest, "metrics", metrics_path)
    _validate_manifest_output(temporal_manifest, "comparisons", comparisons_path)
    _validate_case_provenance(CASE_PROVENANCE_PATH)
    case_bundle = _load_case_bundle(temporal_manifest, temporal_decision, temporal_dir)
    inputs.extend(
        [case_bundle.blends_path, case_bundle.targets_path, case_bundle.metadata_path]
    )
    inputs = list(dict.fromkeys(path.resolve() for path in inputs))

    metrics = _ordered_metrics(pd.read_csv(metrics_path), metrics_path)
    comparisons = _ordered_comparisons(pd.read_csv(comparisons_path), comparisons_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    outputs: list[Path] = []

    metric_source_rows: list[dict[str, object]] = []
    for _, row in metrics.iterrows():
        for metric in METRICS:
            metric_source_rows.append(
                {
                    "model_id": row["model_id"],
                    "display_label": row["display_label"],
                    "metric": metric.key,
                    "value": float(row[metric.key]),
                    "eligible_n": int(row[f"{metric.key}_n"]),
                    "unit": metric.unit,
                    "higher_is_better": metric.higher_is_better,
                }
            )
    metric_source = output_dir / "temporal_five_metric_comparison_source.csv"
    pd.DataFrame(metric_source_rows).to_csv(metric_source, index=False)
    outputs.append(metric_source)
    outputs.extend(plot_metric_comparison(metrics, output_dir, formats, args.dpi))

    forest_source = output_dir / "temporal_paired_improvement_source.csv"
    comparisons.to_csv(forest_source, index=False)
    outputs.append(forest_source)
    outputs.extend(plot_paired_forest(comparisons, output_dir, formats, args.dpi))

    case_source_frame, selected_cases = _case_panel_data(case_bundle)
    case_source = output_dir / "temporal_predeclared_case_heatmaps_source.csv"
    case_source_frame.to_csv(case_source, index=False, float_format="%.6f")
    outputs.append(case_source)
    outputs.extend(
        plot_predeclared_cases(
            case_bundle, selected_cases, output_dir, formats, args.dpi
        )
    )

    manifest = _write_manifest(
        output_dir,
        inputs,
        outputs,
        formats,
        args.dpi,
        temporal_decision,
        case_bundle,
    )
    outputs.append(manifest)

    print(f"Wrote {len(outputs)} frozen-v7 figure artifacts to {output_dir}")
    for path in outputs:
        print(f"  {_relative(path)}  sha256={_sha256(path)}")


if __name__ == "__main__":
    main()
