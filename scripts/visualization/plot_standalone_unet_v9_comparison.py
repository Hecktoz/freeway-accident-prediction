# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.26",
#     "pandas>=2.2",
# ]
# ///
"""Render the standalone U-Net v9 five-metric paper comparison.

The primary figure keeps chronological development and temporal reporting
together.  If supplied, fixed-random robustness is rendered as a separate
figure because it has different fitting and comparator semantics.  Every
panel shows direct metric values and direction-adjusted improvement over
XGBoost; no cross-metric aggregate is computed.

Examples:
    uv run scripts/visualization/plot_standalone_unet_v9_comparison.py \
      --development-dir reports/.../development_evaluation \
      --temporal-dir reports/.../temporal_evaluation
    uv run scripts/visualization/plot_standalone_unet_v9_comparison.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.visualization import STYLE_REPORT  # noqa: E402


DEVELOPMENT_DECISION_SCHEMA = "standalone_unet_v9_development_decision_v1"
TEMPORAL_DECISION_SCHEMA = "standalone_unet_v9_one_use_evaluation_v1"
RANDOM_DECISION_SCHEMA = "standalone_unet_v9_random_robustness_decision_v1"
EVALUATION_MANIFEST_SCHEMAS = {
    "standalone_unet_v9_evaluation_manifest_v1",
    "standalone_unet_v9_random_robustness_manifest_v1",
}
SOURCE_SCHEMA = "standalone_unet_v9_comparison_figure_source_v2"
MANIFEST_SCHEMA = "standalone_unet_v9_comparison_figure_manifest_v2"

STUDY_SCOPE = {
    "freeways": ["N1", "N3"],
    "weather_policy": "no_weather_locked",
    "weather_information_used": False,
    "fixed_feature_schema_slots": 45,
    "non_weather_schema_slots": 35,
    "reserved_weather_schema_slots": 10,
    "reserved_weather_slots_required_value": 0.0,
}
NO_WEATHER_FIGURE_NOTE = (
    "N1/N3; no timestamped weather used—10 reserved weather value/availability "
    "slots fixed to zero."
)

XGBOOST_ID = "xgboost_ensemble"
POINTWISE_ID = "matched_pointwise_ensemble"
UNET_ID = "standalone_unet_ensemble"
MODEL_ORDER = (XGBOOST_ID, POINTWISE_ID, UNET_ID)
MODEL_LABELS = {
    XGBOOST_ID: "XGBoost",
    POINTWISE_ID: "Matched pointwise",
    UNET_ID: "Pure U-Net",
}
MODEL_COLORS = {
    XGBOOST_ID: "#6F6F6F",
    POINTWISE_ID: "#E69F00",
    UNET_ID: "#0072B2",
}
MODEL_HATCHES = {XGBOOST_ID: "//", POINTWISE_ID: "..", UNET_ID: ""}

PRIMARY_ALLOWED_MODELS = {
    XGBOOST_ID,
    POINTWISE_ID,
    UNET_ID,
    *{f"xgboost_seed{seed}" for seed in (42, 7, 123)},
    *{f"standalone_unet_seed{seed}" for seed in (42, 7, 123)},
    *{f"matched_pointwise_seed{seed}" for seed in (42, 7, 123)},
}
RANDOM_ALIASES = {
    XGBOOST_ID: XGBOOST_ID,
    "xgboost_three_seed_metric_mean": XGBOOST_ID,
    "xgboost_seed_mean": XGBOOST_ID,
    "xgboost_seed_metric_mean": XGBOOST_ID,
    UNET_ID: UNET_ID,
    "standalone_unet_three_seed_metric_mean": UNET_ID,
    "standalone_unet_seed_mean": UNET_ID,
}

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "paper_trb_2026"
    / "figures"
    / "standalone_unet_v9_comparison"
)
OUTPUT_BASENAMES = {
    "source": "standalone_unet_v9_comparison_source.csv",
    "primary": "standalone_unet_v9_primary_comparison",
    "random": "standalone_unet_v9_random_robustness",
    "manifest": "standalone_unet_v9_comparison_manifest.json",
}
FORMATS = ("png", "svg", "pdf")


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str
    higher_is_better: bool
    decimals: int


METRICS = (
    MetricSpec("origin_cc_MAE@20", "Origin-core MAE", "km/h", False, 2),
    MetricSpec("cell_MAE", "Cell MAE", "km/h", False, 2),
    MetricSpec("impact_time_AE@20", "Impact-time error", "min", False, 2),
    MetricSpec("recovery_AE@10", "Recovery-time error", "min", False, 2),
    MetricSpec("congestion_F1@20", "Congestion F1", "score", True, 3),
)
METRIC_BY_KEY = {metric.key: metric for metric in METRICS}


@dataclass(frozen=True)
class EvaluationBundle:
    role: str
    stage: str
    stage_label: str
    path: Path
    metrics_path: Path
    decision_path: Path
    manifest_path: Path
    metrics_sha256: str
    decision_sha256: str
    manifest_sha256: str
    decision: dict[str, Any]
    ensemble_rows: pd.DataFrame


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(value: Any, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return normalized


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _validate_manifest_artifact(
    manifest: Mapping[str, Any], *, name: str, path: Path
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get(name), Mapping
    ):
        raise ValueError(f"evaluation manifest lacks artifacts.{name}")
    declaration = artifacts[name]
    declared_path = Path(str(declaration.get("path", ""))).resolve()
    if declared_path != path.resolve():
        raise ValueError(f"evaluation manifest artifacts.{name}.path changed")
    if _require_hash(declaration.get("sha256"), f"artifacts.{name}.sha256") != _sha256(
        path
    ):
        raise ValueError(f"evaluation manifest artifacts.{name} hash changed")
    if int(declaration.get("size_bytes", -1)) != path.stat().st_size:
        raise ValueError(f"evaluation manifest artifacts.{name} size changed")


def _validate_metric_frame(
    frame: pd.DataFrame,
    *,
    role: str,
    expected_n: int,
) -> pd.DataFrame:
    required = {"model_id", "n_incidents"}
    for metric in METRICS:
        required.update({metric.key, f"{metric.key}_n"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"metrics.csv misses columns: {missing}")
    if frame["model_id"].astype(str).duplicated().any():
        raise ValueError("metrics.csv contains duplicate model IDs")

    frame = frame.copy()
    frame["model_id"] = frame["model_id"].astype(str)
    if role == "primary":
        unknown = sorted(set(frame["model_id"]) - PRIMARY_ALLOWED_MODELS)
        if unknown:
            raise ValueError(f"primary metrics.csv has unknown model IDs: {unknown}")
        indexed = frame.set_index("model_id", drop=False)
        missing_models = sorted(set(MODEL_ORDER) - set(indexed.index))
        if missing_models:
            raise ValueError(f"primary evaluation misses ensembles: {missing_models}")
        selected = indexed.loc[list(MODEL_ORDER)].copy()
    else:
        normalized: dict[str, pd.Series] = {}
        for row in frame.itertuples(index=False):
            source_id = str(row.model_id)
            canonical = RANDOM_ALIASES.get(source_id)
            if canonical is None:
                continue
            if canonical in normalized:
                raise ValueError(
                    f"random metrics has duplicate aliases for {canonical}"
                )
            normalized[canonical] = frame.loc[frame["model_id"] == source_id].iloc[0]
        if set(normalized) != {XGBOOST_ID, UNET_ID}:
            raise ValueError(
                "random robustness requires XGBoost and pure U-Net summaries"
            )
        selected = pd.DataFrame([normalized[XGBOOST_ID], normalized[UNET_ID]])
        selected["source_model_id"] = selected["model_id"].astype(str)
        selected["model_id"] = [XGBOOST_ID, UNET_ID]
        if POINTWISE_ID in set(frame["model_id"]):
            raise ValueError(
                "random robustness must not refit the matched pointwise control"
            )

    for _, row in selected.iterrows():
        model_id = str(row["model_id"])
        n = int(row["n_incidents"])
        if n != expected_n:
            raise ValueError(f"{model_id} n_incidents={n}; expected {expected_n}")
        for metric in METRICS:
            value = float(row[metric.key])
            count = int(row[f"{metric.key}_n"])
            if not math.isfinite(value) or count <= 0 or count > n:
                raise ValueError(f"{model_id}/{metric.key} is invalid")
            if metric.higher_is_better:
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{model_id}/{metric.key} is outside [0,1]")
            elif value < 0.0:
                raise ValueError(f"{model_id}/{metric.key} is negative")
    return selected.reset_index(drop=True)


def _load_evaluation(path: Path, *, stage: str) -> EvaluationBundle:
    root = path.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    metrics_path = root / "metrics.csv"
    decision_path = root / "decision.json"
    manifest_path = root / "manifest.json"
    for required in (metrics_path, decision_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    decision = _load_json(decision_path, f"{stage} decision")
    manifest = _load_json(manifest_path, f"{stage} evaluation manifest")
    if manifest.get("schema_version") not in EVALUATION_MANIFEST_SCHEMAS:
        raise ValueError(f"{stage} evaluation manifest schema changed")
    if str(manifest.get("stage", "")).lower() != stage:
        raise ValueError(f"{stage} evaluation manifest stage changed")
    if manifest.get("passed") is not decision.get("passed"):
        raise ValueError(f"{stage} decision/manifest pass state differs")
    _validate_manifest_artifact(manifest, name="metrics.csv", path=metrics_path)
    _validate_manifest_artifact(manifest, name="decision.json", path=decision_path)

    if stage == "development":
        role = "primary"
        stage_label = "Development"
        expected_schema = DEVELOPMENT_DECISION_SCHEMA
        expected_n = 324
    elif stage == "temporal":
        role = "primary"
        stage_label = "Temporal report"
        expected_schema = TEMPORAL_DECISION_SCHEMA
        expected_n = 473
    elif stage == "random_robustness":
        role = "random"
        stage_label = "Random robustness"
        expected_schema = RANDOM_DECISION_SCHEMA
        expected_n = 473
        if (
            decision.get("role") != "post_temporal_secondary_robustness_only"
            or int(decision.get("split_seed", -1)) != 20260713
            or decision.get("matched_pointwise_refit") is not False
            or decision.get("reported_separately_from_chronological") is not True
        ):
            raise ValueError("random robustness decision contract changed")
    else:
        raise ValueError(f"unsupported evaluation stage: {stage}")
    if decision.get("schema_version") != expected_schema:
        raise ValueError(f"{stage} decision schema changed")
    if str(decision.get("stage", "")).lower() != stage:
        raise ValueError(f"{stage} decision stage changed")
    if not isinstance(decision.get("passed"), bool):
        raise ValueError(f"{stage} decision passed must be boolean")
    for key in (
        "config_sha256",
        "protocol_sha256",
        "freeze_record_sha256",
        "backend_lock_sha256",
    ):
        _require_hash(decision.get(key), f"{stage} decision.{key}")
    rung = str(decision.get("selected_rung", "")).lower()
    if rung not in {"g", "gt", "gtr"}:
        raise ValueError(f"{stage} selected rung is invalid")
    metrics = pd.read_csv(metrics_path)
    ensemble_rows = _validate_metric_frame(metrics, role=role, expected_n=expected_n)
    return EvaluationBundle(
        role=role,
        stage=stage,
        stage_label=stage_label,
        path=root,
        metrics_path=metrics_path,
        decision_path=decision_path,
        manifest_path=manifest_path,
        metrics_sha256=_sha256(metrics_path),
        decision_sha256=_sha256(decision_path),
        manifest_sha256=_sha256(manifest_path),
        decision=decision,
        ensemble_rows=ensemble_rows,
    )


def _validate_bundle_chain(bundles: Sequence[EvaluationBundle]) -> None:
    development = bundles[0]
    if development.stage != "development":
        raise AssertionError("development bundle must be first")
    baseline = development.decision
    for bundle in bundles[1:]:
        for key in (
            "config_sha256",
            "protocol_sha256",
            "freeze_record_sha256",
            "backend_lock_sha256",
            "selected_rung",
        ):
            if bundle.decision.get(key) != baseline.get(key):
                raise ValueError(f"evaluation chain changed {key} at {bundle.stage}")
    has_temporal = any(bundle.stage == "temporal" for bundle in bundles)
    has_random = any(bundle.stage == "random_robustness" for bundle in bundles)
    if (has_temporal or has_random) and (
        development.decision.get("passed") is not True
        or development.decision.get("authorize_full_refit_and_temporal") is not True
    ):
        raise PermissionError(
            "downstream reports exist without a passed development gate"
        )
    if has_random and not has_temporal:
        raise PermissionError(
            "random robustness must remain downstream of temporal report"
        )


def _source_rows(bundles: Sequence[EvaluationBundle]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        indexed = bundle.ensemble_rows.set_index("model_id", drop=False)
        model_ids = MODEL_ORDER if bundle.role == "primary" else MODEL_ORDER
        for metric in METRICS:
            xgb_value = float(indexed.loc[XGBOOST_ID, metric.key])
            for model_id in model_ids:
                available = model_id in indexed.index
                if available:
                    source = indexed.loc[model_id]
                    value = float(source[metric.key])
                    metric_n = int(source[f"{metric.key}_n"])
                    n_incidents = int(source["n_incidents"])
                    improvement = (
                        value - xgb_value
                        if metric.higher_is_better
                        else xgb_value - value
                    )
                    source_model_id = str(source.get("source_model_id", model_id))
                    note = ""
                else:
                    value = improvement = np.nan
                    metric_n = 0
                    n_incidents = 473
                    source_model_id = ""
                    note = "Not refit in post-temporal random robustness protocol"
                rows.append(
                    {
                        "schema_version": SOURCE_SCHEMA,
                        "analysis_group": (
                            "chronological_primary"
                            if bundle.role == "primary"
                            else "random_robustness_secondary"
                        ),
                        "stage": bundle.stage,
                        "stage_label": bundle.stage_label,
                        "model_id": model_id,
                        "source_model_id": source_model_id,
                        "model_label": MODEL_LABELS[model_id],
                        "metric": metric.key,
                        "metric_label": metric.label,
                        "unit": metric.unit,
                        "direction": (
                            "higher_is_better"
                            if metric.higher_is_better
                            else "lower_is_better"
                        ),
                        "available": available,
                        "n_incidents": n_incidents,
                        "metric_n": metric_n,
                        "value": value,
                        "xgboost_value": xgb_value,
                        "improvement_vs_xgboost": improvement,
                        "improvement_definition": (
                            "candidate_minus_xgboost"
                            if metric.higher_is_better
                            else "xgboost_minus_candidate"
                        ),
                        "protocol_note": note,
                        "evaluation_passed": bool(bundle.decision["passed"]),
                        "selected_rung": str(bundle.decision["selected_rung"]),
                        "source_metrics_csv_sha256": bundle.metrics_sha256,
                        "source_decision_sha256": bundle.decision_sha256,
                        "source_manifest_sha256": bundle.manifest_sha256,
                        "freeway_scope": "N1/N3",
                        "weather_policy": STUDY_SCOPE["weather_policy"],
                        "weather_information_used": STUDY_SCOPE[
                            "weather_information_used"
                        ],
                        "reserved_weather_schema_slots": STUDY_SCOPE[
                            "reserved_weather_schema_slots"
                        ],
                        "reserved_weather_slots_required_value": STUDY_SCOPE[
                            "reserved_weather_slots_required_value"
                        ],
                        "frozen_config_sha256": bundle.decision["config_sha256"],
                    }
                )
    return pd.DataFrame(rows)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": float(STYLE_REPORT.font_size) - 2.0,
            "axes.titlesize": float(STYLE_REPORT.font_size) - 1.0,
            "axes.labelsize": float(STYLE_REPORT.font_size) - 2.0,
            "xtick.labelsize": float(STYLE_REPORT.font_size) - 3.0,
            "ytick.labelsize": float(STYLE_REPORT.font_size) - 3.0,
            "legend.fontsize": float(STYLE_REPORT.font_size) - 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _format_value(value: float, metric: MetricSpec, *, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return f"{value:{sign}.{metric.decimals}f}"


def _plot_group(
    source: pd.DataFrame,
    *,
    analysis_group: str,
    output_base: Path,
    dpi: int,
) -> list[Path]:
    group = source[source["analysis_group"] == analysis_group].copy()
    if group.empty:
        raise ValueError(f"source has no rows for {analysis_group}")
    stages = group[["stage", "stage_label"]].drop_duplicates()
    stage_names = stages["stage"].tolist()
    stage_labels = stages["stage_label"].tolist()
    is_random = analysis_group == "random_robustness_secondary"
    models = (XGBOOST_ID, UNET_ID) if is_random else MODEL_ORDER

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(16.0, 7.0),
        dpi=dpi,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.08, 1.0]},
    )
    width = 0.22 if len(models) == 3 else 0.28
    offsets = np.linspace(-width, width, len(models))
    x = np.arange(len(stage_names), dtype=float)
    for column, metric in enumerate(METRICS):
        direct_ax = axes[0, column]
        improvement_ax = axes[1, column]
        available_values: list[float] = []
        for offset, model_id in zip(offsets, models, strict=True):
            values: list[float] = []
            improvements: list[float] = []
            for stage in stage_names:
                row = group[
                    (group["stage"] == stage)
                    & (group["model_id"] == model_id)
                    & (group["metric"] == metric.key)
                    & group["available"].astype(bool)
                ]
                if len(row) != 1:
                    raise ValueError(
                        f"plot source row is not unique: {stage}/{model_id}/{metric.key}"
                    )
                values.append(float(row.iloc[0]["value"]))
                improvements.append(float(row.iloc[0]["improvement_vs_xgboost"]))
            available_values.extend(values)
            bars = direct_ax.bar(
                x + offset,
                values,
                width=width,
                color=MODEL_COLORS[model_id],
                hatch=MODEL_HATCHES[model_id],
                edgecolor="white",
                linewidth=0.45,
                label=MODEL_LABELS[model_id],
                zorder=3,
            )
            for bar, value in zip(bars, values, strict=True):
                direct_ax.annotate(
                    _format_value(value, metric),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.4,
                    rotation=90 if len(stage_names) > 1 else 0,
                )
            if model_id != XGBOOST_ID:
                imp_bars = improvement_ax.bar(
                    x + offset,
                    improvements,
                    width=width,
                    color=MODEL_COLORS[model_id],
                    hatch=MODEL_HATCHES[model_id],
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=3,
                )
                for bar, value in zip(imp_bars, improvements, strict=True):
                    improvement_ax.annotate(
                        _format_value(value, metric, signed=True),
                        (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 2 if value >= 0 else -2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if value >= 0 else "top",
                        fontsize=6.6,
                    )
        arrow = "↑" if metric.higher_is_better else "↓"
        direct_ax.set_title(f"{metric.label} {arrow}", pad=5)
        direct_ax.set_xticks(x, stage_labels)
        direct_ax.set_ylim(bottom=0.0)
        direct_ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, zorder=0)
        direct_ax.set_ylabel(f"Value ({metric.unit})" if column == 0 else "")
        direct_ax.margins(y=0.18)

        improvement_ax.axhline(0.0, color="#555555", linewidth=0.75, zorder=2)
        improvement_ax.set_xticks(x, stage_labels)
        improvement_ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, zorder=0)
        improvement_ax.set_ylabel(
            f"Improvement vs XGBoost\n({metric.unit})" if column == 0 else ""
        )
        selected = group[
            (group["metric"] == metric.key)
            & (group["model_id"] != XGBOOST_ID)
            & group["available"].astype(bool)
        ]["improvement_vs_xgboost"].to_numpy(dtype=float)
        bound = max(
            0.02 if metric.higher_is_better else 0.5,
            float(np.max(np.abs(selected))) * 1.35,
        )
        improvement_ax.set_ylim(-bound, bound)
        improvement_ax.text(
            0.98,
            0.96,
            "+ favors model",
            transform=improvement_ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.6,
            color="#4A4A4A",
        )

    title = (
        "Standalone U-Net v9 — chronological development and temporal reporting"
        if not is_random
        else "Standalone U-Net v9 — fixed-random robustness (secondary analysis)"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=len(models),
        frameon=False,
    )
    figure_notes = [NO_WEATHER_FIGURE_NOTE]
    if is_random:
        figure_notes.append(
            "Matched pointwise was not refit for random robustness; structural "
            "support remains chronological."
        )
    fig.text(
        0.5,
        0.005,
        "\n".join(figure_notes),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    paths: list[Path] = []
    for suffix in FORMATS:
        path = output_base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _render(
    *,
    development_dir: Path,
    temporal_dir: Path | None,
    random_dir: Path | None,
    output_dir: Path,
    dpi: int,
    force: bool,
) -> list[Path]:
    if dpi < 150:
        raise ValueError("--dpi must be at least 150 for paper output")
    bundles = [_load_evaluation(development_dir, stage="development")]
    if temporal_dir is not None:
        bundles.append(_load_evaluation(temporal_dir, stage="temporal"))
    if random_dir is not None:
        bundles.append(_load_evaluation(random_dir, stage="random_robustness"))
    _validate_bundle_chain(bundles)
    source = _source_rows(bundles)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        output_dir / OUTPUT_BASENAMES["source"],
        output_dir / OUTPUT_BASENAMES["manifest"],
    ]
    expected.extend(
        output_dir / f"{OUTPUT_BASENAMES['primary']}.{suffix}" for suffix in FORMATS
    )
    if random_dir is not None:
        expected.extend(
            output_dir / f"{OUTPUT_BASENAMES['random']}.{suffix}" for suffix in FORMATS
        )
    existing = [path for path in expected if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite comparison outputs; pass --force: "
            + ", ".join(str(path) for path in existing)
        )

    _set_style()
    with tempfile.TemporaryDirectory(prefix="v9_comparison_", dir=output_dir) as raw:
        staging = Path(raw)
        source_path = staging / OUTPUT_BASENAMES["source"]
        source.to_csv(source_path, index=False)
        outputs = [source_path]
        outputs.extend(
            _plot_group(
                source,
                analysis_group="chronological_primary",
                output_base=staging / OUTPUT_BASENAMES["primary"],
                dpi=dpi,
            )
        )
        if random_dir is not None:
            outputs.extend(
                _plot_group(
                    source,
                    analysis_group="random_robustness_secondary",
                    output_base=staging / OUTPUT_BASENAMES["random"],
                    dpi=dpi,
                )
            )
        input_files = [
            path
            for bundle in bundles
            for path in (
                bundle.metrics_path,
                bundle.decision_path,
                bundle.manifest_path,
            )
        ]
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "no_cross_metric_aggregate": True,
            "random_reported_separately": random_dir is not None,
            "study_scope": STUDY_SCOPE,
            "figure_note": NO_WEATHER_FIGURE_NOTE,
            "frozen_config_sha256": bundles[0].decision["config_sha256"],
            "inputs": [
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in input_files
            ],
            "outputs": [
                {
                    "path": str((output_dir / path.name).resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in outputs
            ],
        }
        manifest_path = staging / OUTPUT_BASENAMES["manifest"]
        _write_json_atomic(manifest_path, manifest)
        outputs.append(manifest_path)
        final_paths: list[Path] = []
        for staged in outputs:
            destination = output_dir / staged.name
            os.replace(staged, destination)
            final_paths.append(destination)
    return final_paths


def _synthetic_metrics(*, random: bool, n: int) -> pd.DataFrame:
    values = {
        XGBOOST_ID: (30.0, 6.0, 20.0, 16.0, 0.50),
        POINTWISE_ID: (28.5, 5.8, 19.5, 15.8, 0.52),
        UNET_ID: (26.0, 5.4, 18.0, 15.0, 0.56),
    }
    models = ("xgboost_seed_metric_mean", UNET_ID) if random else MODEL_ORDER
    rows = []
    for model_id in models:
        row: dict[str, Any] = {"model_id": model_id, "n_incidents": n}
        canonical_id = RANDOM_ALIASES.get(model_id, model_id)
        for metric, value in zip(METRICS, values[canonical_id], strict=True):
            row[metric.key] = value
            row[f"{metric.key}_n"] = n if metric.key == "cell_MAE" else max(1, n // 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_synthetic_evaluation(root: Path, *, stage: str) -> Path:
    root.mkdir(parents=True)
    random = stage == "random_robustness"
    n = 324 if stage == "development" else 473
    metrics_path = root / "metrics.csv"
    decision_path = root / "decision.json"
    manifest_path = root / "manifest.json"
    _synthetic_metrics(random=random, n=n).to_csv(metrics_path, index=False)
    decision: dict[str, Any] = {
        "schema_version": (
            DEVELOPMENT_DECISION_SCHEMA
            if stage == "development"
            else RANDOM_DECISION_SCHEMA
            if random
            else TEMPORAL_DECISION_SCHEMA
        ),
        "stage": stage,
        "passed": True,
        "selected_rung": "gtr",
        "unet_fixed_epoch": 45,
        "pointwise_fixed_epoch": 50,
        "config_sha256": "1" * 64,
        "protocol_sha256": "2" * 64,
        "freeze_record_sha256": "3" * 64,
        "backend_lock_sha256": "4" * 64,
        "authorize_full_refit_and_temporal": stage == "development",
    }
    if random:
        decision.update(
            {
                "role": "post_temporal_secondary_robustness_only",
                "split_seed": 20260713,
                "matched_pointwise_refit": False,
                "reported_separately_from_chronological": True,
            }
        )
    _write_json_atomic(decision_path, decision)
    manifest = {
        "schema_version": (
            "standalone_unet_v9_random_robustness_manifest_v1"
            if random
            else "standalone_unet_v9_evaluation_manifest_v1"
        ),
        "stage": stage,
        "passed": True,
        "artifacts": {
            path.name: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (metrics_path, decision_path)
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return root


def _run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="v9_plot_selftest_") as raw:
        root = Path(raw)
        development = _write_synthetic_evaluation(
            root / "development", stage="development"
        )
        temporal = _write_synthetic_evaluation(root / "temporal", stage="temporal")
        random = _write_synthetic_evaluation(root / "random", stage="random_robustness")
        outputs = _render(
            development_dir=development,
            temporal_dir=temporal,
            random_dir=random,
            output_dir=root / "figures",
            dpi=150,
            force=False,
        )
        expected_names = {
            OUTPUT_BASENAMES["source"],
            OUTPUT_BASENAMES["manifest"],
            *{f"{OUTPUT_BASENAMES['primary']}.{suffix}" for suffix in FORMATS},
            *{f"{OUTPUT_BASENAMES['random']}.{suffix}" for suffix in FORMATS},
        }
        if {path.name for path in outputs} != expected_names:
            raise AssertionError("synthetic render output inventory changed")
        source = pd.read_csv(
            root / "figures" / OUTPUT_BASENAMES["source"],
            dtype={"frozen_config_sha256": "string"},
        )
        if len(source) != 45 or source["improvement_vs_xgboost"].notna().sum() != 40:
            raise AssertionError("synthetic figure-source row contract changed")
        if (
            set(source["weather_policy"]) != {"no_weather_locked"}
            or set(source["weather_information_used"]) != {False}
            or set(source["reserved_weather_schema_slots"]) != {10}
            or set(source["frozen_config_sha256"]) != {"1" * 64}
        ):
            raise AssertionError("synthetic no-weather figure contract changed")
        figure_manifest = _load_json(
            root / "figures" / OUTPUT_BASENAMES["manifest"],
            "synthetic figure manifest",
        )
        if (
            figure_manifest.get("schema_version") != MANIFEST_SCHEMA
            or figure_manifest.get("study_scope") != STUDY_SCOPE
            or figure_manifest.get("frozen_config_sha256") != "1" * 64
        ):
            raise AssertionError("synthetic figure-manifest scope binding changed")
        origin = source[
            (source["stage"] == "development")
            & (source["model_id"] == UNET_ID)
            & (source["metric"] == "origin_cc_MAE@20")
        ].iloc[0]
        f1 = source[
            (source["stage"] == "development")
            & (source["model_id"] == UNET_ID)
            & (source["metric"] == "congestion_F1@20")
        ].iloc[0]
        if not math.isclose(float(origin["improvement_vs_xgboost"]), 4.0):
            raise AssertionError("lower-is-better improvement sign changed")
        if not math.isclose(float(f1["improvement_vs_xgboost"]), 0.06):
            raise AssertionError("higher-is-better improvement sign changed")
        if (
            not (root / "figures" / f"{OUTPUT_BASENAMES['primary']}.png")
            .read_bytes()
            .startswith(b"\x89PNG")
        ):
            raise AssertionError("PNG output is invalid")
        if (
            not (root / "figures" / f"{OUTPUT_BASENAMES['primary']}.pdf")
            .read_bytes()
            .startswith(b"%PDF")
        ):
            raise AssertionError("PDF output is invalid")
        svg = (root / "figures" / f"{OUTPUT_BASENAMES['primary']}.svg").read_text(
            encoding="utf-8"
        )
        if (
            "<svg" not in svg
            or "Origin-core MAE" not in svg
            or "no timestamped weather used" not in svg
        ):
            raise AssertionError("SVG output is invalid")
        metrics_path = development / "metrics.csv"
        metrics_path.write_text(
            metrics_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        try:
            _load_evaluation(development, stage="development")
        except ValueError:
            pass
        else:
            raise AssertionError("tampered evaluator metrics were accepted")
    print(
        "standalone U-Net v9 comparison self-test passed | canonical evaluator "
        "hash chain | all five direct metrics | direction-adjusted improvements | "
        "random robustness separate | PNG/SVG/PDF + source CSV"
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--development-dir", type=Path)
    parser.add_argument("--temporal-dir", type=Path)
    parser.add_argument("--random-robustness-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return _run_self_test()
    if args.development_dir is None:
        raise ValueError("--development-dir is required unless --self-test is used")
    outputs = _render(
        development_dir=args.development_dir,
        temporal_dir=args.temporal_dir,
        random_dir=args.random_robustness_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
        force=args.force,
    )
    print(f"Wrote {len(outputs)} standalone U-Net v9 comparison artifacts:")
    for path in outputs:
        print(f"  {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
