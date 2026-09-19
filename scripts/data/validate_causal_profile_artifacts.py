# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///
"""Validate and hash the paper-specific causal-profile artifacts.

This validator is intentionally strict. It verifies the fixed historical
cutoff and freeway scope, metadata/matrix/mask ID order, spatial and temporal
axes, window metadata, and non-empty historical matrices. It then writes a
JSON manifest containing SHA-256 hashes for every input/output artifact plus a
sidecar SHA-256 for the manifest itself.

Relative data paths resolve under ``--data-root``. The canonical locked files
are read as inputs only; all generated paths should point at a separately named
paper artifact directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
_WINDOW_KEYS = (
    "window_upstream_km",
    "window_downstream_km",
    "window_time_before_min",
    "window_time_after_min",
    "window_spatial_resolution_km",
    "window_resolution_min",
)
_PROFILE_REQUIRED_COLUMNS = {
    "vdid",
    "direction",
    "mileage_m",
    "resolution_min",
    "day_class",
    "peak_class",
    "minute_bucket",
    "n_obs",
    "mean_speed_x10",
    "std_speed_x10",
    "p10_speed_x10",
    "p50_speed_x10",
    "p90_speed_x10",
    "coverage_ratio",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", default=str(PROJECT_ROOT))
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--profile-dataset", required=True)
    p.add_argument("--profile-report", required=True)
    p.add_argument("--metadata", default="accident_samples_5m_metadata.parquet")
    p.add_argument("--samples", default="accident_samples_5m.npz")
    p.add_argument("--historical-matrices", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--features", required=True, help="Immutable paper feature CSV; relative paths resolve under --project-root.")
    p.add_argument("--feature-manifest", required=True, help="Feature-builder manifest JSON; relative paths resolve under --project-root.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--profile-start-date", required=True, help="Expected inclusive profile start date.")
    p.add_argument("--profile-cutoff", required=True, help="Expected inclusive profile end date/cutoff.")
    p.add_argument("--freeways", nargs="+", required=True)
    p.add_argument("--notebook", default=None, help="Optional notebook path to include in the code hashes.")
    p.add_argument("--command", action="append", default=[], help="Executed build command to preserve in the manifest; repeatable.")
    p.add_argument(
        "--canonical-lock-guard-passed",
        action="store_true",
        help="Record that the calling notebook compared canonical lock file stats before and after the build.",
    )
    return p.parse_args()


def _resolve_data_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"VALIDATION FAILED: {message}")


def _normalise_ids(values: Any, label: str) -> list[str]:
    ids = [str(v).strip() for v in values]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in ids:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    _require(not duplicates, f"{label} has duplicate accident IDs: {sorted(duplicates)[:10]}")
    return ids


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    _require(path.exists(), f"NPZ not found: {path}")
    with np.load(str(path), allow_pickle=False) as arr:
        return {key: arr[key] for key in arr.files}


def _file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _file_record(path: Path, project_root: Path) -> dict[str, Any]:
    _require(path.is_file(), f"file not found: {path}")
    print(f"[sha256] {_display_path(path, project_root)}", flush=True)
    return {
        "path": _display_path(path, project_root),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _profile_tree_record(profile_base: Path, project_root: Path) -> dict[str, Any]:
    files = sorted(profile_base.rglob("*.parquet"))
    _require(files, f"profile dataset has no parquet files: {profile_base}")
    records = [_file_record(path, project_root) for path in files]
    tree_digest = hashlib.sha256()
    for record in records:
        relative = Path(record["path"])
        try:
            relative = relative.relative_to(_display_path(profile_base, project_root))
        except (TypeError, ValueError):
            pass
        tree_digest.update(str(relative).encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(bytes.fromhex(record["sha256"]))
    return {
        "path": _display_path(profile_base, project_root),
        "parquet_files": records,
        "bytes": sum(int(record["bytes"]) for record in records),
        "tree_sha256": tree_digest.hexdigest(),
        "tree_sha256_definition": "SHA-256 over sorted relative-path NUL raw-file-SHA256-byte pairs",
    }


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
        status_lines = subprocess.run(
            ["git", "status", "--short"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status_lines), "status_short": status_lines}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _validate_profile(
    profile_base: Path,
    report_path: Path,
    freeways: list[str],
    start_date: str,
    cutoff: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(profile_base.is_dir(), f"profile dataset not found: {profile_base}")
    actual_dirs = sorted(path.name.removeprefix("freeway=") for path in profile_base.glob("freeway=*") if path.is_dir())
    _require(actual_dirs == sorted(freeways), f"profile freeway partitions are {actual_dirs}, expected {sorted(freeways)}")

    row_counts: dict[str, int] = {}
    for freeway in freeways:
        files = sorted((profile_base / f"freeway={freeway}").glob("*.parquet"))
        _require(files, f"no profile parquet files for freeway={freeway}")
        rows = 0
        for path in files:
            parquet = pq.ParquetFile(path)
            columns = set(parquet.schema_arrow.names)
            missing_columns = sorted(_PROFILE_REQUIRED_COLUMNS - columns)
            _require(not missing_columns, f"{path} is missing profile columns: {missing_columns}")
            rows += int(parquet.metadata.num_rows)
        _require(rows > 0, f"profile freeway={freeway} contains zero rows")
        row_counts[freeway] = rows

    _require(report_path.is_file(), f"profile report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("range") == {"start": start_date, "end": cutoff}, f"profile report range is {report.get('range')}")
    report_scope = [str(value) for value in report.get("freeway_scope", [])]
    _require(sorted(report_scope) == sorted(freeways), f"profile report freeway_scope is {report_scope}")
    available = report.get("available_dates", {})
    _require(str(available.get("max")) <= cutoff, f"profile report available-date max {available.get('max')} exceeds cutoff {cutoff}")
    _require(str(available.get("min")) >= start_date, f"profile report available-date min {available.get('min')} precedes start {start_date}")
    report_rows = {str(row.get("freeway")): int(row.get("rows", 0)) for row in report.get("rows_per_freeway", [])}
    _require(report_rows == row_counts, f"profile report row counts {report_rows} differ from parquet counts {row_counts}")
    _require(int(report.get("rows_total", -1)) == sum(row_counts.values()), "profile report rows_total is inconsistent")

    return report, {"freeway_rows": row_counts, "rows_total": sum(row_counts.values())}


def _validate_arrays(
    metadata_path: Path,
    samples_path: Path,
    matrices_path: Path,
    masks_path: Path,
    profile_base: Path,
    freeways: list[str],
) -> tuple[dict[str, Any], list[str]]:
    _require(metadata_path.is_file(), f"sample metadata not found: {metadata_path}")
    metadata = pd.read_parquet(metadata_path)
    for column in ("accident_id", "freeway"):
        _require(column in metadata.columns, f"sample metadata is missing {column!r}")
    metadata["accident_id"] = metadata["accident_id"].astype(str).str.strip()
    metadata["freeway"] = metadata["freeway"].astype(str)
    scoped = metadata[metadata["freeway"].isin(freeways)].copy()
    _require(not scoped.empty, f"sample metadata has no rows in freeway scope {freeways}")
    expected_ids = _normalise_ids(scoped["accident_id"].tolist(), "scoped sample metadata")

    samples = _load_npz(samples_path)
    matrices = _load_npz(matrices_path)
    masks = _load_npz(masks_path)
    for label, artifact in (("samples", samples), ("historical matrices", matrices), ("masks", masks)):
        for key in ("matrix", "accident_ids", "space_offsets_m", "time_offsets_min"):
            _require(key in artifact, f"{label} NPZ is missing {key!r}")

    sample_ids = _normalise_ids(samples["accident_ids"], "observed samples")
    matrix_ids = _normalise_ids(matrices["accident_ids"], "historical matrices")
    mask_ids = _normalise_ids(masks["accident_ids"], "congestion masks")
    _require(matrix_ids == expected_ids, "historical matrix IDs/order do not equal N1/N3-filtered metadata IDs/order")
    _require(mask_ids == matrix_ids, "mask IDs/order do not equal historical matrix IDs/order")
    sample_index = {accident_id: index for index, accident_id in enumerate(sample_ids)}
    missing_from_samples = [accident_id for accident_id in matrix_ids if accident_id not in sample_index]
    _require(not missing_from_samples, f"matrix IDs missing from observed samples: {missing_from_samples[:10]}")

    for key in ("space_offsets_m", "time_offsets_min") + _WINDOW_KEYS:
        if key in samples and key in matrices:
            _require(np.array_equal(samples[key], matrices[key]), f"samples and historical matrices differ on {key}")
        if key in matrices and key in masks:
            _require(np.array_equal(matrices[key], masks[key]), f"historical matrices and masks differ on {key}")

    matrix_values = np.asarray(matrices["matrix"])
    mask_values = np.asarray(masks["matrix"])
    expected_shape = (
        len(matrix_ids),
        len(matrices["space_offsets_m"]),
        len(matrices["time_offsets_min"]),
    )
    _require(matrix_values.shape == expected_shape, f"historical matrix shape is {matrix_values.shape}, expected {expected_shape}")
    _require(mask_values.shape == expected_shape, f"mask shape is {mask_values.shape}, expected {expected_shape}")
    all_empty = ~np.isfinite(matrix_values).any(axis=(1, 2))
    _require(not bool(all_empty.any()), f"{int(all_empty.sum())} historical matrices are all empty")
    mask_unique = np.unique(mask_values)
    _require(set(mask_unique.tolist()).issubset({0, 1}), f"mask contains values outside {{0,1}}: {mask_unique.tolist()}")

    if "profile_dataset" in matrices:
        recorded_profile = Path(str(np.asarray(matrices["profile_dataset"]).item())).resolve()
        _require(recorded_profile == profile_base.resolve(), f"historical matrices record profile {recorded_profile}, expected {profile_base}")
    if "freeway_scope" in matrices:
        matrix_scope = sorted(str(value) for value in np.asarray(matrices["freeway_scope"]).tolist())
        _require(matrix_scope == sorted(freeways), f"historical matrices freeway_scope is {matrix_scope}")

    route_counts = {str(key): int(value) for key, value in scoped["freeway"].value_counts().sort_index().items()}
    return {
        "selected_accidents": len(matrix_ids),
        "route_counts": route_counts,
        "matrix_shape": list(matrix_values.shape),
        "historical_finite_cell_ratio": float(np.isfinite(matrix_values).mean()),
        "historical_all_empty_accidents": int(all_empty.sum()),
        "origin_cc_nonempty_accidents": int(mask_values.any(axis=(1, 2)).sum()),
        "origin_cc_cell_share": float(mask_values.mean()),
        "id_alignment": "metadata == historical matrices == masks; historical IDs are an ordered subset of observed samples",
    }, matrix_ids


def _validate_features(
    feature_path: Path,
    feature_manifest_path: Path,
    matrix_ids: list[str],
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the immutable feature artifact and its causal-weather audit."""
    _require(feature_path.is_file(), f"paper feature CSV not found: {feature_path}")
    _require(feature_manifest_path.is_file(), f"paper feature manifest not found: {feature_manifest_path}")
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    _require(bool(feature_manifest.get("paper_valid")), "feature manifest is not marked paper_valid=true")
    _require(bool(feature_manifest.get("immutable_write")), "feature manifest is not marked immutable_write=true")

    features = pd.read_csv(feature_path, dtype={"task_id": str})
    _require("task_id" in features.columns, "paper feature CSV is missing task_id")
    feature_ids = _normalise_ids(features["task_id"].astype(str).str.strip(), "paper feature CSV")
    feature_id_set = set(feature_ids)
    missing = [accident_id for accident_id in matrix_ids if accident_id not in feature_id_set]
    _require(not missing, f"N1/N3 historical-matrix IDs missing from paper features: {missing[:10]}")

    selected = features.set_index(features["task_id"].astype(str).str.strip()).loc[matrix_ids]
    for column in ("highway_1", "highway_3"):
        _require(column in selected.columns, f"paper feature CSV is missing {column}")
    _require(
        bool(((selected["highway_1"] == 1) | (selected["highway_3"] == 1)).all()),
        "a modeled N1/N3 row is not encoded as highway_1 or highway_3 in the feature table",
    )
    for column in ("highway_4", "highway_6"):
        if column in selected.columns:
            _require(bool((selected[column] == 0).all()), f"a modeled N1/N3 row has {column}=1")

    artifacts = feature_manifest.get("artifacts", {})
    feature_entry = artifacts.get("features", {})
    actual_feature_hash = _file_sha256(feature_path)
    _require(feature_entry.get("sha256") == actual_feature_hash, "feature CSV SHA-256 differs from its feature manifest")
    _require(int(feature_manifest.get("feature_schema", {}).get("rows", -1)) == len(features), "feature manifest row count differs from CSV")
    _require(
        int(feature_manifest.get("feature_schema", {}).get("unique_task_ids", -1)) == len(feature_ids),
        "feature manifest unique-task count differs from CSV",
    )

    weather = feature_manifest.get("sources", {}).get("weather", {})
    mode = str(weather.get("mode"))
    _require(mode in {"omitted", "causal_backward"}, f"unsupported/noncausal paper weather mode: {mode}")
    _require(weather.get("match_direction") == "backward", "feature weather match direction is not backward")
    _require(int(weather.get("future_timestamp_violations", -1)) == 0, "feature weather audit reports future timestamp violations")
    _require(bool(weather.get("causal_validated")), "feature weather audit is not marked causal_validated=true")

    weather_value_columns = ["low_temp", "high_humidity", "strong_wind", "raining", "low_visibility"]
    weather_availability_columns = [f"avail_{column}" for column in weather_value_columns]
    for column in weather_value_columns + weather_availability_columns:
        _require(column in features.columns, f"paper feature CSV is missing weather column {column}")
    if mode == "omitted":
        _require(
            bool((features[weather_value_columns + weather_availability_columns].fillna(0) == 0).all().all()),
            "weather mode is omitted but at least one weather value/availability cell is nonzero",
        )
    else:
        _require(float(weather.get("max_staleness_minutes", -1)) == 90.0, "causal weather max staleness is not 90 minutes")
        _require(int(weather.get("over_staleness_limit_violations", -1)) == 0, "feature weather audit reports over-staleness violations")

    companion_records: dict[str, Any] = {}
    for name, entry in sorted(artifacts.items()):
        artifact_path_value = entry.get("path")
        _require(bool(artifact_path_value), f"feature manifest artifact {name!r} has no path")
        artifact_path = _resolve_project_path(project_root, str(artifact_path_value))
        record = _file_record(artifact_path, project_root)
        _require(entry.get("sha256") == record["sha256"], f"feature companion hash mismatch: {name}")
        companion_records[name] = record
    companion_records["manifest"] = _file_record(feature_manifest_path, project_root)

    source_records: dict[str, Any] = {}
    sources = feature_manifest.get("sources", {})
    accident_source = sources.get("accident_csv", {})
    if accident_source.get("path"):
        accident_source_path = _resolve_project_path(project_root, str(accident_source["path"]))
        accident_record = _file_record(accident_source_path, project_root)
        _require(accident_source.get("sha256") == accident_record["sha256"], "feature accident-source hash mismatch")
        source_records["accident_csv"] = accident_record
    geometry_records: list[dict[str, Any]] = []
    for entry in sources.get("geometry_files", []):
        geometry_path = _resolve_project_path(project_root, str(entry["path"]))
        geometry_record = _file_record(geometry_path, project_root)
        _require(entry.get("sha256") == geometry_record["sha256"], f"feature geometry-source hash mismatch: {geometry_path}")
        geometry_records.append(geometry_record)
    source_records["geometry_files"] = geometry_records
    if weather.get("source"):
        weather_source_path = _resolve_project_path(project_root, str(weather["source"]))
        weather_record = _file_record(weather_source_path, project_root)
        _require(weather.get("source_sha256") == weather_record["sha256"], "feature weather-source hash mismatch")
        source_records["weather"] = weather_record

    weather_matches_path = None
    if "weather_matches" in artifacts:
        weather_matches_path = _resolve_project_path(project_root, str(artifacts["weather_matches"]["path"]))
    if mode == "causal_backward" and weather_matches_path is not None:
        matches = pd.read_csv(weather_matches_path)
        matched = matches[matches["weather_match_status"].eq("matched")].copy()
        if not matched.empty:
            observation = pd.to_datetime(matched["weather_observation_timestamp"], errors="coerce")
            dispatch = pd.to_datetime(matched["dispatch_timestamp"], errors="coerce")
            staleness = pd.to_numeric(matched["weather_staleness_minutes"], errors="coerce")
            _require(bool((observation <= dispatch).all()), "weather-match audit contains a post-dispatch selected observation")
            _require(bool(((staleness >= 0) & (staleness <= 90)).all()), "weather-match audit has matched rows outside 0..90 minutes staleness")

    stats = {
        "feature_rows": len(features),
        "feature_columns": len(features.columns),
        "modeled_ids_present": len(matrix_ids),
        "weather_mode": mode,
        "weather_matched_timestamp_count": int(weather.get("matched_timestamp_count", 0)),
        "weather_future_timestamp_violations": int(weather.get("future_timestamp_violations", 0)),
        "feature_paper_valid": True,
        "feature_immutable_write": True,
    }
    return stats, companion_records, source_records


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    freeways = list(dict.fromkeys(str(value) for value in args.freeways))
    _require(len(freeways) == len(args.freeways), f"duplicate values in --freeways: {args.freeways}")

    profile_base = _resolve_data_path(data_root, args.profile_dataset)
    report_path = _resolve_data_path(data_root, args.profile_report)
    metadata_path = _resolve_data_path(data_root, args.metadata)
    samples_path = _resolve_data_path(data_root, args.samples)
    matrices_path = _resolve_data_path(data_root, args.historical_matrices)
    masks_path = _resolve_data_path(data_root, args.masks)
    feature_path = _resolve_project_path(project_root, args.features)
    feature_manifest_path = _resolve_project_path(project_root, args.feature_manifest)
    manifest_path = _resolve_data_path(data_root, args.manifest)
    _require(
        feature_path.name == "incident_features_paper_trb_2026.csv",
        f"unexpected immutable paper feature filename: {feature_path.name}",
    )

    generated = {profile_base, report_path, matrices_path, masks_path, manifest_path}
    canonical_outputs = {
        data_root / "historical_speed_profile_5m",
        data_root / "metadata" / "historical_profile_report.json",
        data_root / "historical_matrices_5m.npz",
        data_root / "congestion_masks_origin_cc_5m.npz",
    }
    _require(not (generated & canonical_outputs), "a generated paper path resolves to a canonical locked output path")

    report, profile_stats = _validate_profile(
        profile_base,
        report_path,
        freeways,
        args.profile_start_date,
        args.profile_cutoff,
    )
    array_stats, matrix_ids = _validate_arrays(
        metadata_path,
        samples_path,
        matrices_path,
        masks_path,
        profile_base,
        freeways,
    )
    feature_stats, feature_records, feature_source_records = _validate_features(
        feature_path,
        feature_manifest_path,
        matrix_ids,
        project_root,
    )

    code_paths = [
        PROJECT_ROOT / "scripts" / "data" / "build_historical_profile.py",
        PROJECT_ROOT / "scripts" / "data" / "build_historical_matrices.py",
        PROJECT_ROOT / "scripts" / "data" / "build_congestion_masks.py",
        PROJECT_ROOT / "scripts" / "features" / "build_incident_features.py",
        Path(__file__).resolve(),
    ]
    if args.notebook:
        code_paths.append(_resolve_project_path(project_root, args.notebook))

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "name": "TRB 2026 causal historical profile, N1/N3 scope",
            "profile_start_date_inclusive": args.profile_start_date,
            "profile_cutoff_inclusive": args.profile_cutoff,
            "freeways": freeways,
            "resolution_min": int(report.get("resolution_min", 5)),
            "canonical_lock_guard_passed": bool(args.canonical_lock_guard_passed),
        },
        "commands": list(args.command),
        "git": _git_state(project_root),
        "inputs": {
            "sample_metadata": _file_record(metadata_path, project_root),
            "observed_samples": _file_record(samples_path, project_root),
            "feature_sources": feature_source_records,
        },
        "code": [_file_record(path, project_root) for path in code_paths],
        "outputs": {
            "historical_profile": _profile_tree_record(profile_base, project_root),
            "profile_report": _file_record(report_path, project_root),
            "historical_matrices": _file_record(matrices_path, project_root),
            "origin_cc_masks": _file_record(masks_path, project_root),
            "incident_features": feature_records,
        },
        "validation": {
            **profile_stats,
            **array_stats,
            **feature_stats,
            "profile_report_range": report.get("range"),
            "profile_available_dates": report.get("available_dates"),
            "status": "passed",
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_sha = _file_sha256(manifest_path)
    sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    sidecar_path.write_text(f"{manifest_sha}  {manifest_path.name}\n", encoding="ascii")

    print("\n" + "=" * 72)
    print("CAUSAL PROFILE ARTIFACT VALIDATION: PASSED")
    print("=" * 72)
    print(f"Profile cutoff: {args.profile_cutoff} | freeways: {freeways}")
    print(f"Accidents: {array_stats['selected_accidents']} | matrices: {array_stats['matrix_shape']}")
    print(f"Manifest: {manifest_path}")
    print(f"Manifest SHA-256: {manifest_sha}")
    print(f"SHA sidecar: {sidecar_path}")


if __name__ == "__main__":
    main()
