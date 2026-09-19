"""Model inference wrapper.

Two surfaces:

1. `load(name, run_id=None, experiment=None)` — resolves a checkpoint under
   `output/models/<name>/<run_id>/` and returns a `TrafficModel`.

2. `predict_accident_impact(accident, data_root, ...)` — GUI-facing path.
   Takes a fresh accident dict (time / freeway / direction / mileage), fetches
   live VD data, assembles the input tensor in memory, runs the model, and
   returns predicted heatmap + impact duration + recovery curves.

The GUI in `scripts/gui/app.py` depends only on `predict_accident_impact`. See
`docs/DEPLOYMENT.md` for the deployment-side contract.

`predict_accident_impact` loads the checkpoint first when no injectable
`forward_fn` is supplied, then lets the model window drive live/offline VD
assembly, historical-profile lookup, tensor creation, model forward,
post-processing, and coverage reporting. The injectable `forward_fn` remains
available for fake-model smoke tests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from .windows import (
    DEFAULT_WINDOW,
    SampleWindow,
    absolute_mileage_axis_m,
    coverage as window_coverage,
    rasterize_window,
    slice_station_df,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "output" / "models"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

# Free-flow target — used to derive impact duration. Pulled from the project
# convention of "speed back to ~80 km/h" as a non-congested threshold.
FREE_FLOW_THRESHOLD_KMH = 80.0

CHAMPION_SPEC_PATH = PROJECT_ROOT / "reports" / "v1_champion.json"

# Peak window definitions mirror build_historical_profile.py exactly.
_PEAK_AM = (7 * 60, 9 * 60)
_PEAK_PM = (17 * 60, 19 * 60)

LiveVDFetcher = Callable[[datetime, datetime], "FetchResult"]
ForwardFn = Callable[["ModelInput"], np.ndarray]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


class ModelNotAvailableError(RuntimeError):
    """Raised when no checkpoint can be loaded.

    Carries the assembled `ModelInput` so the GUI can still display fetch /
    profile / coverage info even when the model itself isn't wired yet.
    """

    def __init__(self, message: str, model_input: "ModelInput | None" = None) -> None:
        super().__init__(message)
        self.model_input = model_input


@dataclass(frozen=True)
class AccidentInput:
    """Operator-supplied accident metadata.

    `report_time` is Taiwan local wall time, naive (no tz), matching the
    timestamp convention used everywhere else in this project (see
    docs/DATA_DESIGN.md).
    """

    report_time: datetime
    freeway: str
    direction: str
    mileage_km: float
    features: dict[str, Any] | None = None

    @property
    def mileage_m(self) -> int:
        return int(round(self.mileage_km * 1000))


@dataclass
class FetchResult:
    """Return type for a `LiveVDFetcher`.

    Wraps the live station-5m DataFrame plus a status dict so the GUI can show
    per-minute success counts. Synthetic test fetchers may return `status=None`.
    """

    station_5m: pd.DataFrame
    status: dict[str, Any] | None = None


@dataclass
class ModelInput:
    """All artifacts assembled before the model.forward call.

    Channels are kept separate so each architecture can pick what it consumes:
    the matrix U-Net only needs `observed_pre`; feature-fusion adds
    `incident_features`; profile-aware models read `historical_baseline`.
    """

    accident: AccidentInput
    window: SampleWindow
    observed_pre: np.ndarray              # (n_space, n_time_pre) — live VD speeds
    historical_baseline: np.ndarray       # (n_space, n_time)     — typical speed
    incident_features: dict[str, Any] | None
    coverage_pre: dict[str, Any]          # fill stats for observed_pre
    coverage_baseline: dict[str, Any]     # fill stats for historical_baseline
    fetch_status: dict[str, Any] | None


@dataclass
class AccidentPrediction:
    """Result returned to the GUI for one accident.

    `heatmap` is shaped `(window.n_space, window.n_time)` with predicted speed
    in km/h. `recovery_curves` is keyed by absolute mileage_km and holds the
    speed series for that mileage slice over `window.time_offsets_min`.
    """

    heatmap: np.ndarray
    window: SampleWindow
    duration_min: float
    recovery_curves: dict[float, np.ndarray]
    coverage: dict[str, Any]
    input_echo: dict[str, Any] = field(default_factory=dict)


class TrafficModel(Protocol):
    window: SampleWindow

    def predict(self, accident_id: str, hist_matrix: np.ndarray | None = None) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Model loader — Phase D implementation.
# ---------------------------------------------------------------------------


def _load_champion_spec() -> dict:
    if not CHAMPION_SPEC_PATH.exists():
        raise FileNotFoundError(
            f"Champion spec not found at {CHAMPION_SPEC_PATH}. "
            "Run the round-1 ablation and write reports/v1_champion.json first."
        )
    with open(CHAMPION_SPEC_PATH, encoding="utf-8") as fh:
        return json.load(fh)


class _CheckpointModel:
    """TrafficModel backed by a loaded PyTorch checkpoint.

    Implements the TrafficModel protocol via `.window` attribute and
    `.predict()`. Also exposes `.predict_from_input(ModelInput)` for the
    fresh-accident path used by `predict_accident_impact`.
    """

    def __init__(
        self,
        nn_model: Any,
        arch: str,
        method: str,
        input_type: str,
        backbone: str,
        coordinate_channels: bool,
        hist_validity_channel: bool,
        hist_missing_fill: str,
        feature_columns: list[str] | None,
        cfg: Any,  # WindowConfig from model_architectures
        window: SampleWindow,
    ) -> None:
        self._nn_model = nn_model
        self._arch = arch
        self._method = method
        self._input_type = input_type
        self._backbone = backbone
        self._coordinate_channels = bool(coordinate_channels)
        self._hist_validity_channel = bool(hist_validity_channel)
        self._hist_missing_fill = str(hist_missing_fill)
        if self._input_type != "vd+hist":
            raise ValueError("U-Net inference is locked to input_type='vd+hist'")
        if self._coordinate_channels:
            raise ValueError("coordinate planes are outside the locked 3-channel U-Net contract")
        if self._hist_validity_channel:
            raise ValueError("legacy 4-channel U-Net checkpoints are no longer supported")
        if self._hist_missing_fill != "nearest":
            raise ValueError("3-channel U-Net inference requires nearest-space historical fill")
        self._feature_columns = feature_columns
        self._cfg = cfg
        self.window = window

    def predict(self, accident_id: str, hist_matrix: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError(
            "predict(accident_id) is the training-NPZ interface. "
            "For live inference use predict_from_input(ModelInput)."
        )

    def predict_from_input(self, model_input: ModelInput) -> np.ndarray:
        """Assemble the input tensor from ModelInput and run forward pass."""
        import torch
        from .model_architectures import PAD_TARGET, SPEED_MAX

        cfg = self._cfg
        obs = model_input.observed_pre  # (n_space, n_time_pre), km/h or NaN

        n_space = cfg.n_space
        n_time = cfg.n_time
        n_before = cfg.n_before
        n_after = cfg.n_after
        pad_h = cfg.pad_h
        pad_w = cfg.pad_w

        # Normalize km/h → [0, 1] (same as training: arr / SPEED_MAX).
        # observed_pre comes from build_observed_pre which returns km/h.
        speed_norm = np.where(np.isnan(obs), 0.0, np.clip(obs, 0.0, SPEED_MAX) / SPEED_MAX).astype(np.float32)
        obs_mask = np.isfinite(obs).astype(np.float32)

        channels: list[np.ndarray] = []
        # Locked channel 1/2: pre-incident observed speed and validity.
        full_speed = np.zeros((n_space, n_time), dtype=np.float32)
        full_mask = np.zeros((n_space, n_time), dtype=np.float32)
        n_pre_cols = min(speed_norm.shape[1], n_before)
        full_speed[:, :n_pre_cols] = speed_norm[:, :n_pre_cols]
        full_mask[:, :n_pre_cols] = obs_mask[:, :n_pre_cols]
        channels += [full_speed, full_mask]
        # Locked channel 3: causal historical speed with nearest-space fill.
        hist = model_input.historical_baseline.copy()
        for time_idx in range(hist.shape[1]):
            col = hist[:, time_idx]
            valid_idx = np.flatnonzero(np.isfinite(col))
            missing_idx = np.flatnonzero(~np.isfinite(col))
            if not len(missing_idx):
                continue
            if not len(valid_idx):
                raise ValueError(
                    "historical profile has no valid spatial value at "
                    f"time column {time_idx}"
                )
            nearest = np.abs(missing_idx[:, None] - valid_idx[None, :]).argmin(axis=1)
            col[missing_idx] = col[valid_idx[nearest]]
            hist[:, time_idx] = col
        channels.append((np.clip(hist, 0.0, SPEED_MAX) / SPEED_MAX).astype(np.float32))
        n_base_channels = len(channels)
        if self._coordinate_channels:
            # Match AccidentSampleDataset exactly: signed physical offsets from
            # the incident/dispatch origin, normalized to roughly [-1, 1].
            space = (
                np.arange(n_space, dtype=np.float32) * cfg.spatial_resolution_km
                - cfg.downstream_km
            ) / max(cfg.upstream_km, cfg.downstream_km, 1)
            time = (
                np.arange(n_time, dtype=np.float32) * cfg.resolution_min
                - cfg.time_before_min
            ) / max(cfg.time_before_min, cfg.time_after_min, 1)
            channels += [
                np.repeat(space[:, None], n_time, axis=1),
                np.repeat(time[None, :], n_space, axis=0),
            ]

        # Pad spatial+time to PAD_TARGET × PAD_TARGET.
        padded_base = np.pad(
            np.stack(channels[:n_base_channels], axis=0),
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="reflect",
        ).astype(np.float32, copy=False)
        if self._coordinate_channels:
            padded_coord = np.zeros((2, PAD_TARGET, PAD_TARGET), dtype=np.float32)
            for coord_idx, coord in enumerate(channels[n_base_channels:]):
                padded_coord[
                    coord_idx,
                    pad_h : pad_h + n_space,
                    pad_w : pad_w + n_time,
                ] = coord
            padded = np.concatenate([padded_base, padded_coord], axis=0)
        else:
            padded = padded_base
        x = torch.from_numpy(padded).unsqueeze(0)

        feat_np = self._build_feature_vector(model_input.incident_features)
        feat_t = torch.from_numpy(feat_np).unsqueeze(0)  # (1, feature_dim)

        with torch.no_grad():
            if self._arch == "unet_feature_fusion":
                pred_norm, _ = self._nn_model(x, feat_t)
            else:
                pred_norm = self._nn_model(x)

        pred_kmh = pred_norm.squeeze(0).numpy() * SPEED_MAX  # (n_space, n_after)

        # Compose heatmap: pre-accident columns from live VD (km/h), post from model.
        heatmap = np.full((n_space, n_time), np.nan, dtype=np.float32)
        n_pre = min(obs.shape[1], n_before)
        if n_pre > 0:
            heatmap[:, :n_pre] = obs[:, :n_pre]  # already in km/h
        heatmap[:, n_before : n_before + n_after] = pred_kmh

        return heatmap

    def _build_feature_vector(self, incident_features: dict | None) -> np.ndarray:
        if not self._feature_columns:
            return np.zeros(0, dtype=np.float32)
        vec = np.zeros(len(self._feature_columns), dtype=np.float32)
        if incident_features:
            for i, col in enumerate(self._feature_columns):
                val = incident_features.get(col)
                if val is not None:
                    try:
                        vec[i] = float(val)
                    except (TypeError, ValueError):
                        pass
        return vec


def load(
    name: str | None = None,
    run_id: str | None = None,
    models_root: Path | str | None = None,
    *,
    experiment: str | None = None,
) -> TrafficModel:
    """Load a trained checkpoint and return a ``TrafficModel``.

    Call with no arguments to load the v1 champion from
    ``reports/v1_champion.json``.  For an explicit checkpoint pass ``name``
    (architecture name), ``run_id``, and optionally ``experiment`` (required
    for feature-fusion runs which store each method in a sub-directory).
    """
    import torch
    from .model_architectures import WindowConfig, build_feature_fusion_unet, build_matrix_unet

    root = Path(models_root) if models_root is not None else DEFAULT_MODELS_ROOT

    if name is None and run_id is None:
        spec = _load_champion_spec()
        arch = spec["arch"]
        _run_id = spec["run_id"]
        _experiment = spec.get("experiment")
        feature_dim = spec["feature_dim"]
        feature_columns = spec["feature_columns"]
        method = spec["method"]
        input_type = spec.get("input_type", "vd")
        backbone = spec.get("backbone", "simple")
        simple_base_channels = int(spec.get("simple_base_channels", 64))
        coordinate_channels = bool(spec.get("coordinate_channels", False))
        hist_validity_channel = bool(spec.get("hist_validity_channel", False))
        hist_missing_fill = str(spec.get("hist_missing_fill", "nearest"))
        input_channels = spec.get("input_channels")
        window_spec = spec.get("window", {})
        cfg = WindowConfig(**window_spec) if window_spec else WindowConfig()
        ckpt_path = PROJECT_ROOT / spec["checkpoint_rel"]
    else:
        arch = name or "unet_feature_fusion"
        _run_id = run_id or ""
        _experiment = experiment
        feature_dim = 32
        feature_columns = None
        method = experiment or "direct_concat"
        input_type = "vd+hist"
        backbone = "simple"
        simple_base_channels = 64
        coordinate_channels = False
        hist_validity_channel = False
        hist_missing_fill = "nearest"
        input_channels = None
        cfg = WindowConfig()

        if _experiment:
            ckpt_path = root / arch / _run_id / _experiment / "checkpoints" / "best_model.pth"
        else:
            ckpt_path = root / arch / _run_id / "checkpoints" / "best_model.pth"

        # Try experiment-level config, then run-level config (feature-fusion
        # stores run_config.json one directory above the experiment folder).
        for _cfg_parent in (ckpt_path.parent.parent, ckpt_path.parent.parent.parent):
            _rcfg_path = _cfg_parent / "run_config.json"
            if _rcfg_path.exists():
                with open(_rcfg_path, encoding="utf-8") as fh:
                    run_cfg = json.load(fh)
                # run-level config nests settings under "config" key
                _inner = run_cfg.get("config", run_cfg)
                feature_dim = _inner.get("feature_dim", run_cfg.get("feature_dim", feature_dim))
                feature_columns = run_cfg.get("feature_columns", feature_columns)
                method = _inner.get("method", run_cfg.get("method", method))
                input_type = _inner.get("input_type", run_cfg.get("input_type", input_type))
                backbone = _inner.get("backbone", run_cfg.get("backbone", backbone))
                simple_base_channels = int(
                    _inner.get(
                        "simple_base_channels",
                        run_cfg.get("simple_base_channels", simple_base_channels),
                    )
                )
                coordinate_channels = bool(
                    _inner.get(
                        "coordinate_channels",
                        run_cfg.get("coordinate_channels", coordinate_channels),
                    )
                )
                hist_validity_channel = bool(
                    _inner.get(
                        "hist_validity_channel",
                        run_cfg.get("hist_validity_channel", hist_validity_channel),
                    )
                )
                hist_missing_fill = str(
                    _inner.get(
                        "hist_missing_fill",
                        run_cfg.get("hist_missing_fill", hist_missing_fill),
                    )
                )
                input_channels = _inner.get(
                    "input_channels",
                    run_cfg.get("input_channels", input_channels),
                )
                window_keys = {
                    "upstream_km",
                    "downstream_km",
                    "time_before_min",
                    "time_after_min",
                    "spatial_resolution_km",
                    "resolution_min",
                }
                window_kwargs = {k: _inner[k] for k in window_keys if k in _inner}
                if window_kwargs:
                    cfg = WindowConfig(**window_kwargs)
                break

        # All unet_feature_fusion variants share the same 32-column feature set.
        # Fall back to champion spec when feature_columns is not stored in run_config.
        if feature_columns is None and arch == "unet_feature_fusion":
            try:
                feature_columns = _load_champion_spec().get("feature_columns")
            except FileNotFoundError:
                pass

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Ensure training completed and the path in reports/v1_champion.json is correct."
        )

    if arch == "unet_feature_fusion":
        nn_model = build_feature_fusion_unet(
            feature_dim=feature_dim,
            method=method,
            use_tiid=False,
            cfg=cfg,
            input_type=input_type,
            backbone=backbone,
        )
    elif arch == "unet_matrix":
        if input_channels is None:
            input_channels = 3
        if (
            input_type != "vd+hist"
            or int(input_channels) != 3
            or hist_validity_channel
            or coordinate_channels
            or hist_missing_fill != "nearest"
        ):
            raise ValueError(
                "checkpoint does not satisfy the locked 3-channel U-Net contract; "
                "legacy 4-channel/coordinate/input-ablation checkpoints are excluded"
            )
        nn_model = build_matrix_unet(
            cfg=cfg,
            input_type=input_type,
            backbone=backbone,
            in_channels=int(input_channels) if input_channels is not None else None,
            simple_base_channels=simple_base_channels,
        )
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Expected 'unet_matrix' or 'unet_feature_fusion'.")

    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    nn_model.load_state_dict(state_dict)
    nn_model.eval()

    sample_window = SampleWindow(
        upstream_km=cfg.upstream_km,
        downstream_km=cfg.downstream_km,
        time_before_min=cfg.time_before_min,
        time_after_min=cfg.time_after_min,
        spatial_resolution_km=cfg.spatial_resolution_km,
        resolution_min=cfg.resolution_min,
    )

    return _CheckpointModel(
        nn_model=nn_model,
        arch=arch,
        method=method,
        input_type=input_type,
        backbone=backbone,
        coordinate_channels=coordinate_channels,
        hist_validity_channel=hist_validity_channel,
        hist_missing_fill=hist_missing_fill,
        feature_columns=feature_columns,
        cfg=cfg,
        window=sample_window,
    )


# ---------------------------------------------------------------------------
# Default live-VD fetcher — wraps live_vd.fetch_live_station_5m.
# ---------------------------------------------------------------------------


def _default_live_fetcher(accident: AccidentInput, window: SampleWindow) -> LiveVDFetcher:
    """Build a fetcher pre-bound to this accident's spatial filter.

    Imported lazily so callers using a fake fetcher don't pay the import cost.
    """

    def _fetch(start: datetime, end: datetime) -> FetchResult:
        from .live_vd import fetch_live_station_5m

        df, status = fetch_live_station_5m(
            start=start,
            end=end,
            freeway=accident.freeway,
            direction=accident.direction,
            mileage_m=accident.mileage_m,
            upstream_km=window.upstream_km,
            downstream_km=window.downstream_km,
        )
        return FetchResult(station_5m=df, status=status.to_dict())

    return _fetch


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_against_vd_dim(accident: AccidentInput, data_root: Path) -> None:
    dim_path = data_root / "metadata" / "vd_dim.parquet"
    if not dim_path.exists():
        # Soft-fail — operating without vd_dim is allowed, just no bounds check.
        logger.warning("vd_dim.parquet not found at %s; skipping route-bounds check.", dim_path)
        return

    import pyarrow.parquet as pq

    dim = pq.read_table(str(dim_path)).to_pandas()
    route = dim[(dim["freeway"] == accident.freeway) & (dim["direction"] == accident.direction)]
    if route.empty:
        raise ValueError(
            f"No VDs registered for {accident.freeway}-{accident.direction}. "
            f"Check that this freeway/direction is in the project's VD universe."
        )
    min_m = int(route["mileage_m"].min())
    max_m = int(route["mileage_m"].max())
    if not (min_m <= accident.mileage_m <= max_m):
        raise ValueError(
            f"Mileage {accident.mileage_km:.3f} km is outside VD coverage for "
            f"{accident.freeway}-{accident.direction} ({min_m / 1000:.1f}-{max_m / 1000:.1f} km)."
        )


# ---------------------------------------------------------------------------
# Historical baseline assembly
# ---------------------------------------------------------------------------


def _peak_class(minute_of_day: int) -> str:
    if _PEAK_AM[0] <= minute_of_day < _PEAK_AM[1]:
        return "peak"
    if _PEAK_PM[0] <= minute_of_day < _PEAK_PM[1]:
        return "peak"
    return "nonpeak"


def _bucket_minute(ts: pd.Timestamp, resolution_min: int) -> int:
    return ((ts.hour * 60 + ts.minute) // resolution_min) * resolution_min


def make_profile_index(
    profile_df: pd.DataFrame,
    freeway: str,
    direction: str,
) -> dict:
    """Pre-build a lookup dict for fast historical baseline assembly.

    Returns a dict mapping (day_class?, peak_class?, minute_bucket) → DataFrame
    slice with columns [vdid, mileage_m, mean_speed_x10].  Call once per
    (freeway, direction) and pass the result as `profile_index` to
    `build_historical_baseline` to avoid rebuilding the hash index on every
    accident call.
    """
    p = profile_df
    if "freeway" in p.columns:
        p = p[p["freeway"] == freeway]
    p = p[p["direction"] == direction]
    if p.empty:
        return {"_has_day_class": False, "_has_peak_class": False}

    has_dc = "day_class" in p.columns
    has_pc = "peak_class" in p.columns
    key_cols = (["day_class"] if has_dc else []) + (["peak_class"] if has_pc else []) + ["minute_bucket"]

    cols = ["vdid", "mileage_m", "mean_speed_x10"]
    index: dict = {"_has_day_class": has_dc, "_has_peak_class": has_pc}
    for key, grp in p.groupby(key_cols):
        # normalise to always-tuple so callers can build keys uniformly
        k = key if isinstance(key, tuple) else (key,)
        index[k] = grp[cols].reset_index(drop=True)
    return index


def build_historical_baseline(
    accident: AccidentInput,
    window: SampleWindow,
    profile_df: pd.DataFrame | None = None,
    *,
    profile_index: dict | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Materialize the historical-baseline (n_space, n_time) matrix.

    Pass either a raw `profile_df` DataFrame (slow path, rebuilds index each
    call) or a pre-built `profile_index` dict from `make_profile_index` (fast
    path, O(1) lookup per timestamp).  For batch precomputation always use the
    fast path.
    """
    from .calendar import day_class

    _empty = np.full((window.n_space, window.n_time), np.nan, dtype=np.float32)
    _empty_meta: dict[str, Any] = {"filled_cells": 0, "total_cells": int(_empty.size), "filled_ratio": 0.0}

    report_time = pd.Timestamp(accident.report_time)
    starts = report_time - pd.Timedelta(minutes=window.time_before_min)
    timestamps = pd.date_range(
        start=starts,
        periods=window.n_time,
        freq=pd.Timedelta(minutes=window.resolution_min),
    )

    if profile_index is not None:
        # Fast path: O(1) dict lookup per timestamp, no per-call merge overhead.
        has_dc = profile_index.get("_has_day_class", False)
        has_pc = profile_index.get("_has_peak_class", False)

        frames: list[pd.DataFrame] = []
        for ts in timestamps:
            key_parts: list[Any] = []
            if has_dc:
                key_parts.append(day_class(ts.date()))
            if has_pc:
                key_parts.append(_peak_class(ts.hour * 60 + ts.minute))
            key_parts.append(_bucket_minute(ts, window.resolution_min))
            subset = profile_index.get(tuple(key_parts))
            if subset is not None:
                frames.append(subset.assign(timestamp=ts))

        if not frames:
            return _empty, _empty_meta

        station_like = pd.concat(frames, ignore_index=True)
        station_like["freeway"] = accident.freeway
        station_like["direction"] = accident.direction
        station_like = station_like.rename(columns={"mean_speed_x10": "speed_x10"})
        station_like["mileage_m"] = station_like["mileage_m"].astype(int)

    else:
        # Slow path (backward-compatible): merge on the raw DataFrame.
        if profile_df is None or profile_df.empty:
            return _empty, _empty_meta

        profile = profile_df
        if "freeway" in profile.columns:
            profile = profile[profile["freeway"] == accident.freeway]
        profile = profile[profile["direction"] == accident.direction]
        if profile.empty:
            return _empty, _empty_meta

        has_day_class = "day_class" in profile.columns
        has_peak_class = "peak_class" in profile.columns
        merge_cols = ["minute_bucket"]
        ts_records: dict[str, Any] = {
            "timestamp": timestamps,
            "minute_bucket": [_bucket_minute(ts, window.resolution_min) for ts in timestamps],
        }
        if has_day_class:
            ts_records["day_class"] = [day_class(ts.date()) for ts in timestamps]
            merge_cols.append("day_class")
        if has_peak_class:
            ts_records["peak_class"] = [_peak_class(ts.hour * 60 + ts.minute) for ts in timestamps]
            merge_cols.append("peak_class")

        merged = pd.DataFrame(ts_records).merge(
            profile[merge_cols + ["vdid", "mileage_m", "mean_speed_x10"]],
            on=merge_cols,
            how="inner",
        )
        if merged.empty:
            return _empty, _empty_meta

        station_like = merged[["timestamp", "vdid", "mileage_m", "mean_speed_x10"]].copy()
        station_like["freeway"] = accident.freeway
        station_like["direction"] = accident.direction
        station_like = station_like.rename(columns={"mean_speed_x10": "speed_x10"})
        station_like["mileage_m"] = station_like["mileage_m"].astype(int)
    sliced = slice_station_df(
        station_like,
        accident_mileage_m=accident.mileage_m,
        direction=accident.direction,
        freeway=accident.freeway,
        report_time=report_time,
        window=window,
    )
    matrix = rasterize_window(sliced, window, value_col="speed_x10")
    return matrix.astype(np.float32), window_coverage(matrix)


# ---------------------------------------------------------------------------
# Observed (live) matrix assembly
# ---------------------------------------------------------------------------


def build_observed_pre(
    accident: AccidentInput,
    window: SampleWindow,
    fetch_result: FetchResult,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterize live VD readings over the pre-accident time slice.

    The model only sees data up to t=0; rows after the report time are
    discarded even if the fetcher returned them, so a too-eager test fetcher
    can't leak future information.
    """
    n_time_pre = window.time_before_min // window.resolution_min + 1
    pre_window = SampleWindow(
        upstream_km=window.upstream_km,
        downstream_km=window.downstream_km,
        time_before_min=window.time_before_min,
        time_after_min=0,
        spatial_resolution_km=window.spatial_resolution_km,
        resolution_min=window.resolution_min,
    )

    df = fetch_result.station_5m
    if df is None or df.empty:
        empty = np.full((window.n_space, n_time_pre), np.nan, dtype=np.float32)
        return empty, {"filled_cells": 0, "total_cells": int(empty.size), "filled_ratio": 0.0}

    sliced = slice_station_df(
        df,
        accident_mileage_m=accident.mileage_m,
        direction=accident.direction,
        freeway=accident.freeway,
        report_time=pd.Timestamp(accident.report_time),
        window=pre_window,
    )
    matrix = rasterize_window(sliced, pre_window, value_col="speed_x10")
    return matrix.astype(np.float32), window_coverage(matrix)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def derive_duration_min(
    heatmap: np.ndarray,
    accident: AccidentInput,
    window: SampleWindow,
    threshold_kmh: float = FREE_FLOW_THRESHOLD_KMH,
) -> float:
    """Return minutes from t=0 until the accident-mileage cell is back to free-flow.

    Scans the post-accident slice of the heatmap at the accident's mileage row;
    duration is the first time-offset >= 0 where speed >= threshold and stays
    above for the remainder of the window. Falls back to the full window length
    when recovery is never observed.
    """
    abs_mileage_m = absolute_mileage_axis_m(accident.mileage_m, accident.direction, window)
    space_idx = int(np.argmin(np.abs(abs_mileage_m - accident.mileage_m)))

    time_offsets = window.time_offsets_min
    post_mask = time_offsets >= 0
    if not post_mask.any():
        return 0.0

    post_speeds = heatmap[space_idx, post_mask]
    post_times = time_offsets[post_mask]

    above = post_speeds >= threshold_kmh
    if not above.any():
        return float(post_times[-1])

    # First index where the trailing slice is uniformly above threshold.
    for idx in range(len(above)):
        if above[idx:].all():
            return float(post_times[idx])
    return float(post_times[-1])


def build_recovery_curves(
    heatmap: np.ndarray,
    accident: AccidentInput,
    window: SampleWindow,
    sample_km_offsets: tuple[int, ...] = (-5, 0, 5, 10, 20),
) -> dict[float, np.ndarray]:
    """Per-mileage speed curves at a few representative slices."""
    abs_mileage_m = absolute_mileage_axis_m(accident.mileage_m, accident.direction, window)
    out: dict[float, np.ndarray] = {}
    for delta_km in sample_km_offsets:
        target_m = accident.mileage_m + (
            -delta_km * 1000 if accident.direction in ("S", "E") else delta_km * 1000
        )
        if not (abs_mileage_m.min() <= target_m <= abs_mileage_m.max()):
            continue
        space_idx = int(np.argmin(np.abs(abs_mileage_m - target_m)))
        out[round(abs_mileage_m[space_idx] / 1000.0, 1)] = heatmap[space_idx].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def predict_accident_impact(
    accident: AccidentInput,
    data_root: Path | str | None = None,
    fetch_live_vd: LiveVDFetcher | None = None,
    forward_fn: ForwardFn | None = None,
    model_name: str | None = None,
    run_id: str | None = None,
    experiment: str | None = None,
    window: SampleWindow | None = None,
) -> AccidentPrediction:
    """Run end-to-end inference for one fresh accident.

    See `docs/DEPLOYMENT.md` for the consumer-side contract.
    """
    if pd.Timestamp(accident.report_time) > pd.Timestamp.now():
        raise ValueError("Accident report_time is in the future.")
    if accident.mileage_km <= 0:
        raise ValueError("Mileage must be positive.")

    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT

    # 0. Resolve model and window.
    # When using the real checkpoint, load it early so its window drives all
    # artifact assembly.  If loading fails, fall through with a safe window and
    # raise ModelNotAvailableError after the model_input is assembled so the
    # GUI can still show fetch / coverage diagnostics.
    _model: TrafficModel | None = None
    _load_error: Exception | None = None
    if forward_fn is None:
        try:
            _model = load(model_name, run_id=run_id, experiment=experiment)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            _load_error = exc
    win = (
        window
        if window is not None
        else (_model.window if _model is not None else DEFAULT_WINDOW)
    )

    # 1. Validate against VD coverage.
    _validate_against_vd_dim(accident, root)

    # 2. Live VD fetch (pre-accident slice only).
    fetcher = fetch_live_vd or _default_live_fetcher(accident, win)
    fetch_start = pd.Timestamp(accident.report_time) - pd.Timedelta(minutes=win.time_before_min)
    fetch_end = pd.Timestamp(accident.report_time)
    fetch_result = fetcher(fetch_start.to_pydatetime(), fetch_end.to_pydatetime())

    observed_pre, cov_pre = build_observed_pre(accident, win, fetch_result)

    # 3. Historical baseline.
    profile_df = _load_profile_for_accident(accident, root, win)
    baseline, cov_baseline = build_historical_baseline(accident, win, profile_df)

    model_input = ModelInput(
        accident=accident,
        window=win,
        observed_pre=observed_pre,
        historical_baseline=baseline,
        incident_features=accident.features,
        coverage_pre=cov_pre,
        coverage_baseline=cov_baseline,
        fetch_status=fetch_result.status,
    )

    # 4. Forward.
    if forward_fn is not None:
        heatmap = forward_fn(model_input)
    elif _load_error is not None:
        raise ModelNotAvailableError(
            f"Checkpoint could not be loaded: {_load_error}. "
            "Upstream pipeline succeeded — see model_input on this exception.",
            model_input=model_input,
        ) from _load_error
    else:
        heatmap = _adapt_traffic_model(_model, model_input)

    if heatmap.shape != (win.n_space, win.n_time):
        raise ValueError(
            f"forward_fn returned shape {heatmap.shape}, expected ({win.n_space}, {win.n_time})"
        )

    # 5. Post-process.
    duration = derive_duration_min(heatmap, accident, win)
    curves = build_recovery_curves(heatmap, accident, win)

    coverage = {
        "filled_ratio_pre": cov_pre["filled_ratio"],
        "filled_ratio_baseline": cov_baseline["filled_ratio"],
        "vd_count": int(fetch_result.station_5m["vdid"].nunique()) if not fetch_result.station_5m.empty else 0,
        "live_fetch_status": fetch_result.status,
        "model_status": "checkpoint" if forward_fn is None else "forward_fn",
    }
    return AccidentPrediction(
        heatmap=heatmap.astype(np.float32),
        window=win,
        duration_min=duration,
        recovery_curves=curves,
        coverage=coverage,
        input_echo={
            "report_time": accident.report_time.isoformat(),
            "freeway": accident.freeway,
            "direction": accident.direction,
            "mileage_km": accident.mileage_km,
        },
    )


def _adapt_traffic_model(model: TrafficModel, model_input: ModelInput) -> np.ndarray:
    """Delegate to _CheckpointModel.predict_from_input for live-accident inference."""
    if not hasattr(model, "predict_from_input"):
        raise TypeError(
            f"{type(model).__name__} does not implement predict_from_input(). "
            "Load the model via inference.load() to get a compatible wrapper."
        )
    return model.predict_from_input(model_input)  # type: ignore[union-attr]


def _load_profile_for_accident(
    accident: AccidentInput,
    data_root: Path,
    window: SampleWindow,
) -> pd.DataFrame:
    """Read the historical profile slice for this accident's freeway."""
    base = data_root / f"historical_speed_profile_{window.resolution_min}m"
    if not base.exists():
        logger.warning(
            "historical_speed_profile_%dm not found at %s; baseline channel will be empty. "
            "Build with scripts/data/build_historical_profile.py.",
            window.resolution_min,
            base,
        )
        return pd.DataFrame()

    from .data_loaders import read_historical_profile

    try:
        return read_historical_profile(
            resolution_min=window.resolution_min,
            data_root=data_root,
            freeway=accident.freeway,
        )
    except FileNotFoundError:
        return pd.DataFrame()
