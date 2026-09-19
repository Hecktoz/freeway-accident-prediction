# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "torch",
#     "torch-directml; sys_platform == 'win32'",
#     "tqdm",
#     "pyarrow",
# ]
# ///
# ruff: noqa: E402
"""
Train feature-fusion U-Net variants and an optional TIID decay head.

Supported feature-injection methods:
- direct_concat: broadcast a learned 8-channel feature embedding as extra input channels.
- mid_inject:    add a learned 512-d feature embedding to the bottleneck feature map.
- attention:     gate the bottleneck channels with a sigmoid feature embedding.

Optional TIID head:
- Predicts a monotonic non-increasing decay curve over the post-accident horizon.
- Penalty = decay * baseline-speed * tiid_scale, subtracted from the base U-Net forecast.

Inputs:
- `data/accident_samples_5m.npz` (built by build_accident_samples.py)
- For formal paper runs, the immutable `incident_features_paper_trb_2026.csv`
  and its sibling manifest (built by build_incident_features.py; value columns
  plus paired avail_* indicators; `avail_mainline` is QC-only and excluded).
  Continuous fields are standardized from the training split only; binary and
  no-weather fields remain unchanged.

Outputs (per experiment, named `<method>` or `<method>_tiid`):
- `output/models/unet_feature_fusion/<run_id>/<experiment>/checkpoints/best_model.pth`
- `output/models/unet_feature_fusion/<run_id>/<experiment>/plots/curves.png`
- `output/models/unet_feature_fusion/<run_id>/<experiment>/eval/metrics.csv` and `summary.txt`
- `output/models/unet_feature_fusion/<run_id>/comparison_summary.csv` and `.md`

Usage:
  uv run scripts/models/unet_feature_fusion.py --epochs 30
  uv run scripts/models/unet_feature_fusion.py --methods direct_concat,mid_inject --with-tiid
  uv run scripts/models/unet_feature_fusion.py --epochs 50 --methods attention \
      --eval-protocol v1 --input-type vd+hist --feature-set t0 \
      --loss speed_pace_hybrid --speed-loss-lambda 1 --pace-loss-lambda 1 \
      --pace-v-min-kmh 5 \
      --feature-csv output/feature_engineering/incident_features_paper_trb_2026.csv \
      --historical-matrices-npz data/paper_trb_2026/historical_matrices_5m_pre20240401_n1n3.npz \
      --congestion-mask-npz data/paper_trb_2026/congestion_masks_origin_cc_5m_pre20240401_n1n3.npz
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import os as _os
if _os.environ.get("MPLBACKEND", "").startswith("module://"):
    _os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.data_loaders import load_samples
from scripts.lib.model_architectures import ALL_BACKBONES, WindowConfig, build_feature_fusion_unet
from scripts.lib.windows import SampleWindow, subset
from scripts.models.feature_normalization import (
    fit_transform_train_only,
    identity_manifest,
)
from scripts.models.paper_protocol import (
    DEFAULT_ALLOWED_FREEWAYS,
    DEFAULT_SPLIT_SEED,
    artifact_manifest,
    cohort_manifest,
    configure_reproducibility,
    feature_schema_manifest,
    hash_strings,
    load_samples_artifact,
    make_data_loader,
    parse_allowed_freeways,
    split_indices,
    write_split_membership,
)
from scripts.models.unet_matrix import load_congestion_masks_aligned


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "models" / "unet_feature_fusion"
DEFAULT_FEATURE_CSV = PROJECT_ROOT / "output" / "feature_engineering" / "incident_features_44.csv"

SPEED_MAX = 140.0
PAD_TARGET = 64
FREE_FLOW_KMH = 80.0
# eval_protocol_v1 (2026-05-29): upstream congested weight `w` swept over {2,4,7,10}
# (upstream non-congested = 1, downstream = 0); upstream loss/metric zone capped at 20 km.
CONGESTION_WEIGHT_DEFAULT = 4.0
PACE_V_MIN_KMH_DEFAULT = 5.0
SPEED_LOSS_LAMBDA_DEFAULT = 1.0
PACE_LOSS_LAMBDA_DEFAULT = 1.0
CONGESTION_UPSTREAM_KM_DEFAULT = 20
TRAIN_END_DATE = "2024-11-30"
VAL_END_DATE   = "2025-02-28"

# Feature columns zeroed out for each time-progressive training variant.
# See reports/feature_time_grouping.md for rationale.
_T0_ZERO_COLS: frozenset[str] = frozenset({
    "injuries", "fatalities", "lane_blockage_rate",
    "special_accident", "accident_A1", "accident_A2", "accident_A3",
})
_T5_ZERO_COLS: frozenset[str] = frozenset({
    "injuries", "fatalities", "lane_blockage_rate",
})

ALL_METHODS = ("direct_concat", "mid_inject", "attention")


def select_device(preference: str = "auto") -> tuple[torch.device, str]:
    if preference == "cpu":
        return torch.device("cpu"), "cpu (forced)"
    if torch.cuda.is_available():
        return torch.device("cuda"), f"cuda ({torch.cuda.get_device_name(0)})"
    try:
        import torch_directml
        dml = torch_directml.device()
        return dml, f"directml ({torch_directml.device_name(dml.index)})"
    except Exception:
        pass
    return torch.device("cpu"), "cpu"


@dataclass
class TrainConfig:
    upstream_km: int = 25
    downstream_km: int = 10
    time_before_min: int = 30
    time_after_min: int = 145
    spatial_resolution_km: int = 1
    resolution_min: int = 5
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-4
    val_split: float = 0.15
    random_state: int = 42
    congestion_kmh: float = 60.0
    congestion_weight: float = CONGESTION_WEIGHT_DEFAULT  # upstream congested weight `w`
    congestion_upstream_km: int = CONGESTION_UPSTREAM_KM_DEFAULT  # loss/metric upstream cap (0~+20)
    tiid_lambda: float = 0.3
    min_before_coverage: float = 0.0
    min_filled_ratio: float = 0.8
    # eval_protocol_v1 fields
    eval_protocol: str = ""
    data_scope: str = "dispatch_only"
    # eval_protocol_v1 (2026-05-29): current round grounded on t0; t5/t15 paused
    # pending the feature data-spec replan (planning_next_steps_20260529 §0/§8).
    feature_set: str = "t0"
    input_type: str = "vd+hist"
    backbone: str = "simple"  # simple | residual_attention
    loss_name: str = "upstream_congestion_weighted"
    checkpoint_selection_metric: str = "validation_loss"  # validation_loss | cell_mae
    speed_loss_lambda: float = SPEED_LOSS_LAMBDA_DEFAULT
    pace_loss_lambda: float = PACE_LOSS_LAMBDA_DEFAULT
    pace_v_min_kmh: float = PACE_V_MIN_KMH_DEFAULT
    congestion_mask_npz: str = ""
    train_end_date: str = TRAIN_END_DATE
    val_end_date: str = VAL_END_DATE
    split_mode: str = "time"
    split_seed: int = DEFAULT_SPLIT_SEED
    allowed_freeways: str = ",".join(DEFAULT_ALLOWED_FREEWAYS)
    deterministic_mode: str = "warn"
    device_preference: str = "auto"
    historical_matrices_npz: str = ""
    smoke: bool = False
    validation_only: bool = False
    export_val_predictions: bool = False
    eval_checkpoint: str = ""
    export_test_predictions: bool = False

    @property
    def base_in_channels(self) -> int:
        """Locked matrix stream: VD speed, VD validity, historical speed."""
        if self.input_type != "vd+hist":
            raise ValueError("feature-fusion U-Net is locked to the 3-channel vd+hist input")
        return 3

    @property
    def n_space(self) -> int:
        return (self.upstream_km + self.downstream_km) // self.spatial_resolution_km + 1

    @property
    def n_time(self) -> int:
        return (self.time_before_min + self.time_after_min) // self.resolution_min + 1

    @property
    def n_before(self) -> int:
        return self.time_before_min // self.resolution_min

    @property
    def n_after(self) -> int:
        return self.n_time - self.n_before

    @property
    def pad_h(self) -> int:
        return (PAD_TARGET - self.n_space) // 2

    @property
    def pad_w(self) -> int:
        return (PAD_TARGET - self.n_time) // 2


@dataclass
class ExperimentSpec:
    method: str
    use_tiid: bool

    @property
    def tag(self) -> str:
        return f"{self.method}{'_tiid' if self.use_tiid else ''}"


# ============================================================
# Dataset
# ============================================================


def compute_decay_target(speed_norm: np.ndarray, n_before: int) -> np.ndarray:
    """Build a monotonic non-increasing decay target from the future trajectory.

    decay_target[t] = clip((baseline - mean_speed_t) / baseline, 0, 1), accumulated min.
    Used as supervision for the optional TIID head.
    """
    before = speed_norm[:, :n_before]
    after = speed_norm[:, n_before:]
    baseline = np.nanmean(before)
    if np.isnan(baseline) or baseline <= 1e-6:
        return np.zeros(after.shape[1], dtype=np.float32)
    future_mean = np.nanmean(after, axis=0)
    impact = (baseline - future_mean) / baseline
    impact = np.nan_to_num(impact, nan=0.0)
    impact = np.clip(impact, 0.0, 1.0)
    return np.minimum.accumulate(impact).astype(np.float32)


class AccidentFeatureDataset(Dataset):
    def __init__(
        self,
        matrices: np.ndarray,
        accident_ids: np.ndarray,
        feature_array: np.ndarray,
        config: TrainConfig,
        hist_matrices: np.ndarray | None = None,
    ) -> None:
        if matrices.shape[0] != feature_array.shape[0]:
            raise ValueError("matrix count and feature row count must match")
        self.matrices = matrices.astype(np.float32)
        self.accident_ids = accident_ids
        self.features = feature_array.astype(np.float32)
        self.config = config
        self.hist_matrices = hist_matrices.astype(np.float32) if hist_matrices is not None else None
        _ = config.base_in_channels
        if self.hist_matrices is None:
            raise ValueError("3-channel feature-fusion U-Net requires causal historical matrices")

    def __len__(self) -> int:
        return self.matrices.shape[0]

    @staticmethod
    def _nearest_space_fill(matrix: np.ndarray) -> np.ndarray:
        """Fill historical NaNs from the nearest row in the same time column."""
        out = matrix.copy()
        for time_idx in range(out.shape[1]):
            col = out[:, time_idx]
            valid_idx = np.flatnonzero(np.isfinite(col))
            missing_idx = np.flatnonzero(~np.isfinite(col))
            if not len(missing_idx):
                continue
            if not len(valid_idx):
                raise ValueError(
                    f"historical profile has no valid spatial value at time column {time_idx}"
                )
            nearest = np.abs(missing_idx[:, None] - valid_idx[None, :]).argmin(axis=1)
            col[missing_idx] = col[valid_idx[nearest]]
            out[:, time_idx] = col
        return out

    def __getitem__(self, idx: int):
        cfg = self.config
        arr = self.matrices[idx]
        speed_norm = np.where(np.isnan(arr), np.nan, np.clip(arr, 0.0, SPEED_MAX) / SPEED_MAX)

        vd_spd = np.nan_to_num(speed_norm.copy(), nan=0.0)
        vd_spd[:, cfg.n_before:] = 0.0
        vd_val = np.zeros((cfg.n_space, cfg.n_time), dtype=np.float32)
        vd_val[:, : cfg.n_before] = (~np.isnan(speed_norm[:, : cfg.n_before])).astype(np.float32)
        historical = self._nearest_space_fill(self.hist_matrices[idx])
        hist_spd = (np.clip(historical, 0.0, SPEED_MAX) / SPEED_MAX).astype(np.float32)
        channels = [vd_spd, vd_val, hist_spd]

        x = torch.from_numpy(np.stack(channels, axis=0)).float()
        x = F.pad(x, (cfg.pad_w, cfg.pad_w, cfg.pad_h, cfg.pad_h), mode="reflect")

        target_raw = speed_norm[:, cfg.n_before:]
        valid = (~np.isnan(target_raw)).astype(np.float32)
        target = np.nan_to_num(target_raw, nan=0.0)
        decay = compute_decay_target(speed_norm, cfg.n_before)

        return (
            x,
            torch.from_numpy(target).float(),
            torch.from_numpy(valid).float(),
            torch.from_numpy(self.features[idx]).float(),
            torch.from_numpy(decay).float(),
            str(self.accident_ids[idx]),
        )


# ============================================================
# Model
# ============================================================


class FeatureFusionUNet(nn.Module):
    def __init__(self, feature_dim: int, method: str, use_tiid: bool, config: TrainConfig) -> None:
        super().__init__()
        if method not in ALL_METHODS:
            raise ValueError(f"Unsupported method: {method}")
        self.method = method
        self.use_tiid = use_tiid
        self.cfg = config

        base_ch = config.base_in_channels
        if method == "direct_concat":
            self.feature_to_input = nn.Sequential(
                nn.Linear(feature_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.Tanh(),
            )
            in_ch = base_ch + 8
        else:
            in_ch = base_ch

        if method == "mid_inject":
            self.feature_to_bottleneck = nn.Linear(feature_dim, 512)
        if method == "attention":
            self.feature_to_gate = nn.Sequential(nn.Linear(feature_dim, 512), nn.Sigmoid())

        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))

        self.dec4 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())

        self.final = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid())

        if self.use_tiid:
            self.decay_head = nn.Sequential(
                nn.Linear(512 + feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, config.n_after),
            )
            self.tiid_scale = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))

    def forward(self, x: torch.Tensor, feat: torch.Tensor):
        cfg = self.cfg

        if self.method == "direct_concat":
            feat_ch = self.feature_to_input(feat).unsqueeze(-1).unsqueeze(-1)
            feat_map = feat_ch.expand(-1, -1, x.shape[2], x.shape[3])
            x_in = torch.cat([x, feat_map], dim=1)
        else:
            x_in = x

        e1 = self.enc1(x_in)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        if self.method == "mid_inject":
            e4 = e4 + self.feature_to_bottleneck(feat).unsqueeze(-1).unsqueeze(-1)
        if self.method == "attention":
            e4 = e4 * self.feature_to_gate(feat).unsqueeze(-1).unsqueeze(-1)

        d4 = torch.cat([self.dec4(e4), e3], dim=1)
        d3 = torch.cat([self.dec3(d4), e2], dim=1)
        d2 = torch.cat([self.dec2(d3), e1], dim=1)
        d1 = self.dec1(d2)

        out = self.final(d1)
        out = out[:, :, cfg.pad_h : cfg.pad_h + cfg.n_space, cfg.pad_w : cfg.pad_w + cfg.n_time]
        out = out.squeeze(1)
        base_pred = out[:, :, cfg.n_before:]

        if not self.use_tiid:
            return base_pred, None

        pooled = F.adaptive_avg_pool2d(e4, 1).flatten(1)
        decay_curve = torch.sigmoid(self.decay_head(torch.cat([pooled, feat], dim=1)))
        decay_curve = torch.cummin(decay_curve, dim=1).values

        context = x[:, 0, cfg.pad_h : cfg.pad_h + cfg.n_space, cfg.pad_w : cfg.pad_w + cfg.n_before]
        baseline = context.mean(dim=(1, 2))
        penalty = decay_curve * baseline.unsqueeze(1) * torch.clamp(self.tiid_scale, min=0.0)
        penalty = penalty.unsqueeze(1).expand(-1, cfg.n_space, -1)
        pred = torch.clamp(base_pred - penalty, 0.0, 1.0)
        return pred, decay_curve


class CongestionWeightedLoss(nn.Module):
    def __init__(self, congestion_kmh: float, congestion_weight: float) -> None:
        super().__init__()
        self.threshold = congestion_kmh / SPEED_MAX
        self.w = congestion_weight

    def forward(self, pred, target, valid_mask, congestion_mask=None):
        l1 = torch.abs(pred - target)
        cong = (target < self.threshold).float()
        weight = 1.0 + (self.w - 1.0) * cong
        weighted = l1 * weight * valid_mask
        return weighted.sum() / valid_mask.sum().clamp(min=1)


class UpstreamCongestionWeightedLoss(nn.Module):
    """eval_protocol_v1 default loss (2026-05-29 weighting).

    Upstream `0~+congestion_upstream_km` (default 20 km) zone: congested → w
    (sweep {2,4,7,10}), non-congested → 1, downstream / beyond +20 km → 0.
    `congestion_mask` (origin_cc contour), when supplied, defines congested cells;
    otherwise `target < threshold` is the fallback (eval_protocol_v1 §3.1).
    """

    def __init__(self, upstream_start_idx: int, upstream_end_idx: int,
                 congested_weight: float, congestion_threshold: float = 60.0) -> None:
        super().__init__()
        self.up_start = upstream_start_idx
        self.up_end = upstream_end_idx
        self.w = congested_weight
        self.thresh = congestion_threshold / SPEED_MAX

    def forward(self, pred, target, valid_mask, congestion_mask=None):
        # Upstream-only, capped at +congestion_upstream_km. Proper weighted mean:
        # denominator = sum of weights over valid cells.
        s, e = self.up_start, self.up_end
        up_valid = valid_mask[:, s:e, :]
        up_tgt = target[:, s:e, :]
        if congestion_mask is not None:
            cong = congestion_mask[:, s:e, :].float()
        else:
            cong = (up_tgt < self.thresh).float()
        weight = (1.0 + (self.w - 1.0) * cong) * up_valid
        l1 = torch.abs(pred[:, s:e, :] - up_tgt) * weight
        return l1.sum() / weight.sum().clamp(min=1)


class FullFieldCongestionWeightedLoss(nn.Module):
    """Matched paper loss: unit weight over the field and weight ``w`` on AIC.

    The AIC focus is restricted to the configured upstream evaluation window;
    every other valid target cell, including downstream and farther-upstream
    context, retains unit weight.  Concatenation is used instead of in-place
    slice assignment for CPU/DirectML parity and empty edge segments are
    omitted for cropped spatial windows.
    """

    def __init__(
        self,
        upstream_start_idx: int,
        upstream_end_idx: int,
        congested_weight: float,
        congestion_threshold: float,
    ) -> None:
        super().__init__()
        if congested_weight < 1:
            raise ValueError("congested_weight must be >= 1")
        self.up_start = upstream_start_idx
        self.up_end = upstream_end_idx
        self.w = float(congested_weight)
        self.thresh = congestion_threshold / SPEED_MAX

    def forward(self, pred, target, valid_mask, congestion_mask=None):
        s, e = self.up_start, self.up_end
        if congestion_mask is not None:
            focus = congestion_mask[:, s:e, :].to(pred.dtype)
        else:
            focus = (target[:, s:e, :] < self.thresh).to(pred.dtype)
        parts = []
        if s > 0:
            parts.append(torch.ones_like(valid_mask[:, :s, :]))
        parts.append(1.0 + (self.w - 1.0) * focus)
        if e < valid_mask.shape[1]:
            parts.append(torch.ones_like(valid_mask[:, e:, :]))
        multiplier = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        weight = multiplier * valid_mask
        return (torch.abs(pred - target) * weight).sum() / weight.sum().clamp(min=1)


class CongestionOnlyLoss(nn.Module):
    def __init__(self, congestion_threshold: float = 60.0) -> None:
        super().__init__()
        self.thresh = congestion_threshold / SPEED_MAX

    def forward(self, pred, target, valid_mask, congestion_mask=None):
        cong = (target < self.thresh).float()
        mask = valid_mask * cong
        n = mask.sum().clamp(min=1)
        if n < 1:
            return (torch.abs(pred - target) * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        return (torch.abs(pred - target) * mask).sum() / n


class SpeedPaceHybridLoss(nn.Module):
    """General speed MAE plus upstream true-pace MAE.

    The speed term is normalized speed MAE over all valid cells. The pace term
    is true pace, 60/max(v, v_min), in min/km over the upstream loss zone, scaled
    by the physically possible pace range so lambda values are interpretable.
    """

    def __init__(
        self,
        upstream_start_idx: int,
        upstream_end_idx: int,
        speed_lambda: float,
        pace_lambda: float,
        pace_v_min_kmh: float,
    ) -> None:
        super().__init__()
        if pace_v_min_kmh <= 0 or pace_v_min_kmh >= SPEED_MAX:
            raise ValueError(f"pace_v_min_kmh must be in (0, {SPEED_MAX}), got {pace_v_min_kmh}")
        self.up_start = upstream_start_idx
        self.up_end = upstream_end_idx
        self.speed_lambda = speed_lambda
        self.pace_lambda = pace_lambda
        self.v_min = pace_v_min_kmh
        self.pace_span = (60.0 / pace_v_min_kmh) - (60.0 / SPEED_MAX)

    def forward(self, pred, target, valid_mask, congestion_mask=None):
        speed_loss = (torch.abs(pred - target) * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        s, e = self.up_start, self.up_end
        up_valid = valid_mask[:, s:e, :]
        pred_kmh = torch.clamp(pred[:, s:e, :] * SPEED_MAX, min=self.v_min)
        target_kmh = torch.clamp(target[:, s:e, :] * SPEED_MAX, min=self.v_min)
        pred_pace = 60.0 / pred_kmh
        target_pace = 60.0 / target_kmh
        pace_loss = (
            (torch.abs(pred_pace - target_pace) / self.pace_span) * up_valid
        ).sum() / up_valid.sum().clamp(min=1)

        return self.speed_lambda * speed_loss + self.pace_lambda * pace_loss


def _upstream_start_idx(cfg: "TrainConfig") -> int:
    return cfg.downstream_km // cfg.spatial_resolution_km


def _upstream_end_idx(cfg: "TrainConfig") -> int:
    """Exclusive row index capping the upstream loss/metric zone at +congestion_upstream_km."""
    cap = _upstream_start_idx(cfg) + cfg.congestion_upstream_km // cfg.spatial_resolution_km + 1
    return min(cap, cfg.n_space)


def build_loss(loss_name: str, cfg: "TrainConfig") -> nn.Module:
    up_start = _upstream_start_idx(cfg)
    up_end = _upstream_end_idx(cfg)
    if loss_name == "full_field_congestion_weighted":
        return FullFieldCongestionWeightedLoss(
            up_start,
            up_end,
            cfg.congestion_weight,
            cfg.congestion_kmh,
        )
    if loss_name == "upstream_congestion_weighted":
        return UpstreamCongestionWeightedLoss(up_start, up_end, cfg.congestion_weight, cfg.congestion_kmh)
    if loss_name == "speed_pace_hybrid":
        return SpeedPaceHybridLoss(
            up_start,
            up_end,
            cfg.speed_loss_lambda,
            cfg.pace_loss_lambda,
            cfg.pace_v_min_kmh,
        )
    if loss_name == "congestion_only_mae":
        return CongestionOnlyLoss(cfg.congestion_kmh)
    return CongestionWeightedLoss(cfg.congestion_kmh, 1.0)


def _impact_time(speed_kmh: np.ndarray, upstream_start_idx: int, res_min: int, mask: np.ndarray,
                 ff: float = FREE_FLOW_KMH, upstream_end_idx: int | None = None) -> float:
    # Upstream zone 0~+20 km (eval_protocol_v1 §4.3); upstream_end_idx=None means to the edge.
    for j in range(speed_kmh.shape[1] - 1, -1, -1):
        vm = mask[upstream_start_idx:upstream_end_idx, j]
        if vm.sum() == 0:
            continue
        if (speed_kmh[upstream_start_idx:upstream_end_idx, j] * vm).sum() / vm.sum() < ff:
            return float((j + 1) * res_min)
    return 0.0


def _recovery_time(speed_kmh: np.ndarray, upstream_start_idx: int, res_min: int, mask: np.ndarray,
                   ff: float = FREE_FLOW_KMH, sustained: int = 3) -> float:
    near_end = min(speed_kmh.shape[0], upstream_start_idx + 11)
    near = speed_kmh[upstream_start_idx:near_end, :]
    near_m = mask[upstream_start_idx:near_end, :]
    nt = near.shape[1]
    for j in range(nt):
        c = 0
        for k in range(j, min(j + sustained, nt)):
            vm = near_m[:, k]
            if vm.sum() == 0:
                break
            if (near[:, k] * vm).sum() / vm.sum() >= ff:
                c += 1
            else:
                break
        if c >= sustained:
            return float(j * res_min)
    return float(nt * res_min)


def compute_operational_metrics(pred_kmh: np.ndarray, tgt_kmh: np.ndarray, mask: np.ndarray,
                                 upstream_start_idx: int, res_min: int, cong_kmh: float = 60.0,
                                 upstream_end_idx: int | None = None,
                                 congestion_mask: np.ndarray | None = None) -> dict:
    # impact time + congestion MAE over upstream 0~+20 km; recovery stays 0~+10 km.
    # congested cells = origin_cc contour when given, else speed<cong_kmh fallback.
    it_obs  = _impact_time(tgt_kmh,  upstream_start_idx, res_min, mask, upstream_end_idx=upstream_end_idx)
    it_pred = _impact_time(pred_kmh, upstream_start_idx, res_min, mask, upstream_end_idx=upstream_end_idx)
    rt_obs  = _recovery_time(tgt_kmh,  upstream_start_idx, res_min, mask)
    rt_pred = _recovery_time(pred_kmh, upstream_start_idx, res_min, mask)
    up_obs  = tgt_kmh[upstream_start_idx:upstream_end_idx, :]
    up_pred = pred_kmh[upstream_start_idx:upstream_end_idx, :]
    up_m    = mask[upstream_start_idx:upstream_end_idx, :]
    if congestion_mask is not None:
        cong = (congestion_mask[upstream_start_idx:upstream_end_idx, :] > 0) & (up_m > 0)
    else:
        cong = (up_obs < cong_kmh) & (up_m > 0)
    uc_mae = float(np.abs(up_pred[cong] - up_obs[cong]).mean()) if cong.sum() > 0 else float("nan")
    return {
        "impact_time_obs_min": it_obs, "impact_time_pred_min": it_pred,
        "impact_time_err_min": it_pred - it_obs,
        "recovery_time_obs_min": rt_obs, "recovery_time_pred_min": rt_pred,
        "recovery_time_err_min": rt_pred - rt_obs,
        "upstream_congestion_mae_kmh": uc_mae,
    }


def time_based_split(ids: np.ndarray, data_root: Path, train_end: str, val_end: str):
    meta = pd.read_parquet(data_root / "accident_samples_5m_metadata.parquet")
    train_idx, val_idx, test_idx, _ = split_indices(
        ids,
        meta,
        split_mode="time",
        split_seed=DEFAULT_SPLIT_SEED,
        train_end=train_end,
        val_end=val_end,
    )
    return train_idx, val_idx, test_idx


def compute_metrics(
    pred,
    target,
    valid_mask,
    congestion_kmh: float,
    upstream_start_idx: int = 0,
    upstream_end_idx: int | None = None,
    congestion_mask=None,
) -> dict:
    """Cell-wise MAE/RMSE + upstream diagnostics + optional origin_cc MAE.

    Upstream zone is `[upstream_start_idx:upstream_end_idx]` (0~+20 km per
    eval_protocol_v1 §4.1; `upstream_end_idx=None` means to the matrix edge).
    When `congestion_mask` is supplied, `upstream_congestion_mae` is the
    origin_cc masked MAE; otherwise it falls back to the speed<congestion_kmh selector.
    Supports 2-D single-sample (n_space, n_after) and 3-D batch (B, n_space, n_after).
    """
    p = pred * SPEED_MAX
    t = target * SPEED_MAX
    valid = valid_mask.bool()
    _nan = float("nan")
    if valid.sum() == 0:
        return {
            "mae": _nan,
            "rmse": _nan,
            "upstream_mae": _nan,
            "speed60_mae": _nan,
            "origin_cc_mae": _nan,
            "upstream_congestion_mae": _nan,
            "f1": _nan,
            "has_congestion": False,
            "has_origin_cc": False,
        }
    pv = p[valid]
    tv = t[valid]
    mae  = torch.abs(pv - tv).mean().item()
    rmse = torch.sqrt(((pv - tv) ** 2).mean()).item()

    if pred.dim() == 3:
        up_p = p[:, upstream_start_idx:upstream_end_idx, :]
        up_t = t[:, upstream_start_idx:upstream_end_idx, :]
        up_m = valid_mask[:, upstream_start_idx:upstream_end_idx, :].bool()
        up_c = congestion_mask[:, upstream_start_idx:upstream_end_idx, :].bool() if congestion_mask is not None else None
    else:
        up_p = p[upstream_start_idx:upstream_end_idx, :]
        up_t = t[upstream_start_idx:upstream_end_idx, :]
        up_m = valid_mask[upstream_start_idx:upstream_end_idx, :].bool()
        up_c = congestion_mask[upstream_start_idx:upstream_end_idx, :].bool() if congestion_mask is not None else None

    if up_m.sum() == 0:
        return {
            "mae": mae,
            "rmse": rmse,
            "upstream_mae": _nan,
            "speed60_mae": _nan,
            "origin_cc_mae": _nan,
            "upstream_congestion_mae": _nan,
            "f1": _nan,
            "has_congestion": False,
            "has_origin_cc": False,
        }

    up_pv = up_p[up_m]
    up_tv = up_t[up_m]
    upstream_mae = torch.abs(up_pv - up_tv).mean().item()
    pc = up_pv < congestion_kmh
    tc = up_tv < congestion_kmh
    has_congestion = bool(tc.any().item())
    speed60_mae = torch.abs(up_pv[tc] - up_tv[tc]).mean().item() if has_congestion else _nan
    if has_congestion:
        tp = (pc & tc).sum().float()
        fp = (pc & ~tc).sum().float()
        fn = (~pc & tc).sum().float()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = (2 * prec * rec / (prec + rec + 1e-8)).item()
    else:
        f1 = _nan
    origin_cc_mae = _nan
    has_origin_cc = False
    if up_c is not None:
        cc = up_c & up_m
        has_origin_cc = bool(cc.any().item())
        if has_origin_cc:
            origin_cc_mae = torch.abs(up_p[cc] - up_t[cc]).mean().item()
    upstream_congestion_mae = origin_cc_mae if up_c is not None else speed60_mae
    return {
        "mae": mae,
        "rmse": rmse,
        "upstream_mae": upstream_mae,
        "speed60_mae": speed60_mae,
        "origin_cc_mae": origin_cc_mae,
        "upstream_congestion_mae": upstream_congestion_mae,
        "f1": f1,
        "has_congestion": has_congestion,
        "has_origin_cc": has_origin_cc,
    }


def plot_curves(history, save_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.plot(history["val_mae"], color="steelblue", label="val MAE")
    ax2.axhline(15, color="red", linestyle="--", label="target")
    ax2.set_title("Validation MAE")
    ax2.grid(alpha=0.3)
    ax2.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ============================================================
# Data prep
# ============================================================


def _apply_feature_time_mask(feat_array: np.ndarray, feat_cols: list[str], feature_set: str) -> np.ndarray:
    """Zero out columns unavailable at the given operational stage (t0/t5/t15).

    For each stage-masked value column we also zero its ``avail_<col>``
    indicator, so the model sees ``effective_avail = source_avail AND
    stage_avail``. Columns absent from the table (e.g. the legacy 33-col file
    without availability indicators) are simply skipped.
    """
    if feature_set == "t15":
        return feat_array
    zero_set = _T0_ZERO_COLS if feature_set == "t0" else _T5_ZERO_COLS
    avail_zero_set = {f"avail_{c}" for c in zero_set}
    result = feat_array.copy()
    for i, col in enumerate(feat_cols):
        if col in zero_set or col in avail_zero_set:
            result[:, i] = 0.0
    return result


def filter_subset_align_features(
    samples,
    feature_df: pd.DataFrame,
    target_window: SampleWindow,
    min_before_coverage: float,
    min_filled_ratio: float,
    feature_set: str = "t15",
    allowed_freeways: tuple[str, ...] | None = None,
):
    """Subset matrices, drop low-coverage samples, and align to feature rows by accident_id."""
    feat_index = feature_df.set_index("task_id")
    # `avail_mainline` is a data-quality flag for analysis/filtering only (added 2026-05-29);
    # it is NOT a model input, so the feature dimension stays locked even if the column is
    # present in the CSV (see planning_next_steps_20260529 §8 / build_incident_features.py).
    _QC_ONLY_COLS = {"task_id", "avail_mainline"}
    feat_cols = [c for c in feature_df.columns if c not in _QC_ONLY_COLS]
    feature_dim = len(feat_cols)
    meta = samples.metadata.copy()
    meta["accident_id"] = meta["accident_id"].astype(str).str.strip()
    filled_series = meta["filled_ratio"] if "filled_ratio" in meta.columns else pd.Series(1.0, index=meta.index)
    filled_by_id = dict(zip(meta["accident_id"], filled_series))
    freeway_by_id = {
        str(row.accident_id): str(row.freeway).strip().upper()
        for row in meta[["accident_id", "freeway"]].itertuples(index=False)
    } if "freeway" in meta.columns else {}
    allowed_set = set(allowed_freeways) if allowed_freeways is not None else None

    matrices, ids, features = [], [], []
    for i in range(len(samples)):
        accident_id = str(samples.accident_ids[i]).strip()
        if allowed_set is not None and freeway_by_id.get(accident_id) not in allowed_set:
            continue
        filled_ratio = float(filled_by_id.get(accident_id, 0.0) or 0.0)
        if filled_ratio < min_filled_ratio:
            continue
        if accident_id not in feat_index.index:
            continue
        m_full = samples.matrix[i]
        m, _ = subset(
            m_full,
            samples.window,
            upstream_km=target_window.upstream_km,
            downstream_km=target_window.downstream_km,
            time_before_min=target_window.time_before_min,
            time_after_min=target_window.time_after_min,
        )
        n_before = target_window.time_before_min // target_window.resolution_min
        before_block = m[:, :n_before]
        if min_before_coverage > 0 and (
            before_block.size == 0 or float(np.isnan(before_block).mean()) > 1.0 - min_before_coverage
        ):
            continue
        matrices.append(m)
        ids.append(accident_id)
        feat_row = pd.to_numeric(feat_index.loc[accident_id, feat_cols], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        features.append(feat_row)

    if not matrices:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=str),
            np.zeros((0, feature_dim), dtype=np.float32),
            feature_dim,
        )
    feat_arr = _apply_feature_time_mask(
        np.stack(features, axis=0).astype(np.float32), feat_cols, feature_set
    )
    return (
        np.stack(matrices, axis=0).astype(np.float32),
        np.asarray(ids),
        feat_arr,
        feature_dim,
    )


def load_hist_aligned(
    ids: np.ndarray,
    target_window: SampleWindow,
    data_root: Path,
    resolution_min: int,
    hist_tag: str = "",
    historical_matrices_npz: Path | None = None,
    require_complete: bool = False,
) -> np.ndarray:
    """Load historical-baseline matrices and align/subset them to `ids` order."""
    suffix = f"_{hist_tag}" if hist_tag else ""
    hist_path = (
        Path(historical_matrices_npz)
        if historical_matrices_npz is not None
        else data_root / f"historical_matrices_{resolution_min}m{suffix}.npz"
    )
    if not hist_path.exists():
        raise SystemExit(
            f"historical matrices NPZ not found: {hist_path}\n"
            f"Build it first: uv run scripts/data/build_historical_matrices.py"
            f"{' --tag ' + hist_tag if hist_tag else ''}"
        )
    arr = np.load(str(hist_path), allow_pickle=False)
    src_window = SampleWindow(
        upstream_km=int(arr["window_upstream_km"]), downstream_km=int(arr["window_downstream_km"]),
        time_before_min=int(arr["window_time_before_min"]), time_after_min=int(arr["window_time_after_min"]),
        spatial_resolution_km=int(arr["window_spatial_resolution_km"]), resolution_min=int(arr["window_resolution_min"]),
    )
    source_ids = arr["accident_ids"]
    source_matrix = arr["matrix"]
    id_to_idx = {str(a).strip(): i for i, a in enumerate(source_ids)}
    empty = np.full((target_window.n_space, target_window.n_time), np.nan, dtype=np.float32)
    out: list[np.ndarray] = []
    n_missing = 0
    for acc_id in ids:
        j = id_to_idx.get(str(acc_id).strip())
        if j is None:
            out.append(empty.copy())
            n_missing += 1
            continue
        sub, _ = subset(
            source_matrix[j], src_window,
            upstream_km=target_window.upstream_km, downstream_km=target_window.downstream_km,
            time_before_min=target_window.time_before_min, time_after_min=target_window.time_after_min,
        )
        out.append(sub.astype(np.float32))
    if n_missing:
        message = f"{n_missing}/{len(ids)} accidents missing from hist NPZ"
        if require_complete:
            raise SystemExit(message)
        print(f"  [warn] {message} -> NaN baseline")
    return np.stack(out, axis=0).astype(np.float32)


# ============================================================
# Single-experiment train loop
# ============================================================


def train_experiment(
    spec: ExperimentSpec,
    cfg: TrainConfig,
    feature_dim: int,
    train_ds: AccidentFeatureDataset,
    val_ds: AccidentFeatureDataset,
    output_root: Path,
    device: torch.device,
    target_window: SampleWindow,
    test_ds: "AccidentFeatureDataset | None" = None,
    cong_by_id: dict[str, np.ndarray] | None = None,
    protocol_info: dict | None = None,
) -> dict:
    exp_dir = output_root / spec.tag
    ckpt_dir = exp_dir / "checkpoints"
    plot_dir = exp_dir / "plots"
    eval_dir = exp_dir / "eval"
    for d in (ckpt_dir, plot_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Reset every experiment to the same base seed so results are independent
    # of method ordering in --methods.
    experiment_seed_info = configure_reproducibility(cfg.random_state, cfg.deterministic_mode)
    train_loader = make_data_loader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        training_seed=cfg.random_state,
    )
    val_loader = make_data_loader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        training_seed=cfg.random_state,
    )
    test_loader = (
        make_data_loader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            training_seed=cfg.random_state,
        )
        if test_ds
        else None
    )
    window_cfg = WindowConfig(
        upstream_km=cfg.upstream_km,
        downstream_km=cfg.downstream_km,
        time_before_min=cfg.time_before_min,
        time_after_min=cfg.time_after_min,
        spatial_resolution_km=cfg.spatial_resolution_km,
        resolution_min=cfg.resolution_min,
    )
    model = build_feature_fusion_unet(
        feature_dim=feature_dim,
        method=spec.method,
        use_tiid=spec.use_tiid,
        cfg=window_cfg,
        input_type=cfg.input_type,
        backbone=cfg.backbone,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion_speed = build_loss(cfg.loss_name, cfg).to(device)
    criterion_decay = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_cell_mae": [],
        "val_mae": [],
        "val_f1": [],
    }
    best_val_loss = float("inf")
    best_selection_value = float("inf")
    best_epoch = 0
    evaluation_only = bool(cfg.eval_checkpoint)
    print(f"\n=== {'Evaluating' if evaluation_only else 'Training'} {spec.tag} ===")
    print(f"train={len(train_ds)} | val={len(val_ds)} | backbone={cfg.backbone} | params={n_params:,}")

    def _gather_cmask(ids_batch, dev):
        if not cong_by_id:
            return None
        stacked = np.stack([cong_by_id[str(a)] for a in ids_batch], axis=0)
        return torch.from_numpy(stacked).to(dev)

    for epoch in range(1, 1 if evaluation_only else cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, m, feat, decay_target, ids_b in tqdm(train_loader, desc=f"{spec.tag} ep{epoch:02d}", leave=False):
            x, y, m, feat, decay_target = x.to(device), y.to(device), m.to(device), feat.to(device), decay_target.to(device)
            cmask = _gather_cmask(ids_b, device)
            optimizer.zero_grad()
            pred, decay_pred = model(x, feat)
            loss = criterion_speed(pred, y, m, cmask)
            if spec.use_tiid and decay_pred is not None:
                loss = loss + cfg.tiid_lambda * criterion_decay(decay_pred, decay_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        all_pred, all_y, all_m, all_c = [], [], [], []
        with torch.no_grad():
            for x, y, m, feat, decay_target, ids_b in val_loader:
                x, y, m, feat, decay_target = x.to(device), y.to(device), m.to(device), feat.to(device), decay_target.to(device)
                cmask = _gather_cmask(ids_b, device)
                pred, decay_pred = model(x, feat)
                loss = criterion_speed(pred, y, m, cmask)
                if spec.use_tiid and decay_pred is not None:
                    loss = loss + cfg.tiid_lambda * criterion_decay(decay_pred, decay_target)
                val_loss += loss.item()
                all_pred.append(pred.cpu())
                all_y.append(y.cpu())
                all_m.append(m.cpu())
                if cmask is not None:
                    all_c.append(cmask.cpu())
        val_loss /= max(len(val_loader), 1)
        up_start_i = _upstream_start_idx(cfg)
        cat_c = torch.cat(all_c) if all_c else None
        metrics = compute_metrics(torch.cat(all_pred), torch.cat(all_y), torch.cat(all_m),
                                  cfg.congestion_kmh, up_start_i, _upstream_end_idx(cfg), cat_c)
        cat_pred = torch.cat(all_pred)
        cat_y = torch.cat(all_y)
        cat_m = torch.cat(all_m)
        per_incident_num = (torch.abs(cat_pred - cat_y) * cat_m).flatten(start_dim=1).sum(dim=1)
        per_incident_den = cat_m.flatten(start_dim=1).sum(dim=1)
        present = per_incident_den > 0
        if not bool(present.any()):
            raise RuntimeError("validation split has no incident with a valid target cell")
        val_cell_mae = float(
            (per_incident_num[present] / per_incident_den[present]).mean().item() * SPEED_MAX
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cell_mae"].append(val_cell_mae)
        history["val_mae"].append(metrics["upstream_congestion_mae"])
        history["val_f1"].append(metrics["f1"])
        f1_str = f"{metrics['f1']:.3f}" if not np.isnan(metrics["f1"]) else "n/a"
        occ_str = f"{metrics['origin_cc_mae']:.2f}" if not np.isnan(metrics["origin_cc_mae"]) else "n/a"
        print(
            f"{spec.tag} | ep{epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"cell_MAE={val_cell_mae:.3f} km/h | "
            f"origin_cc_MAE@{cfg.congestion_upstream_km}={occ_str} | "
            f"up_MAE@{cfg.congestion_upstream_km}={metrics['upstream_mae']:.2f} | F1={f1_str}"
        )
        cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        selection_value = (
            val_cell_mae
            if cfg.checkpoint_selection_metric == "cell_mae"
            else val_loss
        )
        if selection_value < best_selection_value:
            best_selection_value = selection_value
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(cpu_sd, ckpt_dir / "best_model.pth")
        torch.save(cpu_sd, ckpt_dir / "last_model.pth")
        scheduler.step()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if evaluation_only:
        selected_checkpoint_path = Path(cfg.eval_checkpoint).resolve()
        if not selected_checkpoint_path.is_file():
            raise FileNotFoundError(f"evaluation checkpoint not found: {selected_checkpoint_path}")
        source_metrics_path = selected_checkpoint_path.parent.parent / "eval" / "metrics_v1.json"
        if source_metrics_path.is_file():
            source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
            source_selection = source_metrics.get("checkpoint_selection", {})
            best_epoch = int(source_selection.get("best_epoch", 0))
            best_selection_value = float(source_selection.get("best_selection_value", float("nan")))
            best_val_loss = float(source_selection.get("best_val_loss", float("nan")))
        selected_manifest = artifact_manifest(selected_checkpoint_path)
        checkpoint_artifacts = {
            "best_model": selected_manifest,
            "last_model": selected_manifest,
            "evaluation_checkpoint": selected_manifest,
        }
    else:
        selected_checkpoint_path = ckpt_dir / "best_model.pth"
        checkpoint_artifacts = {
            "best_model": artifact_manifest(selected_checkpoint_path),
            "last_model": artifact_manifest(ckpt_dir / "last_model.pth"),
        }
        plot_curves(history, plot_dir / "curves.png")

    # Per-sample metrics with best model.
    model.load_state_dict(torch.load(selected_checkpoint_path, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    up_start = _upstream_start_idx(cfg)
    up_end = _upstream_end_idx(cfg)
    rows = []
    validation_predictions: list[np.ndarray] = []
    validation_targets: list[np.ndarray] = []
    validation_valid_masks: list[np.ndarray] = []
    validation_ids: list[str] = []
    with torch.no_grad():
        for x, y, m, feat, _, ids_batch in val_loader:
            pred, _ = model(x.to(device), feat.to(device))
            pred = pred.cpu()
            for i, acc_id in enumerate(ids_batch):
                validation_predictions.append(pred[i].numpy())
                validation_targets.append(y[i].numpy())
                validation_valid_masks.append(m[i].numpy())
                validation_ids.append(str(acc_id).strip())
                cmask_np = cong_by_id.get(str(acc_id)) if cong_by_id else None
                sm = compute_metrics(
                    pred[i],
                    y[i],
                    m[i],
                    cfg.congestion_kmh,
                    up_start,
                    up_end,
                    torch.from_numpy(cmask_np) if cmask_np is not None else None,
                )
                operational = compute_operational_metrics(
                    pred[i].numpy() * SPEED_MAX,
                    y[i].numpy() * SPEED_MAX,
                    m[i].numpy(),
                    upstream_start_idx=up_start,
                    res_min=cfg.resolution_min,
                    cong_kmh=cfg.congestion_kmh,
                    upstream_end_idx=up_end,
                    congestion_mask=cmask_np,
                )
                rows.append({
                    "acc_id": acc_id,
                    "method": spec.method,
                    "use_tiid": int(spec.use_tiid),
                    "backbone": cfg.backbone,
                    "mae_kmh": sm["mae"],
                    "rmse_kmh": sm["rmse"],
                    "upstream_mae_kmh": sm["upstream_mae"],
                    "upstream_congestion_mae_kmh": sm["upstream_congestion_mae"],
                    "origin_cc_mae_kmh": sm["origin_cc_mae"],
                    "speed60_mae_kmh": sm["speed60_mae"],
                    "congestion_f1": sm["f1"],
                    "has_congestion": sm["has_congestion"],
                    "has_origin_cc": sm["has_origin_cc"],
                    "impact_time_ae_min": abs(operational["impact_time_err_min"]),
                    "recovery_time_ae_min": abs(operational["recovery_time_err_min"]),
                    "n_valid_cells": int(m[i].sum().item()),
                })
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(eval_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    prediction_artifact: dict | None = None
    if cfg.export_val_predictions:
        if not cfg.validation_only:
            raise RuntimeError("validation prediction export requires validation_only=True")
        result_ids = np.asarray(validation_ids, dtype=str)
        expected_ids = np.asarray([str(value).strip() for value in val_ds.accident_ids], dtype=str)
        if not np.array_equal(result_ids, expected_ids):
            raise RuntimeError("validation prediction accident-ID alignment failure")
        prediction_kmh = np.stack(validation_predictions).astype(np.float32) * SPEED_MAX
        target_kmh = np.stack(validation_targets).astype(np.float32) * SPEED_MAX
        valid_mask = np.stack(validation_valid_masks).astype(np.uint8)
        historical_future_kmh = val_ds.hist_matrices[:, :, cfg.n_before:].astype(np.float32)
        origin_cc_mask = np.stack(
            [
                cong_by_id.get(acc_id, np.zeros((cfg.n_space, cfg.n_after), dtype=np.uint8))
                if cong_by_id
                else np.zeros((cfg.n_space, cfg.n_after), dtype=np.uint8)
                for acc_id in result_ids
            ],
            axis=0,
        ).astype(np.uint8)
        if not (
            prediction_kmh.shape
            == target_kmh.shape
            == valid_mask.shape
            == historical_future_kmh.shape
            == origin_cc_mask.shape
        ):
            raise RuntimeError("validation prediction bundle has inconsistent field shapes")
        prediction_path = eval_dir / "predictions_val.npz"
        checkpoint_manifest = artifact_manifest(selected_checkpoint_path)
        runner_manifest = artifact_manifest(Path(__file__))
        np.savez_compressed(
            prediction_path,
            schema_version=np.asarray("unet_feature_fusion_validation_predictions_v1"),
            model_family=np.asarray(f"{cfg.backbone}_feature_fusion_unet"),
            feature_fusion_method=np.asarray(spec.method),
            split=np.asarray("val"),
            stage=np.asarray("development"),
            fit_scope=np.asarray("train_only_model_checkpoint"),
            accident_ids=result_ids,
            prediction_kmh=prediction_kmh,
            target_kmh=target_kmh,
            valid_mask=valid_mask,
            historical_future_kmh=historical_future_kmh,
            origin_cc_mask=origin_cc_mask,
            training_seed=np.asarray(cfg.random_state, dtype=np.int64),
            validation_membership_sha256=np.asarray(hash_strings(result_ids.tolist(), sort=True)),
            ordered_ids_sha256=np.asarray(hash_strings(result_ids.tolist(), sort=False)),
            checkpoint_sha256=np.asarray(checkpoint_manifest["sha256"] or ""),
            runner_sha256=np.asarray(runner_manifest["sha256"] or ""),
            space_offsets_m=target_window.space_offsets_m.astype(np.int32),
            time_offsets_min=target_window.time_offsets_min[cfg.n_before:].astype(np.int32),
        )
        prediction_artifact = artifact_manifest(prediction_path)
        print(f"Validation predictions: {prediction_path}")

    def _nm(v): return float(np.nanmean(v)) if len(v) else float("nan")
    up_mae_v = eval_df["upstream_mae_kmh"].dropna().tolist()
    uc_mae_v = eval_df["upstream_congestion_mae_kmh"].dropna().tolist()
    occ_mae_v = eval_df.loc[eval_df["has_origin_cc"].astype(bool), "origin_cc_mae_kmh"].dropna().tolist()
    f1_v     = eval_df.loc[eval_df["has_congestion"].astype(bool), "congestion_f1"].dropna().tolist()
    impact_ae_v = eval_df["impact_time_ae_min"].dropna().tolist()
    recovery_ae_v = eval_df["recovery_time_ae_min"].dropna().tolist()
    no_cong  = int((~eval_df["has_congestion"].astype(bool)).sum())
    no_occ   = int((~eval_df["has_origin_cc"].astype(bool)).sum())

    # Test set metrics
    test_rows: list[dict] = []
    test_predictions: list[np.ndarray] = []
    test_targets: list[np.ndarray] = []
    test_valid_masks: list[np.ndarray] = []
    test_ids: list[str] = []
    if test_loader:
        with torch.no_grad():
            for x, y, m, feat, _, ids_batch in test_loader:
                pred, _ = model(x.to(device), feat.to(device))
                pred = pred.cpu()
                for i, acc_id in enumerate(ids_batch):
                    test_predictions.append(pred[i].numpy())
                    test_targets.append(y[i].numpy())
                    test_valid_masks.append(m[i].numpy())
                    test_ids.append(str(acc_id).strip())
                    cmask_np = cong_by_id.get(str(acc_id)) if cong_by_id else None
                    sm = compute_metrics(
                        pred[i],
                        y[i],
                        m[i],
                        cfg.congestion_kmh,
                        up_start,
                        up_end,
                        torch.from_numpy(cmask_np) if cmask_np is not None else None,
                    )
                    operational = compute_operational_metrics(
                        pred[i].numpy() * SPEED_MAX,
                        y[i].numpy() * SPEED_MAX,
                        m[i].numpy(),
                        upstream_start_idx=up_start,
                        res_min=cfg.resolution_min,
                        cong_kmh=cfg.congestion_kmh,
                        upstream_end_idx=up_end,
                        congestion_mask=cmask_np,
                    )
                    test_rows.append({"acc_id": acc_id, **sm, **operational})
        pd.DataFrame(test_rows).to_csv(eval_dir / "metrics_test.csv", index=False)

    test_prediction_artifact: dict | None = None
    if cfg.export_test_predictions:
        if not evaluation_only or test_ds is None:
            raise RuntimeError("test prediction export requires a frozen --eval-checkpoint and test split")
        result_ids = np.asarray(test_ids, dtype=str)
        expected_ids = np.asarray([str(value).strip() for value in test_ds.accident_ids], dtype=str)
        if not np.array_equal(result_ids, expected_ids):
            raise RuntimeError("test prediction accident-ID alignment failure")
        prediction_kmh = np.stack(test_predictions).astype(np.float32) * SPEED_MAX
        target_kmh = np.stack(test_targets).astype(np.float32) * SPEED_MAX
        valid_mask = np.stack(test_valid_masks).astype(np.uint8)
        historical_future_kmh = test_ds.hist_matrices[:, :, cfg.n_before:].astype(np.float32)
        origin_cc_mask = np.stack(
            [
                cong_by_id.get(acc_id, np.zeros((cfg.n_space, cfg.n_after), dtype=np.uint8))
                if cong_by_id
                else np.zeros((cfg.n_space, cfg.n_after), dtype=np.uint8)
                for acc_id in result_ids
            ],
            axis=0,
        ).astype(np.uint8)
        if not (
            prediction_kmh.shape
            == target_kmh.shape
            == valid_mask.shape
            == historical_future_kmh.shape
            == origin_cc_mask.shape
        ):
            raise RuntimeError("test prediction bundle has inconsistent field shapes")
        prediction_path = eval_dir / "predictions_test.npz"
        checkpoint_manifest = artifact_manifest(selected_checkpoint_path)
        runner_manifest = artifact_manifest(Path(__file__))
        np.savez_compressed(
            prediction_path,
            schema_version=np.asarray("unet_feature_fusion_test_predictions_v1"),
            model_family=np.asarray(f"{cfg.backbone}_feature_fusion_unet"),
            feature_fusion_method=np.asarray(spec.method),
            split=np.asarray("test"),
            stage=np.asarray("formal_temporal_evaluation"),
            fit_scope=np.asarray("frozen_validation_selected_checkpoint_no_retraining"),
            accident_ids=result_ids,
            prediction_kmh=prediction_kmh,
            target_kmh=target_kmh,
            valid_mask=valid_mask,
            historical_future_kmh=historical_future_kmh,
            origin_cc_mask=origin_cc_mask,
            training_seed=np.asarray(cfg.random_state, dtype=np.int64),
            test_membership_sha256=np.asarray(hash_strings(result_ids.tolist(), sort=True)),
            ordered_ids_sha256=np.asarray(hash_strings(result_ids.tolist(), sort=False)),
            checkpoint_sha256=np.asarray(checkpoint_manifest["sha256"] or ""),
            runner_sha256=np.asarray(runner_manifest["sha256"] or ""),
            space_offsets_m=target_window.space_offsets_m.astype(np.int32),
            time_offsets_min=target_window.time_offsets_min[cfg.n_before:].astype(np.int32),
        )
        test_prediction_artifact = artifact_manifest(prediction_path)
        print(f"Test predictions: {prediction_path}")

    metrics_v1: dict = {
        "eval_protocol": cfg.eval_protocol, "run_id": output_root.name,
        "experiment": spec.tag, "loss_name": cfg.loss_name,
        "speed_loss_lambda": cfg.speed_loss_lambda,
        "pace_loss_lambda": cfg.pace_loss_lambda,
        "pace_v_min_kmh": cfg.pace_v_min_kmh,
        "data_scope": cfg.data_scope, "feature_set": cfg.feature_set, "input_type": cfg.input_type,
        "backbone": cfg.backbone,
        "training_seed": cfg.random_state,
        "random_seed": cfg.random_state,
        "split_seed": cfg.split_seed,
        "split_mode": cfg.split_mode if cfg.eval_protocol == "v1" else "legacy",
        "reproducibility": experiment_seed_info,
        "protocol": protocol_info or {},
        "feature_normalization": (protocol_info or {}).get("feature_normalization", {}),
        "checkpoint_selection": {
            "criterion": (
                "minimum_validation_cell_mae"
                if cfg.checkpoint_selection_metric == "cell_mae"
                else "minimum_validation_weighted_loss"
            ),
            "best_epoch": int(best_epoch),
            "best_selection_value": float(best_selection_value),
            "selection_units": (
                "km/h" if cfg.checkpoint_selection_metric == "cell_mae" else "normalized_loss"
            ),
            "best_val_loss": float(best_val_loss),
        },
        "checkpoint_artifacts": checkpoint_artifacts,
        "prediction_artifacts": {
            **({"validation_predictions": prediction_artifact} if prediction_artifact else {}),
            **({"test_predictions": test_prediction_artifact} if test_prediction_artifact else {}),
        },
        "val": {
            "n": len(rows),
            "cell_MAE": float(eval_df["mae_kmh"].mean()),
            "upstream_MAE": _nm(up_mae_v),
            f"origin_cc_MAE@{cfg.congestion_upstream_km}": _nm(occ_mae_v),
            "upstream_congestion_MAE": _nm(uc_mae_v),
            "congestion_F1": _nm(f1_v),
            f"congestion_F1@{cfg.congestion_upstream_km}": _nm(f1_v),
            f"impact_time_AE@{cfg.congestion_upstream_km}": _nm(impact_ae_v),
            "recovery_AE@10": _nm(recovery_ae_v),
            "no_congestion_count": no_cong,
            "no_origin_cc_count": no_occ,
        },
    }
    if test_rows:
        t_df = pd.DataFrame(test_rows)
        t_up = t_df["upstream_mae"].dropna().tolist()
        t_uc = t_df["upstream_congestion_mae"].dropna().tolist()
        t_occ = t_df.loc[t_df["has_origin_cc"].astype(bool), "origin_cc_mae"].dropna().tolist()
        t_f1 = t_df.loc[t_df["has_congestion"].astype(bool), "f1"].dropna().tolist()
        t_impact_ae = t_df["impact_time_err_min"].abs().dropna().tolist()
        t_recovery_ae = t_df["recovery_time_err_min"].abs().dropna().tolist()
        t_noo = int((~t_df["has_origin_cc"].astype(bool)).sum())
        metrics_v1["test"] = {
            "n": len(test_rows),
            "cell_MAE": _nm(t_df["mae"].tolist()),
            "upstream_MAE": _nm(t_up),
            f"origin_cc_MAE@{cfg.congestion_upstream_km}": _nm(t_occ),
            "upstream_congestion_MAE": _nm(t_uc),
            "congestion_F1": _nm(t_f1),
            f"congestion_F1@{cfg.congestion_upstream_km}": _nm(t_f1),
            f"impact_time_AE@{cfg.congestion_upstream_km}": _nm(t_impact_ae),
            "recovery_AE@10": _nm(t_recovery_ae),
            "no_origin_cc_count": t_noo,
        }
    (eval_dir / "metrics_v1.json").write_text(json.dumps(metrics_v1, indent=2), encoding="utf-8")

    summary_text = (
        f"experiment={spec.tag}\nmethod={spec.method}\nuse_tiid={spec.use_tiid}\n"
        f"loss={cfg.loss_name}\neval_protocol={cfg.eval_protocol}\n"
        f"loss_params=speed_lambda={cfg.speed_loss_lambda}, "
        f"pace_lambda={cfg.pace_loss_lambda}, pace_v_min={cfg.pace_v_min_kmh} km/h\n"
        f"data_scope={cfg.data_scope}\nfeature_set={cfg.feature_set}\nbackbone={cfg.backbone}\n"
        f"train_samples={len(train_ds)}\nval_samples={len(val_ds)}\n"
        f"cell_MAE={eval_df['mae_kmh'].mean():.4f}\n"
        f"origin_cc_MAE@{cfg.congestion_upstream_km}={_nm(occ_mae_v):.2f} km/h  "
        f"(excl {no_occ}/{len(rows)} no-origin_cc)\n"
        f"upstream_MAE={_nm(up_mae_v):.2f} km/h\n"
        f"congestion_F1={_nm(f1_v):.4f}  (excl {no_cong}/{len(rows)} no-cong)\n"
        f"impact_time_AE@{cfg.congestion_upstream_km}={_nm(impact_ae_v):.2f} min\n"
        f"recovery_AE@10={_nm(recovery_ae_v):.2f} min\n"
        f"best_val_loss={best_val_loss:.6f}\n"
        f"best_epoch={best_epoch}\n"
    )
    (eval_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    return {
        "experiment": spec.tag,
        "method": spec.method,
        "use_tiid": bool(spec.use_tiid),
        "backbone": cfg.backbone,
        "loss_name": cfg.loss_name,
        "eval_protocol": cfg.eval_protocol,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "mean_mae_kmh": float(eval_df["mae_kmh"].mean()),
        "mean_rmse_kmh": float(eval_df["rmse_kmh"].mean()),
        "upstream_MAE": _nm(up_mae_v),
        f"origin_cc_MAE@{cfg.congestion_upstream_km}": _nm(occ_mae_v),
        "upstream_congestion_MAE": _nm(uc_mae_v),
        "congestion_F1": _nm(f1_v),
        f"congestion_F1@{cfg.congestion_upstream_km}": _nm(f1_v),
        f"impact_time_AE@{cfg.congestion_upstream_km}": _nm(impact_ae_v),
        "recovery_AE@10": _nm(recovery_ae_v),
        "no_congestion_count": no_cong,
        "no_origin_cc_count": no_occ,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "checkpoint_selection_criterion": (
            "minimum_validation_cell_mae"
            if cfg.checkpoint_selection_metric == "cell_mae"
            else "minimum_validation_weighted_loss"
        ),
        "best_selection_value": float(best_selection_value),
        "best_checkpoint_sha256": checkpoint_artifacts["best_model"]["sha256"],
        "last_checkpoint_sha256": checkpoint_artifacts["last_model"]["sha256"],
        "training_seed": cfg.random_state,
        "split_seed": cfg.split_seed,
        "split_mode": cfg.split_mode if cfg.eval_protocol == "v1" else "legacy",
        "feature_normalization_parameters_sha256": (
            (protocol_info or {})
            .get("feature_normalization", {})
            .get("hashes", {})
            .get("parameters_sha256")
        ),
        "experiment_dir": str(exp_dir),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train feature-fusion U-Net variants on accident samples NPZ.")
    p.add_argument("--samples-npz", default=None, help="Default: data/accident_samples_<resolution>m.npz")
    p.add_argument("--feature-csv", default=str(DEFAULT_FEATURE_CSV))
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--run-id", default=None)
    p.add_argument("--methods", default="direct_concat,mid_inject,attention")
    p.add_argument("--with-tiid", action="store_true", help="Also train one TIID experiment")
    p.add_argument("--tiid-method", default="mid_inject")
    p.add_argument("--upstream-km", type=int, default=25)
    p.add_argument("--downstream-km", type=int, default=10)
    p.add_argument("--time-before-min", type=int, default=30)
    p.add_argument("--time-after-min", type=int, default=145)
    p.add_argument("--spatial-resolution-km", type=int, default=1)
    p.add_argument("--resolution-min", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--congestion-kmh", type=float, default=60.0)
    p.add_argument("--congestion-weight", type=float, default=CONGESTION_WEIGHT_DEFAULT,
                   help="Upstream congested-cell weight w (non-congested=1, downstream=0); "
                        "eval_protocol_v1 sweep {2,4,7,10}, default 4.")
    p.add_argument("--congestion-upstream-km", type=int, default=CONGESTION_UPSTREAM_KM_DEFAULT,
                   help="Upstream zone (km) for congestion-weighted loss and upstream_congestion_MAE "
                        "(eval_protocol_v1 §3.1/§4.1; default 20).")
    p.add_argument("--tiid-lambda", type=float, default=0.3)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument(
        "--training-seed",
        type=int,
        default=42,
        help="Model/DataLoader RNG seed. Sweep this independently of --split-seed.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Deprecated alias for --training-seed (kept for existing commands).",
    )
    p.add_argument("--max-accidents", type=int, default=None)
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
    # eval_protocol_v1 args
    p.add_argument(
        "--eval-protocol",
        default="",
        help="Set to 'v1' to use the paper train/val/test protocol and operational metrics.",
    )
    p.add_argument("--loss", default="upstream_congestion_weighted", dest="loss_name",
                   choices=["full_field_congestion_weighted", "upstream_congestion_weighted", "congestion_only_mae", "cell_mae", "speed_pace_hybrid"])
    p.add_argument(
        "--checkpoint-selection",
        choices=["validation_loss", "cell_mae"],
        default="validation_loss",
        help=(
            "Epoch-selection metric. 'cell_mae' uses the paper's mean of "
            "per-incident valid-cell MAE; legacy runs use validation_loss."
        ),
    )
    p.add_argument("--speed-loss-lambda", type=float, default=SPEED_LOSS_LAMBDA_DEFAULT,
                   help="Lambda for the speed MAE term in speed_pace_hybrid.")
    p.add_argument("--pace-loss-lambda", type=float, default=PACE_LOSS_LAMBDA_DEFAULT,
                   help="Lambda for the true-pace MAE term in speed_pace_hybrid.")
    p.add_argument("--pace-v-min-kmh", type=float, default=PACE_V_MIN_KMH_DEFAULT,
                   help="Minimum speed floor for true pace 60/max(v, v_min), in km/h.")
    p.add_argument("--data-scope", default="dispatch_only")
    p.add_argument("--feature-set", default="t0", choices=["t0", "t5", "t15"],
                   help="Feature availability time. Default t0; t5/t15 paused pending the "
                        "feature data-spec replan (planning_next_steps_20260529 §0/§8).")
    p.add_argument(
        "--input-type",
        default="vd+hist",
        choices=["vd+hist"],
        help="Locked 3-channel matrix stream: VD speed, VD validity, causal historical speed.",
    )
    p.add_argument("--backbone", default="simple", choices=ALL_BACKBONES,
                   help="UNet backbone: simple keeps the legacy architecture; residual_attention is the larger v2.")
    p.add_argument("--hist-tag", default="", help="Tag suffix of the historical matrices NPZ (matches the samples --tag; empty = dispatch).")
    p.add_argument(
        "--historical-matrices-npz",
        default="",
        help="Explicit historical-matrix NPZ. Takes precedence over --hist-tag.",
    )
    p.add_argument("--congestion-mask-npz", default="",
                   help="origin_cc mask NPZ (required under --eval-protocol v1). When set, "
                        "loss and upstream_congestion_MAE use accident-caused congestion cells.")
    p.add_argument("--train-end-date", default=TRAIN_END_DATE)
    p.add_argument("--val-end-date", default=VAL_END_DATE)
    p.add_argument(
        "--split-mode",
        choices=["time", "random"],
        default="time",
        help="Paper split under --eval-protocol v1: time is primary; random is robustness only and keeps the same counts.",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Fixed random-robustness split seed; do not change across training seeds.",
    )
    p.add_argument(
        "--allowed-freeways",
        default=",".join(DEFAULT_ALLOWED_FREEWAYS),
        help="Explicit modeled freeway allow-list (paper default: N1,N3).",
    )
    p.add_argument(
        "--deterministic-mode",
        choices=["warn", "error", "off"],
        default="warn",
        help="PyTorch deterministic-kernel policy. 'warn' is portable to DirectML; use 'error' for strict audit.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu"],
        default="auto",
        help="Execution backend. Use cpu to exclude DirectML from formal training.",
    )
    p.add_argument(
        "--samples-metadata-parquet",
        default="",
        help="Metadata parquet paired with --samples-npz; default is the sister *_metadata.parquet file.",
    )
    p.add_argument("--smoke", action="store_true", help="Smoke test: 5-epoch quick validation")
    p.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Train and evaluate the chronological validation split without running "
            "test inference or writing test metrics. Use for adaptive candidate screens."
        ),
    )
    p.add_argument(
        "--export-val-predictions",
        action="store_true",
        help=(
            "Write eval/predictions_val.npz for development-only AIC detection-rate "
            "and false-alarm evaluation. Requires --validation-only."
        ),
    )
    p.add_argument(
        "--eval-checkpoint",
        default="",
        help="Evaluate one frozen validation-selected checkpoint without retraining.",
    )
    p.add_argument(
        "--export-test-predictions",
        action="store_true",
        help="Write eval/predictions_test.npz; requires --eval-checkpoint.",
    )
    p.add_argument(
        "--protocol-dry-run",
        action="store_true",
        help="Load/filter/split/hash artifacts and write run_config.json without training.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.export_val_predictions and not args.validation_only:
        raise SystemExit("--export-val-predictions requires --validation-only")
    if args.export_test_predictions and not args.eval_checkpoint:
        raise SystemExit("--export-test-predictions requires --eval-checkpoint")
    if args.eval_checkpoint and args.validation_only:
        raise SystemExit("--eval-checkpoint cannot be combined with --validation-only")
    if args.eval_checkpoint and args.export_val_predictions:
        raise SystemExit("--eval-checkpoint cannot be combined with --export-val-predictions")
    if args.eval_checkpoint and args.smoke:
        raise SystemExit("--eval-checkpoint cannot be combined with --smoke")
    if (
        args.eval_protocol == "v1"
        and not args.validation_only
        and not args.protocol_dry_run
        and not args.eval_checkpoint
    ):
        raise SystemExit(
            "formal eval_protocol v1 feature-fusion training requires --validation-only; "
            "freeze a validation-selected checkpoint before any temporal test evaluation"
        )
    training_seed = args.random_state if args.random_state is not None else args.training_seed
    allowed_freeways = parse_allowed_freeways(args.allowed_freeways)
    if args.eval_protocol == "v1":
        if set(allowed_freeways) != set(DEFAULT_ALLOWED_FREEWAYS):
            raise SystemExit("formal eval_protocol v1 is locked to --allowed-freeways N1,N3")
        if args.feature_set != "t0":
            raise SystemExit("formal eval_protocol v1 is locked to --feature-set t0")
        if args.split_seed != DEFAULT_SPLIT_SEED:
            raise SystemExit(
                f"formal eval_protocol v1 is locked to --split-seed {DEFAULT_SPLIT_SEED}"
            )
        if args.train_end_date != TRAIN_END_DATE or args.val_end_date != VAL_END_DATE:
            raise SystemExit(
                "formal eval_protocol v1 split dates are locked to "
                f"{TRAIN_END_DATE} and {VAL_END_DATE}"
            )
        if args.min_filled_ratio != 0.8 or args.min_before_coverage != 0.0:
            raise SystemExit(
                "formal eval_protocol v1 cohort is locked to filled_ratio>=0.8 "
                "with no extra pre-input coverage filter"
            )
        if args.max_accidents is not None:
            raise SystemExit(
                "formal eval_protocol v1 uses the full eligible cohort; --max-accidents is disabled"
            )
        if not args.congestion_mask_npz:
            raise SystemExit(
                "formal eval_protocol v1 requires an explicit --congestion-mask-npz; "
                "speed<60 fallback masks are diagnostic only"
            )
        if "hist" in args.input_type and not args.historical_matrices_npz:
            raise SystemExit(
                "formal eval_protocol v1 with historical input requires an explicit "
                "--historical-matrices-npz; legacy --hist-tag fallback is disabled"
            )
    cfg = TrainConfig(
        upstream_km=args.upstream_km,
        downstream_km=args.downstream_km,
        time_before_min=args.time_before_min,
        time_after_min=args.time_after_min,
        spatial_resolution_km=args.spatial_resolution_km,
        resolution_min=args.resolution_min,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        random_state=training_seed,
        congestion_kmh=args.congestion_kmh,
        congestion_weight=args.congestion_weight,
        congestion_upstream_km=args.congestion_upstream_km,
        tiid_lambda=args.tiid_lambda,
        min_before_coverage=args.min_before_coverage,
        min_filled_ratio=args.min_filled_ratio,
        eval_protocol=args.eval_protocol,
        data_scope=args.data_scope,
        feature_set=args.feature_set,
        input_type=args.input_type,
        backbone=args.backbone,
        loss_name=args.loss_name,
        checkpoint_selection_metric=args.checkpoint_selection,
        speed_loss_lambda=args.speed_loss_lambda,
        pace_loss_lambda=args.pace_loss_lambda,
        pace_v_min_kmh=args.pace_v_min_kmh,
        congestion_mask_npz=args.congestion_mask_npz or "",
        train_end_date=args.train_end_date,
        val_end_date=args.val_end_date,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        allowed_freeways=",".join(allowed_freeways),
        deterministic_mode=args.deterministic_mode,
        device_preference=args.device,
        historical_matrices_npz=args.historical_matrices_npz or "",
        smoke=args.smoke,
        validation_only=args.validation_only,
        export_val_predictions=args.export_val_predictions,
        eval_checkpoint=args.eval_checkpoint or "",
        export_test_predictions=args.export_test_predictions,
    )
    if cfg.smoke:
        cfg.epochs = 5
    if cfg.pad_h * 2 + cfg.n_space != PAD_TARGET or cfg.pad_w * 2 + cfg.n_time != PAD_TARGET:
        raise SystemExit(f"window ({cfg.n_space}, {cfg.n_time}) cannot symmetrically pad to {PAD_TARGET}; choose even sizes.")

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    device, device_msg = select_device(cfg.device_preference)
    print(f"Device: {device_msg}")
    print(f"Run id: {run_id}")
    print(f"Output: {output_root}")
    print(f"Window: ({cfg.n_space} space x {cfg.n_time} time) | input cols 0..{cfg.n_before-1}, target cols {cfg.n_before}..{cfg.n_time-1}")
    if cfg.eval_protocol:
        print(
            f"eval_protocol={cfg.eval_protocol} | loss={cfg.loss_name} | data_scope={cfg.data_scope} | "
            f"backbone={cfg.backbone} | smoke={cfg.smoke}"
        )

    feature_csv = Path(args.feature_csv)
    if not feature_csv.exists():
        raise SystemExit(f"feature CSV not found: {feature_csv}")
    feature_df = pd.read_csv(feature_csv, dtype={"task_id": str})
    feature_df["task_id"] = feature_df["task_id"].astype(str).str.strip()
    if feature_df["task_id"].duplicated().any():
        duplicate_ids = feature_df.loc[feature_df["task_id"].duplicated(keep=False), "task_id"].head(5).tolist()
        raise SystemExit(f"feature CSV has duplicate task_id rows; examples={duplicate_ids}")
    feature_columns = [c for c in feature_df.columns if c not in {"task_id", "avail_mainline"}]
    try:
        role_reference_array = feature_df[feature_columns].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"feature CSV contains a non-numeric model field: {exc}") from exc

    feature_source_manifest_path = feature_csv.with_name(f"{feature_csv.stem}_manifest.json")
    feature_source_manifest: dict | None = None
    feature_declared_dtypes: dict[str, str] | None = None
    locked_no_weather_zero_columns: list[str] = []
    if feature_source_manifest_path.exists():
        try:
            feature_source_manifest = json.loads(
                feature_source_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"cannot read feature manifest {feature_source_manifest_path}: {exc}"
            ) from exc

    if cfg.eval_protocol == "v1":
        if len(feature_columns) != 45:
            raise SystemExit(
                "formal Track-B requires the immutable 45-field model schema; "
                f"found {len(feature_columns)} fields"
            )
        if feature_source_manifest is None:
            raise SystemExit(
                "formal feature-fusion runs require the immutable sibling feature manifest: "
                f"{feature_source_manifest_path}"
            )
        if not bool(feature_source_manifest.get("paper_valid")):
            raise SystemExit(
                f"feature manifest is not paper-valid: {feature_source_manifest_path}"
            )
        if not bool(feature_source_manifest.get("immutable_write")):
            raise SystemExit(
                f"formal feature artifact was not written immutably: {feature_source_manifest_path}"
            )
        declared_hash = (
            feature_source_manifest.get("artifacts", {})
            .get("features", {})
            .get("sha256")
        )
        actual_hash = artifact_manifest(feature_csv)["sha256"]
        if not declared_hash or declared_hash != actual_hash:
            raise SystemExit(
                "feature CSV hash does not match its immutable manifest: "
                f"declared={declared_hash}, actual={actual_hash}"
            )
        declared_columns = (
            feature_source_manifest.get("feature_schema", {})
            .get("model_feature_columns")
        )
        if declared_columns != feature_columns:
            raise SystemExit(
                "feature CSV model-column order does not match its immutable manifest"
            )
        declared_schema = feature_source_manifest.get("feature_schema", {}).get("columns")
        if not isinstance(declared_schema, list):
            raise SystemExit("immutable feature manifest has no declared column/dtype schema")
        feature_declared_dtypes = {
            str(entry.get("name")): str(entry.get("dtype"))
            for entry in declared_schema
            if isinstance(entry, dict) and entry.get("name") is not None
        }
        missing_dtypes = [
            column for column in feature_columns if column not in feature_declared_dtypes
        ]
        if missing_dtypes:
            raise SystemExit(
                "immutable feature manifest is missing declared dtypes for: "
                f"{missing_dtypes}"
            )
        declared_rows = feature_source_manifest.get("feature_schema", {}).get("rows")
        if declared_rows != len(feature_df):
            raise SystemExit(
                "feature CSV row count does not match its immutable manifest: "
                f"declared={declared_rows}, actual={len(feature_df)}"
            )
        weather_cfg = feature_source_manifest.get("configuration", {})
        if weather_cfg.get("weather_match_direction") != "backward":
            raise SystemExit(
                "formal feature manifest must use backward weather matching "
                "(latest observation at or before dispatch)"
            )
        if weather_cfg.get("weather_mode") == "omitted":
            weather_availability = (
                feature_source_manifest.get("sources", {})
                .get("weather", {})
                .get("availability", {})
            )
            weather_value_columns = list(weather_availability)
            locked_no_weather_zero_columns = [
                column
                for value_column in weather_value_columns
                for column in (value_column, f"avail_{value_column}")
            ]
            missing_weather_columns = [
                column for column in locked_no_weather_zero_columns if column not in feature_df.columns
            ]
            if missing_weather_columns:
                raise SystemExit(
                    "omitted-weather manifest is missing value/availability fields: "
                    f"{missing_weather_columns}"
                )
            nonzero_weather_columns = [
                column
                for column in locked_no_weather_zero_columns
                if np.count_nonzero(
                    pd.to_numeric(feature_df[column], errors="coerce").fillna(0).to_numpy()
                )
            ]
            if nonzero_weather_columns:
                raise SystemExit(
                    "omitted-weather paper artifact must keep weather values and availability "
                    f"at zero; nonzero columns={nonzero_weather_columns}"
                )

    samples_npz_path = (
        Path(args.samples_npz)
        if args.samples_npz
        else PROJECT_ROOT / "data" / f"accident_samples_{cfg.resolution_min}m.npz"
    )
    samples_metadata_path = (
        Path(args.samples_metadata_parquet)
        if args.samples_metadata_parquet
        else samples_npz_path.with_name(f"{samples_npz_path.stem}_metadata.parquet")
    )
    samples = (
        load_samples(data_root=PROJECT_ROOT / "data", resolution_min=cfg.resolution_min)
        if not args.samples_npz
        else load_samples_artifact(samples_npz_path, samples_metadata_path)
    )
    target_window = SampleWindow(
        upstream_km=cfg.upstream_km,
        downstream_km=cfg.downstream_km,
        time_before_min=cfg.time_before_min,
        time_after_min=cfg.time_after_min,
        spatial_resolution_km=cfg.spatial_resolution_km,
        resolution_min=cfg.resolution_min,
    )
    matrices, ids, features, feature_dim = filter_subset_align_features(
        samples,
        feature_df,
        target_window,
        cfg.min_before_coverage,
        cfg.min_filled_ratio,
        feature_set=cfg.feature_set,
        allowed_freeways=allowed_freeways,
    )
    print(
        f"Samples after coverage filter + feature align: {len(matrices)} | feature_dim={feature_dim} "
        f"(min_filled_ratio={cfg.min_filled_ratio}, min_before_coverage={cfg.min_before_coverage})"
    )
    if len(matrices) < 2:
        raise SystemExit("Need at least 2 aligned samples for train/val split.")
    if args.max_accidents is not None and len(matrices) > args.max_accidents:
        matrices = matrices[: args.max_accidents]
        ids = ids[: args.max_accidents]
        features = features[: args.max_accidents]

    cohort_info = cohort_manifest(
        samples.accident_ids,
        ids,
        samples.metadata,
        allowed_freeways,
    )
    print(
        f"Paper scope: {','.join(allowed_freeways)} | route-excluded={cohort_info['route_excluded_n']} | "
        f"retained routes={cohort_info['retained_freeway_counts']}"
    )

    hist_matrices = None
    historical_path: Path | None = None
    if "hist" in cfg.input_type:
        historical_path = (
            Path(args.historical_matrices_npz)
            if args.historical_matrices_npz
            else PROJECT_ROOT / "data" / f"historical_matrices_{cfg.resolution_min}m{'_' + args.hist_tag if args.hist_tag else ''}.npz"
        )
        hist_matrices = load_hist_aligned(
            ids,
            target_window,
            PROJECT_ROOT / "data",
            cfg.resolution_min,
            args.hist_tag,
            historical_path,
            require_complete=(cfg.eval_protocol == "v1"),
        )
        print(f"Input type: {cfg.input_type} | base_in_channels={cfg.base_in_channels} | hist matrices: {hist_matrices.shape}")
    else:
        print(f"Input type: {cfg.input_type} | base_in_channels={cfg.base_in_channels}")

    cong_by_id: dict[str, np.ndarray] = {}
    if cfg.congestion_mask_npz:
        cong_aligned = load_congestion_masks_aligned(
            ids,
            target_window,
            Path(cfg.congestion_mask_npz),
            require_complete=(cfg.eval_protocol == "v1"),
        )
        cong_by_id = {str(a): cong_aligned[i] for i, a in enumerate(ids)}
        print(
            f"Congestion mask: {Path(cfg.congestion_mask_npz).name} | "
            f"origin_cc cell share={float(cong_aligned.mean()):.4%}"
        )

    if cfg.eval_protocol == "v1":
        train_idx, val_idx, test_idx, split_info = split_indices(
            ids,
            samples.metadata,
            split_mode=cfg.split_mode,
            split_seed=cfg.split_seed,
            train_end=cfg.train_end_date,
            val_end=cfg.val_end_date,
        )
        if len(train_idx) < 2 or len(val_idx) < 1:
            raise SystemExit(f"Paper split too small: train={len(train_idx)} val={len(val_idx)}")
        print(
            f"Paper split ({cfg.split_mode}): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} | "
            f"split_seed={cfg.split_seed} (training_seed={cfg.random_state})"
        )
    else:
        from sklearn.model_selection import train_test_split as _tts
        train_idx, val_idx = _tts(np.arange(len(matrices)), test_size=cfg.val_split, random_state=cfg.random_state)
        test_idx = np.array([], dtype=np.int64)
        split_info = {
            "mode": "legacy_random_train_val",
            "primary_split": "legacy",
            "split_seed": cfg.random_state,
            "splits": {
                "train": {"n": int(len(train_idx)), "membership_sha256": hash_strings(ids[train_idx].tolist(), sort=True)},
                "val": {"n": int(len(val_idx)), "membership_sha256": hash_strings(ids[val_idx].tolist(), sort=True)},
                "test": {"n": 0, "membership_sha256": hash_strings([], sort=True)},
            },
        }

    split_indices_by_name = {"train": train_idx, "val": val_idx, "test": test_idx}
    split_membership_artifact = write_split_membership(
        output_root / "split_membership.csv",
        ids,
        samples.metadata,
        split_indices_by_name,
    )
    split_info["membership_artifact"] = split_membership_artifact

    if cfg.eval_protocol == "v1":
        features, feature_normalization_info = fit_transform_train_only(
            features,
            feature_columns,
            train_idx,
            role_reference_array=role_reference_array,
            declared_dtypes=feature_declared_dtypes,
            split_indices_by_name=split_indices_by_name,
            sample_ids=ids,
        )
        feature_normalization_info["source_weather_mode"] = (
            feature_source_manifest.get("configuration", {}).get("weather_mode")
            if feature_source_manifest is not None
            else None
        )
        feature_normalization_info["role_inference_source"] = (
            "immutable_feature_manifest_declared_schema"
        )
        feature_normalization_info["locked_no_weather_zero_columns"] = (
            locked_no_weather_zero_columns
        )
        if locked_no_weather_zero_columns:
            zero_indices = [feature_columns.index(column) for column in locked_no_weather_zero_columns]
            if np.count_nonzero(features[:, zero_indices]):
                raise SystemExit("no-weather value/availability zeros changed during normalization")
        print(
            "Track-B normalization: train-only z-score for "
            f"{len(feature_normalization_info['continuous_columns'])} continuous fields; "
            f"{len(feature_normalization_info['binary_columns'])} binary fields unchanged | "
            f"parameters={feature_normalization_info['hashes']['parameters_sha256'][:12]}..."
        )
    else:
        # Preserve the pre-paper behavior exactly for legacy runs.
        feature_normalization_info = identity_manifest(features, feature_columns)

    feature_normalization_path = output_root / "feature_normalization.json"
    feature_normalization_path.write_text(
        json.dumps(feature_normalization_info, indent=2), encoding="utf-8"
    )
    feature_normalization_artifact = artifact_manifest(feature_normalization_path)

    def _hist_subset(idx: np.ndarray) -> np.ndarray | None:
        return hist_matrices[idx] if hist_matrices is not None else None

    train_ds = AccidentFeatureDataset(
        matrices[train_idx],
        ids[train_idx],
        features[train_idx],
        cfg,
        _hist_subset(train_idx),
    )
    val_ds = AccidentFeatureDataset(
        matrices[val_idx],
        ids[val_idx],
        features[val_idx],
        cfg,
        _hist_subset(val_idx),
    )
    test_ds: AccidentFeatureDataset | None = (
        AccidentFeatureDataset(
            matrices[test_idx],
            ids[test_idx],
            features[test_idx],
            cfg,
            _hist_subset(test_idx),
        )
        if len(test_idx) > 0
        else None
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ALL_METHODS:
            raise SystemExit(f"unknown method: {m}; allowed: {ALL_METHODS}")
    specs = [ExperimentSpec(method=m, use_tiid=False) for m in methods]
    if args.with_tiid:
        specs.append(ExperimentSpec(method=args.tiid_method, use_tiid=True))

    seed_info = configure_reproducibility(cfg.random_state, cfg.deterministic_mode)
    feature_schema_info = feature_schema_manifest(feature_df, feature_columns, feature_csv)
    feature_schema_info["source_manifest"] = {
        "artifact": artifact_manifest(feature_source_manifest_path),
        "paper_valid": (
            feature_source_manifest.get("paper_valid")
            if feature_source_manifest is not None else None
        ),
        "immutable_write": (
            feature_source_manifest.get("immutable_write")
            if feature_source_manifest is not None else None
        ),
        "configuration": (
            feature_source_manifest.get("configuration")
            if feature_source_manifest is not None else None
        ),
    }
    protocol_info = {
        "paper_scope": "/".join(allowed_freeways),
        "allowed_freeways": list(allowed_freeways),
        "cohort": cohort_info,
        "split": split_info,
        "reproducibility": seed_info,
        "feature_schema": feature_schema_info,
        "feature_normalization": feature_normalization_info,
        "artifacts": {
            "samples_npz": artifact_manifest(samples_npz_path),
            "samples_metadata_parquet": artifact_manifest(samples_metadata_path),
            "historical_matrices_npz": artifact_manifest(historical_path),
            "congestion_mask_npz": artifact_manifest(cfg.congestion_mask_npz),
            "feature_manifest_json": artifact_manifest(feature_source_manifest_path),
            "feature_normalization_json": feature_normalization_artifact,
        },
        "evaluation_firewall": {
            "validation_only": cfg.validation_only,
            "evaluation_only": bool(cfg.eval_checkpoint),
            "test_inference_allowed": bool(cfg.eval_checkpoint),
            "test_prediction_export_requested": cfg.export_test_predictions,
        },
    }

    run_config: dict = {
        "run_id": run_id,
        "config": asdict(cfg),
        "experiments": [s.tag for s in specs],
        "feature_dim": feature_dim,
        "feature_columns": feature_columns,
        "feature_normalization": feature_normalization_info,
        "training_seed": cfg.random_state,
        "random_seed": cfg.random_state,
        "split_seed": cfg.split_seed,
        "split_mode": cfg.split_mode if cfg.eval_protocol == "v1" else "legacy",
        "checkpoint_selection_metric": cfg.checkpoint_selection_metric,
        "validation_only": cfg.validation_only,
        "evaluation_only": bool(cfg.eval_checkpoint),
        "protocol": protocol_info,
        "status": "training",
    }
    if cfg.eval_protocol == "v1":
        run_config.update({
            "eval_protocol": "v1",
            "data_scope": cfg.data_scope,
            "feature_set": cfg.feature_set,
            "input_type": cfg.input_type,
            "backbone": cfg.backbone,
            "loss": cfg.loss_name,
            "speed_loss_lambda": cfg.speed_loss_lambda,
            "pace_loss_lambda": cfg.pace_loss_lambda,
            "pace_v_min_kmh": cfg.pace_v_min_kmh,
            "congestion_mask_npz": cfg.congestion_mask_npz,
            "train_date_range": [
                split_info["splits"]["train"].get("date_min"),
                split_info["splits"]["train"].get("date_max"),
            ],
            "val_date_range": [
                split_info["splits"]["val"].get("date_min"),
                split_info["splits"]["val"].get("date_max"),
            ],
            "test_date_range": [
                split_info["splits"]["test"].get("date_min"),
                split_info["splits"]["test"].get("date_max"),
            ],
            "chronological_reference_cutoffs": {
                "train_end_date": cfg.train_end_date,
                "val_end_date": cfg.val_end_date,
            },
            "training_seed": cfg.random_state,
            "random_seed": cfg.random_state,
            "split_seed": cfg.split_seed,
            "split_mode": cfg.split_mode,
        })
    (output_root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    if args.protocol_dry_run:
        run_config["status"] = "protocol_validated"
        (output_root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        print(f"Protocol dry run complete. Output: {output_root}")
        return

    results = []
    failed = []
    for spec in specs:
        try:
            results.append(train_experiment(
                spec=spec,
                cfg=cfg,
                feature_dim=feature_dim,
                train_ds=train_ds,
                val_ds=val_ds,
                output_root=output_root,
                device=device,
                target_window=target_window,
                test_ds=test_ds if not cfg.validation_only else None,
                cong_by_id=cong_by_id,
                protocol_info=protocol_info,
            ))
            pd.DataFrame(results).to_csv(output_root / "comparison_summary.csv", index=False)
        except Exception:
            traceback.print_exc()
            failed.append(spec.tag)

    if not results:
        raise SystemExit("All experiments failed.")

    summary_df = pd.DataFrame(results)
    lines = [
        f"| experiment | method | backbone | use_tiid | loss | train | val | cell_MAE | origin_cc_MAE@{cfg.congestion_upstream_km} | upstream_MAE@{cfg.congestion_upstream_km} | congestion_F1@{cfg.congestion_upstream_km} | best_val_loss |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['experiment']} | {row['method']} | {row.get('backbone', '')} | {int(row['use_tiid'])} | "
            f"{row.get('loss_name', '')} | {int(row.get('train_samples', 0))} | {int(row.get('val_samples', 0))} | "
            f"{row['mean_mae_kmh']:.2f} | "
            f"{row.get(f'origin_cc_MAE@{cfg.congestion_upstream_km}', float('nan')):.2f} | "
            f"{row.get('upstream_MAE', float('nan')):.2f} | "
            f"{row.get('congestion_F1', float('nan')):.4f} | {row['best_val_loss']:.6f} |"
        )
    (output_root / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_config["status"] = "complete" if not failed else "partial"
    run_config["completed_experiments"] = [str(row["experiment"]) for row in results]
    run_config["failed_experiments"] = failed
    (output_root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"Done. Output: {output_root}")
    if failed:
        print(f"Failed experiments: {failed}")


if __name__ == "__main__":
    main()
