# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "torch-directml; sys_platform == 'win32'",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scikit-learn",
#     "tqdm",
#     "pyarrow",
# ]
# ///
# ruff: noqa: E402
"""
Speed-matrix U-Net for accident congestion forecasting.

Architecture (from the legacy U-Net baseline):
- Input: speed matrix for `time_before_min` minutes pre-accident (default: -30 to -5).
- Output: speed matrix for `time_after_min` minutes post-accident (default: 0 to +145).
- 4-layer encoder/decoder with skip connections, 2 input channels (speed value + observed mask).
- Congestion-weighted L1 loss (eval_protocol_v1, 2026-05-29): downstream weight 0,
  upstream non-congested weight 1, upstream congested weight `w`;
  loss + operational metrics are capped to the upstream `0~+20 km` zone.
- LOCKED spec (2026-06-03, reports/loss_window_decision_20260603.md): loss window 20 km,
  origin_cc mask (--congestion-mask-npz), w=4. Champion selection uses the composite score
  (eval_protocol_v1 §5 / scripts/experiments/select_composite.py), not a lone metric.

Inputs:
- `data/accident_samples_5m.npz` (built by `scripts/data/build_accident_samples.py`)
- `data/accident_samples_5m_metadata.parquet` (for filtering / split)

Outputs:
- `output/models/unet_matrix/<run_id>/checkpoints/best_model.pth`
- `output/models/unet_matrix/<run_id>/checkpoints/last_model.pth`
- `output/models/unet_matrix/<run_id>/plots/`
- `output/models/unet_matrix/<run_id>/eval/metrics.csv` and `summary.txt`
- `output/models/unet_matrix/<run_id>/run_config.json`

Usage:
  uv run scripts/models/unet_matrix.py --epochs 50
  # Locked spec (2026-06-03): vd+hist, win20, origin_cc mask, w=4
  uv run scripts/models/unet_matrix.py --epochs 50 --eval-protocol v1 \
      --data-scope dispatch_only --input-type vd+hist --feature-set t0 \
      --loss upstream_congestion_weighted --congestion-upstream-km 20 --congestion-weight 4 \
      --historical-matrices-npz data/paper_trb_2026/historical_matrices_5m_pre20240401_n1n3.npz \
      --congestion-mask-npz data/paper_trb_2026/congestion_masks_origin_cc_5m_pre20240401_n1n3.npz
  # True-pace hybrid candidate (pace = 60 / max(v, 5 km/h), min/km)
  uv run scripts/models/unet_matrix.py --epochs 50 --eval-protocol v1 \
      --data-scope dispatch_only --input-type vd+hist --feature-set t0 \
      --loss speed_pace_hybrid --speed-loss-lambda 1 --pace-loss-lambda 1 \
      --pace-v-min-kmh 5 --congestion-upstream-km 20 \
      --historical-matrices-npz data/paper_trb_2026/historical_matrices_5m_pre20240401_n1n3.npz \
      --congestion-mask-npz data/paper_trb_2026/congestion_masks_origin_cc_5m_pre20240401_n1n3.npz
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Colab sets MPLBACKEND=module://matplotlib_inline.backend_inline, which is
# unavailable in the isolated uv venv.  Override to a non-interactive backend
# before matplotlib is imported.
import os as _os
if _os.environ.get("MPLBACKEND", "").startswith("module://"):
    _os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.data_loaders import load_samples
from scripts.lib.model_architectures import ALL_BACKBONES, WindowConfig, build_matrix_unet
from scripts.lib.windows import SampleWindow, subset
from scripts.models.paper_protocol import (
    DEFAULT_ALLOWED_FREEWAYS,
    DEFAULT_SPLIT_SEED,
    artifact_manifest,
    cohort_manifest,
    configure_reproducibility,
    hash_strings,
    load_samples_artifact,
    make_data_loader,
    parse_allowed_freeways,
    split_indices,
    write_split_membership,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "models" / "unet_matrix"

SPEED_MAX = 140.0
CONGESTION_KMH_DEFAULT = 60.0
# eval_protocol_v1 (2026-05-29): upstream congested weight `w`, swept over {2,4,7,10}
# (upstream non-congested = 1, downstream = 0). 4 is the mid-sweep default.
CONGESTION_WEIGHT_DEFAULT = 4.0
SECONDARY_CONGESTION_WEIGHT_DEFAULT = 2.0
# True-pace hybrid loss defaults. Pace is 60 / max(speed_kmh, v_min), in min/km.
PACE_V_MIN_KMH_DEFAULT = 5.0
SPEED_LOSS_LAMBDA_DEFAULT = 1.0
PACE_LOSS_LAMBDA_DEFAULT = 1.0
FULL_FIELD_LOSS_LAMBDA_DEFAULT = 0.0
# Upstream zone (km) over which the congestion-weighted loss and upstream_congestion_MAE
# apply. Locked to 20 km by eval_protocol_v1 §3.1/§4.1 (origin_cc p99≈14, max=26). The
# matrix window may extend further upstream; cells beyond +20 km are not weighted/scored.
CONGESTION_UPSTREAM_KM_DEFAULT = 20
PAD_TARGET = 64  # U-Net input must be a multiple of 16
FREE_FLOW_KMH = 80.0  # speed threshold for "recovered"

# eval_protocol_v1 time split defaults
TRAIN_END_DATE = "2024-11-30"
VAL_END_DATE = "2025-02-28"


@dataclass
class TrainConfig:
    upstream_km: int = 25
    downstream_km: int = 10
    time_before_min: int = 30
    time_after_min: int = 145
    spatial_resolution_km: int = 1
    resolution_min: int = 5
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    val_split: float = 0.15  # unused when eval_protocol is set
    random_state: int = 42
    congestion_kmh: float = CONGESTION_KMH_DEFAULT
    congestion_weight: float = CONGESTION_WEIGHT_DEFAULT  # upstream congested weight `w`
    secondary_congestion_weight: float = SECONDARY_CONGESTION_WEIGHT_DEFAULT
    congestion_upstream_km: int = CONGESTION_UPSTREAM_KM_DEFAULT  # loss/metric upstream cap (0~+20)
    min_before_coverage: float = 0.0
    min_filled_ratio: float = 0.8
    samples_npz: str = ""
    output_dir: str = ""
    run_id: str = ""
    # eval_protocol_v1 fields
    eval_protocol: str = ""
    data_scope: str = "dispatch_only"
    # eval_protocol_v1 (2026-05-29): current round grounded on t0; t5/t15 paused
    # pending the feature data-spec replan (see reports/planning_next_steps_20260529.md §0/§8).
    feature_set: str = "t0"
    input_type: str = "vd+hist"
    hist_validity_channel: bool = False
    hist_missing_fill: str = "nearest"
    coordinate_channels: bool = False
    backbone: str = "simple"  # simple | residual_attention
    simple_base_channels: int = 64
    loss_name: str = "upstream_congestion_weighted"
    checkpoint_selection_metric: str = "validation_loss"  # validation_loss | cell_mae
    speed_loss_lambda: float = SPEED_LOSS_LAMBDA_DEFAULT
    pace_loss_lambda: float = PACE_LOSS_LAMBDA_DEFAULT
    pace_v_min_kmh: float = PACE_V_MIN_KMH_DEFAULT
    full_field_loss_lambda: float = FULL_FIELD_LOSS_LAMBDA_DEFAULT
    # Optional origin_cc congestion-mask NPZ (build_congestion_masks.py). When set, the
    # congestion-weighted loss selects congested cells from this mask instead of speed<60
    # (eval_protocol_v1 §3.1 origin_cc variant). Empty = speed<60 fallback.
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
    eval_checkpoint: str = ""
    export_val_predictions: bool = False
    export_test_predictions: bool = False

    @property
    def n_space(self) -> int:
        return (self.upstream_km + self.downstream_km) // self.spatial_resolution_km + 1

    @property
    def n_time(self) -> int:
        return (self.time_before_min + self.time_after_min) // self.resolution_min + 1

    @property
    def n_before(self) -> int:
        # Cells covering pre-accident times (-time_before_min .. -resolution_min).
        # Cell at t=0 is the first target cell, not input.
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

    @property
    def in_channels(self) -> int:
        """Return the locked three-channel U-Net input width."""
        if self.input_type != "vd+hist":
            raise ValueError("U-Net is locked to input_type='vd+hist'")
        if self.hist_validity_channel:
            raise ValueError("the legacy fourth historical-validity channel is removed")
        if self.hist_missing_fill != "nearest":
            raise ValueError("3-channel U-Net requires hist_missing_fill='nearest'")
        if self.coordinate_channels:
            raise ValueError("coordinate channels are outside the locked 3-channel contract")
        return 3


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


# ============================================================
# Dataset
# ============================================================


class AccidentSampleDataset(Dataset):
    """Wraps a stack of (n_space, n_time) speed matrices as torch samples.

    Per item:
      - x: (3, PAD_TARGET, PAD_TARGET) reflect-padded VD speed, VD validity,
        and nearest-space-filled causal historical speed
      - y: (n_space, n_after) normalized target speed
      - m: (n_space, n_after) binary valid mask
      - acc_id: accident id string
    """

    def __init__(
        self,
        matrices: np.ndarray,        # (N, n_space, n_time) observed VD speeds in km/h, NaN allowed
        accident_ids: np.ndarray,    # (N,) str
        config: TrainConfig,
        hist_matrices: np.ndarray | None = None,  # (N, n_space, n_time) historical baseline, km/h
    ) -> None:
        self.matrices = matrices.astype(np.float32)
        self.accident_ids = accident_ids
        self.config = config
        self.hist_matrices = hist_matrices.astype(np.float32) if hist_matrices is not None else None
        _ = config.in_channels  # validate the locked input contract at construction
        if self.hist_matrices is None:
            raise ValueError("3-channel U-Net requires causal historical matrices")

    def __len__(self) -> int:
        return self.matrices.shape[0]

    @staticmethod
    def _nearest_space_fill(matrix: np.ndarray) -> np.ndarray:
        """Fill NaNs from the nearest valid spatial row at the same time.

        The frozen historical profile is causal. This operation uses no
        observed post-dispatch target and never borrows across time columns.
        """
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
        arr = self.matrices[idx]  # (n_space, n_time) observed VD

        speed_norm = np.where(
            np.isnan(arr),
            np.nan,
            np.clip(arr, 0.0, SPEED_MAX) / SPEED_MAX,
        )

        # Channel 1: observed speed with future cells masked to 0.
        vd_spd = np.nan_to_num(speed_norm.copy(), nan=0.0)
        vd_spd[:, cfg.n_before:] = 0.0
        # Channel 2: observed validity over the pre-incident input window.
        vd_val = np.zeros((cfg.n_space, cfg.n_time), dtype=np.float32)
        vd_val[:, : cfg.n_before] = (~np.isnan(speed_norm[:, : cfg.n_before])).astype(np.float32)
        # Channel 3: causal historical speed over the full window. The removed
        # validity plane is replaced by a deterministic nearest-space fill.
        h = self._nearest_space_fill(self.hist_matrices[idx])
        h_norm = np.clip(h, 0.0, SPEED_MAX) / SPEED_MAX
        channels = [vd_spd, vd_val, h_norm.astype(np.float32)]

        x = torch.from_numpy(np.stack(channels, axis=0)).float()
        x = F.pad(x, (cfg.pad_w, cfg.pad_w, cfg.pad_h, cfg.pad_h), mode="reflect")
        # Target is ALWAYS the observed VD post-incident field, regardless of input_type.
        target_raw = speed_norm[:, cfg.n_before:]
        valid_mask = (~np.isnan(target_raw)).astype(np.float32)
        target = np.nan_to_num(target_raw, nan=0.0)

        y = torch.from_numpy(target).float()
        m = torch.from_numpy(valid_mask).float()

        return x, y, m, str(self.accident_ids[idx])


# ============================================================
# Model
# ============================================================


class SimpleUNet(nn.Module):
    """Four-layer encoder-decoder with skip connections.

    Input  : (B, 2, PAD_TARGET, PAD_TARGET)
    Output : (B, n_space, n_after) sigmoid-normalized speed
    """

    def __init__(self, n_space: int, n_time: int, n_before: int, pad_h: int, pad_w: int, in_channels: int = 2) -> None:
        super().__init__()
        self.n_space = n_space
        self.n_time = n_time
        self.n_before = n_before
        self.pad_h = pad_h
        self.pad_w = pad_w

        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))

        self.dec4 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())

        self.final = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = torch.cat([self.dec4(e4), e3], dim=1)
        d3 = torch.cat([self.dec3(d4), e2], dim=1)
        d2 = torch.cat([self.dec2(d3), e1], dim=1)
        d1 = self.dec1(d2)

        out = self.final(d1)
        out = out[:, :, self.pad_h : self.pad_h + self.n_space, self.pad_w : self.pad_w + self.n_time]
        out = out.squeeze(1)
        out = out[:, :, self.n_before:]
        return out


# ============================================================
# Loss & metrics
# ============================================================


class CongestionWeightedLoss(nn.Module):
    def __init__(self, congestion_kmh: float, congestion_weight: float) -> None:
        super().__init__()
        self.threshold = congestion_kmh / SPEED_MAX
        self.w = congestion_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor,
                congestion_mask: torch.Tensor | None = None) -> torch.Tensor:
        l1 = torch.abs(pred - target)
        cong = (target < self.threshold).float()
        weight = 1.0 + (self.w - 1.0) * cong
        weighted = l1 * weight * valid_mask
        n_valid = valid_mask.sum().clamp(min=1)
        return weighted.sum() / n_valid


class IncidentMacroCellMAELoss(nn.Module):
    """Mean of per-incident valid-cell MAE, matching the paper cell metric."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        congestion_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del congestion_mask
        numerator = (
            torch.abs(pred - target) * valid_mask
        ).flatten(start_dim=1).sum(dim=1)
        denominator = valid_mask.flatten(start_dim=1).sum(dim=1)
        present = (denominator > 0).to(pred.dtype)
        per_incident = numerator / denominator.clamp(min=1)
        return (per_incident * present).sum() / present.sum().clamp(min=1)


class UpstreamCongestionWeightedLoss(nn.Module):
    """eval_protocol_v1 default loss (2026-05-29 weighting).

    Spatial weights over the upstream `0~+congestion_upstream_km` (default 20 km) zone:
      - upstream congested        -> w  (configurable; sweep {2,4,7,10})
      - upstream non-congested    -> 1
      - downstream / beyond +20km  -> 0  (excluded from the denominator)

    The matrix axis is ordered from downstream to upstream:
    `[-downstream_km, ..., 0, ..., +upstream_km]`. `upstream_start_idx` is the
    accident-point row; `upstream_end_idx` (exclusive) caps the weighted zone.
    When `congestion_mask` is supplied (origin_cc contour, eval_protocol_v1 §3.1),
    it defines the congested cells; otherwise `target < threshold` is the fallback.

    Loss = Σ(m·w·|ŷ−y|) / Σ(m·w) — a proper weighted mean (denominator = sum of
    weights over valid cells), so the value is a bounded weighted MAE and is not
    inflated by larger `w` (aligned to eval_protocol_v1 §3.1, 2026-06-03).
    """

    def __init__(self, upstream_start_idx: int, upstream_end_idx: int,
                 congested_weight: float, congestion_threshold: float = 60.0) -> None:
        super().__init__()
        self.up_start = upstream_start_idx
        self.up_end = upstream_end_idx
        self.w = congested_weight
        self.thresh = congestion_threshold / SPEED_MAX

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor,
                congestion_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Upstream-only, capped at +congestion_upstream_km. Proper weighted mean:
        # denominator = sum of weights over valid cells (Σ m·w), per eval_protocol_v1 §3.1.
        s, e = self.up_start, self.up_end
        up_valid = valid_mask[:, s:e, :]
        up_tgt = target[:, s:e, :]
        if congestion_mask is not None:
            cong = congestion_mask[:, s:e, :].float()
        else:
            cong = (up_tgt < self.thresh).float()
        weight = (1.0 + (self.w - 1.0) * cong) * up_valid  # non-congested → 1×, congested → w×, invalid → 0
        l1 = torch.abs(pred[:, s:e, :] - up_tgt) * weight
        return l1.sum() / weight.sum().clamp(min=1)


class FullFieldCongestionWeightedLoss(nn.Module):
    """LSTM-matched field MAE: base 1 everywhere and AIC weight ``w``.

    Only AIC cells inside the locked upstream 0--20 km zone receive the higher
    weight.  Every other valid output cell, including downstream and the final
    5 km of upstream context, retains unit weight.  The denominator is the
    complete valid weighted mass, matching the paper-baseline harness.
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

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        congestion_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s, e = self.up_start, self.up_end
        if congestion_mask is not None:
            focus = congestion_mask[:, s:e, :].to(pred.dtype)
        else:
            focus = (target[:, s:e, :] < self.thresh).to(pred.dtype)
        # Do not use in-place slice assignment here.  torch-directml 0.2.5 can
        # silently leave the destination unchanged, which collapses this loss
        # back to ordinary cell MAE.  Concatenation gives identical CPU/DML
        # semantics and keeps the weighting tensor independent of prediction.
        # DirectML rejects ``torch.cat`` when one of its inputs has zero
        # spatial length.  That boundary occurs when the AIC focus reaches
        # the first or last row of a cropped model window (for example an
        # upstream-15 km model with a 0--15 km AIC focus).  Build the same
        # tensor without including empty edge segments.
        parts = []
        if s > 0:
            parts.append(torch.ones_like(valid_mask[:, :s, :]))
        parts.append(1.0 + (self.w - 1.0) * focus)
        if e < valid_mask.shape[1]:
            parts.append(torch.ones_like(valid_mask[:, e:, :]))
        multiplier = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        weight = multiplier * valid_mask
        return (torch.abs(pred - target) * weight).sum() / weight.sum().clamp(min=1)


class FullFieldSpeed60WeightedLoss(nn.Module):
    """Full-field weighted MAE with weight ``w`` on observed speed<60 cells.

    The supplied origin-connected congestion mask is intentionally ignored: it
    remains available to the evaluator, while training weights are derived from
    every valid target cell whose observed speed is below the threshold.
    """

    def __init__(self, congested_weight: float, congestion_threshold: float) -> None:
        super().__init__()
        if congested_weight < 1:
            raise ValueError("congested_weight must be >= 1")
        self.w = float(congested_weight)
        self.thresh = congestion_threshold / SPEED_MAX

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        congestion_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del congestion_mask
        speed60 = (target < self.thresh).to(pred.dtype)
        weight = (1.0 + (self.w - 1.0) * speed60) * valid_mask
        return (torch.abs(pred - target) * weight).sum() / weight.sum().clamp(min=1)


class UpstreamCongestionFullFieldAuxLoss(nn.Module):
    """AIC-weighted upstream MAE plus a separately normalized field MAE.

    The first term is exactly the locked upstream 0--20 km objective: valid
    non-AIC cells have weight 1 and AIC cells have weight ``w``.  The auxiliary
    term gives every valid output cell unit weight.  Keeping the two means
    separately normalized makes ``full_field_lambda`` interpretable and avoids
    changing the AIC weight merely because the output window contains more
    ordinary cells.
    """

    def __init__(
        self,
        upstream_start_idx: int,
        upstream_end_idx: int,
        congested_weight: float,
        congestion_threshold: float,
        full_field_lambda: float,
    ) -> None:
        super().__init__()
        if full_field_lambda < 0:
            raise ValueError("full_field_lambda must be non-negative")
        self.upstream = UpstreamCongestionWeightedLoss(
            upstream_start_idx,
            upstream_end_idx,
            congested_weight,
            congestion_threshold,
        )
        self.full_field_lambda = float(full_field_lambda)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        congestion_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        upstream_loss = self.upstream(
            pred, target, valid_mask, congestion_mask
        )
        full_field_loss = (
            torch.abs(pred - target) * valid_mask
        ).sum() / valid_mask.sum().clamp(min=1)
        return upstream_loss + self.full_field_lambda * full_field_loss


class UpstreamTieredCongestionLoss(nn.Module):
    """Upstream weighted MAE using speed<60 alone or AIC/speed<60 tiers.

    ``speed60`` gives every valid speed<60 cell the configured primary weight.
    ``aic_speed60`` gives origin_cc cells the primary weight and other valid
    speed<60 cells the secondary weight. In both modes, ordinary upstream cells
    have weight 1 and cells outside 0--20 km have weight 0.
    """

    def __init__(
        self,
        upstream_start_idx: int,
        upstream_end_idx: int,
        congestion_threshold: float,
        primary_weight: float,
        secondary_weight: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in {"speed60", "aic_speed60"}:
            raise ValueError(f"unsupported tiered congestion-loss mode: {mode}")
        if primary_weight < 1 or secondary_weight < 1:
            raise ValueError("congestion weights must be >= 1")
        self.up_start = upstream_start_idx
        self.up_end = upstream_end_idx
        self.thresh = congestion_threshold / SPEED_MAX
        self.primary_weight = primary_weight
        self.secondary_weight = secondary_weight
        self.mode = mode

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        congestion_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s, e = self.up_start, self.up_end
        up_valid = valid_mask[:, s:e, :]
        up_tgt = target[:, s:e, :]
        speed60 = up_tgt < self.thresh
        if self.mode == "speed60":
            weight = torch.where(speed60, self.primary_weight, 1.0)
        else:
            if congestion_mask is None:
                raise ValueError("aic_speed60 loss requires an origin_cc congestion mask")
            origin_cc = congestion_mask[:, s:e, :] > 0
            weight = torch.where(
                origin_cc,
                self.primary_weight,
                torch.where(speed60, self.secondary_weight, 1.0),
            )
        effective = weight * up_valid
        l1 = torch.abs(pred[:, s:e, :] - up_tgt) * effective
        return l1.sum() / effective.sum().clamp(min=1)


class CongestionOnlyLoss(nn.Module):
    def __init__(self, congestion_threshold: float = 60.0) -> None:
        super().__init__()
        self.thresh = congestion_threshold / SPEED_MAX

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor,
                congestion_mask: torch.Tensor | None = None) -> torch.Tensor:
        cong = (target < self.thresh).float()
        mask = valid_mask * cong
        n = mask.sum().clamp(min=1)
        if n < 1:
            return (torch.abs(pred - target) * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        return (torch.abs(pred - target) * mask).sum() / n


class SpeedPaceHybridLoss(nn.Module):
    """General speed MAE plus upstream true-pace MAE.

    Speed term:
      mean(|v_hat - v| / SPEED_MAX) over all valid target cells.

    Pace term:
      mean(|60/max(v_hat, v_min) - 60/max(v, v_min)| / pace_span)
      over the upstream loss zone, where pace_span is the physically possible
      min/km range between v_min and SPEED_MAX. The scaling keeps the loss
      dimensionless while preserving the true-pace interpretation.
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

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        congestion_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        speed_err = torch.abs(pred - target)
        speed_loss = (speed_err * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        s, e = self.up_start, self.up_end
        up_valid = valid_mask[:, s:e, :]
        pred_kmh = torch.clamp(pred[:, s:e, :] * SPEED_MAX, min=self.v_min)
        target_kmh = torch.clamp(target[:, s:e, :] * SPEED_MAX, min=self.v_min)
        pred_pace = 60.0 / pred_kmh
        target_pace = 60.0 / target_kmh
        pace_err = torch.abs(pred_pace - target_pace) / self.pace_span
        pace_loss = (pace_err * up_valid).sum() / up_valid.sum().clamp(min=1)

        return self.speed_lambda * speed_loss + self.pace_lambda * pace_loss


class OriginCCPaceAuxLoss(nn.Module):
    """Locked origin_cc-weighted speed loss plus an upstream true-pace auxiliary.

    Base term is exactly the eval_protocol_v1 `upstream_congestion_weighted`
    loss: upstream non-origin_cc cells weight 1, origin_cc cells weight w, and
    downstream / beyond-window cells weight 0. The auxiliary pace term is a
    normalized true-pace MAE over all valid cells in the same upstream loss zone.
    """

    def __init__(
        self,
        upstream_start_idx: int,
        upstream_end_idx: int,
        congested_weight: float,
        congestion_threshold: float,
        pace_lambda: float,
        pace_v_min_kmh: float,
    ) -> None:
        super().__init__()
        if pace_v_min_kmh <= 0 or pace_v_min_kmh >= SPEED_MAX:
            raise ValueError(f"pace_v_min_kmh must be in (0, {SPEED_MAX}), got {pace_v_min_kmh}")
        self.up_start = upstream_start_idx
        self.up_end = upstream_end_idx
        self.w = congested_weight
        self.thresh = congestion_threshold / SPEED_MAX
        self.pace_lambda = pace_lambda
        self.v_min = pace_v_min_kmh
        self.pace_span = (60.0 / pace_v_min_kmh) - (60.0 / SPEED_MAX)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        congestion_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s, e = self.up_start, self.up_end
        up_valid = valid_mask[:, s:e, :]
        up_pred = pred[:, s:e, :]
        up_tgt = target[:, s:e, :]
        if congestion_mask is not None:
            cong = congestion_mask[:, s:e, :].float()
        else:
            cong = (up_tgt < self.thresh).float()
        weight = (1.0 + (self.w - 1.0) * cong) * up_valid
        base_loss = (torch.abs(up_pred - up_tgt) * weight).sum() / weight.sum().clamp(min=1)

        pred_kmh = torch.clamp(up_pred * SPEED_MAX, min=self.v_min)
        target_kmh = torch.clamp(up_tgt * SPEED_MAX, min=self.v_min)
        pred_pace = 60.0 / pred_kmh
        target_pace = 60.0 / target_kmh
        pace_err = torch.abs(pred_pace - target_pace) / self.pace_span
        pace_loss = (pace_err * up_valid).sum() / up_valid.sum().clamp(min=1)

        return base_loss + self.pace_lambda * pace_loss


def _upstream_start_idx(cfg: "TrainConfig") -> int:
    return cfg.downstream_km // cfg.spatial_resolution_km


def _upstream_end_idx(cfg: "TrainConfig") -> int:
    """Exclusive row index capping the upstream loss/metric zone at +congestion_upstream_km.

    Clamped to n_space so a smaller matrix window degrades gracefully.
    """
    cap = _upstream_start_idx(cfg) + cfg.congestion_upstream_km // cfg.spatial_resolution_km + 1
    return min(cap, cfg.n_space)


def build_loss(loss_name: str, cfg: "TrainConfig") -> nn.Module:
    up_start = _upstream_start_idx(cfg)
    up_end = _upstream_end_idx(cfg)
    if loss_name == "upstream_congestion_weighted":
        return UpstreamCongestionWeightedLoss(up_start, up_end, cfg.congestion_weight, cfg.congestion_kmh)
    if loss_name == "full_field_congestion_weighted":
        return FullFieldCongestionWeightedLoss(
            up_start,
            up_end,
            cfg.congestion_weight,
            cfg.congestion_kmh,
        )
    if loss_name == "full_field_speed60_weighted":
        return FullFieldSpeed60WeightedLoss(
            cfg.congestion_weight,
            cfg.congestion_kmh,
        )
    if loss_name == "upstream_congestion_fullfield_aux":
        return UpstreamCongestionFullFieldAuxLoss(
            up_start,
            up_end,
            cfg.congestion_weight,
            cfg.congestion_kmh,
            cfg.full_field_loss_lambda,
        )
    if loss_name == "upstream_speed60_weighted":
        return UpstreamTieredCongestionLoss(
            up_start,
            up_end,
            cfg.congestion_kmh,
            cfg.congestion_weight,
            cfg.secondary_congestion_weight,
            mode="speed60",
        )
    if loss_name == "upstream_aic_speed60_weighted":
        return UpstreamTieredCongestionLoss(
            up_start,
            up_end,
            cfg.congestion_kmh,
            cfg.congestion_weight,
            cfg.secondary_congestion_weight,
            mode="aic_speed60",
        )
    if loss_name == "speed_pace_hybrid":
        return SpeedPaceHybridLoss(
            up_start,
            up_end,
            cfg.speed_loss_lambda,
            cfg.pace_loss_lambda,
            cfg.pace_v_min_kmh,
        )
    if loss_name == "origin_cc_pace_aux":
        return OriginCCPaceAuxLoss(
            up_start,
            up_end,
            cfg.congestion_weight,
            cfg.congestion_kmh,
            cfg.pace_loss_lambda,
            cfg.pace_v_min_kmh,
        )
    if loss_name == "congestion_only_mae":
        return CongestionOnlyLoss(cfg.congestion_kmh)
    if loss_name == "incident_macro_cell_mae":
        return IncidentMacroCellMAELoss()
    # default: cell_mae (same as original CongestionWeightedLoss with weight=1)
    return CongestionWeightedLoss(cfg.congestion_kmh, 1.0)


# ── Operational metrics (eval_protocol_v1) ────────────────────────────────────

def _impact_time(speed_matrix_kmh: np.ndarray, upstream_start_idx: int, resolution_min: int,
                 valid_mask: np.ndarray, free_flow: float = FREE_FLOW_KMH,
                 upstream_end_idx: int | None = None) -> float:
    """Return time (minutes from t=0) until upstream zone stays above free_flow.

    The upstream zone is `[upstream_start_idx:upstream_end_idx]` (0~+20 km per
    eval_protocol_v1 §4.3); `upstream_end_idx=None` means to the matrix edge.
    """
    upstream = speed_matrix_kmh[upstream_start_idx:upstream_end_idx, :]
    up_mask  = valid_mask[upstream_start_idx:upstream_end_idx, :]
    n_time = speed_matrix_kmh.shape[1]
    for j in range(n_time - 1, -1, -1):
        valid_cells = up_mask[:, j]
        if valid_cells.sum() == 0:
            continue
        mean_v = (upstream[:, j] * valid_cells).sum() / valid_cells.sum()
        if mean_v < free_flow:
            return float((j + 1) * resolution_min)
    return 0.0


def _recovery_time(speed_matrix_kmh: np.ndarray, upstream_start_idx: int, resolution_min: int,
                   valid_mask: np.ndarray, free_flow: float = FREE_FLOW_KMH,
                   sustained_steps: int = 3) -> float:
    """Return time until the accident-to-10-km-upstream zone recovers."""
    near_end = min(speed_matrix_kmh.shape[0], upstream_start_idx + 11)
    near = speed_matrix_kmh[upstream_start_idx:near_end, :]
    near_mask = valid_mask[upstream_start_idx:near_end, :]
    n_time = near.shape[1]
    for j in range(n_time):
        consec = 0
        for k in range(j, min(j + sustained_steps, n_time)):
            valid = near_mask[:, k]
            if valid.sum() == 0:
                break
            mean_v = (near[:, k] * valid).sum() / valid.sum()
            if mean_v >= free_flow:
                consec += 1
            else:
                break
        if consec >= sustained_steps:
            return float(j * resolution_min)
    return float(n_time * resolution_min)


def compute_operational_metrics(
    pred_kmh: np.ndarray,
    target_kmh: np.ndarray,
    valid_mask: np.ndarray,
    upstream_start_idx: int,
    resolution_min: int,
    congestion_kmh: float = CONGESTION_KMH_DEFAULT,
    free_flow_kmh: float = FREE_FLOW_KMH,
    upstream_end_idx: int | None = None,
    congestion_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute eval_protocol_v1 operational metrics for a single accident.

    `upstream_end_idx` (exclusive) caps the impact-time and congestion-MAE zone at
    +20 km (recovery time stays at 0~+10 km). `congestion_mask`, if given, is the
    origin_cc contour (boolean, same shape as `target_kmh`) used to select congested
    cells for upstream_congestion_MAE; otherwise speed<congestion_kmh is the fallback.
    """
    # impact time (upstream 0~+20 km)
    it_obs  = _impact_time(target_kmh, upstream_start_idx, resolution_min, valid_mask, free_flow_kmh, upstream_end_idx)
    it_pred = _impact_time(pred_kmh,   upstream_start_idx, resolution_min, valid_mask, free_flow_kmh, upstream_end_idx)

    # recovery time (0~10 km upstream from the accident point)
    rt_obs  = _recovery_time(target_kmh, upstream_start_idx, resolution_min, valid_mask, free_flow_kmh)
    rt_pred = _recovery_time(pred_kmh,   upstream_start_idx, resolution_min, valid_mask, free_flow_kmh)

    # upstream congestion MAE (upstream 0~+20 km; congested cells = origin_cc contour, fallback speed<60)
    up_obs  = target_kmh[upstream_start_idx:upstream_end_idx, :]
    up_pred = pred_kmh[upstream_start_idx:upstream_end_idx, :]
    up_mask = valid_mask[upstream_start_idx:upstream_end_idx, :]
    if congestion_mask is not None:
        cong_mask = (congestion_mask[upstream_start_idx:upstream_end_idx, :] > 0) & (up_mask > 0)
    else:
        cong_mask = (up_obs < congestion_kmh) & (up_mask > 0)
    if cong_mask.sum() > 0:
        up_cong_mae = float(np.abs(up_pred[cong_mask] - up_obs[cong_mask]).mean())
    else:
        up_cong_mae = float("nan")

    return {
        "impact_time_obs_min": it_obs,
        "impact_time_pred_min": it_pred,
        "impact_time_err_min": it_pred - it_obs,
        "recovery_time_obs_min": rt_obs,
        "recovery_time_pred_min": rt_pred,
        "recovery_time_err_min": rt_pred - rt_obs,
        "upstream_congestion_mae_kmh": up_cong_mae,
    }


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    congestion_kmh: float,
    upstream_start_idx: int = 0,
    upstream_end_idx: int | None = None,
    congestion_mask: torch.Tensor | None = None,
) -> dict:
    """Cell-wise MAE/RMSE + upstream diagnostics + optional origin_cc MAE.

    The upstream zone is `[upstream_start_idx:upstream_end_idx]` (0~+20 km per
    eval_protocol_v1 §4.1; `upstream_end_idx=None` means to the matrix edge).
    `upstream_mae` remains the all-valid upstream diagnostic. When
    `congestion_mask` is supplied, `origin_cc_mae` and `upstream_congestion_mae`
    are MAE over accident-caused congestion cells only. Without the mask,
    `upstream_congestion_mae` falls back to the speed<congestion_kmh selector.
    Supports both 2-D single-sample (n_space, n_after) and 3-D batch (B, n_space, n_after).
    """
    pred_kmh = pred * SPEED_MAX
    target_kmh = target * SPEED_MAX
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

    pv = pred_kmh[valid]; tv = target_kmh[valid]
    mae  = torch.abs(pv - tv).mean().item()
    rmse = torch.sqrt(((pv - tv) ** 2).mean()).item()

    # Upstream zone slice (capped at +20 km) — handles both 2-D and 3-D inputs
    if pred.dim() == 3:
        up_pred = pred_kmh[:, upstream_start_idx:upstream_end_idx, :]
        up_tgt  = target_kmh[:, upstream_start_idx:upstream_end_idx, :]
        up_mask = valid_mask[:, upstream_start_idx:upstream_end_idx, :].bool()
        up_cmask = (
            congestion_mask[:, upstream_start_idx:upstream_end_idx, :].bool()
            if congestion_mask is not None else None
        )
    else:
        up_pred = pred_kmh[upstream_start_idx:upstream_end_idx, :]
        up_tgt  = target_kmh[upstream_start_idx:upstream_end_idx, :]
        up_mask = valid_mask[upstream_start_idx:upstream_end_idx, :].bool()
        up_cmask = (
            congestion_mask[upstream_start_idx:upstream_end_idx, :].bool()
            if congestion_mask is not None else None
        )

    if up_mask.sum() == 0:
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

    up_p = up_pred[up_mask]; up_t = up_tgt[up_mask]
    upstream_mae = torch.abs(up_p - up_t).mean().item()

    pc = up_p < congestion_kmh; tc = up_t < congestion_kmh
    has_congestion = bool(tc.any().item())
    speed60_mae = torch.abs(up_p[tc] - up_t[tc]).mean().item() if has_congestion else _nan
    if has_congestion:
        tp  = (pc & tc).sum().float(); fp = (pc & ~tc).sum().float(); fn = (~pc & tc).sum().float()
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        f1   = (2 * prec * rec / (prec + rec + 1e-8)).item()
    else:
        f1 = _nan

    origin_cc_mae = _nan
    has_origin_cc = False
    if up_cmask is not None:
        cc = up_cmask & up_mask
        has_origin_cc = bool(cc.any().item())
        if has_origin_cc:
            origin_cc_mae = torch.abs(up_pred[cc] - up_tgt[cc]).mean().item()

    upstream_congestion_mae = origin_cc_mae if up_cmask is not None else speed60_mae
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


# ============================================================
# Plot helpers (English labels for cross-platform font safety)
# ============================================================


def _plot_matrix(ax, data, title, vmin, vmax, cmap):
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time step", fontsize=8)
    ax.set_ylabel("Mileage cell", fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def plot_predictions(samples, save_path: Path, title: str = "") -> None:
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 3.5))
    if n == 1:
        axes = axes[None, :]
    for i, s in enumerate(samples):
        before_kmh = s["before"] * SPEED_MAX
        pred_kmh = s["pred"] * SPEED_MAX
        target_kmh = s["target"] * SPEED_MAX
        error = pred_kmh - target_kmh
        error[s["valid"] == 0] = np.nan
        _plot_matrix(axes[i, 0], before_kmh, f"Before\n{s['acc_id']}", 0, SPEED_MAX, "RdYlGn")
        f1_lbl = f"{s['f1']:.2f}" if not np.isnan(s.get("f1", float("nan"))) else "n/a"
        _plot_matrix(axes[i, 1], pred_kmh, f"Predicted\nupMAE={s.get('upstream_mae', s['mae']):.1f}|F1={f1_lbl}", 0, SPEED_MAX, "RdYlGn")
        _plot_matrix(axes[i, 2], target_kmh, "Ground truth", 0, SPEED_MAX, "RdYlGn")
        _plot_matrix(axes[i, 3], error, "Error (pred - target)", -40, 40, "RdBu_r")
    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_curves(history: dict, save_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss curves")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(history["val_mae"], color="steelblue", label="val MAE")
    ax2.axhline(15, color="red", linestyle="--", label="target < 15 km/h")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (km/h)")
    ax2.set_title("Validation MAE")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ============================================================
# Training entry
# ============================================================


def time_based_split(
    ids: np.ndarray,
    data_root: Path,
    train_end: str,
    val_end: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split sample indices into train/val/test by report_time.

    Returns arrays of integer indices (into `ids`) for train, val, test.
    Samples with report_time > val_end go to test.
    """
    meta_path = data_root / "accident_samples_5m_metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    meta = pd.read_parquet(meta_path)
    train_idx, val_idx, test_idx, _ = split_indices(
        ids,
        meta,
        split_mode="time",
        split_seed=DEFAULT_SPLIT_SEED,
        train_end=train_end,
        val_end=val_end,
    )
    return train_idx, val_idx, test_idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the baseline speed-matrix U-Net on accident samples NPZ.")
    p.add_argument("--samples-npz", default=None)
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--run-id", default=None)
    p.add_argument("--upstream-km", type=int, default=25)
    p.add_argument("--downstream-km", type=int, default=10)
    p.add_argument("--time-before-min", type=int, default=30)
    p.add_argument("--time-after-min", type=int, default=145)
    p.add_argument("--spatial-resolution-km", type=int, default=1)
    p.add_argument("--resolution-min", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Coupled L2 weight decay for Adam. Legacy runs use 0; conservative regularization screen uses 1e-5.",
    )
    p.add_argument("--val-split", type=float, default=0.15, help="Ignored when --eval-protocol is set")
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
    p.add_argument("--congestion-kmh", type=float, default=CONGESTION_KMH_DEFAULT)
    p.add_argument("--congestion-weight", type=float, default=CONGESTION_WEIGHT_DEFAULT,
                   help="Upstream congested-cell weight w (non-congested=1, downstream=0); "
                        "eval_protocol_v1 sweep {2,4,7,10}, default 4.")
    p.add_argument(
        "--secondary-congestion-weight",
        type=float,
        default=SECONDARY_CONGESTION_WEIGHT_DEFAULT,
        help="Non-origin_cc speed<60 weight in upstream_aic_speed60_weighted (default 2).",
    )
    p.add_argument("--congestion-upstream-km", type=int, default=CONGESTION_UPSTREAM_KM_DEFAULT,
                   help="Upstream zone (km) for congestion-weighted loss and upstream_congestion_MAE "
                        "(eval_protocol_v1 §3.1/§4.1; default 20). Matrix window may extend further.")
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
    # eval_protocol_v1 flags
    p.add_argument(
        "--eval-protocol",
        default="",
        help="Set to 'v1' to use the paper train/val/test protocol and operational metrics.",
    )
    p.add_argument("--loss", default="upstream_congestion_weighted",
                   choices=[
                       "cell_mae",
                       "incident_macro_cell_mae",
                       "full_field_congestion_weighted",
                       "full_field_speed60_weighted",
                       "upstream_congestion_weighted",
                       "upstream_congestion_fullfield_aux",
                       "upstream_speed60_weighted",
                       "upstream_aic_speed60_weighted",
                       "congestion_only_mae",
                       "speed_pace_hybrid",
                       "origin_cc_pace_aux",
                   ])
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
    p.add_argument(
        "--full-field-loss-lambda",
        type=float,
        default=FULL_FIELD_LOSS_LAMBDA_DEFAULT,
        help=(
            "Weight of the separately normalized all-valid-cell MAE term in "
            "upstream_congestion_fullfield_aux. Pass explicitly for ablations."
        ),
    )
    p.add_argument("--data-scope", default="dispatch_only", help="Ablation axis: data source label")
    p.add_argument("--feature-set", default="t0", choices=["t0", "t5", "t15"],
                   help="Ablation axis: feature availability time. Default t0; t5/t15 paused "
                        "pending the feature data-spec replan (planning_next_steps_20260529 §0/§8).")
    p.add_argument(
        "--input-type",
        default="vd+hist",
        choices=["vd+hist"],
        help="Locked 3-channel input: VD speed, VD validity, causal historical speed.",
    )
    p.add_argument("--backbone", default="simple", choices=ALL_BACKBONES,
                   help="UNet backbone: simple keeps the legacy architecture; residual_attention is the larger v2.")
    p.add_argument(
        "--simple-base-channels",
        type=int,
        choices=[48, 64],
        default=64,
        help="Width of the simple U-Net first stage. 64 is legacy; 48 is the conservative 0.75x ablation.",
    )
    p.add_argument("--hist-tag", default="", help="Tag suffix of the historical matrices NPZ (matches the samples --tag; empty = dispatch).")
    p.add_argument(
        "--historical-matrices-npz",
        default="",
        help="Explicit historical-matrix NPZ. Takes precedence over --hist-tag.",
    )
    p.add_argument("--congestion-mask-npz", default="",
                   help="origin_cc mask NPZ (required under --eval-protocol v1). When set, "
                        "the congestion-weighted loss uses origin_cc cells instead of speed<60 "
                        "(eval_protocol_v1 §3.1 origin_cc variant).")
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
    p.add_argument("--smoke", action="store_true", help="Smoke test: 5 epochs, skip heavy eval plotting")
    p.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Train and evaluate the chronological validation split without running "
            "test inference or writing test metrics. Use for adaptive candidate screens."
        ),
    )
    p.add_argument(
        "--eval-checkpoint",
        default="",
        help=(
            "Load a weights-only checkpoint and skip training. The caller must pass the "
            "same architecture/input arguments used to create the checkpoint."
        ),
    )
    p.add_argument(
        "--export-val-predictions",
        action="store_true",
        help=(
            "Write eval/predictions_val.npz with raw validation fields. This is only "
            "allowed together with --validation-only."
        ),
    )
    p.add_argument(
        "--export-test-predictions",
        action="store_true",
        help=(
            "Write eval/predictions_test.npz with raw formal temporal-evaluation "
            "fields. This is only allowed with --eval-checkpoint and without "
            "--validation-only."
        ),
    )
    p.add_argument(
        "--protocol-dry-run",
        action="store_true",
        help="Load/filter/split/hash artifacts and write run_config.json without training.",
    )
    p.add_argument(
        "--keep-ids", default=None,
        help=(
            "Path to a CSV with column 'accident_id' (e.g. output/quality/accident_keep_ids.csv "
            "from check_accident_quality.py). When set, only those IDs are used in train AND val. "
            "Test set is NEVER filtered — it always reflects the raw time-window distribution."
        ),
    )
    return p.parse_args()


def filter_and_subset(
    samples,
    target_window: SampleWindow,
    min_before_coverage: float,
    min_filled_ratio: float,
    allowed_freeways: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Subset the window and apply route plus canonical coverage filters.

    ``allowed_freeways=None`` preserves the historical helper behavior for
    external diagnostic scripts.  Paper entry points pass ``("N1", "N3")``
    explicitly.
    """
    matrices = []
    ids = []
    meta = samples.metadata.copy()
    meta["accident_id"] = meta["accident_id"].astype(str).str.strip()
    filled_series = meta["filled_ratio"] if "filled_ratio" in meta.columns else pd.Series(1.0, index=meta.index)
    filled_by_id = dict(zip(meta["accident_id"], filled_series))
    freeway_by_id = {
        str(row.accident_id): str(row.freeway).strip().upper()
        for row in meta[["accident_id", "freeway"]].itertuples(index=False)
    } if "freeway" in meta.columns else {}
    allowed_set = set(allowed_freeways) if allowed_freeways is not None else None
    for i in range(len(samples)):
        acc_id = str(samples.accident_ids[i]).strip()
        if allowed_set is not None and freeway_by_id.get(acc_id) not in allowed_set:
            continue
        filled_ratio = float(filled_by_id.get(acc_id, 0.0) or 0.0)
        if filled_ratio < min_filled_ratio:
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
        nan_ratio = float(np.isnan(before_block).mean()) if before_block.size else 1.0
        if min_before_coverage > 0 and nan_ratio > 1.0 - min_before_coverage:
            continue
        matrices.append(m)
        ids.append(acc_id)
    if not matrices:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=str)
    return np.stack(matrices, axis=0).astype(np.float32), np.asarray(ids)


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
            out.append(empty.copy()); n_missing += 1; continue
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


def load_congestion_masks_aligned(
    ids: np.ndarray,
    target_window: SampleWindow,
    mask_path: Path,
    require_complete: bool = False,
) -> np.ndarray:
    """Load origin_cc congestion masks, subset to the window, slice to post-accident cols.

    Returns `(N, n_space, n_after)` uint8 aligned to `ids` order — the same shape as
    the training target `y`. Accidents missing from the NPZ get an all-zero mask (no
    congested cells). Build the NPZ with `scripts/data/build_congestion_masks.py`.
    """
    if not mask_path.exists():
        raise SystemExit(
            f"congestion mask NPZ not found: {mask_path}\n"
            f"Build it first: uv run scripts/data/build_congestion_masks.py"
        )
    arr = np.load(str(mask_path), allow_pickle=False)
    src_window = SampleWindow(
        upstream_km=int(arr["window_upstream_km"]), downstream_km=int(arr["window_downstream_km"]),
        time_before_min=int(arr["window_time_before_min"]), time_after_min=int(arr["window_time_after_min"]),
        spatial_resolution_km=int(arr["window_spatial_resolution_km"]), resolution_min=int(arr["window_resolution_min"]),
    )
    n_before = target_window.time_before_min // target_window.resolution_min
    source_ids = arr["accident_ids"]
    source_matrix = arr["matrix"]
    id_to_idx = {str(a).strip(): i for i, a in enumerate(source_ids)}
    empty = np.zeros((target_window.n_space, target_window.n_time - n_before), dtype=np.uint8)
    out: list[np.ndarray] = []
    n_missing = 0
    for acc_id in ids:
        j = id_to_idx.get(str(acc_id).strip())
        if j is None:
            out.append(empty.copy()); n_missing += 1; continue
        sub, _ = subset(
            source_matrix[j], src_window,
            upstream_km=target_window.upstream_km, downstream_km=target_window.downstream_km,
            time_before_min=target_window.time_before_min, time_after_min=target_window.time_after_min,
        )
        out.append(sub[:, n_before:].astype(np.uint8))  # post-accident cols, matches target y
    if n_missing:
        message = f"{n_missing}/{len(ids)} accidents missing from congestion mask NPZ"
        if require_complete:
            raise SystemExit(message)
        print(f"  [warn] {message} -> zero mask")
    return np.stack(out, axis=0).astype(np.uint8)


def _eval_per_sample(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    cong_by_id: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """Run best-checkpoint model; return per-sample dicts with metrics + raw arrays."""
    up_start = _upstream_start_idx(cfg)
    up_end = _upstream_end_idx(cfg)
    results: list[dict] = []
    model.eval()
    with torch.no_grad():
        for x, y, m, ids_batch in tqdm(loader, desc="Eval", leave=False):
            x = x.to(device); y = y.to(device); m = m.to(device)
            pred = model(x)
            before = x[:, 0, cfg.pad_h : cfg.pad_h + cfg.n_space, cfg.pad_w : cfg.pad_w + cfg.n_before]
            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            m_np = m.cpu().numpy()
            before_np = before.cpu().numpy()
            for i, acc_id in enumerate(ids_batch):
                cmask_np = None
                if cong_by_id:
                    cmask_np = cong_by_id.get(str(acc_id))
                sm = compute_metrics(
                    torch.from_numpy(pred_np[i]),
                    torch.from_numpy(y_np[i]),
                    torch.from_numpy(m_np[i]),
                    cfg.congestion_kmh,
                    upstream_start_idx=up_start,
                    upstream_end_idx=up_end,
                    congestion_mask=torch.from_numpy(cmask_np) if cmask_np is not None else None,
                )
                operational = compute_operational_metrics(
                    pred_np[i] * SPEED_MAX,
                    y_np[i] * SPEED_MAX,
                    m_np[i],
                    upstream_start_idx=up_start,
                    resolution_min=cfg.resolution_min,
                    congestion_kmh=cfg.congestion_kmh,
                    upstream_end_idx=up_end,
                    congestion_mask=cmask_np,
                )
                results.append({
                    "acc_id": acc_id,
                    "before": before_np[i],
                    "pred": pred_np[i],
                    "target": y_np[i],
                    "valid": m_np[i],
                    **sm,
                    **operational,
                })
    return results


def _export_validation_predictions(
    output_path: Path,
    results: list[dict],
    dataset: AccidentSampleDataset,
    cfg: TrainConfig,
    target_window: SampleWindow,
    cong_by_id: dict[str, np.ndarray],
    checkpoint_path: Path,
) -> dict:
    """Export an auditable development-only prediction bundle.

    The bundle intentionally contains validation data only.  Targets and masks
    make offline metric checks possible, while the historical field supports
    predeclared blend/bias diagnostics without rebuilding the data pipeline.
    """
    if not cfg.validation_only:
        raise RuntimeError("validation prediction export requires validation_only=True")
    result_ids = np.asarray([str(row["acc_id"]).strip() for row in results], dtype=str)
    expected_ids = np.asarray([str(value).strip() for value in dataset.accident_ids], dtype=str)
    if not np.array_equal(result_ids, expected_ids):
        raise RuntimeError("validation prediction accident-ID alignment failure")

    prediction_kmh = np.stack([row["pred"] for row in results]).astype(np.float32) * SPEED_MAX
    target_kmh = np.stack([row["target"] for row in results]).astype(np.float32) * SPEED_MAX
    valid_mask = np.stack([row["valid"] for row in results]).astype(np.uint8)
    if dataset.hist_matrices is None:
        historical_future_kmh = np.full_like(prediction_kmh, np.nan, dtype=np.float32)
    else:
        historical_future_kmh = dataset.hist_matrices[:, :, cfg.n_before:].astype(np.float32)
    origin_cc_mask = np.stack(
        [
            cong_by_id.get(acc_id, np.zeros((cfg.n_space, cfg.n_after), dtype=np.uint8))
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

    checkpoint = artifact_manifest(checkpoint_path)
    runner = artifact_manifest(Path(__file__))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray("unet_matrix_validation_predictions_v1"),
        model_family=np.asarray(f"{cfg.backbone}_matrix_unet"),
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
        checkpoint_sha256=np.asarray(checkpoint["sha256"] or ""),
        runner_sha256=np.asarray(runner["sha256"] or ""),
        space_offsets_m=target_window.space_offsets_m.astype(np.int32),
        time_offsets_min=target_window.time_offsets_min[cfg.n_before:].astype(np.int32),
    )
    return artifact_manifest(output_path)


def _export_test_predictions(
    output_path: Path,
    results: list[dict],
    dataset: AccidentSampleDataset,
    cfg: TrainConfig,
    target_window: SampleWindow,
    cong_by_id: dict[str, np.ndarray],
    checkpoint_path: Path,
    expected_membership_sha256: str | None,
) -> dict:
    """Export a frozen-checkpoint temporal-test prediction bundle.

    This path is deliberately separate from the development export so the
    formal test artifact cannot be produced by an adaptive training run.
    """
    if cfg.validation_only:
        raise RuntimeError("test prediction export requires validation_only=False")
    if not cfg.eval_checkpoint:
        raise RuntimeError("test prediction export requires an eval checkpoint")
    result_ids = np.asarray([str(row["acc_id"]).strip() for row in results], dtype=str)
    expected_ids = np.asarray([str(value).strip() for value in dataset.accident_ids], dtype=str)
    if not np.array_equal(result_ids, expected_ids):
        raise RuntimeError("test prediction accident-ID alignment failure")
    membership_sha256 = hash_strings(result_ids.tolist(), sort=True)
    if (
        expected_membership_sha256
        and membership_sha256 != expected_membership_sha256
    ):
        raise RuntimeError(
            "test prediction membership does not match the formal split: "
            f"expected={expected_membership_sha256}, actual={membership_sha256}"
        )

    prediction_kmh = np.stack([row["pred"] for row in results]).astype(np.float32) * SPEED_MAX
    target_kmh = np.stack([row["target"] for row in results]).astype(np.float32) * SPEED_MAX
    valid_mask = np.stack([row["valid"] for row in results]).astype(np.uint8)
    if dataset.hist_matrices is None:
        historical_future_kmh = np.full_like(prediction_kmh, np.nan, dtype=np.float32)
    else:
        historical_future_kmh = dataset.hist_matrices[:, :, cfg.n_before:].astype(np.float32)
    origin_cc_mask = np.stack(
        [
            cong_by_id.get(acc_id, np.zeros((cfg.n_space, cfg.n_after), dtype=np.uint8))
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

    checkpoint = artifact_manifest(checkpoint_path)
    runner = artifact_manifest(Path(__file__))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray("unet_matrix_test_predictions_v1"),
        model_family=np.asarray(f"{cfg.backbone}_matrix_unet"),
        split=np.asarray("test"),
        stage=np.asarray("formal_temporal_evaluation"),
        fit_scope=np.asarray("train_only_model_checkpoint"),
        accident_ids=result_ids,
        prediction_kmh=prediction_kmh,
        target_kmh=target_kmh,
        valid_mask=valid_mask,
        historical_future_kmh=historical_future_kmh,
        origin_cc_mask=origin_cc_mask,
        training_seed=np.asarray(cfg.random_state, dtype=np.int64),
        membership_sha256=np.asarray(membership_sha256),
        test_membership_sha256=np.asarray(membership_sha256),
        ordered_ids_sha256=np.asarray(hash_strings(result_ids.tolist(), sort=False)),
        checkpoint_sha256=np.asarray(checkpoint["sha256"] or ""),
        runner_sha256=np.asarray(runner["sha256"] or ""),
        space_offsets_m=target_window.space_offsets_m.astype(np.int32),
        time_offsets_min=target_window.time_offsets_min[cfg.n_before:].astype(np.int32),
    )
    return artifact_manifest(output_path)


def main() -> None:
    args = parse_args()
    if args.export_val_predictions and not args.validation_only:
        raise SystemExit("--export-val-predictions requires --validation-only")
    if args.export_test_predictions and not args.eval_checkpoint:
        raise SystemExit("--export-test-predictions requires --eval-checkpoint")
    if args.export_test_predictions and args.validation_only:
        raise SystemExit("--export-test-predictions cannot be combined with --validation-only")
    if args.export_test_predictions and args.protocol_dry_run:
        raise SystemExit("--export-test-predictions cannot be combined with --protocol-dry-run")
    if args.export_test_predictions and args.export_val_predictions:
        raise SystemExit(
            "--export-test-predictions cannot be combined with --export-val-predictions"
        )
    if args.eval_checkpoint and args.smoke:
        raise SystemExit("--eval-checkpoint cannot be combined with --smoke")
    eval_checkpoint_path = Path(args.eval_checkpoint).resolve() if args.eval_checkpoint else None
    if eval_checkpoint_path is not None and not eval_checkpoint_path.is_file():
        raise SystemExit(f"--eval-checkpoint not found: {eval_checkpoint_path}")
    if (
        args.eval_protocol == "v1"
        and not args.validation_only
        and eval_checkpoint_path is None
        and not args.protocol_dry_run
    ):
        raise SystemExit(
            "formal eval_protocol v1 training requires --validation-only; "
            "run temporal test evaluation only from a frozen --eval-checkpoint"
        )
    if args.backbone != "simple" and args.simple_base_channels != 64:
        raise SystemExit("--simple-base-channels only applies to --backbone simple")
    if args.weight_decay < 0:
        raise SystemExit("--weight-decay must be non-negative")
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
        if args.keep_ids:
            raise SystemExit("formal eval_protocol v1 uses the full eligible cohort; --keep-ids is disabled")
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
        epochs=args.smoke and 5 or args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        random_state=training_seed,
        congestion_kmh=args.congestion_kmh,
        congestion_weight=args.congestion_weight,
        secondary_congestion_weight=args.secondary_congestion_weight,
        congestion_upstream_km=args.congestion_upstream_km,
        min_before_coverage=args.min_before_coverage,
        min_filled_ratio=args.min_filled_ratio,
        samples_npz=args.samples_npz or "",
        output_dir="",
        run_id=args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        eval_protocol=args.eval_protocol,
        data_scope=args.data_scope,
        feature_set=args.feature_set,
        input_type=args.input_type,
        hist_validity_channel=False,
        hist_missing_fill="nearest",
        coordinate_channels=False,
        backbone=args.backbone,
        simple_base_channels=args.simple_base_channels,
        loss_name=args.loss,
        checkpoint_selection_metric=args.checkpoint_selection,
        speed_loss_lambda=args.speed_loss_lambda,
        pace_loss_lambda=args.pace_loss_lambda,
        pace_v_min_kmh=args.pace_v_min_kmh,
        full_field_loss_lambda=args.full_field_loss_lambda,
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
        eval_checkpoint=str(eval_checkpoint_path) if eval_checkpoint_path is not None else "",
        export_val_predictions=args.export_val_predictions,
        export_test_predictions=args.export_test_predictions,
    )

    if cfg.downstream_km % cfg.spatial_resolution_km != 0:
        raise SystemExit("--downstream-km must be divisible by --spatial-resolution-km")
    if cfg.time_before_min % cfg.resolution_min != 0:
        raise SystemExit("--time-before-min must be divisible by --resolution-min")

    if cfg.n_space % 4 != 0 and cfg.pad_h * 2 + cfg.n_space != PAD_TARGET:
        # Pad must be exact for the symmetric reflect to land on PAD_TARGET.
        raise SystemExit(
            f"window space ({cfg.n_space}) cannot symmetrically pad to {PAD_TARGET}; "
            "choose upstream_km + downstream_km so that n_space is even."
        )
    if cfg.n_time % 4 != 0 and cfg.pad_w * 2 + cfg.n_time != PAD_TARGET:
        raise SystemExit(
            f"window time ({cfg.n_time}) cannot symmetrically pad to {PAD_TARGET}; "
            "choose time_before_min + time_after_min so that n_time is even."
        )

    output_dir = Path(args.output_root) / cfg.run_id
    cfg.output_dir = str(output_dir)
    source_run_config_path: Path | None = None
    source_run_config: dict = {}
    if eval_checkpoint_path is not None:
        source_run_dir = eval_checkpoint_path.parent.parent.resolve()
        if output_dir.resolve() == source_run_dir:
            raise SystemExit(
                "evaluation output must not reuse the source checkpoint run directory"
            )
        source_run_config_path = source_run_dir / "run_config.json"
        if cfg.eval_protocol == "v1" and not source_run_config_path.is_file():
            raise SystemExit(
                "formal eval_protocol v1 requires the checkpoint's source run_config.json: "
                f"{source_run_config_path}"
            )
        if source_run_config_path.is_file():
            try:
                source_run_config = json.loads(
                    source_run_config_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(
                    f"cannot read checkpoint source run_config: {source_run_config_path}: {exc}"
                ) from exc
    ckpt_dir = output_dir / "checkpoints"
    plot_dir = output_dir / "plots"
    eval_dir = output_dir / "eval"
    for d in (ckpt_dir, plot_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    device, device_msg = select_device(cfg.device_preference)
    print(f"Device: {device_msg}")
    print(f"Run id: {cfg.run_id}")
    print(f"Output: {output_dir}")
    print(f"Window: ({cfg.n_space} space x {cfg.n_time} time) | input cols 0..{cfg.n_before-1}, target cols {cfg.n_before}..{cfg.n_time-1}")

    # ---- Load samples ----
    npz_path = Path(args.samples_npz) if args.samples_npz else None
    samples_npz_path = npz_path or (PROJECT_ROOT / "data" / f"accident_samples_{cfg.resolution_min}m.npz")
    samples_metadata_path = (
        Path(args.samples_metadata_parquet)
        if args.samples_metadata_parquet
        else samples_npz_path.with_name(f"{samples_npz_path.stem}_metadata.parquet")
    )
    samples = load_samples(
        data_root=PROJECT_ROOT / "data",
        resolution_min=cfg.resolution_min,
    ) if npz_path is None else load_samples_artifact(npz_path, samples_metadata_path)

    target_window = SampleWindow(
        upstream_km=cfg.upstream_km,
        downstream_km=cfg.downstream_km,
        time_before_min=cfg.time_before_min,
        time_after_min=cfg.time_after_min,
        spatial_resolution_km=cfg.spatial_resolution_km,
        resolution_min=cfg.resolution_min,
    )
    matrices, ids = filter_and_subset(
        samples,
        target_window,
        cfg.min_before_coverage,
        cfg.min_filled_ratio,
        allowed_freeways=allowed_freeways,
    )
    cohort_info = cohort_manifest(
        samples.accident_ids,
        ids,
        samples.metadata,
        allowed_freeways,
    )
    print(
        f"Samples after coverage filter: {len(matrices)}/{len(samples)} "
        f"(min_filled_ratio={cfg.min_filled_ratio}, min_before_coverage={cfg.min_before_coverage})"
    )
    print(
        f"Paper scope: {','.join(allowed_freeways)} | route-excluded={cohort_info['route_excluded_n']} | "
        f"retained routes={cohort_info['retained_freeway_counts']}"
    )
    if len(matrices) == 0:
        raise SystemExit("No samples left after filter; loosen --min-filled-ratio/--min-before-coverage or rebuild samples.")

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
        print(f"Input type: {cfg.input_type} | in_channels={cfg.in_channels} | hist matrices: {hist_matrices.shape}")
    else:
        print(f"Input type: {cfg.input_type} | in_channels={cfg.in_channels}")

    # Optional origin_cc congestion mask for the weighted loss (eval_protocol_v1 §3.1 variant).
    cong_by_id: dict[str, np.ndarray] = {}
    if cfg.congestion_mask_npz:
        cong_aligned = load_congestion_masks_aligned(
            ids,
            target_window,
            Path(cfg.congestion_mask_npz),
            require_complete=(cfg.eval_protocol == "v1"),
        )
        cong_by_id = {str(a): cong_aligned[i] for i, a in enumerate(ids)}
        cong_share = float(cong_aligned.mean())
        loss_selector = {
            "full_field_speed60_weighted": "speed<60 over the full field (origin_cc retained for evaluation only)",
            "upstream_speed60_weighted": "speed<60 (origin_cc retained for evaluation only)",
            "upstream_aic_speed60_weighted": "origin_cc=4 with non-origin_cc speed<60=2",
        }.get(cfg.loss_name, "origin_cc")
        print(f"Congestion mask: {Path(cfg.congestion_mask_npz).name} | origin_cc cell share={cong_share:.4%} "
              f"(loss congested-cell selector = {loss_selector})")

    def _gather_cmask(ids_batch, device) -> "torch.Tensor | None":
        """Stack per-accident origin_cc masks for a batch, or None when disabled."""
        if not cong_by_id:
            return None
        stacked = np.stack([cong_by_id[str(a)] for a in ids_batch], axis=0)
        return torch.from_numpy(stacked).to(device)

    if cfg.eval_protocol == "v1":
        train_idx, val_idx, test_idx, split_info = split_indices(
            ids,
            samples.metadata,
            split_mode=cfg.split_mode,
            split_seed=cfg.split_seed,
            train_end=cfg.train_end_date,
            val_end=cfg.val_end_date,
        )
        print(
            f"Paper split ({cfg.split_mode}): train={len(train_idx)} | val={len(val_idx)} | "
            f"test={len(test_idx)} | split_seed={cfg.split_seed} (training_seed={cfg.random_state})"
        )
    else:
        # legacy random split (backward compat)
        rng = np.random.default_rng(cfg.random_state)
        all_idx = np.arange(len(matrices))
        rng.shuffle(all_idx)
        split = int(len(all_idx) * (1 - cfg.val_split))
        train_idx, val_idx, test_idx = all_idx[:split], all_idx[split:], np.array([], dtype=int)
        print(f"Random split: train={len(train_idx)} | val={len(val_idx)}")
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

    # Quality filter on train+val — test always keeps the raw time-window distribution.
    if args.keep_ids:
        keep_df = pd.read_csv(args.keep_ids, dtype={"accident_id": str})
        keep_set = set(keep_df["accident_id"].str.strip())
        train_mask = np.array([str(ids[i]) in keep_set for i in train_idx])
        val_mask   = np.array([str(ids[i]) in keep_set for i in val_idx])
        n_train_before, n_val_before = len(train_idx), len(val_idx)
        train_idx = train_idx[train_mask]
        val_idx   = val_idx[val_mask]
        print(
            f"Quality filter ({Path(args.keep_ids).name}): "
            f"train {len(train_idx)}/{n_train_before} | val {len(val_idx)}/{n_val_before} | "
            f"test {len(test_idx)} unchanged"
        )

    if len(train_idx) < 2 or len(val_idx) < 1:
        raise SystemExit(
            f"Split too small after optional quality filtering: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )

    split_membership_artifact = write_split_membership(
        output_dir / "split_membership.csv",
        ids,
        samples.metadata,
        {"train": train_idx, "val": val_idx, "test": test_idx},
    )
    split_info["actual_after_optional_quality_filter_counts"] = {
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
    }
    split_info["membership_artifact"] = split_membership_artifact
    split_info["optional_keep_ids"] = artifact_manifest(args.keep_ids)

    def _hist(idx):
        return hist_matrices[idx] if hist_matrices is not None else None

    train_ds = AccidentSampleDataset(matrices[train_idx], ids[train_idx], cfg, _hist(train_idx))
    val_ds = AccidentSampleDataset(matrices[val_idx], ids[val_idx], cfg, _hist(val_idx))
    test_ds = AccidentSampleDataset(matrices[test_idx], ids[test_idx], cfg, _hist(test_idx)) if len(test_idx) else None
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds) if test_ds else 0}")

    seed_info = configure_reproducibility(cfg.random_state, cfg.deterministic_mode)
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
        if test_ds and not cfg.validation_only
        else None
    )
    if cfg.export_test_predictions and test_loader is None:
        raise RuntimeError(
            "formal test prediction export requested, but no test loader was constructed"
        )

    matrix_input_schema: list[dict[str, object]] = []
    if "vd" in cfg.input_type:
        matrix_input_schema += [
            {
                "name": "observed_vd_speed",
                "dtype": "float32",
                "transform": f"clip_0_{SPEED_MAX:g}_divide_{SPEED_MAX:g}",
                "future_policy": "zero_masked",
                "padding": "reflect",
            },
            {
                "name": "observed_vd_validity",
                "dtype": "float32",
                "transform": "finite_indicator_pre_dispatch_only",
                "padding": "reflect",
            },
        ]
    if "hist" in cfg.input_type:
        matrix_input_schema.append(
            {
                "name": "causal_historical_speed",
                "dtype": "float32",
                "transform": f"clip_0_{SPEED_MAX:g}_divide_{SPEED_MAX:g}",
                "missing_fill": cfg.hist_missing_fill,
                "padding": "reflect",
            }
        )
        if cfg.hist_validity_channel:
            matrix_input_schema.append(
                {
                    "name": "causal_historical_validity",
                    "dtype": "float32",
                    "transform": "finite_indicator",
                    "padding": "reflect",
                }
            )
    if cfg.coordinate_channels:
        matrix_input_schema += [
            {
                "name": "normalized_signed_space_from_incident",
                "dtype": "float32",
                "normalizer_km": max(cfg.upstream_km, cfg.downstream_km, 1),
                "padding": "constant_zero_sentinel",
            },
            {
                "name": "normalized_signed_time_from_dispatch",
                "dtype": "float32",
                "normalizer_min": max(cfg.time_before_min, cfg.time_after_min, 1),
                "padding": "constant_zero_sentinel",
            },
        ]
    protocol_info = {
        "paper_scope": "/".join(allowed_freeways),
        "allowed_freeways": list(allowed_freeways),
        "cohort": cohort_info,
        "split": split_info,
        "reproducibility": seed_info,
        "feature_schema": {
            "mode": "matrix_plus_deterministic_coordinates" if cfg.coordinate_channels else "matrix_only",
            "feature_count": len(matrix_input_schema),
            "ordered_schema": matrix_input_schema,
            "ordered_schema_sha256": hash_strings(
                [
                    json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    for item in matrix_input_schema
                ],
                sort=False,
            ),
        },
        "evaluation_firewall": {
            "validation_only": cfg.validation_only,
            "test_loader_constructed": test_loader is not None,
            "test_inference_allowed": not cfg.validation_only,
            "validation_prediction_export_requested": cfg.export_val_predictions,
            "test_prediction_export_requested": cfg.export_test_predictions,
        },
        "artifacts": {
            "runner_script": artifact_manifest(Path(__file__)),
            "samples_npz": artifact_manifest(samples_npz_path),
            "samples_metadata_parquet": artifact_manifest(samples_metadata_path),
            "historical_matrices_npz": artifact_manifest(historical_path),
            "congestion_mask_npz": artifact_manifest(cfg.congestion_mask_npz),
            "evaluation_checkpoint": artifact_manifest(eval_checkpoint_path),
        },
        "runner_argv": [str(arg) for arg in sys.argv],
    }
    initial_run_config = asdict(cfg)
    initial_run_config.update({
        "input_channels": cfg.in_channels,
        "training_seed": cfg.random_state,
        "protocol": protocol_info,
        "status": "evaluating_checkpoint" if eval_checkpoint_path is not None else "training",
    })
    (output_dir / "run_config.json").write_text(
        json.dumps(initial_run_config, indent=2),
        encoding="utf-8",
    )
    if args.protocol_dry_run:
        initial_run_config["status"] = "protocol_validated"
        (output_dir / "run_config.json").write_text(
            json.dumps(initial_run_config, indent=2),
            encoding="utf-8",
        )
        print(f"Protocol dry run complete. Output: {output_dir}")
        return

    # ---- Model / loss / optimizer ----
    window_cfg = WindowConfig(
        upstream_km=cfg.upstream_km,
        downstream_km=cfg.downstream_km,
        time_before_min=cfg.time_before_min,
        time_after_min=cfg.time_after_min,
        spatial_resolution_km=cfg.spatial_resolution_km,
        resolution_min=cfg.resolution_min,
    )
    model = build_matrix_unet(
        window_cfg,
        input_type=cfg.input_type,
        backbone=cfg.backbone,
        in_channels=cfg.in_channels,
        simple_base_channels=cfg.simple_base_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Backbone: {cfg.backbone} | Params: {n_params:,}")

    criterion = build_loss(cfg.loss_name, cfg).to(device)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_cell_mae": [],
        "val_mae": [],
        "val_f1": [],
    }
    best_epoch = 0
    best_val_loss: float | None = None
    best_selection_value: float | None = None

    if eval_checkpoint_path is not None:
        if source_run_config:
            expected_fields = {
                "upstream_km": cfg.upstream_km,
                "downstream_km": cfg.downstream_km,
                "time_before_min": cfg.time_before_min,
                "time_after_min": cfg.time_after_min,
                "spatial_resolution_km": cfg.spatial_resolution_km,
                "resolution_min": cfg.resolution_min,
                "input_type": cfg.input_type,
                "backbone": cfg.backbone,
                "simple_base_channels": cfg.simple_base_channels,
                "loss_name": cfg.loss_name,
                "checkpoint_selection_metric": cfg.checkpoint_selection_metric,
                "weight_decay": cfg.weight_decay,
                "eval_protocol": cfg.eval_protocol,
                "data_scope": cfg.data_scope,
                "feature_set": cfg.feature_set,
                "split_mode": cfg.split_mode,
                "split_seed": cfg.split_seed,
                "allowed_freeways": cfg.allowed_freeways,
                "train_end_date": cfg.train_end_date,
                "val_end_date": cfg.val_end_date,
                "hist_validity_channel": cfg.hist_validity_channel,
                "hist_missing_fill": cfg.hist_missing_fill,
                "coordinate_channels": cfg.coordinate_channels,
                "input_channels": cfg.in_channels,
                "training_seed": cfg.random_state,
            }
            mismatches = []
            for field, expected in expected_fields.items():
                observed = source_run_config.get(field)
                if field == "training_seed" and observed is None:
                    observed = source_run_config.get("random_state")
                if observed is not None and observed != expected:
                    mismatches.append(f"{field}: checkpoint={observed!r}, requested={expected!r}")
            source_protocol = source_run_config.get("protocol", {})
            source_artifacts = source_protocol.get("artifacts", {})
            current_artifacts = protocol_info.get("artifacts", {})
            for artifact_name in (
                "samples_npz",
                "samples_metadata_parquet",
                "historical_matrices_npz",
                "congestion_mask_npz",
            ):
                source_sha = source_artifacts.get(artifact_name, {}).get("sha256")
                current_sha = current_artifacts.get(artifact_name, {}).get("sha256")
                if source_sha and current_sha and source_sha != current_sha:
                    mismatches.append(
                        f"{artifact_name}.sha256: checkpoint={source_sha}, requested={current_sha}"
                    )
            source_splits = source_protocol.get("split", {}).get("splits", {})
            current_splits = split_info.get("splits", {})
            for split_name in ("train", "val", "test"):
                source_sha = source_splits.get(split_name, {}).get("membership_sha256")
                current_sha = current_splits.get(split_name, {}).get("membership_sha256")
                if source_sha and current_sha and source_sha != current_sha:
                    mismatches.append(
                        f"{split_name}.membership_sha256: checkpoint={source_sha}, requested={current_sha}"
                    )
            source_checkpoint_artifacts = source_run_config.get("checkpoint_artifacts", {})
            source_checkpoint_key = (
                "best_model"
                if eval_checkpoint_path.name == "best_model.pth"
                else "last_model"
                if eval_checkpoint_path.name == "last_model.pth"
                else "evaluation_checkpoint"
            )
            recorded_checkpoint_sha = source_checkpoint_artifacts.get(
                source_checkpoint_key, {}
            ).get("sha256")
            actual_checkpoint_sha = artifact_manifest(eval_checkpoint_path).get("sha256")
            if (
                recorded_checkpoint_sha
                and actual_checkpoint_sha
                and recorded_checkpoint_sha != actual_checkpoint_sha
            ):
                mismatches.append(
                    "checkpoint.sha256: "
                    f"source={recorded_checkpoint_sha}, actual={actual_checkpoint_sha}"
                )
            if mismatches:
                raise SystemExit(
                    "--eval-checkpoint source run_config is incompatible:\n  "
                    + "\n  ".join(mismatches)
                )
        source_selection = source_run_config.get("checkpoint_selection", {})
        best_epoch = int(source_selection.get("best_epoch", 0) or 0)
        source_best_val_loss = source_selection.get("best_val_loss")
        best_val_loss = (
            float(source_best_val_loss) if source_best_val_loss is not None else None
        )
        source_best_selection_value = source_selection.get("best_selection_value")
        best_selection_value = (
            float(source_best_selection_value)
            if source_best_selection_value is not None
            else best_val_loss
        )
        selected_checkpoint_path = eval_checkpoint_path
        checkpoint_selection = {
            "criterion": "provided_checkpoint",
            "evaluation_only": True,
            "source_checkpoint_selection": source_selection or None,
        }
        checkpoint_artifacts = {
            "evaluation_checkpoint": artifact_manifest(eval_checkpoint_path),
            "source_run_config": artifact_manifest(source_run_config_path),
        }
        train_sec = 0.0
        print(f"Evaluation-only checkpoint: {eval_checkpoint_path}")
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            betas=(0.5, 0.999),
            weight_decay=cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=1e-6
        )
        best_val_loss = float("inf")
        best_selection_value = float("inf")
        t_train = time.time()

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            train_loss = 0.0
            train_loss_weight = 0
            for x, y, m, ids_b in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
                x = x.to(device); y = y.to(device); m = m.to(device)
                cmask = _gather_cmask(ids_b, device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y, m, cmask)
                loss.backward()
                optimizer.step()
                batch_weight = y.shape[0] if cfg.loss_name == "incident_macro_cell_mae" else 1
                train_loss += loss.item() * batch_weight
                train_loss_weight += batch_weight
            train_loss /= max(train_loss_weight, 1)

            model.eval()
            val_loss = 0.0
            val_loss_weight = 0
            all_pred, all_y, all_m, all_c = [], [], [], []
            with torch.no_grad():
                for x, y, m, ids_b in val_loader:
                    x = x.to(device); y = y.to(device); m = m.to(device)
                    cmask = _gather_cmask(ids_b, device)
                    pred = model(x)
                    batch_weight = y.shape[0] if cfg.loss_name == "incident_macro_cell_mae" else 1
                    val_loss += criterion(pred, y, m, cmask).item() * batch_weight
                    val_loss_weight += batch_weight
                    all_pred.append(pred.cpu()); all_y.append(y.cpu()); all_m.append(m.cpu())
                    if cmask is not None:
                        all_c.append(cmask.cpu())
            val_loss /= max(val_loss_weight, 1)
            cat_pred = torch.cat(all_pred); cat_y = torch.cat(all_y); cat_m = torch.cat(all_m)
            cat_c = torch.cat(all_c) if all_c else None
            metrics = compute_metrics(cat_pred, cat_y, cat_m, cfg.congestion_kmh,
                                      _upstream_start_idx(cfg), _upstream_end_idx(cfg), cat_c)
            per_incident_num = (
                torch.abs(cat_pred - cat_y) * cat_m
            ).flatten(start_dim=1).sum(dim=1)
            per_incident_den = cat_m.flatten(start_dim=1).sum(dim=1)
            present = per_incident_den > 0
            if not bool(present.any()):
                raise RuntimeError("validation split has no incident with a valid target cell")
            val_cell_mae = float(
                (per_incident_num[present] / per_incident_den[present]).mean().item()
                * SPEED_MAX
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_cell_mae"].append(val_cell_mae)
            history["val_mae"].append(metrics["upstream_congestion_mae"])
            history["val_f1"].append(metrics["f1"])

            f1_str = f"{metrics['f1']:.3f}" if not np.isnan(metrics["f1"]) else "n/a"
            occ_str = f"{metrics['origin_cc_mae']:.2f}" if not np.isnan(metrics["origin_cc_mae"]) else "n/a"
            print(
                f"Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f} | "
                f"cell_MAE={val_cell_mae:.3f} km/h | "
                f"origin_cc_MAE@20={occ_str} km/h | up_MAE={metrics['upstream_mae']:.2f} km/h | F1={f1_str}"
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

        train_sec = round(time.time() - t_train, 1)
        selected_checkpoint_path = ckpt_dir / "best_model.pth"
        checkpoint_selection = {
            "criterion": (
                "minimum_validation_cell_mae"
                if cfg.checkpoint_selection_metric == "cell_mae"
                else "minimum_validation_weighted_loss"
            ),
            "best_epoch": int(best_epoch),
            "best_selection_value": float(best_selection_value),
            "selection_units": (
                "km/h"
                if cfg.checkpoint_selection_metric == "cell_mae"
                else "normalized_loss"
            ),
            "best_val_loss": float(best_val_loss),
        }
        checkpoint_artifacts = {
            "best_model": artifact_manifest(ckpt_dir / "best_model.pth"),
            "last_model": artifact_manifest(ckpt_dir / "last_model.pth"),
        }
        plot_curves(history, plot_dir / "curves.png")

    # ---- Full evaluation on val (and test, unless firewalled) ----
    model.load_state_dict(
        torch.load(selected_checkpoint_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    model.to(device)

    val_results = _eval_per_sample(model, val_loader, cfg, device, cong_by_id)

    def _nanmean(vals): return float(np.nanmean(vals)) if vals else float("nan")
    def _nanstd(vals):  return float(np.nanstd(vals))  if vals else float("nan")

    mae_vals      = [r["mae"]          for r in val_results]
    rmse_vals     = [r["rmse"]         for r in val_results]
    up_mae_vals   = [r["upstream_mae"] for r in val_results]
    uc_mae_vals   = [r["upstream_congestion_mae"] for r in val_results if not np.isnan(r["upstream_congestion_mae"])]
    occ_mae_vals  = [r["origin_cc_mae"] for r in val_results if r["has_origin_cc"]]
    no_occ_count  = sum(1 for r in val_results if not r["has_origin_cc"])
    f1_vals       = [r["f1"] for r in val_results if r["has_congestion"]]
    impact_ae_vals = [abs(r["impact_time_err_min"]) for r in val_results]
    recovery_ae_vals = [abs(r["recovery_time_err_min"]) for r in val_results]
    no_cong_count = sum(1 for r in val_results if not r["has_congestion"])
    if eval_checkpoint_path is not None:
        selection_line = (
            f"Source best epoch: {best_epoch} "
            f"({source_selection.get('criterion', 'selection')}={best_selection_value:.6f})"
            if best_selection_value is not None
            else "Selection: provided checkpoint (source selection value unavailable)"
        )
    elif cfg.checkpoint_selection_metric == "cell_mae":
        selection_line = (
            f"Best epoch: {best_epoch} (validation cell MAE={best_selection_value:.6f} km/h; "
            f"validation weighted loss={best_val_loss:.6f})"
        )
    else:
        selection_line = (
            f"Best epoch: {best_epoch} (validation weighted loss={best_val_loss:.6f})"
        )

    prediction_artifacts: dict[str, dict] = {}
    if cfg.export_val_predictions:
        prediction_artifacts["validation_predictions"] = _export_validation_predictions(
            eval_dir / "predictions_val.npz",
            val_results,
            val_ds,
            cfg,
            target_window,
            cong_by_id,
            selected_checkpoint_path,
        )
        print(f"Validation predictions: {eval_dir / 'predictions_val.npz'}")

    summary = (
        f"Run id        : {cfg.run_id}\n"
        f"Eval protocol : {cfg.eval_protocol or 'legacy'}\n"
        f"Loss          : {cfg.loss_name}\n"
        f"Loss params   : speed_lambda={cfg.speed_loss_lambda}, "
        f"pace_lambda={cfg.pace_loss_lambda}, pace_v_min={cfg.pace_v_min_kmh} km/h, "
        f"full_field_lambda={cfg.full_field_loss_lambda}\n"
        f"Backbone      : {cfg.backbone}\n"
        f"Data scope    : {cfg.data_scope}\n"
        f"Feature set   : {cfg.feature_set}\n"
        f"Val accidents : {len(val_results)}\n"
        f"--- Cell-wise ---\n"
        f"MAE  : mean={_nanmean(mae_vals):.2f} std={_nanstd(mae_vals):.2f}\n"
        f"RMSE : mean={_nanmean(rmse_vals):.2f}\n"
        f"--- Upstream ---\n"
        f"origin_cc_MAE@20 : {_nanmean(occ_mae_vals):.2f} km/h  "
        f"(excl. {no_occ_count}/{len(val_results)} no-origin_cc samples)\n"
        f"upstream_MAE  : {_nanmean(up_mae_vals):.2f} km/h\n"
        f"congestion_F1 : {_nanmean(f1_vals):.3f}  "
        f"(excl. {no_cong_count}/{len(val_results)} no-congestion samples)\n"
        f"impact_time_AE@20 : {_nanmean(impact_ae_vals):.2f} min\n"
        f"recovery_AE@10 : {_nanmean(recovery_ae_vals):.2f} min\n"
        f"Train sec: {train_sec}\n"
        f"{selection_line}\n"
    )
    (eval_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)

    pd.DataFrame([{
        "acc_id": r["acc_id"],
        "mae_kmh": r["mae"],
        "rmse_kmh": r["rmse"],
        "upstream_mae_kmh": r["upstream_mae"],
        "upstream_congestion_mae_kmh": r["upstream_congestion_mae"],
        "origin_cc_mae_kmh": r["origin_cc_mae"],
        "speed60_mae_kmh": r["speed60_mae"],
        "congestion_f1": r["f1"],
        "impact_time_ae_min": abs(r["impact_time_err_min"]),
        "recovery_time_ae_min": abs(r["recovery_time_err_min"]),
        "has_congestion": r["has_congestion"],
        "has_origin_cc": r["has_origin_cc"],
        "n_valid_cells": int(r["valid"].sum()),
    } for r in val_results]).to_csv(eval_dir / "metrics.csv", index=False)

    # Test set evaluation (only if eval_protocol v1 and test set exists)
    test_results: list[dict] = []
    if test_loader:
        test_results = _eval_per_sample(model, test_loader, cfg, device, cong_by_id)
        pd.DataFrame([{
            "acc_id": r["acc_id"], "mae_kmh": r["mae"], "rmse_kmh": r["rmse"],
            "upstream_mae_kmh": r["upstream_mae"], "origin_cc_mae_kmh": r["origin_cc_mae"],
            "upstream_congestion_mae_kmh": r["upstream_congestion_mae"],
            "speed60_mae_kmh": r["speed60_mae"], "congestion_f1": r["f1"],
            "impact_time_ae_min": abs(r["impact_time_err_min"]),
            "recovery_time_ae_min": abs(r["recovery_time_err_min"]),
            "has_congestion": r["has_congestion"], "has_origin_cc": r["has_origin_cc"],
        } for r in test_results]).to_csv(eval_dir / "metrics_test.csv", index=False)
    if cfg.export_test_predictions:
        if not test_results or test_ds is None:
            raise RuntimeError("formal test prediction export has no evaluated test incidents")
        prediction_artifacts["test_predictions"] = _export_test_predictions(
            eval_dir / "predictions_test.npz",
            test_results,
            test_ds,
            cfg,
            target_window,
            cong_by_id,
            selected_checkpoint_path,
            split_info.get("splits", {}).get("test", {}).get("membership_sha256"),
        )
        print(f"Test predictions: {eval_dir / 'predictions_test.npz'}")

    # metrics_v1.json — used for champion selection
    metrics_v1 = {
        "eval_protocol": cfg.eval_protocol,
        "run_id": cfg.run_id,
        "loss_name": cfg.loss_name,
        "checkpoint_selection_metric": cfg.checkpoint_selection_metric,
        "data_scope": cfg.data_scope,
        "feature_set": cfg.feature_set,
        "input_type": cfg.input_type,
        "input_channels": cfg.in_channels,
        "hist_validity_channel": cfg.hist_validity_channel,
        "hist_missing_fill": cfg.hist_missing_fill,
        "coordinate_channels": cfg.coordinate_channels,
        "backbone": cfg.backbone,
        "simple_base_channels": cfg.simple_base_channels,
        "weight_decay": cfg.weight_decay,
        "validation_only": cfg.validation_only,
        "export_val_predictions": cfg.export_val_predictions,
        "export_test_predictions": cfg.export_test_predictions,
        "speed_loss_lambda": cfg.speed_loss_lambda,
        "pace_loss_lambda": cfg.pace_loss_lambda,
        "pace_v_min_kmh": cfg.pace_v_min_kmh,
        "full_field_loss_lambda": cfg.full_field_loss_lambda,
        "loss_design": {
            "support": (
                "upstream_0_20_km_plus_full_field_aux"
                if cfg.loss_name == "upstream_congestion_fullfield_aux"
                else "upstream_0_20_km"
                if cfg.loss_name.startswith("upstream_")
                else "full_field"
            ),
            "base_weight": 1.0,
            "aic_weight": cfg.congestion_weight if cfg.loss_name in {
                "full_field_congestion_weighted",
                "upstream_congestion_weighted",
                "upstream_congestion_fullfield_aux",
                "upstream_aic_speed60_weighted",
            } else None,
            "speed60_weight": (
                cfg.congestion_weight if cfg.loss_name in {
                    "full_field_speed60_weighted",
                    "upstream_speed60_weighted",
                }
                else cfg.secondary_congestion_weight
                if cfg.loss_name == "upstream_aic_speed60_weighted"
                else 1.0
            ),
            "overlap_rule": (
                "origin_cc_precedence"
                if cfg.loss_name == "upstream_aic_speed60_weighted"
                else "origin_cc"
                if cfg.loss_name in {
                    "full_field_congestion_weighted",
                    "upstream_congestion_weighted",
                    "upstream_congestion_fullfield_aux",
                }
                else "speed60_only"
                if cfg.loss_name in {
                    "full_field_speed60_weighted",
                    "upstream_speed60_weighted",
                }
                else "not_applicable"
            ),
            "full_field_aux_lambda": (
                cfg.full_field_loss_lambda
                if cfg.loss_name == "upstream_congestion_fullfield_aux"
                else 0.0
            ),
            "term_normalization": (
                "mean_of_per_incident_valid_cell_mae"
                if cfg.loss_name == "incident_macro_cell_mae"
                else "separate_weighted_upstream_mean_plus_all_valid_cell_mean"
                if cfg.loss_name == "upstream_congestion_fullfield_aux"
                else "single_weighted_mean"
            ),
        },
        "training_seed": cfg.random_state,
        "random_seed": cfg.random_state,
        "split_seed": cfg.split_seed,
        "split_mode": cfg.split_mode if cfg.eval_protocol == "v1" else "legacy",
        "protocol": protocol_info,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_artifacts": checkpoint_artifacts,
        "outputs": prediction_artifacts,
        "val": {
            "n": len(val_results),
            "cell_MAE": _nanmean(mae_vals),
            "cell_RMSE": _nanmean(rmse_vals),
            "upstream_MAE": _nanmean(up_mae_vals),
            "origin_cc_MAE@20": _nanmean(occ_mae_vals),
            "upstream_congestion_MAE": _nanmean(uc_mae_vals),
            "congestion_F1": _nanmean(f1_vals),
            "congestion_F1@20": _nanmean(f1_vals),
            "impact_time_AE@20": _nanmean(impact_ae_vals),
            "recovery_AE@10": _nanmean(recovery_ae_vals),
            "no_congestion_count": no_cong_count,
            "no_origin_cc_count": no_occ_count,
        },
    }
    if test_results:
        t_up  = [r["upstream_mae"] for r in test_results]
        t_uc  = [r["upstream_congestion_mae"] for r in test_results if not np.isnan(r["upstream_congestion_mae"])]
        t_occ = [r["origin_cc_mae"] for r in test_results if r["has_origin_cc"]]
        t_f1  = [r["f1"] for r in test_results if r["has_congestion"]]
        t_impact_ae = [abs(r["impact_time_err_min"]) for r in test_results]
        t_recovery_ae = [abs(r["recovery_time_err_min"]) for r in test_results]
        t_noc = sum(1 for r in test_results if not r["has_congestion"])
        t_noo = sum(1 for r in test_results if not r["has_origin_cc"])
        metrics_v1["test"] = {
            "n": len(test_results),
            "cell_MAE": _nanmean([r["mae"] for r in test_results]),
            "upstream_MAE": _nanmean(t_up),
            "origin_cc_MAE@20": _nanmean(t_occ),
            "upstream_congestion_MAE": _nanmean(t_uc),
            "congestion_F1": _nanmean(t_f1),
            "congestion_F1@20": _nanmean(t_f1),
            "impact_time_AE@20": _nanmean(t_impact_ae),
            "recovery_AE@10": _nanmean(t_recovery_ae),
            "no_congestion_count": t_noc,
            "no_origin_cc_count": t_noo,
        }
    (eval_dir / "metrics_v1.json").write_text(json.dumps(metrics_v1, indent=2), encoding="utf-8")

    if not cfg.smoke:
        sorted_by_mae = sorted(val_results, key=lambda r: r["upstream_mae"] if not np.isnan(r.get("upstream_mae", float("nan"))) else 9999.0)
        n_show = min(4, len(sorted_by_mae))
        plot_predictions(sorted_by_mae[:n_show], plot_dir / "predictions_best4.png", title="Best 4 (lowest upstream MAE)")
        plot_predictions(sorted_by_mae[-n_show:], plot_dir / "predictions_worst4.png", title="Worst 4 (highest upstream MAE)")

    cfg_dict = asdict(cfg)
    cfg_dict.update({
        "input_channels": cfg.in_channels,
        "train_date_range": [
            split_info.get("splits", {}).get("train", {}).get("date_min"),
            split_info.get("splits", {}).get("train", {}).get("date_max"),
        ],
        "val_date_range": [
            split_info.get("splits", {}).get("val", {}).get("date_min"),
            split_info.get("splits", {}).get("val", {}).get("date_max"),
        ],
        "test_date_range": [
            split_info.get("splits", {}).get("test", {}).get("date_min"),
            split_info.get("splits", {}).get("test", {}).get("date_max"),
        ],
        "chronological_reference_cutoffs": {
            "train_end_date": cfg.train_end_date,
            "val_end_date": cfg.val_end_date,
        },
        "training_seed": cfg.random_state,
        "random_seed": cfg.random_state,
        "split_seed": cfg.split_seed,
        "split_mode": cfg.split_mode if cfg.eval_protocol == "v1" else "legacy",
        "keep_ids": args.keep_ids,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "test_n": len(test_ds) if test_ds else 0,
        "protocol": protocol_info,
        "status": "complete",
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_artifacts": checkpoint_artifacts,
        "outputs": prediction_artifacts,
    })
    (output_dir / "run_config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
    print(f"Done. Output: {output_dir}")


def _load_samples_explicit(npz_path: Path):
    """Backward-compatible wrapper around the exact paper-artifact loader."""
    return load_samples_artifact(npz_path)


if __name__ == "__main__":
    main()
