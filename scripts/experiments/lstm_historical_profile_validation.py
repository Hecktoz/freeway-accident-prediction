# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "matplotlib",
#     "scikit-learn",
#     "tqdm",
# ]
# ///
"""Validation-only exploratory LSTM with causal historical-speed context.

This script answers one narrow sensitivity question without changing the
paper's frozen LSTM benchmark: what happens when the same spatially blind
per-cell LSTM is given the causal historical-speed profile at that location?

The comparison preserves the locked N1/N3 cohort, chronological split,
ACR-weighted x4 objective, hidden width, optimizer, schedule, epoch budget,
batch size, and validation checkpoint rule. It changes only the input/head
needed to expose the historical trajectory:

- encoder: six pre-report steps x [VD speed, VD validity, historical speed];
- forecast covariate: the 30-step post-report historical-speed trajectory;
- output: a direct 30-step forecast from the final encoder state plus that
  deterministic covariate.

The historical artifact predates the study period and missing values are
nearest-filled in space within each time column, matching the current U-Net.
The script is permanently firewalled from test tensor construction, inference,
prediction export, and metric calculation.

Examples
--------
Smoke gate:
  uv run scripts/experiments/lstm_historical_profile_validation.py \
    --validation-only --smoke \
    --run-id smoke_histcov_20260729 \
    --output-root output/models/exploratory/lstm_historical_profile_20260729/smoke

Declared three-seed validation run:
  uv run scripts/experiments/lstm_historical_profile_validation.py \
    --validation-only --epochs 80 --seeds 42,7,123 \
    --run-id histcov_aicw4_20260729 \
    --output-root output/models/exploratory/lstm_historical_profile_20260729/full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines._harness import (
    CONGESTION_UPSTREAM_KM,
    CONGESTION_WEIGHT,
    SPEED_MAX,
    BaselineData,
    BaselineResult,
    add_protocol_arguments,
    artifact_manifest,
    configure_reproducibility,
    congestion_weights,
    load_baseline_data,
    loss_design_manifest,
    objective_mae_kmh,
    print_seed_table,
    protocol_kwargs_from_args,
    score_predictions,
    write_result,
    write_seed_summary,
)
from scripts.baselines.lstm_percell import _LSTMCell, _weighted_l1

MODEL_NAME = "lstm_hist_profile_exploratory"
LOSS_NAME = "full_field_aic_weighted"
EXPECTED_VAL_HASH = "1d63e2cbb45acc7d498082400734c06788e32b7f418d2545d02186d8255433f8"
DECLARED_SEEDS = (42, 7, 123)
INPUT_FEATURES = (
    "observed_vd_speed_pre_report",
    "observed_vd_validity_pre_report",
    "causal_historical_speed_pre_report",
)


def _nearest_space_fill(matrix: np.ndarray) -> np.ndarray:
    """Match the U-Net's causal within-time historical-profile fill."""
    out = np.asarray(matrix, dtype=np.float32).copy()
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


def _cell_arrays(
    data: BaselineData,
    indices: np.ndarray,
    loss_name: str = LOSS_NAME,
) -> dict[str, np.ndarray]:
    """Build only an explicitly supplied split; callers never pass test indices."""
    if data.hist is None:
        raise ValueError("historical-profile LSTM requires data.hist")
    nb, na, ns = data.n_before, data.n_after, data.n_space
    if nb != 6 or na != 30:
        raise ValueError(f"expected 6 pre-report and 30 post-report steps, got {nb}, {na}")

    n_rows = len(indices) * ns
    enc = np.zeros((n_rows, nb, len(INPUT_FEATURES)), dtype=np.float32)
    hist_future = np.zeros((n_rows, na), dtype=np.float32)
    y = np.zeros((n_rows, na), dtype=np.float32)
    m = np.zeros((n_rows, na), dtype=np.float32)
    w = np.zeros((n_rows, na), dtype=np.float32)

    for accident_pos, idx_raw in enumerate(indices):
        idx = int(idx_raw)
        observed = data.matrices[idx]
        observed_norm = (
            np.clip(np.nan_to_num(observed, nan=0.0), 0.0, SPEED_MAX) / SPEED_MAX
        ).astype(np.float32)
        observed_valid = np.isfinite(observed).astype(np.float32)
        historical = _nearest_space_fill(data.hist[idx])
        historical_norm = (np.clip(historical, 0.0, SPEED_MAX) / SPEED_MAX).astype(
            np.float32
        )

        weights = congestion_weights(
            data,
            idx,
            outside_weight=1.0,
            loss_name=loss_name,
        )
        row_start = accident_pos * ns
        row_stop = row_start + ns
        enc[row_start:row_stop, :, 0] = observed_norm[:, :nb]
        enc[row_start:row_stop, :, 1] = observed_valid[:, :nb]
        enc[row_start:row_stop, :, 2] = historical_norm[:, :nb]
        hist_future[row_start:row_stop] = historical_norm[:, nb:]
        y[row_start:row_stop] = observed_norm[:, nb:]
        m[row_start:row_stop] = observed_valid[:, nb:]
        w[row_start:row_stop] = weights

    return {"enc": enc, "hist_future": hist_future, "y": y, "m": m, "w": w}


class HistoricalProfileLSTM(nn.Module):
    """Past-30-minute per-cell LSTM conditioned on the causal history trajectory."""

    def __init__(self, n_after: int = 30, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.n_after = int(n_after)
        self.enc_cell = _LSTMCell(len(INPUT_FEATURES), self.hidden)
        self.head = nn.Sequential(
            nn.Linear(self.hidden + self.n_after, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.n_after),
            nn.Sigmoid(),
        )

    def forward(
        self,
        enc: torch.Tensor,
        historical_future: torch.Tensor,
    ) -> torch.Tensor:
        batch = enc.shape[0]
        h = enc.new_zeros(batch, self.hidden)
        c = enc.new_zeros(batch, self.hidden)
        for time_idx in range(enc.shape[1]):
            h, c = self.enc_cell(enc[:, time_idx, :], h, c)
        return self.head(torch.cat([h, historical_future], dim=1))


def _predict_split(
    model: HistoricalProfileLSTM,
    arrays: dict[str, np.ndarray],
    data: BaselineData,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(arrays["enc"]), 8192):
            stop = start + 8192
            enc = torch.from_numpy(arrays["enc"][start:stop]).to(device)
            hist_future = torch.from_numpy(
                arrays["hist_future"][start:stop]
            ).to(device)
            output = model(enc, hist_future)
            predictions.append((output * SPEED_MAX).cpu().numpy())
    flat = np.concatenate(predictions, axis=0).astype(np.float32)
    if flat.shape[0] % data.n_space:
        raise ValueError("flat prediction rows do not divide into accidents")
    return flat.reshape(flat.shape[0] // data.n_space, data.n_space, data.n_after)


def _train_one_seed(
    data: BaselineData,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    *,
    seed: int,
    epochs: int,
    hidden: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[HistoricalProfileLSTM, float, int, list[dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_weighted_mask = train_arrays["w"] * train_arrays["m"]
    keep = train_weighted_mask.sum(axis=1) > 0
    enc = torch.from_numpy(train_arrays["enc"][keep]).to(device)
    hist_future = torch.from_numpy(train_arrays["hist_future"][keep]).to(device)
    target = torch.from_numpy(train_arrays["y"][keep]).to(device)
    weighted_mask = torch.from_numpy(train_weighted_mask[keep]).to(device)

    val_enc = torch.from_numpy(val_arrays["enc"]).to(device)
    val_hist_future = torch.from_numpy(val_arrays["hist_future"]).to(device)
    val_target = torch.from_numpy(val_arrays["y"]).to(device)
    val_weighted_mask = torch.from_numpy(
        val_arrays["w"] * val_arrays["m"]
    ).to(device)

    model = HistoricalProfileLSTM(n_after=data.n_after, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    best_val = float("inf")
    best_epoch = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(enc), generator=generator).to(device)
        train_loss_sum = 0.0
        batch_count = 0
        for start in range(0, len(enc), batch_size):
            batch = permutation[start : start + batch_size]
            optimizer.zero_grad()
            prediction = model(enc[batch], hist_future[batch])
            loss = _weighted_l1(
                prediction,
                target[batch],
                weighted_mask[batch],
            )
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            batch_count += 1
        train_loss = train_loss_sum / max(batch_count, 1)

        model.eval()
        with torch.no_grad():
            val_prediction = model(val_enc, val_hist_future)
            val_loss = float(
                _weighted_l1(
                    val_prediction,
                    val_target,
                    val_weighted_mask,
                ).item()
            )
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise FloatingPointError(
                f"non-finite loss at epoch {epoch}: train={train_loss}, val={val_loss}"
            )
        history.append(
            {
                "epoch": int(epoch),
                "train_normalized_weighted_l1": train_loss,
                "val_normalized_weighted_l1": val_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"    epoch {epoch:03d} | train_wL1={train_loss:.5f} | "
                f"val_wL1={val_loss:.5f} | best={best_val:.5f}"
            )

    model.load_state_dict(best_state)
    return model, best_val, best_epoch, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Required safety acknowledgement; the script has no test path.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DECLARED_SEEDS),
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-accidents", type=int, default=0)
    add_protocol_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.validation_only:
        raise SystemExit(
            "--validation-only is required; this exploratory script must not evaluate test"
        )
    if args.smoke:
        epochs = 3
        seeds = [42]
        max_train_accidents = args.max_train_accidents or 120
    else:
        epochs = int(args.epochs)
        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
        max_train_accidents = int(args.max_train_accidents)
        if tuple(seeds) != DECLARED_SEEDS:
            raise SystemExit(
                f"full run seeds must match the predeclared order {DECLARED_SEEDS}, got {seeds}"
            )

    device = torch.device("cpu")
    print(
        f"Run={args.run_id} | validation_only=True | seeds={seeds} | "
        f"epochs={epochs} | device=cpu"
    )
    data = load_baseline_data(
        input_type="vd+hist",
        with_features=False,
        **protocol_kwargs_from_args(args),
    )
    split_manifest = data.paper_protocol["split"]["splits"]
    actual_val_hash = split_manifest["val"]["membership_sha256"]
    if actual_val_hash != EXPECTED_VAL_HASH:
        raise RuntimeError(
            f"validation membership changed: {actual_val_hash} != {EXPECTED_VAL_HASH}"
        )
    if len(data.train_idx) != 1028 or len(data.val_idx) != 324 or len(data.test_idx) != 473:
        raise RuntimeError(
            "split counts changed: "
            f"train={len(data.train_idx)}, val={len(data.val_idx)}, test={len(data.test_idx)}"
        )

    train_indices = data.train_idx
    if max_train_accidents and len(train_indices) > max_train_accidents:
        train_indices = train_indices[:max_train_accidents]
        print(f"[smoke/subset] train accidents={len(train_indices)}")

    print("Assembling train and validation arrays only ...")
    assembly_start = time.time()
    train_arrays = _cell_arrays(data, train_indices)
    val_arrays = _cell_arrays(data, data.val_idx)
    print(
        f"  assembled in {time.time() - assembly_start:.1f}s | "
        f"train_cells={len(train_arrays['enc'])} | val_cells={len(val_arrays['enc'])}"
    )

    output_root = Path(args.output_root)
    summary_rows: list[dict] = []
    for seed in seeds:
        print(f"\n=== exploratory seed {seed} ===")
        seed_start = time.time()
        reproducibility = configure_reproducibility(seed, args.deterministic_mode)
        model, best_val, best_epoch, history = _train_one_seed(
            data,
            train_arrays,
            val_arrays,
            seed=seed,
            epochs=epochs,
            hidden=args.hidden,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
        )
        val_prediction = _predict_split(model, val_arrays, data, device)
        expected_shape = (len(data.val_idx), data.n_space, data.n_after)
        if val_prediction.shape != expected_shape or not np.isfinite(val_prediction).all():
            raise RuntimeError(
                f"invalid validation prediction: {val_prediction.shape}, "
                f"finite={np.isfinite(val_prediction).all()}"
            )
        val_rows, val_aggregate = score_predictions(
            val_prediction,
            data.val_idx,
            data,
            "val",
        )
        val_objective_mae = objective_mae_kmh(
            val_prediction,
            data.val_idx,
            data,
            LOSS_NAME,
        )

        seed_run_id = f"{args.run_id}_s{seed}"
        seed_dir = output_root / MODEL_NAME / seed_run_id
        checkpoint_dir = seed_dir / "checkpoints"
        eval_dir = seed_dir / "eval"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        checkpoint_path = checkpoint_dir / "best_model.pth"
        torch.save(
            {
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "architecture": "past30_direct_multihorizon_with_historical_covariate",
                "paper_status": "exploratory_only",
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "best_val_normalized_training_loss": float(best_val),
                "loss_name": LOSS_NAME,
                "hidden": int(args.hidden),
                "input_features": list(INPUT_FEATURES),
                "forecast_covariates": [
                    "causal_historical_speed_post_report_steps_0_to_145_min"
                ],
                "historical_gap_handling": "nearest_space_within_time_column",
                "n_after": int(data.n_after),
                "parameter_count": int(parameter_count),
            },
            checkpoint_path,
        )
        checkpoint_manifest = artifact_manifest(checkpoint_path)
        pd.DataFrame(history).to_csv(seed_dir / "training_history.csv", index=False)
        np.savez_compressed(
            eval_dir / "predictions_val.npz",
            schema_version=np.asarray("exploratory_lstm_hist_profile_val_v1"),
            paper_status=np.asarray("exploratory_only"),
            split=np.asarray("val"),
            accident_ids=np.asarray(
                [str(data.ids[int(index)]) for index in data.val_idx],
                dtype=str,
            ),
            prediction_kmh=val_prediction.astype(np.float32),
            validation_membership_sha256=np.asarray(actual_val_hash),
            training_seed=np.asarray(seed, dtype=np.int64),
            test_inference_performed=np.asarray(False),
        )

        result = BaselineResult(
            model=MODEL_NAME,
            seed=seed,
            loss_name=LOSS_NAME,
            mask="origin_cc",
            val_rows=val_rows,
            val_agg=val_aggregate,
            test_rows=[],
            test_agg={},
            best_val_loss=val_objective_mae,
            extra_config={
                "paper_status": "exploratory_only_not_for_model_selection",
                "hidden": int(args.hidden),
                "parameter_count": int(parameter_count),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "epochs": int(epochs),
                "input_type": "past_30min_vd_valid_plus_causal_historical_profile",
                "input_features": list(INPUT_FEATURES),
                "forecast_covariates": [
                    "causal_historical_speed_post_report_steps_0_to_145_min"
                ],
                "historical_gap_handling": "nearest_space_within_time_column",
                "uses_historical_profile": True,
                "uses_incident_features": False,
                "uses_space_offset": False,
                "uses_neighbouring_cells": False,
                "smoke": bool(args.smoke),
                "n_train_accidents": len(train_indices),
                "n_validation_accidents": len(data.val_idx),
                "congestion_weight": float(CONGESTION_WEIGHT),
                "congestion_upstream_km": int(CONGESTION_UPSTREAM_KM),
                "loss_design": loss_design_manifest(LOSS_NAME),
                "best_val_loss_unit": "km/h",
                "best_val_normalized_training_loss": float(best_val),
                "checkpoint_selection": {
                    "criterion": "minimum_validation_weighted_loss",
                    "best_epoch": int(best_epoch),
                },
                "model_artifact": checkpoint_manifest,
                "runner_artifact": artifact_manifest(Path(__file__)),
                "reproducibility": reproducibility,
                "test_firewall": {
                    "validation_only": True,
                    "shared_source_artifact_contains_test_rows": True,
                    "test_split_membership_recorded_in_protocol": True,
                    "test_arrays_constructed": False,
                    "test_inference_performed": False,
                    "test_predictions_written": False,
                    "test_metrics_computed": False,
                },
                "paper_protocol": data.paper_protocol,
            },
        )
        written_dir = write_result(result, seed_run_id, output_root)
        if written_dir != seed_dir:
            raise RuntimeError(f"unexpected output path: {written_dir} != {seed_dir}")
        summary_rows.append(result.summary_row())
        print(
            f"  seed {seed} done in {time.time() - seed_start:.1f}s | "
            f"best_epoch={best_epoch} | val_cell_MAE="
            f"{val_aggregate['val/cell_MAE']:.4f} km/h | "
            f"test_inference=False"
        )

    print("\nExploratory validation panel:")
    print_seed_table(summary_rows)
    summary_path = write_seed_summary(
        summary_rows,
        MODEL_NAME,
        args.run_id,
        output_root,
    )
    run_manifest = {
        "schema_version": "exploratory_lstm_historical_profile_run_v1",
        "paper_status": "exploratory_only_not_for_model_selection_or_formal_claims",
        "validation_only": True,
        "smoke": bool(args.smoke),
        "seeds": seeds,
        "epochs": epochs,
        "validation_membership_sha256": actual_val_hash,
        "test_firewall": {
            "test_arrays_constructed": False,
            "test_inference_performed": False,
            "test_predictions_written": False,
            "test_metrics_computed": False,
        },
        "summary": artifact_manifest(summary_path),
    }
    manifest_path = summary_path.parent / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[summary] {summary_path}")
    print("[firewall] no test arrays, inference, predictions, or metrics were produced")


if __name__ == "__main__":
    main()
