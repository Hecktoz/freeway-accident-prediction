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
# ruff: noqa: E402
# NOTE: torch-directml is intentionally NOT a dependency. Its custom torch build
# omits the fused LSTM cell kernel (aten::_thnn_fused_lstm_cell) for BOTH the DML
# and CPU backends, so any nn.LSTM forward fails. Stock CPU torch has the kernel;
# select_device() falls back to CPU when torch_directml is not importable.
"""Baseline 1a — past-30-minute per-cell LSTM (eval_protocol_v1).

Model class
-----------
A deliberately *spatially-blind* recurrent baseline: each (accident, mileage
cell) is one independent sequence. The shared LSTM cannot see neighbouring
cells, so the gap to the U-Net candidates isolates the value of modelling the
2-D speed field jointly (the paper's framing).

Canonical per-(accident, cell) sample:
- input  : exactly six pre-incident 5-minute steps (-30..-5 min) ×
  [observed VD speed, observed validity]
- target : all 30 post-incident steps (0..145 min) at the same cell
- excluded inputs: causal historical profile, future decoder covariates,
  incident features, space offset, and neighbouring cells

The encoder state is projected directly to the full 30-step horizon. Three
full-field masked-L1 objectives are supported while holding architecture,
inputs, and supervision support fixed:
1. ``full_field_mae``: every valid target cell weight 1;
2. ``full_field_speed60_weighted``: observed speed<60 km/h weight 4;
3. ``full_field_aic_weighted``: origin_cc AIC-mask cells weight 4.

Protocol: 3 seeds {42,7,123} × 80 epochs, the same explicit N1/N3 cohort, and
either the chronological primary split or fixed random robustness split.

Usage:
  uv run scripts/baselines/lstm_percell.py --smoke
  uv run scripts/baselines/lstm_percell.py                       # 3 seeds × 80 ep
  uv run scripts/baselines/lstm_percell.py --epochs 80 --seeds 42,7,123
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines._harness import (  # noqa: E402
    CONGESTION_UPSTREAM_KM,
    CONGESTION_WEIGHT,
    DEFAULT_OUTPUT_ROOT,
    PAPER_SEEDS,
    SPEED_MAX,
    BaselineData,
    BaselineResult,
    add_protocol_arguments,
    artifact_manifest,
    congestion_weights,
    configure_reproducibility,
    loss_design_manifest,
    load_baseline_data,
    make_run_id,
    objective_mae_kmh,
    print_seed_table,
    protocol_kwargs_from_args,
    score_predictions,
    write_result,
    write_seed_summary,
)
from scripts.models.unet_matrix import select_device  # noqa: E402

ENC_IN = 4   # legacy frozen-checkpoint compatibility only
DEC_IN = 2   # legacy frozen-checkpoint compatibility only
PAST30_IN = 2  # observed VD speed, observed validity


# ────────────────────────────────────────────────────────────────────────────
# Sample assembly (BaselineData -> per-cell tensors)
# ────────────────────────────────────────────────────────────────────────────


def _cell_arrays(data: BaselineData, indices: np.ndarray, loss_name: str):
    """Flatten accidents×cells into per-cell sequence arrays.

    Returns a dict of stacked arrays. Row r corresponds to (accident k in
    `indices`, cell s) with r = k * n_space + s, so val/test predictions reshape
    back to `(len(indices), n_space, n_after)` directly.
    """
    nb, na, ns = data.n_before, data.n_after, data.n_space
    if nb != 6 or data.target_window.time_before_min != 30:
        raise ValueError(
            f"past-30-minute LSTM requires six 5-minute input steps, got nb={nb}"
        )

    enc = np.zeros((len(indices) * ns, nb, PAST30_IN), dtype=np.float32)
    y = np.zeros((len(indices) * ns, na), dtype=np.float32)
    m = np.zeros((len(indices) * ns, na), dtype=np.float32)
    w = np.zeros((len(indices) * ns, na), dtype=np.float32)

    for k, idx in enumerate(indices):
        idx = int(idx)
        vd = data.matrices[idx]                      # (n_space, n_time) km/h NaN
        vd_norm = np.clip(np.nan_to_num(vd, nan=0.0), 0, SPEED_MAX) / SPEED_MAX
        vd_val = (~np.isnan(vd)).astype(np.float32)

        if loss_name == "full_field_mae":
            wcell = np.ones((ns, na), dtype=np.float32)
        else:
            wcell = congestion_weights(
                data,
                idx,
                outside_weight=1.0,
                loss_name=loss_name,
            )

        base = k * ns
        for s in range(ns):
            r = base + s
            enc[r, :, 0] = vd_norm[s, :nb]
            enc[r, :, 1] = vd_val[s, :nb]
            y[r] = vd_norm[s, nb:]
            m[r] = vd_val[s, nb:]
            w[r] = wcell[s]
    return {"enc": enc, "y": y, "m": m, "w": w}


# ────────────────────────────────────────────────────────────────────────────
# Model
# ───────────────────────────────────────────────────────────────────────────


class _LSTMCell(nn.Module):
    """Hand-rolled LSTM cell (Linear + pointwise ops only).

    Avoids `aten::_thnn_fused_lstm_cell`, which the torch-directml build omits on
    both DML and CPU. Built from `nn.Linear`/sigmoid/tanh so it runs on any
    backend; the per-cell sequences are short (<=30 steps) so the Python loop is
    cheap.
    """

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.x2h = nn.Linear(in_dim, 4 * hidden)
        self.h2h = nn.Linear(hidden, 4 * hidden)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        i, f, g, o = (self.x2h(x) + self.h2h(h)).chunk(4, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class PerCellLSTM(nn.Module):
    """Legacy encoder-decoder architecture for frozen-checkpoint replay only."""

    def __init__(self, n_static: int, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.enc_cell = _LSTMCell(ENC_IN, hidden)
        self.bridge = nn.Sequential(nn.Linear(hidden + n_static, hidden), nn.Tanh())
        self.dec_cell = _LSTMCell(DEC_IN, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, enc: torch.Tensor, dec: torch.Tensor, stat: torch.Tensor) -> torch.Tensor:
        B = enc.shape[0]
        h = enc.new_zeros(B, self.hidden)
        c = enc.new_zeros(B, self.hidden)
        for t in range(enc.shape[1]):                   # encode pre-accident steps
            h, c = self.enc_cell(enc[:, t, :], h, c)
        h = self.bridge(torch.cat([h, stat], dim=1))    # inject static context
        c = torch.zeros_like(h)
        outs = []
        for t in range(dec.shape[1]):                   # decode over post hist profile
            h, c = self.dec_cell(dec[:, t, :], h, c)
            outs.append(self.head(h))                   # (B, 1)
        return torch.cat(outs, dim=1)                   # (B, n_after)


class Past30LSTM(nn.Module):
    """Spatially blind LSTM using only the six pre-incident VD observations."""

    def __init__(self, n_after: int, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.n_after = int(n_after)
        self.enc_cell = _LSTMCell(PAST30_IN, self.hidden)
        self.head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.n_after),
            nn.Sigmoid(),
        )

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        batch = enc.shape[0]
        h = enc.new_zeros(batch, self.hidden)
        c = enc.new_zeros(batch, self.hidden)
        for time_idx in range(enc.shape[1]):
            h, c = self.enc_cell(enc[:, time_idx, :], h, c)
        return self.head(h)


def _weighted_l1(pred: torch.Tensor, y: torch.Tensor, wm: torch.Tensor) -> torch.Tensor:
    return (torch.abs(pred - y) * wm).sum() / wm.sum().clamp(min=1)


# ────────────────────────────────────────────────────────────────────────────
# Train + predict one seed
# ────────────────────────────────────────────────────────────────────────────


def _to_device(d: dict, device, keys) -> dict:
    return {k: torch.from_numpy(d[k]).to(device) for k in keys}


def train_one_seed(
    data: BaselineData,
    train_arr: dict,
    val_arr: dict,
    test_arr: dict | None,
    seed: int,
    epochs: int,
    hidden: int,
    batch_size: int,
    lr: float,
    device,
) -> tuple[Past30LSTM, float, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Keep cells with at least one weighted-valid target. Under the full-field
    # objective this is every observed cell; under the loss-matched sensitivity
    # it is the supervised upstream zone only.
    wm_train = train_arr["w"] * train_arr["m"]
    keep = wm_train.sum(axis=1) > 0
    enc = torch.from_numpy(train_arr["enc"][keep]).to(device)
    y = torch.from_numpy(train_arr["y"][keep]).to(device)
    wm = torch.from_numpy(wm_train[keep]).to(device)
    n = enc.shape[0]

    # Validation uses the same explicit objective support as training.
    v_enc = torch.from_numpy(val_arr["enc"]).to(device)
    v_y = torch.from_numpy(val_arr["y"]).to(device)
    v_wm = torch.from_numpy(val_arr["w"] * val_arr["m"]).to(device)

    model = Past30LSTM(data.n_after, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    best_val = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        tl = 0.0
        nb_batches = 0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(enc[b])
            loss = _weighted_l1(pred, y[b], wm[b])
            loss.backward()
            opt.step()
            tl += loss.item()
            nb_batches += 1
        tl /= max(nb_batches, 1)

        model.eval()
        with torch.no_grad():
            vpred = model(v_enc)
            vl = _weighted_l1(vpred, v_y, v_wm).item()
        if vl < best_val:
            best_val = vl
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        sched.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"    epoch {epoch:03d} | train_wL1={tl:.4f} | val_wL1={vl:.4f} | best={best_val:.4f}")

    model.load_state_dict(best_state)
    return model, best_val, best_epoch


def predict_split(model: Past30LSTM, arr: dict, data: BaselineData, device) -> np.ndarray:
    """Predict a split's full field -> (n_acc, n_space, n_after) km/h."""
    model.eval()
    enc = torch.from_numpy(arr["enc"]).to(device)
    out_list = []
    with torch.no_grad():
        for i in range(0, enc.shape[0], 8192):
            sl = slice(i, i + 8192)
            out_list.append((model(enc[sl]) * SPEED_MAX).cpu().numpy())
    flat = np.concatenate(out_list, axis=0).astype(np.float32)   # (n_acc*n_space, n_after)
    n_acc = flat.shape[0] // data.n_space
    return flat.reshape(n_acc, data.n_space, data.n_after)


# ────────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seeds", default=",".join(str(s) for s in PAPER_SEEDS))
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--loss",
        default="full_field_mae",
        choices=[
            "full_field_mae",
            "full_field_speed60_weighted",
            "full_field_aic_weighted",
        ],
        help=(
            "Hold full-field support fixed: equal valid-cell weights, observed "
            "speed<60 weight 4, or origin_cc AIC-mask weight 4."
        ),
    )
    p.add_argument("--run-id", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--device", default="cpu", choices=["auto", "cpu"],
                   help="Default cpu: the tiny per-cell net trains fast on CPU and avoids "
                        "torch-directml numerical quirks. auto = cuda/directml/cpu via "
                        "select_device (the hand-rolled LSTM cell runs on any backend).")
    p.add_argument("--smoke", action="store_true",
                   help="Quick check: 3 epochs, seed 42 only, subsample train accidents.")
    p.add_argument("--max-train-accidents", type=int, default=0,
                   help="Cap train accidents (0 = all); set by --smoke.")
    add_protocol_arguments(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        epochs, seeds, max_train = 3, [42], (args.max_train_accidents or 120)
    else:
        epochs = args.epochs
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        max_train = args.max_train_accidents

    if args.device == "cpu":
        device, device_msg = torch.device("cpu"), "cpu (forced)"
    else:
        device, device_msg = select_device()
    print(f"Device: {device_msg}")
    run_id = args.run_id or make_run_id("lstm_percell" + ("_smoke" if args.smoke else ""))
    print(f"Run id: {run_id} | loss={args.loss} | seeds={seeds} | epochs={epochs}")

    data = load_baseline_data(
        input_type="vd",
        with_features=False,
        **protocol_kwargs_from_args(args),
    )

    train_idx = data.train_idx
    if max_train and len(train_idx) > max_train:
        train_idx = train_idx[:max_train]
        print(f"[smoke] subsampled train accidents -> {len(train_idx)}")

    print("Assembling per-cell sequences ...")
    t0 = time.time()
    train_arr = _cell_arrays(data, train_idx, args.loss)
    val_arr = _cell_arrays(data, data.val_idx, args.loss)
    test_arr = _cell_arrays(data, data.test_idx, args.loss) if len(data.test_idx) else None
    print(f"  assembled in {time.time() - t0:.1f}s | "
          f"train cells={train_arr['enc'].shape[0]} | input_shape={train_arr['enc'].shape[1:]}")

    summary_rows: list[dict] = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        t_seed = time.time()
        reproducibility = configure_reproducibility(seed, args.deterministic_mode)
        model, best_val, best_epoch = train_one_seed(
            data, train_arr, val_arr, test_arr, seed, epochs,
            args.hidden, args.batch_size, args.lr, device,
        )
        val_pred = predict_split(model, val_arr, data, device)
        val_rows, val_agg = score_predictions(val_pred, data.val_idx, data, "val")
        val_objective_mae = objective_mae_kmh(val_pred, data.val_idx, data, args.loss)
        test_rows, test_agg = [], {}
        if test_arr is not None:
            test_pred = predict_split(model, test_arr, data, device)
            test_rows, test_agg = score_predictions(test_pred, data.test_idx, data, "test")

        seed_run_id = f"{run_id}_s{seed}"
        out_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
        checkpoint_dir = out_root / "lstm_percell" / seed_run_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "best_model.pth"
        torch.save(
            {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "architecture": "past30_direct_multihorizon",
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "best_val_normalized_training_loss": float(best_val),
                "loss_name": args.loss,
                "hidden": int(args.hidden),
                "input_window_min": 30,
                "input_steps": int(data.n_before),
                "input_features": ["vd_speed", "vd_valid"],
                "n_after": int(data.n_after),
            },
            checkpoint_path,
        )
        checkpoint_artifact = artifact_manifest(checkpoint_path)

        mask_name = {
            "full_field_mae": "none",
            "full_field_speed60_weighted": "speed60",
            "full_field_aic_weighted": "origin_cc",
        }[args.loss]
        result = BaselineResult(
            model="lstm_percell", seed=seed, loss_name=args.loss,
            mask=mask_name,
            val_rows=val_rows, val_agg=val_agg, test_rows=test_rows, test_agg=test_agg,
            best_val_loss=val_objective_mae,
            extra_config={
                "hidden": args.hidden, "batch_size": args.batch_size, "lr": args.lr,
                "epochs": epochs, "input_type": "past_30min_vd_only",
                "feature_set": "none",
                "smoke": bool(args.smoke),
                "input_window_min": 30,
                "input_steps": int(data.n_before),
                "input_features": ["vd_speed", "vd_valid"],
                "forecast_steps": int(data.n_after),
                "uses_historical_profile": False,
                "uses_incident_features": False,
                "uses_space_offset": False,
                "uses_neighbouring_cells": False,
                "congestion_weight": CONGESTION_WEIGHT, "congestion_upstream_km": CONGESTION_UPSTREAM_KM,
                "loss_design": loss_design_manifest(args.loss),
                "n_train_accidents": int(len(train_idx)),
                "best_val_loss_unit": "km/h",
                "best_val_normalized_training_loss": float(best_val),
                "checkpoint_selection": {
                    "criterion": "minimum_validation_weighted_loss",
                    "best_epoch": int(best_epoch),
                },
                "model_artifact": checkpoint_artifact,
                "reproducibility": reproducibility,
                "paper_protocol": data.paper_protocol,
            },
        )
        out_dir = write_result(result, seed_run_id, args.output_root)
        summary_rows.append(result.summary_row())
        print(f"  seed {seed} done in {time.time() - t_seed:.1f}s -> {out_dir}")

    print("\nPer-seed panel (val):")
    print_seed_table(summary_rows)
    summary_path = write_seed_summary(summary_rows, "lstm_percell", run_id, args.output_root)
    print(f"\n[summary] {summary_path}")


if __name__ == "__main__":
    main()
