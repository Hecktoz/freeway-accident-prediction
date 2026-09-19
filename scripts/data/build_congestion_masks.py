# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scipy",
#     "pandas",
# ]
# ///
"""Precompute per-accident origin_cc congestion masks aligned to the sample window.

For every accident ID in the selected historical-matrix NPZ, this builds the
eval_protocol_v1 (2026-05-29) ground-truth congestion mask `origin_cc` from the
ID-aligned observed speed matrix and its historical baseline. The output layout
mirrors the historical matrices (same IDs, axes, and window metadata) so the
trainer can crop it to the research window with the same `subset()` operation.

The mask is the connected component(s) of the abs drop-mask that touch the accident
origin anchor zone (`t∈[0,15] min, s∈[0,3] km`); see scripts/lib/congestion_mask.py
and reports/upstream_extent_20260528_summary_tw.md.

Inputs
------
- data/accident_samples_5m.npz       — observed speed (km/h), (N, S, T)
- data/historical_matrices_5m.npz    — same-shape historical baseline

Output
------
- data/congestion_masks_origin_cc_5m.npz  — uint8 mask (N, S, T) + axes/window meta

Usage
-----
  uv run scripts/data/build_congestion_masks.py
  uv run scripts/data/build_congestion_masks.py --tag sms     # SMS-tagged samples
  uv run scripts/data/build_congestion_masks.py --limit 50    # debug
  uv run scripts/data/build_congestion_masks.py --historical-matrices paper_trb_2026/historical_matrices_5m_pre20240401_n1n3.npz --out data/paper_trb_2026/congestion_masks_origin_cc_5m_pre20240401_n1n3.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.lib.congestion_mask import (  # noqa: E402
    ANCHOR_S_KM_MAX_DEFAULT,
    ANCHOR_S_KM_MIN_DEFAULT,
    ANCHOR_T_MAX_DEFAULT,
    ANCHOR_T_MIN_DEFAULT,
    CONG_KMH_DEFAULT,
    DELTA_MIN_DEFAULT,
    PERSIST_N_DEFAULT,
    origin_cc_congestion_mask,
)

_WINDOW_KEYS = (
    "window_upstream_km", "window_downstream_km", "window_time_before_min",
    "window_time_after_min", "window_spatial_resolution_km", "window_resolution_min",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--tag", default="", help="Sample tag suffix (e.g. 'sms'); matches build_accident_samples --tag.")
    p.add_argument(
        "--samples",
        default=None,
        help="Observed sample NPZ path. Relative paths resolve under --data-root. Default: accident_samples_5m[_tag].npz.",
    )
    p.add_argument(
        "--historical-matrices",
        default=None,
        help="Historical-matrix NPZ path. Relative paths resolve under --data-root. Default: historical_matrices_5m[_tag].npz.",
    )
    p.add_argument("--out", default=None, help="Output NPZ path (default: data/congestion_masks_origin_cc_5m[_tag].npz)")
    p.add_argument("--delta-min", type=float, default=DELTA_MIN_DEFAULT)
    p.add_argument("--cong-kmh", type=float, default=CONG_KMH_DEFAULT)
    p.add_argument("--persist-n", type=int, default=PERSIST_N_DEFAULT)
    p.add_argument("--anchor-t-max-min", type=int, default=ANCHOR_T_MAX_DEFAULT)
    p.add_argument("--anchor-s-max-km", type=float, default=ANCHOR_S_KM_MAX_DEFAULT)
    p.add_argument("--limit", type=int, default=None, help="Limit to first N accidents (debug).")
    return p.parse_args()


def _load_npz(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"NPZ not found: {path}")
    with np.load(str(path), allow_pickle=False) as arr:
        return {k: arr[k] for k in arr.files}


def _resolve_data_path(data_root: Path, value: str | None, default_name: str) -> Path:
    path = Path(value) if value else Path(default_name)
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


def _normalise_ids(values: np.ndarray, label: str) -> list[str]:
    ids = [str(v).strip() for v in values]
    if len(ids) != len(set(ids)):
        seen: set[str] = set()
        duplicates: list[str] = []
        for accident_id in ids:
            if accident_id in seen and accident_id not in duplicates:
                duplicates.append(accident_id)
            seen.add(accident_id)
        raise SystemExit(f"{label} contains duplicate accident_ids: {duplicates[:10]}")
    return ids


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    suffix = f"_{args.tag}" if args.tag else ""

    obs_path = _resolve_data_path(data_root, args.samples, f"accident_samples_5m{suffix}.npz")
    base_path = _resolve_data_path(data_root, args.historical_matrices, f"historical_matrices_5m{suffix}.npz")
    obs = _load_npz(obs_path)
    base = _load_npz(base_path)

    if not np.array_equal(obs["space_offsets_m"], base["space_offsets_m"]):
        raise SystemExit("space axes differ between observed and baseline NPZ")
    if not np.array_equal(obs["time_offsets_min"], base["time_offsets_min"]):
        raise SystemExit("time axes differ between observed and baseline NPZ")
    for key in _WINDOW_KEYS:
        if key in obs and key in base and not np.array_equal(obs[key], base[key]):
            raise SystemExit(f"window metadata differs between observed and baseline NPZ: {key}")

    obs_ids = _normalise_ids(obs["accident_ids"], "observed samples")
    base_ids = _normalise_ids(base["accident_ids"], "historical matrices")
    obs_index = {accident_id: i for i, accident_id in enumerate(obs_ids)}
    missing = [accident_id for accident_id in base_ids if accident_id not in obs_index]
    if missing:
        raise SystemExit(
            f"{len(missing)} historical-matrix accident_ids are absent from observed samples: {missing[:10]}"
        )
    obs_positions = np.asarray([obs_index[accident_id] for accident_id in base_ids], dtype=np.int64)
    if obs_ids == base_ids:
        print("Accident alignment: exact ID/order match")
    else:
        print(
            f"Accident alignment: selected/reordered {len(base_ids)} of {len(obs_ids)} observed samples "
            "to match the named historical matrices"
        )

    S = base["space_offsets_m"].astype(np.int32)
    T = base["time_offsets_min"].astype(np.int32)
    observed = obs["matrix"][obs_positions].astype(np.float32)
    baseline = base["matrix"].astype(np.float32)
    if observed.shape != baseline.shape:
        raise SystemExit(f"observed and historical matrix shapes differ after ID alignment: {observed.shape} vs {baseline.shape}")
    n_total = observed.shape[0]
    n = n_total if args.limit is None else min(int(args.limit), n_total)

    masks = np.zeros((n, observed.shape[1], observed.shape[2]), dtype=np.uint8)
    n_nonempty = 0
    for i in range(n):
        m = origin_cc_congestion_mask(
            observed[i], baseline[i], S, T,
            delta_min=args.delta_min, cong_kmh=args.cong_kmh, persist_n=args.persist_n,
            anchor_t_min=ANCHOR_T_MIN_DEFAULT, anchor_t_max=args.anchor_t_max_min,
            anchor_s_km_min=ANCHOR_S_KM_MIN_DEFAULT, anchor_s_km_max=args.anchor_s_max_km,
        )
        if m.any():
            n_nonempty += 1
        masks[i] = m.astype(np.uint8)

    out_path = Path(args.out) if args.out else (data_root / f"congestion_masks_origin_cc_5m{suffix}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "matrix": masks,
        "accident_ids": base["accident_ids"][:n],
        "space_offsets_m": S,
        "time_offsets_min": T,
        "variant": np.asarray("origin_cc"),
        "delta_min": np.asarray(args.delta_min, dtype=np.float32),
        "cong_kmh": np.asarray(args.cong_kmh, dtype=np.float32),
        "persist_n": np.asarray(args.persist_n, dtype=np.int32),
        "anchor_t_max_min": np.asarray(args.anchor_t_max_min, dtype=np.int32),
        "anchor_s_max_km": np.asarray(args.anchor_s_max_km, dtype=np.float32),
        "source_samples": np.asarray(str(obs_path)),
        "source_historical_matrices": np.asarray(str(base_path)),
    }
    for k in _WINDOW_KEYS:
        if k in base:
            payload[k] = base[k]

    np.savez_compressed(str(out_path), **payload)
    pct = 100.0 * n_nonempty / max(n, 1)
    print(f"[npz] {out_path}")
    print(f"  accidents: {n}/{n_total} | non-empty origin_cc masks: {n_nonempty} ({pct:.1f}%) | "
          f"congested-cell share: {masks.mean():.4%}")
    print(f"  params: delta_min={args.delta_min}, cong_kmh={args.cong_kmh}, persist_n={args.persist_n}, "
          f"anchor t<=[{ANCHOR_T_MIN_DEFAULT},{args.anchor_t_max_min}]min, s<=[{ANCHOR_S_KM_MIN_DEFAULT},{args.anchor_s_max_km}]km")


if __name__ == "__main__":
    main()
