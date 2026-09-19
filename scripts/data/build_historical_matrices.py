# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "tqdm",
# ]
# ///
# ruff: noqa: E402
"""
Build per-accident historical-baseline speed matrices aligned to the VD samples.

For each accident in the sample metadata, materialize the event-agnostic
historical baseline (typical non-accident speed for that location / time-of-day /
day-class) over the SAME canonical superset window as
`data/accident_samples_5m.npz` (50 km up + 10 km down, -60..+240 min, 1 km x 5 min).

The output stacks one (n_space, n_time) matrix per accident in the exact order of
the sample metadata, so the model scripts can load it next to the VD NPZ and feed
it as extra "history" input channels (the `vd+hist` input type).

Usage:
  uv run scripts/data/build_historical_matrices.py
  uv run scripts/data/build_historical_matrices.py --tag sms
  uv run scripts/data/build_historical_matrices.py --profile-dataset paper_trb_2026/historical_speed_profile_5m_pre20240401_n1n3 --freeways N1 N3 --output data/paper_trb_2026/historical_matrices_5m_pre20240401_n1n3.npz --fail-on-empty
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.inference import AccidentInput, build_historical_baseline, make_profile_index
from scripts.lib.windows import SampleWindow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-accident historical-baseline matrices aligned to the VD samples NPZ.")
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--resolution-min", type=int, default=5, choices=[1, 5])
    p.add_argument("--tag", default="", help="Suffix matching the VD samples tag (e.g. 'sms'). Empty = dispatch.")
    p.add_argument("--metadata", default=None, help="Override metadata parquet path.")
    p.add_argument("--output", default=None, help="Override output NPZ path.")
    p.add_argument(
        "--profile-dataset",
        default=None,
        help="Historical-profile directory name or path. Relative names resolve under --data-root. Default: historical_speed_profile_<resolution>m.",
    )
    p.add_argument(
        "--freeways",
        nargs="+",
        default=None,
        help="Keep only metadata rows on these freeways, preserving their original order (e.g. N1 N3).",
    )
    p.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="Abort without writing if any selected accident has an all-empty historical matrix.",
    )
    # Superset window must match build_accident_samples defaults so the matrices align.
    p.add_argument("--upstream-km", type=int, default=50)
    p.add_argument("--downstream-km", type=int, default=10)
    p.add_argument("--time-before-min", type=int, default=60)
    p.add_argument("--time-after-min", type=int, default=240)
    p.add_argument("--spatial-resolution-km", type=int, default=1)
    return p.parse_args()


def _resolve_profile_base(data_root: Path, value: str | None, resolution_min: int) -> Path:
    path = Path(value) if value else Path(f"historical_speed_profile_{resolution_min}m")
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


def _read_named_profile(profile_base: Path, freeway: str) -> pd.DataFrame:
    """Read one freeway partition from an explicitly selected profile dataset."""
    freeway_dir = profile_base / f"freeway={freeway}"
    files = sorted(freeway_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No profile parquet files found for freeway={freeway}: {freeway_dir}")
    return ds.dataset(files, format="parquet").to_table().to_pandas()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    suffix = f"_{args.tag}" if args.tag else ""
    res = args.resolution_min
    profile_base = _resolve_profile_base(data_root, args.profile_dataset, res)

    meta_path = Path(args.metadata) if args.metadata else data_root / f"accident_samples_{res}m{suffix}_metadata.parquet"
    out_path = Path(args.output) if args.output else data_root / f"historical_matrices_{res}m{suffix}.npz"
    if not meta_path.exists():
        raise SystemExit(f"metadata parquet not found: {meta_path}")

    meta = pd.read_parquet(meta_path)
    meta["accident_id"] = meta["accident_id"].astype(str)
    meta["report_time"] = pd.to_datetime(meta["report_time"])
    if args.freeways:
        freeways = list(dict.fromkeys(str(fw) for fw in args.freeways))
        before = len(meta)
        meta = meta[meta["freeway"].astype(str).isin(freeways)].copy()
        print(f"Freeway scope: {freeways} | kept {len(meta)}/{before} metadata rows")
    else:
        freeways = None
    if meta.empty:
        raise SystemExit("No accidents remain after applying the requested metadata/freeway scope.")
    print(f"Accidents in metadata: {len(meta)}")
    print(f"Profile dataset: {profile_base}")
    if not profile_base.exists():
        raise SystemExit(f"historical profile dataset not found: {profile_base}")

    window = SampleWindow(
        upstream_km=args.upstream_km,
        downstream_km=args.downstream_km,
        time_before_min=args.time_before_min,
        time_after_min=args.time_after_min,
        spatial_resolution_km=args.spatial_resolution_km,
        resolution_min=res,
    )
    print(f"Window: ({window.n_space} space x {window.n_time} time)")

    raw_profile_cache: dict[str, pd.DataFrame] = {}
    profile_index_cache: dict[tuple[str, str], dict] = {}

    matrices: list[np.ndarray] = []
    ids: list[str] = []
    n_empty = 0
    errors: list[tuple[str, str]] = []
    for row in tqdm(meta.itertuples(index=False), total=len(meta), unit="acc"):
        rec = row._asdict()
        fw = str(rec["freeway"])
        direction = str(rec["direction"])
        accident_id = str(rec["accident_id"])
        try:
            if fw not in raw_profile_cache:
                raw_profile_cache[fw] = _read_named_profile(profile_base, fw)
            key = (fw, direction)
            if key not in profile_index_cache:
                profile_index_cache[key] = make_profile_index(raw_profile_cache[fw], fw, direction)
            accident = AccidentInput(
                report_time=pd.Timestamp(rec["report_time"]).to_pydatetime(),
                freeway=fw,
                direction=direction,
                mileage_km=float(rec["mileage_km"]),
            )
            mat, cov = build_historical_baseline(accident, window, profile_index=profile_index_cache[key])
        except Exception as exc:
            mat = np.full((window.n_space, window.n_time), np.nan, dtype=np.float32)
            cov = {"filled_cells": 0}
            errors.append((accident_id, f"{type(exc).__name__}: {exc}"))
        if int(cov.get("filled_cells", 0)) == 0:
            n_empty += 1
        matrices.append(mat.astype(np.float32))
        ids.append(accident_id)

    if errors:
        print(f"WARNING: {len(errors)} accident(s) raised errors while building the baseline.")
        for accident_id, message in errors[:10]:
            print(f"  {accident_id}: {message}")
        if len(errors) > 10:
            print(f"  ... {len(errors) - 10} more")
    if args.fail_on_empty and n_empty:
        raise SystemExit(
            f"Refusing to write: {n_empty}/{len(meta)} selected accidents have all-empty historical matrices."
        )

    stacked = np.stack(matrices, axis=0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        matrix=stacked,
        accident_ids=np.asarray(ids).astype(str),
        space_offsets_m=window.space_offsets_m.astype(np.int32),
        time_offsets_min=window.time_offsets_min.astype(np.int32),
        window_upstream_km=np.int32(window.upstream_km),
        window_downstream_km=np.int32(window.downstream_km),
        window_time_before_min=np.int32(window.time_before_min),
        window_time_after_min=np.int32(window.time_after_min),
        window_spatial_resolution_km=np.int32(window.spatial_resolution_km),
        window_resolution_min=np.int32(window.resolution_min),
        profile_dataset=np.asarray(str(profile_base)),
        metadata_path=np.asarray(str(meta_path.resolve())),
        freeway_scope=np.asarray(freeways or [], dtype=str),
    )

    coverage = float(np.mean(~np.isnan(stacked)))
    print("\n" + "=" * 64)
    print(f"Output: {out_path}")
    print(f"Matrix: {stacked.shape} | cell coverage: {coverage:.1%} | all-empty accidents: {n_empty}")
    print("=" * 64)


if __name__ == "__main__":
    main()
