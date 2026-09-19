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
Pre-training data quality check for accident sample matrices.

Seven filters are evaluated. By default only the overall coverage filter is
active; the behavior matches the current training contract: remove accidents
with `filled_ratio < 0.80` and leave other quality filters as diagnostics.

  F0  low_overall_coverage -- Matrix-level filled_ratio < min_filled_ratio.
                             [active]

  F1  no_low_speed_post   -- No VD reading < speed_threshold in the upstream
                             congestion zone (0–15 km, 0–120 min). Indicates
                             no measurable traffic impact. [diagnostic]

  F2  small_drop_absolute -- Speed drop (pre-mean − post-min) < min_drop_kmh
                             AND relative drop < min_drop_ratio. Pattern is
                             indistinguishable from a non-incident day when
                             evaluated purely within the matrix. [diagnostic]

  F3  pre_congested        -- Median speed in a tight pre-accident probe (±2 km,
                             last 15 min) is already below speed_threshold.
                             Accident causal effect is confounded by existing
                             jam. [optional, off by default]

  F4  overnight            -- Report time between 23:00 and 05:00.
                             Low-volume dynamics differ from daytime.
                             [optional, off by default]

  F5  low_post_coverage    -- Upstream-zone VD coverage < min_zone_coverage
                             in the 0–120 min post-accident window. Speed
                             values come from interpolation, not real sensors.
                             [optional, off by default]

  F6  similar_to_historical -- MAE between actual post-accident speed and the
                             historical p50 baseline at matched (vdid,
                             day_class, minute_bucket) is below
                             hist_mae_threshold. Accident left no detectable
                             fingerprint relative to an ordinary day.
                             [diagnostic]

Matrix axes (default window: up=50 km, down=10 km, −60 to +240 min, 1 km / 5 min):
  Space index 0  = −10 km (downstream)
  Space index 10 = accident location
  Space index 60 = +50 km (upstream)
  Time index  0  = −60 min
  Time index  12 = accident time
  Time index  60 = +240 min

Usage:
  uv run scripts/data/check_accident_quality.py
  uv run scripts/data/check_accident_quality.py --enable-f3 --enable-f4
  uv run scripts/data/check_accident_quality.py --speed-threshold 60 --min-drop-kmh 15 --hist-mae-threshold 8
  uv run scripts/data/check_accident_quality.py --output-csv output/quality/diag.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.calendar import day_class as get_day_class

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_NPZ = DEFAULT_DATA_ROOT / "accident_samples_5m.npz"
DEFAULT_METADATA = DEFAULT_DATA_ROOT / "accident_samples_5m_metadata.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "quality"

# ── matrix geometry ──────────────────────────────────────────────────────────
# Default SampleWindow: upstream=50km, downstream=10km, res=1km/5min → 61×61
ACCIDENT_SPACE_IDX = 10   # space_offset_m == 0 (accident location)
ACCIDENT_TIME_IDX  = 12   # time_offset_min == 0 (accident report time)

# F1/F5/F6: upstream congestion zone, 0–15 km upstream, 0–120 min post
ZONE_S_LO = ACCIDENT_SPACE_IDX           # space index 10 (0 km)
ZONE_S_HI = ACCIDENT_SPACE_IDX + 15      # space index 25 (+15 km upstream)
POST_T_LO  = ACCIDENT_TIME_IDX           # time index 12  (0 min)
POST_T_HI  = ACCIDENT_TIME_IDX + 24      # time index 36  (+120 min)

# F2: full pre-accident baseline, same upstream zone
PRE_T_LO = 0
PRE_T_HI = ACCIDENT_TIME_IDX             # time index 12 (inclusive)

# F3: tight pre-accident probe — ±2 km around accident, last 15 min before
PROBE_S_LO = max(0, ACCIDENT_SPACE_IDX - 2)   # index 8
PROBE_S_HI = ACCIDENT_SPACE_IDX + 2            # index 12
PROBE_T_LO = ACCIDENT_TIME_IDX - 3             # index 9  (−15 min)
PROBE_T_HI = ACCIDENT_TIME_IDX - 1             # index 11 (−5 min)

OVERNIGHT_HOURS = frozenset(range(23, 24)) | frozenset(range(0, 6))


# ── historical profile lookup ─────────────────────────────────────────────────

def load_profile_lookup(data_root: Path) -> tuple[dict, dict]:
    """Load historical speed profiles and build in-memory lookup structures.

    Returns
    -------
    lookup : dict  (vdid, day_class, minute_bucket_int) -> p50_speed_kmh
    vd_by_fd : dict  (freeway, direction) -> (mileage_m_array, vdid_array)
        Both arrays are sorted by mileage_m, enabling np.searchsorted.
    """
    profile_base = data_root / "historical_speed_profile_5m"
    if not profile_base.exists():
        return {}, {}

    cols = ["vdid", "direction", "mileage_m", "day_class", "minute_bucket", "p50_speed_x10"]
    parts: list[pd.DataFrame] = []
    for fw in ("N1", "N3", "N4", "N6"):
        path = profile_base / f"freeway={fw}" / "profile_0.parquet"
        if not path.exists():
            continue
        df = pq.read_table(str(path), columns=cols).to_pandas()
        df["freeway"] = fw
        parts.append(df)

    if not parts:
        return {}, {}

    profile = pd.concat(parts, ignore_index=True)

    keys = list(zip(profile["vdid"], profile["day_class"], profile["minute_bucket"].astype(int)))
    vals = (profile["p50_speed_x10"] / 10.0).tolist()
    lookup = dict(zip(keys, vals))

    vd_locs = (
        profile[["freeway", "direction", "mileage_m", "vdid"]]
        .drop_duplicates(subset=["freeway", "direction", "mileage_m"])
        .copy()
    )
    vd_by_fd: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for (fw, dir_), g in vd_locs.groupby(["freeway", "direction"]):
        g_sorted = g.sort_values("mileage_m")
        vd_by_fd[(fw, dir_)] = (
            g_sorted["mileage_m"].to_numpy(dtype=np.int32),
            g_sorted["vdid"].to_numpy(),
        )

    print(f"  Profile loaded: {len(lookup):,} lookup entries, {len(vd_by_fd)} (freeway, direction) groups")
    return lookup, vd_by_fd


def _nearest_vdid(
    mileage_arr: np.ndarray,
    vdid_arr: np.ndarray,
    target_m: int,
    max_gap_m: int = 3000,
) -> str | None:
    if len(mileage_arr) == 0:
        return None
    idx = int(np.searchsorted(mileage_arr, target_m))
    candidates = []
    if idx < len(mileage_arr):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best_idx = min(candidates, key=lambda i: abs(int(mileage_arr[i]) - target_m))
    if abs(int(mileage_arr[best_idx]) - target_m) > max_gap_m:
        return None
    return str(vdid_arr[best_idx])


def compute_f6_hist_deficit(
    matrix: np.ndarray,
    freeway: str,
    direction: str,
    accident_mileage_m: int,
    report_time: pd.Timestamp,
    lookup: dict,
    vd_by_fd: dict,
) -> float:
    """Return mean speed deficit (km/h) of post-accident speed vs historical p50.

    deficit = mean(historical_p50 − actual_speed) over all paired cells in the
    upstream congestion zone (0–15 km, 0–120 min post-accident).

    Positive value means the accident pushed speed LOWER than the historical
    baseline would predict — i.e., genuine additional congestion.
    Near-zero or negative means no extra slowdown beyond what history expects.

    Returns NaN if fewer than 20 paired cells are found.
    """
    fd_key = (freeway, direction)
    if fd_key not in vd_by_fd:
        return float("nan")

    mileage_arr, vdid_arr = vd_by_fd[fd_key]
    dc = get_day_class(report_time)
    base_min = report_time.hour * 60 + report_time.minute

    deficits: list[float] = []

    for k_km in range(-3, 4):  # ±3 km around accident point (indices 7–13)
        s_idx = ACCIDENT_SPACE_IDX + k_km
        if s_idx < 0 or s_idx >= 61:
            continue
        if direction in ("S", "E"):
            cell_m = accident_mileage_m - k_km * 1000
        else:
            cell_m = accident_mileage_m + k_km * 1000

        vdid = _nearest_vdid(mileage_arr, vdid_arr, cell_m)
        if vdid is None:
            continue

        for step in range(25):  # 0 to 120 min post (25 × 5-min steps)
            t_offset_min = step * 5
            t_idx = ACCIDENT_TIME_IDX + step

            actual = float(matrix[s_idx, t_idx])
            if np.isnan(actual):
                continue

            mb = ((base_min + t_offset_min) // 5 * 5) % 1440
            hist = lookup.get((vdid, dc, mb))
            if hist is None:
                continue

            deficits.append(hist - actual)   # positive = actual slower than history

    if len(deficits) < 20:
        return float("nan")
    return float(np.mean(deficits))


# ── per-accident diagnostics ──────────────────────────────────────────────────

def _region(m: np.ndarray, s_lo: int, s_hi: int, t_lo: int, t_hi: int) -> np.ndarray:
    return m[s_lo : s_hi + 1, t_lo : t_hi + 1]


def compute_diagnostics(matrix: np.ndarray) -> dict:
    post_zone = _region(matrix, ZONE_S_LO,    ZONE_S_HI,    POST_T_LO,   POST_T_HI)
    pre_zone  = _region(matrix, ZONE_S_LO,    ZONE_S_HI,    PRE_T_LO,    PRE_T_HI)
    probe     = _region(matrix, PROBE_S_LO,   PROBE_S_HI,   PROBE_T_LO,  PROBE_T_HI)

    def _safe(fn, arr):
        v = arr[~np.isnan(arr)]
        return float(fn(v)) if len(v) else float("nan")

    post_min    = _safe(np.min,    post_zone)
    post_mean   = _safe(np.mean,   post_zone)
    pre_mean    = _safe(np.mean,   pre_zone)
    probe_med   = _safe(np.median, probe)      # tight pre-accident probe median

    drop_kmh  = (pre_mean - post_min)       if not (np.isnan(pre_mean) or np.isnan(post_min)) else float("nan")
    drop_ratio = (drop_kmh / pre_mean)      if (not np.isnan(drop_kmh) and pre_mean > 0)      else float("nan")

    zone_cells  = post_zone.size
    zone_filled = int(np.count_nonzero(~np.isnan(post_zone)))

    return {
        "pre_mean_kmh":       round(pre_mean,  1),
        "probe_median_kmh":   round(probe_med, 1),   # tight pre-accident probe
        "post_min_kmh":       round(post_min,  1),
        "post_mean_kmh":      round(post_mean, 1),
        "speed_drop_kmh":     round(drop_kmh,  1) if not np.isnan(drop_kmh)   else float("nan"),
        "speed_drop_ratio":   round(drop_ratio, 3) if not np.isnan(drop_ratio) else float("nan"),
        "post_zone_coverage": round(zone_filled / zone_cells, 3) if zone_cells else float("nan"),
    }


def apply_flags(
    diag: dict,
    report_hour: int,
    filled_ratio: float,
    hist_deficit: float,
    speed_threshold: float,
    min_drop_kmh: float,
    min_drop_ratio: float,
    min_zone_coverage: float,
    min_filled_ratio: float,
    hist_deficit_threshold: float,
) -> dict:
    post_min   = diag["post_min_kmh"]
    pre_mean   = diag["pre_mean_kmh"]
    probe_med  = diag["probe_median_kmh"]
    drop_kmh   = diag["speed_drop_kmh"]
    drop_ratio = diag["speed_drop_ratio"]
    zone_cov   = diag["post_zone_coverage"]

    # F1: no speed below threshold in post-accident upstream zone
    f0 = filled_ratio < min_filled_ratio

    # F1: no speed below threshold in post-accident upstream zone
    f1 = np.isnan(post_min) or post_min >= speed_threshold

    # F2: speed drop too small (both absolute and relative)
    f2 = (
        np.isnan(drop_kmh)
        or np.isnan(drop_ratio)
        or (drop_kmh < min_drop_kmh and drop_ratio < min_drop_ratio)
    )

    # F3: tight pre-accident probe already congested
    f3 = (not np.isnan(probe_med)) and (probe_med < speed_threshold)

    # F4: overnight accident
    f4 = report_hour in OVERNIGHT_HOURS

    # F5: low VD coverage in the post-accident upstream zone
    f5 = (not np.isnan(zone_cov)) and (zone_cov < min_zone_coverage)

    # F6: post-accident speed deficit vs historical too small
    # deficit = mean(historical_p50 - actual): positive = actual slower than history
    # remove when deficit < threshold (accident didn't cause extra slowdown vs history)
    f6 = (not np.isnan(hist_deficit)) and (hist_deficit < hist_deficit_threshold)

    return {
        "f0_low_overall_coverage": f0,
        "f1_no_low_speed":    f1,
        "f2_small_drop":      f2,
        "f3_pre_congested":   f3,
        "f4_overnight":       f4,
        "f5_low_cov":         f5,
        "f6_like_historical": f6,
        "hist_deficit_kmh":   round(hist_deficit, 2) if not np.isnan(hist_deficit) else float("nan"),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-training accident sample quality check.")
    p.add_argument("--npz",      default=str(DEFAULT_NPZ))
    p.add_argument("--metadata", default=str(DEFAULT_METADATA))
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--output-csv",      default=None)
    p.add_argument("--output-keep-csv", default=None)

    p.add_argument("--speed-threshold",    type=float, default=60.0,
                   help="km/h: F1 no-low-speed, F3 pre-congestion (default 60)")
    p.add_argument("--min-drop-kmh",       type=float, default=15.0,
                   help="km/h: F2 minimum absolute speed drop (default 15)")
    p.add_argument("--min-drop-ratio",     type=float, default=0.15,
                   help="F2 minimum relative speed drop (default 0.15)")
    p.add_argument("--min-zone-coverage",  type=float, default=0.30,
                   help="F5 minimum post-zone VD coverage (default 0.30)")
    p.add_argument("--min-filled-ratio",   type=float, default=0.80,
                   help="Overall matrix fill threshold (active by default, default 0.80)")
    p.add_argument("--hist-deficit-threshold", type=float, default=8.0,
                   help="km/h: F6 min mean speed deficit vs historical to keep (default 8)")

    # active-filter toggles. Coverage is active by default; others are opt-in diagnostics.
    p.add_argument("--no-filter-coverage", action="store_true")
    p.add_argument("--enable-f1", action="store_true", help="Also filter accidents with no low-speed post zone")
    p.add_argument("--enable-f2", action="store_true", help="Also filter accidents with too-small speed drop")
    p.add_argument("--no-filter-f1", action="store_true")
    p.add_argument("--no-filter-f2", action="store_true")
    p.add_argument("--enable-f3",   action="store_true", help="Add F3 (pre-congestion) to active filters")
    p.add_argument("--enable-f4",   action="store_true", help="Add F4 (overnight) to active filters")
    p.add_argument("--enable-f5",   action="store_true", help="Add F5 (low coverage) to active filters")
    p.add_argument("--enable-f6", action="store_true", help="Also filter accidents similar to historical baseline")
    p.add_argument("--no-filter-f6", action="store_true")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    print(f"Loading NPZ: {args.npz}")
    npz = np.load(args.npz, allow_pickle=False)
    matrices     = npz["matrix"]
    accident_ids = npz["accident_ids"].astype(str)
    print(f"  {len(accident_ids)} accidents, shape {matrices.shape[1:]}")

    print(f"Loading metadata: {args.metadata}")
    meta = pq.read_table(args.metadata).to_pandas().set_index("accident_id")

    print("Loading historical profile …")
    lookup, vd_by_fd = load_profile_lookup(data_root)
    has_profile = bool(lookup)

    rows: list[dict] = []
    for i, aid in enumerate(tqdm(accident_ids, unit="acc", desc="Diagnosing")):
        mat = matrices[i]
        diag = compute_diagnostics(mat)

        row_meta = meta.loc[aid] if aid in meta.index else pd.Series(dtype=object)
        filled_ratio = float(row_meta.get("filled_ratio", 0.0) or 0.0)
        report_time  = row_meta.get("report_time")
        report_ts    = pd.Timestamp(report_time) if pd.notna(report_time) else None
        report_hour  = report_ts.hour if report_ts else -1

        freeway   = str(row_meta.get("freeway", ""))
        direction = str(row_meta.get("direction", ""))
        mileage_m = int(row_meta.get("mileage_m", 0) or 0)

        hist_deficit = float("nan")
        if has_profile and report_ts and freeway and direction and mileage_m:
            hist_deficit = compute_f6_hist_deficit(
                mat, freeway, direction, mileage_m, report_ts, lookup, vd_by_fd
            )

        flags = apply_flags(
            diag,
            report_hour=report_hour,
            filled_ratio=filled_ratio,
            hist_deficit=hist_deficit,
            speed_threshold=args.speed_threshold,
            min_drop_kmh=args.min_drop_kmh,
            min_drop_ratio=args.min_drop_ratio,
            min_zone_coverage=args.min_zone_coverage,
            min_filled_ratio=args.min_filled_ratio,
            hist_deficit_threshold=args.hist_deficit_threshold,
        )

        rows.append({
            "accident_id":  aid,
            "freeway":      freeway,
            "direction":    direction,
            "mileage_km":   row_meta.get("mileage_km"),
            "report_time":  report_time,
            "filled_ratio": filled_ratio,
            **diag,
            **flags,
        })

    df = pd.DataFrame(rows)

    # ── active filter union ───────────────────────────────────────────────────
    active = pd.Series(False, index=df.index)
    if not args.no_filter_coverage:
        active |= df["f0_low_overall_coverage"]
    if args.enable_f1 and not args.no_filter_f1:
        active |= df["f1_no_low_speed"]
    if args.enable_f2 and not args.no_filter_f2:
        active |= df["f2_small_drop"]
    if args.enable_f3:         active |= df["f3_pre_congested"]
    if args.enable_f4:         active |= df["f4_overnight"]
    if args.enable_f5:         active |= df["f5_low_cov"]
    if args.enable_f6 and not args.no_filter_f6 and has_profile:
        active |= df["f6_like_historical"]
    df["filtered_out"] = active

    # ── outputs ───────────────────────────────────────────────────────────────
    out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv  = Path(args.output_csv)      if args.output_csv      else out_dir / "accident_quality.csv"
    keep_csv = Path(args.output_keep_csv) if args.output_keep_csv else out_dir / "accident_keep_ids.csv"

    df.to_csv(out_csv,  index=False, encoding="utf-8-sig", float_format="%.3f")
    keep_ids = df.loc[~df["filtered_out"], "accident_id"]
    keep_ids.to_csv(keep_csv, index=False, encoding="utf-8")
    print(f"\nDiagnostics: {out_csv}")
    print(f"Keep-list:   {keep_csv}  ({len(keep_ids)}/{len(df)} kept)")

    # ── console report ────────────────────────────────────────────────────────
    n   = len(df)
    sep = "=" * 72

    def _pct(k): return f"{df[k].sum():>4}  ({df[k].sum()/n:.1%})"

    print(f"\n{sep}")
    print("ACCIDENT QUALITY REPORT")
    print(sep)
    print(f"Total:  {n}")
    print()

    print("── Filter summary ──────────────────────────────────────────────────")
    print(f"  F0  overall filled_ratio < {args.min_filled_ratio:.0%}:              {_pct('f0_low_overall_coverage')} [active]")
    print(f"  F1  no speed < {args.speed_threshold:.0f} km/h (0–15km, 0–120min):  {_pct('f1_no_low_speed')}")
    print(f"  F2  drop < {args.min_drop_kmh:.0f} km/h AND < {args.min_drop_ratio:.0%} (within-matrix): {_pct('f2_small_drop')}")
    print(f"  F3  pre-congested tight probe (median < {args.speed_threshold:.0f}):  {_pct('f3_pre_congested')}")
    print(f"  F4  overnight (23–05):                         {_pct('f4_overnight')}")
    print(f"  F5  low post-zone VD coverage (< {args.min_zone_coverage:.0%}):       {_pct('f5_low_cov')}")
    if has_profile:
        print(f"  F6  deficit vs history < {args.hist_deficit_threshold:.0f} km/h (no extra slowdown): {_pct('f6_like_historical')}")
    else:
        print("  F6  [skipped — historical profile not found]")
    print()

    n_out = df["filtered_out"].sum()
    print(f"  Active filter union:  removed {n_out} ({n_out/n:.1%})  →  {n-n_out} kept ({(n-n_out)/n:.1%})")
    print()

    print("── Speed drop distribution (pre_mean − post_min) ──────────────────")
    drop = df["speed_drop_kmh"].dropna()
    for pct in (0, 10, 25, 50, 75, 90, 100):
        print(f"  p{pct:3d}: {np.percentile(drop, pct):>6.1f} km/h")
    print()

    print("── Post-zone minimum speed distribution ────────────────────────────")
    pmin = df["post_min_kmh"].dropna()
    for pct in (0, 10, 25, 50, 75, 90, 100):
        print(f"  p{pct:3d}: {np.percentile(pmin, pct):>6.1f} km/h")
    print()

    if has_profile:
        print("── F6 speed deficit vs historical distribution ──────────────────────")
        def_s = df["hist_deficit_kmh"].dropna()
        for pct in (0, 10, 25, 50, 75, 90, 100):
            print(f"  p{pct:3d}: {np.percentile(def_s, pct):>6.1f} km/h")
        print()

    print("── Pairwise overlap (count of accidents flagged by both) ────────────")
    flags_cols = ["f0_low_overall_coverage", "f1_no_low_speed", "f2_small_drop", "f3_pre_congested",
                  "f4_overnight", "f5_low_cov", "f6_like_historical"]
    labels = ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]
    header = "      " + "  ".join(f"{l:>4}" for l in labels)
    print(header)
    for i, (ci, li) in enumerate(zip(flags_cols, labels)):
        row_str = f"  {li}  "
        for j, (cj, lj) in enumerate(zip(flags_cols, labels)):
            if j < i:
                row_str += "      "
            else:
                both = (df[ci] & df[cj]).sum()
                row_str += f"  {both:>4}"
        print(row_str)
    print()

    print("── Filtered-out breakdown by freeway ───────────────────────────────")
    out_df = df[df["filtered_out"]]
    if len(out_df):
        print(out_df.groupby("freeway").size().rename("removed").to_string())
    print(sep)

    # Save per-flag counts for report generation
    return df


if __name__ == "__main__":
    main()
