# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///
# ruff: noqa: E402
"""
Canonical batch heatmap renderer.  Reads the accident samples NPZ and renders
one PNG per accident using the unified style defined in ``scripts/lib/visualization``.

Examples:
  uv run scripts/visualization/plot_heatmap.py --style report --limit 5
  uv run scripts/visualization/plot_heatmap.py --style slide --accident-id 2024040004
  uv run scripts/visualization/plot_heatmap.py --style notebook --start-date 2024-04-01 --end-date 2024-04-30
  uv run scripts/visualization/plot_heatmap.py --style report --force

Output: output/heatmaps_<style>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import os as _os
if _os.environ.get("MPLBACKEND", "").startswith("module://"):
    _os.environ["MPLBACKEND"] = "Agg"

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

from scripts.lib.data_loaders import load_samples
from scripts.lib.windows import subset
from scripts.lib.visualization import (
    STYLE_MAP,
    HeatmapStyle,
    compare_heatmaps,
    heatmap,
)

DEFAULT_DATA_ROOT   = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
RESEARCH_WINDOW = {
    "upstream_km": 25,
    "downstream_km": 10,
    "time_before_min": 30,
    # 5-minute endpoint convention: -30..+145 has 36 cells and covers a
    # 150-minute post-incident prediction horizon.
    "time_after_min": 145,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _accident_meta(row: pd.Series) -> dict:
    """Build accident_meta dict from a metadata DataFrame row."""

    def _ts(v):
        return pd.Timestamp(v) if pd.notna(v) else None

    return {
        "accident_id":   str(row.get("accident_id", "")),
        "freeway":       str(row.get("freeway", "")),
        "direction":     str(row.get("direction", "")),
        "mileage_km":    float(row["mileage_km"]) if pd.notna(row.get("mileage_km")) else None,
        "report_time":   _ts(row.get("report_time")),
        "arrived_time":  _ts(row.get("arrived_time")),
        "completed_time": _ts(row.get("completed_time")),
    }


def _filter_samples(
    samples,
    accident_id: str | None,
    start_date: str | None,
    end_date: str | None,
    limit: int | None,
) -> list[int]:
    """Return indices into samples.matrix matching the filters."""
    meta = samples.metadata
    idx = list(range(len(samples)))

    if accident_id:
        mask = meta["accident_id"].astype(str) == str(accident_id)
        idx = [i for i in idx if mask.iloc[i]]

    if start_date:
        start = pd.Timestamp(start_date).normalize()
        mask = pd.to_datetime(meta["report_time"], errors="coerce").dt.normalize() >= start
        idx = [i for i in idx if mask.iloc[i]]

    if end_date:
        end = pd.Timestamp(end_date).normalize()
        mask = pd.to_datetime(meta["report_time"], errors="coerce").dt.normalize() <= end
        idx = [i for i in idx if mask.iloc[i]]

    if limit is not None:
        idx = idx[:limit]

    return idx


# ── Main rendering loop ───────────────────────────────────────────────────────
def render(
    style: HeatmapStyle,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    resolution_min: int = 5,
    accident_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
    force: bool = False,
    research_window: bool = False,
) -> None:
    samples = load_samples(data_root=data_root, resolution_min=resolution_min)
    source_window = samples.window
    if research_window:
        _, plot_window = subset(samples.matrix[0], source_window, **RESEARCH_WINDOW)
        space_axis_km = plot_window.space_offsets_m.astype(float) / 1000.0
        time_offsets_min = plot_window.time_offsets_min.astype(float)
    else:
        plot_window = source_window
        space_axis_km = samples.space_offsets_m.astype(float) / 1000.0
        time_offsets_min = samples.time_offsets_min.astype(float)

    indices = _filter_samples(samples, accident_id, start_date, end_date, limit)
    if not indices:
        print("No accidents matched the filters.")
        return

    style_name = next(k for k, v in STYLE_MAP.items() if v == style)
    suffix = "_research" if research_window else ""
    out_dir = output_root / f"heatmaps_{style_name}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(indices)} heatmap(s) → {out_dir}")

    for i in indices:
        row = samples.metadata.iloc[i]
        acc_id = str(row.get("accident_id", i))
        out_path = out_dir / f"{acc_id}.png"

        if not force and out_path.exists():
            print(f"  skip {acc_id} (exists)")
            continue

        matrix = samples.matrix[i].astype(float)  # (n_space, n_time)
        if research_window:
            matrix, _ = subset(matrix, source_window, **RESEARCH_WINDOW)
        meta = _accident_meta(row)

        fig = heatmap(
            matrix,
            space_axis_km,
            time_offsets_min,
            meta,
            style=style,
            resolution_min=resolution_min,
            output_path=out_path,
        )
        plt.close(fig)
        print(f"  {acc_id} → {out_path.name}")

    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render accident heatmaps from the samples NPZ using the canonical style."
    )
    p.add_argument(
        "--style",
        choices=list(STYLE_MAP),
        default="report",
        help="Visual preset: notebook (100 dpi), report (150 dpi), slide (150 dpi, large text)",
    )
    p.add_argument("--accident-id",  default=None, help="Render one accident")
    p.add_argument("--start-date",   default=None, help="Inclusive start date YYYY-MM-DD")
    p.add_argument("--end-date",     default=None, help="Inclusive end date YYYY-MM-DD")
    p.add_argument("--limit",        type=int, default=None, help="Render the first N after filters")
    p.add_argument("--resolution",   type=int, default=5, choices=[1, 5], help="NPZ resolution (minutes)")
    p.add_argument("--data-root",    default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--output-dir",   default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument(
        "--research-window",
        action="store_true",
        help="Render the research/model window: -10..+25 km and -30..+145 min (150-min horizon).",
    )
    p.add_argument("--force",        action="store_true", help="Overwrite existing PNGs")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    render(
        style=STYLE_MAP[args.style],
        data_root=Path(args.data_root),
        output_root=Path(args.output_dir),
        resolution_min=args.resolution,
        accident_id=args.accident_id,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        force=args.force,
        research_window=args.research_window,
    )


if __name__ == "__main__":
    main()
