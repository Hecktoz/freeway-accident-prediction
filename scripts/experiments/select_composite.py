# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "tabulate",
# ]
# ///
"""Composite-score model selection for the loss/window ablation (eval_protocol_v1).

Why
---
The ablation (reports/loss_window_ablation_20260601.md) showed that the lone
primary `origin_cc_MAE@20` is monotone in the congestion weight `w`: cranking `w`
minimises congested-cell speed MAE but inflates the impact/recovery-time error and
cell_MAE — i.e. it is gamed by over-predicting congestion, conflicting with the
"事故影響時間" deliverable. Per the 2026-06 decision, selection switches to a
COMPOSITE score that balances congestion fidelity against overall fidelity and the
impact-time deliverable.

Composite (lower = better), baseline-relative to the `cell_mae` run:
    C = Σ_k weight_k * (metric_k / metric_k^baseline)
with Σ weight_k = 1, so the `cell_mae` baseline scores exactly 1.000 and any cell
with C < 1.0 beats the simple baseline on the chosen preference. Metrics:
    origin_cc_MAE@20 (congestion depth),  cell_MAE (overall heatmap fidelity),
    impact_time_AE@20 (duration),         recovery_AE@10 (near-field recovery).

Three preference schemes are reported so the winner's sensitivity is explicit.

Usage
-----
  uv run scripts/experiments/select_composite.py \
      --summary output/models/loss_window_ablation/<run>/summary.csv
  uv run scripts/experiments/select_composite.py --masks origin_cc   # + cellmae baseline
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# metric key -> (column in summary.csv, pretty label). All "lower is better".
METRICS = {
    "cong": ("val/origin_cc_MAE@20", "origin_cc_MAE@20"),
    "cell": ("val/cell_MAE", "cell_MAE"),
    "impact": ("val/impact_time_AE@20", "impact_AE@20"),
    "recovery": ("val/recovery_AE@10", "recovery_AE@10"),
}

# Preference schemes: weights on [cong, cell, impact, recovery], summing to 1.
SCHEMES = {
    "balanced": {"cong": 0.25, "cell": 0.25, "impact": 0.25, "recovery": 0.25},
    "congestion_priority": {"cong": 0.50, "cell": 0.15, "impact": 0.20, "recovery": 0.15},
    "duration_priority": {"cong": 0.30, "cell": 0.15, "impact": 0.35, "recovery": 0.20},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_summary = sorted(
        (PROJECT_ROOT / "output" / "models" / "loss_window_ablation").glob("*/summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    p.add_argument("--summary", default=str(default_summary[-1]) if default_summary else None,
                   help="ablation summary.csv (default: newest under output/models/loss_window_ablation)")
    p.add_argument("--masks", default="origin_cc",
                   help="comma list of masks to keep as candidates (cellmae baseline always kept)")
    p.add_argument("--report-out", default=None,
                   help="default: reports/loss_window_composite_<YYYYMMDD>.md")
    return p.parse_args()


def seed_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per cell_id, keeping the descriptive columns."""
    metric_cols = [c for c in df.columns if c.startswith(("val/", "test/")) or c == "best_val_loss"]
    keys = ["cell_id", "loss_name", "mask", "loss_window_km", "weight"]
    agg = {c: "mean" for c in metric_cols}
    agg["seed"] = "count"
    out = df.groupby(keys, dropna=False).agg(agg).rename(columns={"seed": "n_seeds"}).reset_index()
    return out


def compute_composite(agg: pd.DataFrame) -> pd.DataFrame:
    base = agg[agg["loss_name"] == "cell_mae"]
    if base.empty:
        raise SystemExit("No cell_mae baseline row found — composite is baseline-relative and needs it.")
    base = base.iloc[0]
    out = agg.copy()
    for k, (col, _) in METRICS.items():
        out[f"ratio_{k}"] = out[col] / float(base[col])
    for scheme, w in SCHEMES.items():
        out[f"C_{scheme}"] = sum(w[k] * out[f"ratio_{k}"] for k in METRICS)
    return out


def _fmt_block(df: pd.DataFrame, scheme: str, lines: list[str]) -> None:
    col = f"C_{scheme}"
    w = SCHEMES[scheme]
    ranked = df.sort_values(col).reset_index(drop=True)
    lines.append(f"### Scheme `{scheme}`  "
                 f"(cong={w['cong']}, cell={w['cell']}, impact={w['impact']}, recovery={w['recovery']})\n")
    show = ranked[[
        "cell_id", "loss_window_km", "weight", "n_seeds",
        "val/origin_cc_MAE@20", "val/cell_MAE", "val/impact_time_AE@20", "val/recovery_AE@10", col,
    ]].rename(columns={
        "val/origin_cc_MAE@20": "cong_MAE", "val/cell_MAE": "cell_MAE",
        "val/impact_time_AE@20": "impact_AE", "val/recovery_AE@10": "recov_AE", col: "composite",
    })
    lines.append(show.round(3).to_markdown(index=False))
    win = ranked.iloc[0]
    lines.append(f"\n**Winner:** `{win['cell_id']}` — composite={win[col]:.3f} "
                 f"(<1.0 beats cell_mae baseline)\n")


def main() -> None:
    args = parse_args()
    if not args.summary:
        raise SystemExit("No summary.csv found; pass --summary explicitly.")
    summary_path = Path(args.summary)
    df = pd.read_csv(summary_path)
    agg = seed_aggregate(df)

    keep_masks = {m.strip() for m in args.masks.split(",") if m.strip()}
    cand = agg[(agg["mask"].isin(keep_masks)) | (agg["loss_name"] == "cell_mae")].copy()
    comp = compute_composite(cand)

    lines: list[str] = []
    lines.append(f"# Composite model selection — {date.today().isoformat()}\n")
    lines.append(f"Source: `{summary_path.as_posix()}` | candidates: masks={sorted(keep_masks)} + cell_mae baseline\n")
    lines.append("Composite = baseline-relative weighted sum of "
                 "`origin_cc_MAE@20` / `cell_MAE` / `impact_AE@20` / `recovery_AE@10` "
                 "(cell_mae baseline ≡ 1.000; lower=better).\n")
    for scheme in SCHEMES:
        _fmt_block(comp, scheme, lines)
        lines.append("")

    # Cross-scheme winners summary
    lines.append("## Winner by scheme\n")
    rows = []
    for scheme in SCHEMES:
        w = comp.sort_values(f"C_{scheme}").iloc[0]
        rows.append({"scheme": scheme, "winner": w["cell_id"],
                     "loss_window_km": w["loss_window_km"], "weight": w["weight"],
                     "composite": round(float(w[f"C_{scheme}"]), 3)})
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    lines.append("")

    report = "\n".join(lines)
    out_path = Path(args.report_out) if args.report_out else (
        PROJECT_ROOT / "reports" / f"loss_window_composite_{date.today():%Y%m%d}.md")
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[md] {out_path}")


if __name__ == "__main__":
    main()
