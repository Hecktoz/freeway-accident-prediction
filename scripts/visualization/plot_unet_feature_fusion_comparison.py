# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "pandas",
# ]
# ///
"""Plot the formal chronological comparison of the two U-Net variants."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = PROJECT_ROOT / "reports" / "paper_trb_2026" / "results" / "paper_metrics_summary.csv"
OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "paper_trb_2026"
    / "figures"
    / "unet_feature_fusion"
    / "unet_feature_fusion_metric_comparison.png"
)

MODELS = {
    "unet_matrix:simple": "Matrix U-Net",
    "unet_feature_fusion:residual_attention:attention": "Feature-fusion U-Net",
}
METRICS = [
    ("origin_cc_MAE@20", "Accident-core MAE", "km/h", False),
    ("cell_MAE", "Whole-field MAE", "km/h", False),
    ("impact_time_AE@20", "Impact-time AE", "min", False),
    ("recovery_AE@10", "Recovery-time AE", "min", False),
    ("congestion_F1@20", "Congestion F1", "score", True),
]


def main() -> None:
    frame = pd.read_csv(SUMMARY_CSV)
    frame = frame[
        frame["model"].isin(MODELS)
        & frame["split_mode"].eq("time")
        & frame["objective"].eq("upstream_congestion_weighted")
    ].copy()
    frame = frame.set_index("model").loc[list(MODELS)]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "black",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )
    fig, axes = plt.subplots(1, 5, figsize=(9.2, 2.8))
    colors = ["#777777", "#1f5a85"]
    hatches = ["////", ""]

    for ax, (metric, title, unit, higher_better) in zip(axes, METRICS):
        means = frame[f"{metric}_mean"].astype(float).to_numpy()
        sds = frame[f"{metric}_sd"].astype(float).to_numpy()
        bars = ax.bar(
            [0, 1],
            means,
            yerr=sds,
            capsize=3,
            width=0.62,
            color=colors,
            edgecolor="black",
            linewidth=0.7,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        direction = "higher better" if higher_better else "lower better"
        ax.set_title(f"{title}\n({direction})", pad=5)
        ax.set_ylabel(unit)
        ax.set_xticks([0, 1], ["Matrix", "Fusion"])
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        span = max(float(means.max() - means.min()), float(sds.max()), 0.05)
        top = float((means + sds).max()) + 0.45 * span
        bottom = max(0.0, float((means - sds).min()) - 0.45 * span)
        ax.set_ylim(bottom, top)
        for x, value in enumerate(means):
            label = f"{value:.3f}" if metric == "congestion_F1@20" else f"{value:.2f}"
            ax.text(x, value + sds[x] + 0.08 * span, label, ha="center", va="bottom", fontsize=7.5)

    fig.suptitle(
        "Chronological temporal evaluation (473 incidents; mean and SD across three seeds)",
        fontsize=10,
        y=0.98,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.20, top=0.72, wspace=0.48)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
