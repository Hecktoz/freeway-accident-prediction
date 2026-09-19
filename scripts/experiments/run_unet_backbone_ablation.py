# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Long-run UNet backbone ablation under the locked origin_cc@20 protocol.

This orchestrates full training runs instead of re-implementing the trainers.
Every run uses the same current modeling contract:

- feature_set=t0
- input_type=vd+hist by default
- loss=upstream_congestion_weighted
- congestion_upstream_km=20
- congestion_weight=4
- congestion mask=data/congestion_masks_origin_cc_5m.npz

Outputs are written by the underlying trainers plus a compact experiment report:

  output/models/unet_backbone_ablation/<run_ts>/
    commands.ps1
    summary.json
    report.md

Usage:

  uv run scripts/experiments/run_unet_backbone_ablation.py --dry-run
  uv run scripts/experiments/run_unet_backbone_ablation.py --epochs 80
  uv run scripts/experiments/run_unet_backbone_ablation.py --epochs 100 --with-tiid
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "models" / "unet_backbone_ablation"
DEFAULT_MASK_NPZ = PROJECT_ROOT / "data" / "congestion_masks_origin_cc_5m.npz"


@dataclass
class RunSpec:
    arch: str
    run_id: str
    command: list[str]
    metrics_paths: list[Path]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-ts", default=None, help="Run timestamp. Default: UTC now.")
    p.add_argument("--epochs", type=int, default=80, help="Long-run epochs for each training command.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--input-type", default="vd+hist", choices=["vd", "hist", "vd+hist"])
    p.add_argument("--backbones", default="simple,residual_attention")
    p.add_argument("--feature-methods", default="direct_concat,attention")
    p.add_argument("--mask-npz", default=str(DEFAULT_MASK_NPZ))
    p.add_argument("--hist-tag", default="")
    p.add_argument("--skip-matrix", action="store_true", help="Only run feature-fusion models.")
    p.add_argument("--skip-feature-fusion", action="store_true", help="Only run matrix models.")
    p.add_argument("--with-tiid", action="store_true", help="Also train one TIID head per feature-fusion backbone.")
    p.add_argument("--tiid-method", default="attention")
    p.add_argument("--smoke", action="store_true", help="Pass --smoke to trainers for a quick end-to-end check.")
    p.add_argument("--dry-run", action="store_true", help="Write commands/report without launching training.")
    return p.parse_args()


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _common_train_args(args: argparse.Namespace) -> list[str]:
    out = [
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--eval-protocol", "v1",
        "--data-scope", "dispatch_only",
        "--feature-set", "t0",
        "--input-type", args.input_type,
        "--loss", "upstream_congestion_weighted",
        "--congestion-upstream-km", "20",
        "--congestion-weight", "4",
        "--congestion-mask-npz", args.mask_npz,
    ]
    if args.hist_tag:
        out += ["--hist-tag", args.hist_tag]
    if args.smoke:
        out.append("--smoke")
    return out


def build_specs(args: argparse.Namespace, run_ts: str) -> list[RunSpec]:
    specs: list[RunSpec] = []
    backbones = _csv(args.backbones)
    methods = _csv(args.feature_methods)

    for backbone in backbones:
        if not args.skip_matrix:
            run_id = f"{run_ts}_matrix_{backbone}"
            cmd = [
                "uv", "run", "scripts/models/unet_matrix.py",
                "--run-id", run_id,
                "--backbone", backbone,
                *_common_train_args(args),
            ]
            metrics = [PROJECT_ROOT / "output" / "models" / "unet_matrix" / run_id / "eval" / "metrics_v1.json"]
            specs.append(RunSpec("unet_matrix", run_id, cmd, metrics))

        if not args.skip_feature_fusion:
            run_id = f"{run_ts}_ff_{backbone}"
            cmd = [
                "uv", "run", "scripts/models/unet_feature_fusion.py",
                "--run-id", run_id,
                "--backbone", backbone,
                "--methods", ",".join(methods),
                *_common_train_args(args),
            ]
            if args.with_tiid:
                cmd += ["--with-tiid", "--tiid-method", args.tiid_method]
            experiments = list(methods)
            if args.with_tiid:
                experiments.append(f"{args.tiid_method}_tiid")
            metrics = [
                PROJECT_ROOT / "output" / "models" / "unet_feature_fusion" / run_id / exp / "eval" / "metrics_v1.json"
                for exp in experiments
            ]
            specs.append(RunSpec("unet_feature_fusion", run_id, cmd, metrics))

    return specs


def _quote_ps(parts: list[str]) -> str:
    quoted = []
    for p in parts:
        if any(ch.isspace() for ch in p) or any(ch in p for ch in "&()[]{}^=;!,+'`"):
            quoted.append("'" + p.replace("'", "''") + "'")
        else:
            quoted.append(p)
    return " ".join(quoted)


def load_metric(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def flatten_metric(path: Path, metric: dict | None) -> dict:
    if metric is None:
        return {"metrics_path": str(path.relative_to(PROJECT_ROOT)), "status": "missing"}
    val = metric.get("val", {})
    test = metric.get("test", {})
    return {
        "metrics_path": str(path.relative_to(PROJECT_ROOT)),
        "status": "ok",
        "arch": "unet_feature_fusion" if "experiment" in metric else "unet_matrix",
        "run_id": metric.get("run_id"),
        "experiment": metric.get("experiment", ""),
        "backbone": metric.get("backbone", "simple"),
        "input_type": metric.get("input_type", ""),
        "val_origin_cc_MAE@20": val.get("origin_cc_MAE@20", val.get("upstream_congestion_MAE")),
        "val_cell_MAE": val.get("cell_MAE"),
        "val_upstream_MAE": val.get("upstream_MAE"),
        "val_congestion_F1": val.get("congestion_F1"),
        "test_origin_cc_MAE@20": test.get("origin_cc_MAE@20", test.get("upstream_congestion_MAE")),
        "test_cell_MAE": test.get("cell_MAE"),
        "test_upstream_MAE": test.get("upstream_MAE"),
        "test_congestion_F1": test.get("congestion_F1"),
    }


def write_report(out_dir: Path, rows: list[dict], specs: list[RunSpec], dry_run: bool) -> None:
    lines: list[str] = []
    lines.append("# UNet Backbone Ablation\n")
    lines.append(f"Mode: {'dry-run' if dry_run else 'executed'}\n")
    lines.append("## Protocol\n")
    lines.append("- model window: 25 km upstream / 10 km downstream unless overridden inside trainers")
    lines.append("- metric/loss focus: origin_cc mask, upstream 0~+20 km, w=4")
    lines.append("- feature timing: t0")
    lines.append("- primary metric for this ablation: `origin_cc_MAE@20`; use cell/impact/recovery guards for final champion selection")
    lines.append("")
    lines.append("## Commands\n")
    for spec in specs:
        lines.append(f"- `{_quote_ps(spec.command)}`")
    lines.append("")
    lines.append("## Metrics\n")
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        header = [
            "arch", "run_id", "experiment", "backbone",
            "val_origin_cc_MAE@20", "val_cell_MAE", "val_congestion_F1",
            "test_origin_cc_MAE@20", "test_cell_MAE", "test_congestion_F1",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in ok_rows:
            vals = []
            for key in header:
                val = row.get(key, "")
                vals.append(f"{val:.3f}" if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
    else:
        lines.append("No metrics found yet.")
    lines.append("")
    out_dir.joinpath("report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_ts = args.run_ts or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = DEFAULT_OUTPUT_ROOT / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args, run_ts)
    commands_text = "\n".join(_quote_ps(spec.command) for spec in specs) + "\n"
    (out_dir / "commands.ps1").write_text(commands_text, encoding="utf-8")

    if args.dry_run:
        print(commands_text)
    else:
        for spec in specs:
            print(f"\n=== {spec.arch} | {spec.run_id} ===")
            print(_quote_ps(spec.command))
            subprocess.run(spec.command, cwd=str(PROJECT_ROOT), check=True)

    rows: list[dict] = []
    for spec in specs:
        for path in spec.metrics_paths:
            rows.append(flatten_metric(path, load_metric(path)))
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_report(out_dir, rows, specs, args.dry_run)
    print(f"[ablation] {out_dir}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
